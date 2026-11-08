#!/usr/bin/env python3
"""Original BF16 Qwen evaluation, one independent GPU replica per shard.

No training, no current target, no context truncation, no profile fallback.
Resume requires full run identity and matching input row. Error rows retry.
"""
from __future__ import annotations

import argparse
import fcntl
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

from evaluation_inputs_actual_id import score_prediction, validate_row
from media_inputs import MediaCache, atomic_json, canonical, prepare_inputs, sha_file

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_identity(path, content=False):
    path = Path(path)
    stat = path.stat()
    result = {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
              "ctime_ns": stat.st_ctime_ns}
    if content:
        result["sha256"] = sha_file(path)
    return result


def safe_id(sample_id):
    # Retain ordinary human-readable IDs; append digest only after replacement.
    value = str(sample_id)
    clean = re.sub(r"[^a-zA-Z0-9_.-]", "_", value)
    if clean in ("", ".", "..") or clean != value or len(clean) > 180:
        clean = clean[:150] + "-" + digest(value)[:16]
    return clean


def describe_callable(function):
    """Follow official kernels nn.Module/forward and fallback closure wrappers."""
    import torch
    seen = set()
    for _ in range(12):
        if isinstance(function, torch.nn.Module):
            function = function.forward
        if inspect.ismethod(function):
            function = function.__func__
        if id(function) in seen or not inspect.isfunction(function):
            break
        seen.add(id(function))
        closure = inspect.getclosurevars(function).nonlocals
        inner = closure.get("implementation", closure.get("func"))
        if inner is None:
            break
        function = inner
    source = inspect.getsourcefile(function)
    return {"module": function.__module__, "qualname": function.__qualname__,
            "source_sha256": sha_file(source) if source else None}


def resolved_kernels():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    return {name: describe_callable(getattr(modeling, name)) for name in
            ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
             "causal_conv1d_fn", "causal_conv1d_update")}


def require_expected_run(output, fingerprint, identity):
    """Pin an output directory once, across replicas and allocation resumes."""
    expected_path = output / "expected_run.json"
    expected = {"run_fingerprint": fingerprint, "identity": identity}
    with (output / ".expected_run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if expected_path.exists():
            actual = json.loads(expected_path.read_text())
            if actual != expected:
                raise ValueError("Output expected_run.json differs from this complete run identity; "
                                 "use the original frozen inputs/config/code or a new output directory")
        else:
            atomic_json(expected_path, expected)


def activate_kernels(model, config):
    """Official partial mapping: FLA only; unavailable Mamba stays torch."""
    mode = config.get("kernel_mode", "default")
    if mode == "default":
        return
    if mode != "fla":
        raise ValueError(f"Unknown kernel_mode: {mode}")
    from kernels import LayerRepository, Mode, kernelize, use_kernel_mapping
    mapping = {name: {"cuda": {Mode.INFERENCE: LayerRepository(
        repo_id="kernels-community/fla", revision=config["fla_kernel_revision"], layer_name=name)}}
        for name in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule")}
    with use_kernel_mapping(mapping, inherit_mapping=False):
        kernelize(model, mode=Mode.INFERENCE, device="cuda", use_fallback=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/history_selection_v3/eval.jsonl")
    parser.add_argument("--profile", "--config", dest="profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen3.8-27B")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache/evaluation_rgb")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument("--prepare-only", action="store_true", help="CPU processor smoke; never loads model")
    parser.add_argument("--force", action="store_true", help="Rerun matching successful rows")
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        parser.error("Require 0 <= rank < world-size")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    return args


def identities(args, rows, config):
    import transformers
    import torch

    local_code = Path(__file__).parent
    code = {name: sha_file(local_code / name) for name in
            ("run_model_eval_actual_id.py", "media_inputs.py", "evaluation_inputs.py", "evaluation_inputs_actual_id.py")}
    implementation = {}
    site = Path(transformers.__file__).parent
    for name in ("models/qwen3_5/modeling_qwen3_5.py", "models/qwen3_vl/processing_qwen3_vl.py",
                 "models/qwen3_vl/video_processing_qwen3_vl.py", "models/qwen2_vl/image_processing_qwen2_vl.py",
                 "generation/utils.py"):
        implementation[name] = sha_file(site / name)
    model_files = [file_identity(path, content=path.suffix != ".safetensors") for path in
                   sorted(args.model.iterdir()) if path.is_file() and path.suffix in
                   (".json", ".jinja", ".safetensors", ".txt")]
    # Existing verified content manifest + live stat identity avoids hashing 55GB
    # independently in every replica. Any touched/replaced weight invalidates resume.
    weights_report = ROOT / "reports/weights_verified.json"
    model = {"files": model_files,
             "weights_verified_sha256": sha_file(weights_report) if weights_report.exists() else None,
             "revision_manifest_sha256": sha_file(ROOT / "configs/model_revision.json")}
    paths = set()
    for row in rows:
        for message in row["model_input"]["messages"]:
            if isinstance(message.get("content"), list):
                paths.update(block["path"] for block in message["content"] if block.get("type") in ("video", "image"))
    media_manifest = [file_identity(path) for path in sorted(paths)]
    packages = {}
    for name in ("torch", "torchvision", "transformers", "av", "numpy", "kernels", "flash-attn", "flash-linear-attention", "fla-core", "einops", "causal-conv1d"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    prompt = ROOT / "configs/evaluation/actual_id/task_prompt.txt"
    identity = {"schema_version": 1, "config": config, "config_sha256": sha_file(args.profile),
                "dataset_sha256": sha_file(args.dataset), "dataset_rows": len(rows), "code": code,
                "prompt_sha256": sha_file(prompt), "model_identity": model,
                "media_stat_manifest_sha256": digest(media_manifest),
                "implementation_sha256": implementation, "packages": packages,
                "resolved_kernel_functions_before_model": resolved_kernels(),
                "prepare_only": args.prepare_only, "torch_cuda_version": torch.version.cuda,
                "use_hub_kernels": os.environ.get("USE_HUB_KERNELS", ""),
                "deterministic_generation": True, "compiler_cache_policy": "slurm_job_rank_v1"}
    manifest = ROOT / "configs/evaluation/kernel_environment.json"
    identity["kernel_environment_sha256"] = sha_file(manifest) if manifest.exists() else None
    environment_lock = ROOT / "configs/evaluation/environment.lock"
    identity["environment_lock_sha256"] = sha_file(environment_lock) if environment_lock.exists() else None
    identity["gpu_wrapper_sha256"] = sha_file(ROOT / "scripts/run_gpu_python.sh")
    return identity


def main():
    args = parse_args()
    config = json.loads(args.profile.read_text())
    if config.get("selection_contract") != "actual_id_v1":
        raise ValueError("Require actual_id_v1 profile")
    # Set both compiler caches before importing torch, transformers or Triton.
    # Concurrent Triton writers on scratch produced a missing .ttir artifact.
    # Job/rank namespaces prevent inter-replica and inter-allocation races;
    # low/high stages within one job/rank may reuse these compilation artifacts.
    compiler_root = ROOT / "cache/compilation_actual_id" / ("job_" + safe_id(os.environ.get("SLURM_JOB_ID", "local_" + str(os.getpid())))) / ("rank_" + str(args.rank))
    os.environ["TRITON_CACHE_DIR"] = str(compiler_root / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compiler_root / "inductor")
    threads = config.get("cpu_threads", 2)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    if config["dtype"] != "bfloat16" or config["generation"]["do_sample"]:
        raise ValueError("This evaluator requires original BF16 deterministic inference")
    rows = [json.loads(line) for line in args.dataset.open() if line.strip()]
    for row in rows:
        validate_row(row)
    identifiers = [str(row["sample_id"]) for row in rows]
    if len(set(identifiers)) != len(rows):
        raise ValueError("Dataset sample IDs must be unique")
    if args.sample_id is not None and args.sample_id not in identifiers:
        raise ValueError(f"Unknown sample ID: {args.sample_id}")
    args.output.mkdir(parents=True, exist_ok=True)
    identity = identities(args, rows, config)
    fingerprint = digest(identity)
    require_expected_run(args.output, fingerprint, identity)
    atomic_json(args.output / "runs" / f"{fingerprint}.json", identity)
    selected = [(index, row) for index, row in enumerate(rows)
                if index % args.world_size == args.rank and
                (args.sample_id is None or str(row["sample_id"]) == args.sample_id)]
    if args.limit:
        selected = selected[:args.limit]
    existing = 0
    work = []
    for index, row in selected:
        path = args.output / "records" / f"{safe_id(row['sample_id'])}.json"
        row_digest = digest(row)
        if path.exists() and not args.force:
            try:
                record = json.loads(path.read_text())
                if (record.get("run_fingerprint") == fingerprint and record.get("row_sha256") == row_digest
                        and record.get("status") == ("prepared" if args.prepare_only else "ok")):
                    existing += 1
                    continue
            except (ValueError, OSError):
                pass
        work.append((index, row, path, row_digest))
    print(canonical({"event": "start", "rank": args.rank, "world_size": args.world_size,
                     "selected": len(selected), "existing": existing, "pending": len(work),
                     "run_fingerprint": fingerprint, "profile": config["name"],
                     "triton_cache_dir": os.environ["TRITON_CACHE_DIR"],
                     "torchinductor_cache_dir": os.environ["TORCHINDUCTOR_CACHE_DIR"]}), flush=True)
    if not work:
        return
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, token=False)
    cache = MediaCache(args.cache, config["profile"], config.get("cache_max_gib", 64),
                       config.get("cache_minimum_free_gib", 30))
    model = None
    load_seconds = None
    if not args.prepare_only:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable: launch with scripts/run_gpu_python.sh inside allocated srun")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("One visible GPU per replica required; assign CUDA_VISIBLE_DEVICES before launch")
        start = time.perf_counter()
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True, token=False, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation=config["attention_implementation"],
            use_kernels=config.get("use_kernels", False))
        model.eval()
        activate_kernels(model, config)
        load_seconds = time.perf_counter() - start
        print(canonical({"event": "model_loaded", "seconds": load_seconds,
                         "gpu": torch.cuda.get_device_name(0), "allocated_bytes": torch.cuda.memory_allocated(),
                         "resolved_kernel_functions": resolved_kernels()}), flush=True)
    errors = completed = 0
    began = time.perf_counter()
    for index, row, path, row_digest in work:
        start = time.perf_counter()
        record = {"schema_version": 1, "sample_id": row["sample_id"], "dataset_index": index,
                  "run_fingerprint": fingerprint, "row_sha256": row_digest, "rank": args.rank,
                  "world_size": args.world_size, "profile": config["name"], "audit": row["audit"],
                  "supervision": row["supervision"], "raw_output": None, "scores": None,
                  "generated_tokens": None, "generated_token_ids": None,
                  "created_at": datetime.now(timezone.utc).isoformat(), "model_load_seconds": load_seconds}
        if model is not None:
            record["resolved_kernel_functions"] = resolved_kernels()
        inputs = generated = None
        if model is not None:
            torch.cuda.reset_peak_memory_stats()
        try:
            inputs, visual_audit = prepare_inputs(row["model_input"], processor, cache)
            record.update(visual_audit)
            record["prepare_seconds"] = time.perf_counter() - start
            if inputs["input_ids"].shape[-1] + config["generation"]["max_new_tokens"] > 262144:
                raise ValueError("Full input exceeds native context; truncation prohibited")
            if args.prepare_only:
                record["status"] = "prepared"
            else:
                inputs = inputs.to("cuda:0")
                torch.cuda.synchronize()
                generation_start = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**inputs, **config["generation"])
                torch.cuda.synchronize()
                record["generation_seconds"] = time.perf_counter() - generation_start
                token_ids = generated[0, inputs["input_ids"].shape[-1]:].tolist()
                raw = processor.tokenizer.decode(token_ids, skip_special_tokens=False)
                record.update(status="ok", raw_output=raw, generated_tokens=len(token_ids),
                              generated_token_ids=token_ids, scores=score_prediction(raw, row),
                              hit_generation_limit=len(token_ids) >= config["generation"]["max_new_tokens"])
            completed += 1
        except Exception as exc:
            errors += 1
            record.update(status="error", error_type=type(exc).__name__, error=str(exc),
                          traceback=traceback.format_exc(), oom=isinstance(exc, torch.OutOfMemoryError),
                          scores={"valid": False, "strict_correct": False, "resolved_correct": False,
                                  "error": "runtime_error"})
        finally:
            record["total_seconds"] = time.perf_counter() - start
            if model is not None:
                record.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                              gpu_name=torch.cuda.get_device_name(0))
            atomic_json(path, record)
            print(canonical({"event": "row", "sample_id": row["sample_id"], "rank": args.rank,
                             "status": record["status"], "seconds": record["total_seconds"],
                             "input_tokens": record.get("input_tokens"), "scores": record["scores"],
                             "error": record.get("error")}), flush=True)
            del inputs, generated
            if model is not None and record["status"] == "error":
                import gc
                gc.collect()
                torch.cuda.empty_cache()
    atomic_json(args.output / "shards" / f"{fingerprint}.rank-{args.rank}.json",
                {"run_fingerprint": fingerprint, "rank": args.rank, "world_size": args.world_size,
                 "selected": len(selected), "existing": existing, "completed": completed,
                 "errors": errors, "seconds": time.perf_counter() - began})
    if errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
