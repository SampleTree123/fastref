"""Local, frozen Qwen backend with the previously validated high input profile.

No Qwen-Agent dependency installation: reuse official Transformers media/chat
templates and our validated FLA setup, retaining the actual model weights.
"""
from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / "scripts/evaluation"
if str(EVALUATION) not in sys.path:
    sys.path.insert(0, str(EVALUATION))
from contracts import digest, parse_tool_calls
from media_inputs import MediaCache, atomic_json, prepare_inputs, sha_file


class QwenBackend:
    def __init__(self, config=None, *, rank=0, model=None):
        self.config = json.loads((ROOT / "configs/evaluation/actual_id/high.json").read_text())
        self.config["generation"]["max_new_tokens"] = 1024
        # Do not evict the useful previous experiment's media just to start this run.
        self.config["cache_max_gib"] = 96
        if config:
            for key, value in config.items():
                if isinstance(value, dict) and isinstance(self.config.get(key), dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
        compiler_root = ROOT / "cache/compilation_editing_agent" / (
            "job_" + os.environ.get("SLURM_JOB_ID", "local_" + str(os.getpid()))) / f"rank_{rank}"
        os.environ["TRITON_CACHE_DIR"] = str(compiler_root / "triton")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compiler_root / "inductor")
        threads = self.config.get("cpu_threads", 2)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        from run_model_eval_actual_id import activate_kernels
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one allocated GPU to each Qwen backend")
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        if self.config["dtype"] != "bfloat16" or self.config["generation"].get("do_sample"):
            raise ValueError("Use original BF16 and deterministic generation")
        self.model_path = Path(model or ROOT / "models/Qwen3.8-27B")
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True, token=False)
        self.cache = MediaCache(ROOT / "cache/evaluation_rgb", self.config["profile"],
                                self.config["cache_max_gib"], self.config["cache_minimum_free_gib"])
        start = time.monotonic()
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_path, local_files_only=True, token=False, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation=self.config["attention_implementation"],
            use_kernels=self.config.get("use_kernels", False))
        self.model.eval().requires_grad_(False)
        activate_kernels(self.model, self.config)
        self.identity = {"config": self.config, "model": str(self.model_path.resolve()),
                         "model_revision_sha256": sha_file(ROOT / "configs/model_revision.json"),
                         "weights_verified_sha256": sha_file(ROOT / "reports/weights_verified.json"),
                         "backend_sha256": sha_file(__file__),
                         "media_inputs_sha256": sha_file(EVALUATION / "media_inputs.py")}
        print(json.dumps({"event": "qwen_backend_loaded", "seconds": time.monotonic() - start,
                          "gpu": torch.cuda.get_device_name(0), "identity": digest(self.identity)}), flush=True)

    def call(self, messages, tools, output_path=None):
        import torch
        start = time.monotonic()
        model_input = {"messages": deepcopy(messages), "tools": deepcopy(tools),
                       "enable_thinking": False, "add_vision_id": False}
        has_media = any(block.get("type") in ("image", "video") for message in messages
                        if isinstance(message.get("content"), list) for block in message["content"])
        if has_media:
            inputs, audit = prepare_inputs(model_input, self.processor, self.cache)
        else:
            text = self.processor.apply_chat_template(messages, tools=tools, tokenize=False,
                add_generation_prompt=True, enable_thinking=False, add_vision_id=False)
            inputs = self.processor(text=text, return_tensors="pt", truncation=False)
            audit = {"media": [], "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "input_tokens": inputs["input_ids"].numel()}
        generation = self.config["generation"]
        if inputs["input_ids"].shape[-1] + generation["max_new_tokens"] > 262144:
            raise ValueError("Full input exceeds native context; no silent truncation")
        prepare_seconds = time.monotonic() - start
        inputs = inputs.to("cuda:0")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        began = time.monotonic()
        try:
            with torch.inference_mode():
                generated = self.model.generate(**inputs, **generation)
            torch.cuda.synchronize()
            token_ids = generated[0, inputs["input_ids"].shape[-1]:].tolist()
            raw = self.processor.tokenizer.decode(token_ids, skip_special_tokens=False)
            result = {"raw_output": raw, "tool_calls": [], "parse_error": None,
                      "generated_tokens": len(token_ids),
                      "hit_generation_limit": len(token_ids) >= generation["max_new_tokens"],
                      "prepare_seconds": prepare_seconds, "generation_seconds": time.monotonic() - began,
                      "total_seconds": time.monotonic() - start,
                      "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(), **audit}
            try:
                result["tool_calls"] = parse_tool_calls(raw, tools)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                result["parse_error"] = str(exc)
            if output_path:
                atomic_json(output_path, {"messages": messages, "tools": tools, **result})
            return result
        finally:
            del inputs
            if "generated" in locals():
                del generated

    def close(self):
        del self.model
        gc.collect()
        import torch
        torch.cuda.empty_cache()
