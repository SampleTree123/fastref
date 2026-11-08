"""Native LTX editing for an arbitrary source video and current edit references.

Owner: scripts/inference/editing_tool_README.md. No target video, dataset index,
history label or reference-selection answer is accepted by this interface.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

from project_layout import ROOT, resolve_path


REFERENCE_TYPES = {"motion", "identity", "hair", "garment", "scene", "outfit", "camera"}
DEFAULT_NEGATIVE = ("blurry, out of focus, excessive noise, compression artifacts, flickering, "
                    "jittery movement, distorted proportions, deformed facial features, extra limbs, "
                    "disfigured hands, unnatural transitions, inconsistent lighting")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(canonical(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def normalize_request(raw):
    allowed = {"request_id", "source_video", "prompt", "references", "output_dir", "seed"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("Only current source, prompt and editing references are accepted")
    required = {"request_id", "source_video", "prompt", "references", "output_dir"}
    if not required <= raw.keys():
        raise ValueError("Missing required editing request fields")
    rid = raw["request_id"]
    if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", rid):
        raise ValueError("request_id must be a safe nonempty filename identifier")
    prompt = raw["prompt"]
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16000:
        raise ValueError("prompt must be nonempty text of at most16000characters")
    seed = raw.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative integer below2**63")
    def media_path(value):
        if not isinstance(value, str) or not Path(value).is_absolute() or not Path(value).is_file():
            raise ValueError(f"Media must be an existing absolute file path: {value}")
        return str(Path(value).resolve())
    source = media_path(raw["source_video"])
    output = raw["output_dir"]
    if not isinstance(output, str) or not Path(output).is_absolute():
        raise ValueError("output_dir must be absolute")
    if not isinstance(raw["references"], list):
        raise ValueError("references must be a list, possibly empty")
    refs, ids = [], set()
    for ref in raw["references"]:
        if not isinstance(ref, dict) or set(ref) - {"reference_id", "type", "path", "view"}:
            raise ValueError("Reference requires only reference_id/type/path/view")
        refid = ref.get("reference_id")
        if not isinstance(refid, (str, int)) or isinstance(refid, bool) or str(refid) in ids:
            raise ValueError("Reference IDs must be distinct strings or integers")
        if ref.get("type") not in REFERENCE_TYPES:
            raise ValueError("Unknown editing reference type")
        if not isinstance(ref.get("view", ""), str):
            raise ValueError("Reference view must be text")
        ids.add(str(refid))
        refs.append({"reference_id": refid, "type": ref["type"], "path": media_path(ref["path"]),
                     "view": ref.get("view", "")})
    return {"request_id": rid, "source_video": source, "prompt": prompt.strip(),
            "references": refs, "output_dir": str(Path(output).resolve()), "seed": seed}


def dynamic_video_profile(base, metadata):
    """Keep training sampling, accept original30fps or newly generated12fps RGB."""
    if min(metadata["width"], metadata["height"], metadata["frames"]) <= 0:
        raise ValueError("Video requires positive geometry and frame count")
    fps = Fraction(metadata["fps"])
    if fps <= 0 or fps.denominator != 1:
        raise ValueError("Current edited-chain video preprocessing requires integer constant FPS")
    p = deepcopy(base)
    p.update(source_dimensions=[[metadata["width"], metadata["height"]]],
             source_fps=int(fps), source_frames=metadata["frames"])
    return p


class NativeVideoEditor:
    def __init__(self, config, checkpoint, cache_dir, *, steps=40, ref_cfg=1.0):
        import torch
        import yaml
        from encoding_helpers import weight_paths, sha_file
        from native_reference_inference import load_inference_models
        from ltx_trainer.model_loader import load_text_encoder, load_embeddings_processor, embedding_weight_paths
        from ltx_pipelines.utils.blocks.video_decoder import VideoDecoder

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one allocated GPU per editing worker")
        self.device = torch.device("cuda:0")
        self.config_path = Path(config).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.cfg = yaml.safe_load(self.config_path.read_text())
        self.steps, self.ref_cfg = steps, ref_cfg
        if steps < 2 or ref_cfg < 1:
            raise ValueError("Require steps>=2 and referenceCFG>=1")
        record_path = resolve_path("runtime/models/native_model_revision.json")
        self.weights_record, self.weights = weight_paths(record_path, resolve_path("runtime/verification/native_weights_verified.json"))
        receipt = json.loads(self.checkpoint.with_suffix(".contract.json").read_text())
        self.checkpoint_sha = sha_file(self.checkpoint)
        expected = receipt["files"][self.checkpoint.name]
        if self.checkpoint_sha != expected["sha256"] or self.checkpoint.stat().st_size != expected["bytes"]:
            raise ValueError("Checkpoint differs from committed training receipt")
        identity = receipt["identity"]
        if (identity["config"]["lora"] != self.cfg["lora"] or
                identity["config"]["reference_protocol"] != self.cfg["reference_protocol"] or
                identity["record_sha256"]["models/native_model_revision.json"] != sha_file(record_path)):
            raise ValueError("Checkpoint and current model/LoRA/reference settings differ")
        self.profile = json.loads((ROOT / "configs/data/synthetic_original_resolution/media.json").read_text())
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hashes = {}
        self._shared_media = {}
        self._checked_artifacts = set()
        self._index_shared_encodings()
        self.identity = {
            "schema": "datamake_editing_tool_v1", "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha, "config_sha256": sha_file(self.config_path),
            "reference_protocol": self.cfg["reference_protocol"], "weights_sha256": sha_file(record_path),
            "profile": self.profile, "steps": steps, "cfg_scale": 3.0, "stg_scale": 1.0,
            "guidance_rescale": 0.7, "stg_blocks": [28], "ref_cfg": ref_cfg,
            "negative_prompt": DEFAULT_NEGATIVE, "runtime_sha256": sha_file(Path(__file__)),
            "generated_source_policy": "reencode_actual_generated_rgb_with_training_sampling",
            "text_policy": "official_Gemma_left_padding_no_automatic_enhancement"}
        self.identity_sha = digest(self.identity)
        self.model, self.processor, self.scale = load_inference_models(
            self.weights, self.cfg["lora"], checkpoint=self.checkpoint,
            protocol=self.cfg["reference_protocol"], device=self.device, ref_cfg=ref_cfg)
        # Keep the expensive model weights once in RAM. Move text-only components
        # onto the GPU for each new prompt, then return them to CPU before denoising.
        self.text_encoder = load_text_encoder(self.weights["text_encoders"], device="cpu", dtype=torch.bfloat16).eval().requires_grad_(False)
        text_processor = load_embeddings_processor(
            embedding_weight_paths(self.weights["diffusion_models"], self.weights["text_encoders"]),
            self.weights["text_encoders"], device="cpu", dtype=torch.bfloat16).eval().requires_grad_(False)
        self.feature_extractor = text_processor.feature_extractor
        text_processor.feature_extractor = None
        del text_processor
        self.vae = None
        self.decoder = VideoDecoder(str(self.weights["vae"]), dtype=torch.bfloat16, device=self.device)
        negative = self.encode_prompt(DEFAULT_NEGATIVE)
        from ltx_core.text_encoders.gemma import convert_to_additive_mask
        with torch.inference_mode():
            self.negative_context, _, mask = self.processor.create_embeddings(
                negative["video_prompt_embeds"].to(self.device), None,
                convert_to_additive_mask(negative["prompt_attention_mask"].to(self.device), torch.bfloat16))
        if mask is None or not bool((mask == 1).all()):
            raise ValueError("Invalid negative connector context")
        gc.collect()
        torch.cuda.empty_cache()

    def _index_shared_encodings(self):
        """Read only media receipts; never read supervised target sample tensors."""
        root = ROOT / "data/uedata/encodings"
        expected_vae = next(x["sha256"] for x in self.weights_record["files"] if x["path"].split("/")[0] == "vae")
        for suffix in ("scope_train_and_development.json", "scope_test.json"):
            path = root / "variants/outfit_camera_first_frame" / suffix
            if not path.exists():
                continue
            meta = json.loads(path.read_text())
            if meta.get("status") != "passed" or meta["encoding_identity"]["vae_sha256"] != expected_vae:
                raise ValueError("Shared VAE encoding receipt mismatch")
            if meta["profile"] != self.profile:
                raise ValueError("Shared media preprocessing profile differs")
            for item in meta["media"].values():
                if item["status"] != "passed":
                    raise ValueError("Shared media encoding incomplete")
                key = (item["media_sha256_verified"], tuple(item["canvas_wh"]), item["num_frames"])
                self._shared_media[key] = (root, item)

    def content_sha(self, path):
        from encoding_helpers import sha_file
        p = Path(path)
        st = p.stat()
        key = (str(p.resolve()), st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if key not in self._hashes:
            self._hashes[key] = sha_file(p)
        return self._hashes[key]

    def encode_prompt(self, prompt):
        import torch
        key = digest({"text": prompt, "model": self.identity["weights_sha256"], "policy": self.identity["text_policy"]})
        path = self.cache_dir / "text" / f"{key}.pt"
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
        self.text_encoder.to(self.device)
        self.feature_extractor.to(self.device)
        try:
            with torch.inference_mode():
                hidden, mask = self.text_encoder.encode([prompt])[0]
                features, _ = self.feature_extractor(hidden, mask, "left")
                if not bool(torch.isfinite(features).all()):
                    raise ValueError("Non-finite text features")
                result = {"video_prompt_embeds": features.cpu().contiguous(),
                          "prompt_attention_mask": mask.cpu().contiguous()}
                del hidden, mask, features
        finally:
            self.text_encoder.to("cpu")
            self.feature_extractor.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
        self._save_tensor(path, result)
        return result

    @staticmethod
    def _save_tensor(path, value):
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".pt", dir=path.parent)
        os.close(fd)
        try:
            torch.save(value, tmp)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def encode_media(self, path, *, video, canvas=None):
        import numpy as np
        import torch
        from PIL import Image
        import prepare_datamake_media as media
        from native_image_cache_contract import aligned_canvas, prepare_image
        from ltx_trainer.model_loader import load_video_vae_encoder
        if video:
            metadata = media.probe_video(path)
            profile = dynamic_video_profile(self.profile, metadata)
            actual_canvas = profile["video_canvas"]["landscape" if metadata["width"] > metadata["height"] else "portrait"]
            if canvas is not None and actual_canvas != canvas:
                raise ValueError("Motion reference orientation differs from selected source")
            canvas, frames = actual_canvas, 121
        else:
            with Image.open(path) as image:
                canvas = aligned_canvas(list(image.size))
            frames = 1
        source_sha = self.content_sha(path)
        shared = self._shared_media.get((source_sha, tuple(canvas), frames))
        if shared:
            from cache_integrity import verify_payload
            root, item = shared
            artifact = item["artifact"]
            payload_path = (root / artifact["path"]).resolve()
            if not payload_path.is_relative_to(root.resolve()):
                raise ValueError("Shared encoding escapes root")
            if str(payload_path) not in self._checked_artifacts:
                verify_payload(payload_path, artifact)
                self._checked_artifacts.add(str(payload_path))
            value = torch.load(payload_path, map_location="cpu", weights_only=True)
            self._check_latent(value, canvas, frames)
            return value, {"source_sha256": source_sha, "canvas_wh": canvas, "frames": frames,
                           "reuse": "existing_training_encoding", "artifact": str(payload_path)}
        key = digest({"source_sha": source_sha, "canvas": canvas, "frames": frames,
                      "model": self.identity["weights_sha256"], "profile": self.profile,
                      "policy": self.identity["generated_source_policy"]})
        cache = self.cache_dir / "media" / f"{key}.pt"
        if cache.exists():
            value = torch.load(cache, map_location="cpu", weights_only=True)
            self._check_latent(value, canvas, frames)
            return value, {"source_sha256": source_sha, "canvas_wh": canvas, "frames": frames,
                           "reuse": "editing_cache", "artifact": str(cache)}
        if video:
            images, pixel_info = media.load_video_frames(Path(path), canvas, profile)
        else:
            with Image.open(path) as image:
                image, pixel_info = prepare_image(image)
            images = [image]
        pixels = torch.from_numpy(np.stack([np.asarray(frame) for frame in images]))
        pixels = pixels.float().div_(255).sub_(0.5).div_(0.5).permute(3, 0, 1, 2).unsqueeze(0).to(self.device, torch.bfloat16)
        if self.vae is None:
            self.vae = load_video_vae_encoder(self.weights["vae"], device="cpu", dtype=torch.bfloat16).eval().requires_grad_(False)
        self.vae.to(self.device)
        try:
            with torch.inference_mode():
                latent = self.vae(pixels)
                value = {"latents": latent[0].cpu().contiguous(), "fps": 12.0,
                         "num_frames": latent.shape[2], "height": latent.shape[3], "width": latent.shape[4]}
                del latent
        finally:
            del pixels
            self.vae.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
        self._check_latent(value, canvas, frames)
        self._save_tensor(cache, value)
        return value, {"source_sha256": source_sha, "canvas_wh": canvas, "frames": frames,
                       "reuse": "new_encoding", "artifact": str(cache), "pixels": pixel_info}

    @staticmethod
    def _check_latent(value, canvas, frames):
        import torch
        expected = (128, (frames - 1) // 8 + 1, canvas[1] // 32, canvas[0] // 32)
        latent = value["latents"]
        if (tuple(latent.shape) != expected or latent.dtype != torch.bfloat16 or
                value["fps"] != 12 or not bool(torch.isfinite(latent).all()) or
                tuple(value[k] for k in ("num_frames", "height", "width")) != expected[1:]):
            raise ValueError("Cached latent geometry, precision, FPS or values differ")

    def generate(self, raw):
        import torch
        from ltx_pipelines.utils.media_io import encode_video
        from native_reference_inference import generate_latent
        request = normalize_request(raw)
        media_sha = [self.content_sha(request["source_video"])] + [self.content_sha(r["path"]) for r in request["references"]]
        execution_sha = digest({"request": request, "media_sha256": media_sha, "worker": self.identity_sha})
        output = Path(request["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        receipt = output / "editing_generation.json"
        video_path = output / "edited.mp4"
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old["execution_sha256"] != execution_sha:
                raise ValueError("Output already belongs to another editing request or runtime")
            if old.get("status") == "ok" and video_path.is_file():
                if self.content_sha(video_path) != old["video_sha256"]:
                    raise ValueError("Previously generated video changed")
                return {**old, "reused": True}
        elif video_path.exists():
            raise ValueError("Existing edited.mp4 has no committed provenance")
        began = time.monotonic()
        record = {"request_id": request["request_id"], "status": "running", "request": request,
                  "request_sha256": digest(raw), "execution_sha256": execution_sha,
                  "worker_identity_sha256": self.identity_sha, "video_path": str(video_path),
                  "input_media_sha256": media_sha, "started_unix": time.time()}
        atomic_json(receipt, record)
        source, source_audit = self.encode_media(request["source_video"], video=True)
        references, ref_audit = [], []
        for i, ref in enumerate(request["references"]):
            latent, audit = self.encode_media(ref["path"], video=ref["type"] == "motion", canvas=source_audit["canvas_wh"])
            references.append({"index": i, "type": ref["type"], "view": ref["view"], "latent": latent})
            ref_audit.append({"reference_id": ref["reference_id"], **audit})
        conditions = self.encode_prompt(request["prompt"])
        encode_seconds = time.monotonic() - began
        # Explicit B=1 assembly, matching collate_single_sample without target fields.
        def gpu_latent(raw_latent):
            return {**raw_latent, "latents": raw_latent["latents"].unsqueeze(0).to(self.device)}
        source = gpu_latent(source)
        references = [{**ref, "latent": gpu_latent(ref["latent"])} for ref in references]
        conditions = {k: v.to(self.device) for k, v in conditions.items()}
        torch.cuda.reset_peak_memory_stats()
        step_start = time.monotonic()
        latent = generate_latent(self.model, self.processor, source=source, references=references,
            conditions=conditions, scale_factors=self.scale, protocol=self.cfg["reference_protocol"],
            seed=request["seed"], steps=self.steps, negative_context=self.negative_context,
            cfg_scale=3.0, stg_scale=1.0, guidance_rescale=0.7, stg_blocks=(28,),
            guidance_batch_size=1, ref_cfg=self.ref_cfg)
        torch.cuda.synchronize()
        denoise_seconds = time.monotonic() - step_start
        denoise_peak = torch.cuda.max_memory_allocated()
        # Releasing conditioning tensors before decode bounds peak memory.
        del source, references, conditions
        decode_start = time.monotonic()
        with torch.inference_mode():
            frames = self.decoder(latent, generator=torch.Generator(device=self.device).manual_seed(request["seed"]))
            partial = output / "edited.partial.mp4"
            encode_video(frames, fps=12, audio=None, output_path=str(partial), video_chunks_number=1, thread_count=2)
        del latent, frames
        os.replace(partial, video_path)
        record.update(status="ok", video_sha256=self.content_sha(video_path), bytes=video_path.stat().st_size,
                      source_encoding=source_audit, reference_encodings=ref_audit,
                      timings={"encoding_seconds": encode_seconds, "denoise_seconds": denoise_seconds,
                               "decode_seconds": time.monotonic() - decode_start,
                               "total_seconds": time.monotonic() - began},
                      denoise_peak_allocated_bytes=denoise_peak,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), finished_unix=time.time())
        atomic_json(receipt, record)
        return record
