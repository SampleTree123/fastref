"""Shared encoder utilities: local weights, tensor checks and atomic metadata writes.

No model loading or GPU work happens on import. Used by data preparation,
training and inference; this is not a stand-alone test script.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import prepare_datamake_media as media
from project_layout import ROOT, resolve_path

def sha_file(path):
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)

def weight_paths(record_path, verified_path):
    record = json.loads(resolve_path(record_path).read_text())
    verified = json.loads(resolve_path(verified_path).read_text())
    if verified.get("status") != "passed" or record["revision"] != verified["revision"]:
        raise ValueError("Native weights lack matching completed SHA verification")
    verified_files = {x["path"]: x for x in verified["files"]}
    paths = {}
    root = resolve_path(record["local_dir"]).resolve()
    for item in record["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("Missing/changed native weight file: " + str(path))
        evidence = verified_files.get(item["path"], {})
        if not evidence.get("sha256_verified") or any(evidence.get(k) != item[k] for k in ("bytes", "sha256")):
            raise ValueError("Weight digest differs from verified evidence")
        group = item["path"].split("/")[0]
        if group in paths:
            raise ValueError("Expected exactly one file per native component")
        paths[group] = path
    if set(paths) != {"diffusion_models", "text_encoders", "vae"}:
        raise ValueError("Expected transformer, packed text encoder, and video VAE")
    return record, paths

def shape_check(shape, frames, canvas, profile):
    expected = [1, *media.latent_shape(frames, *canvas, profile)["shape_cfhw"]]
    if list(shape) != expected:
        raise ValueError(f"Native VAE shape mismatch: actual={list(shape)}, expected={expected}")

def tensor_info(tensor, *, mask=False):
    import torch
    if not torch.isfinite(tensor).all().item():
        raise ValueError("Non-finite cached tensor")
    if tensor.requires_grad:
        raise ValueError("Encoding probe accidentally retained gradients")
    result = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
              "tensor_bytes": tensor.numel() * tensor.element_size(), "finite": True}
    if mask:
        if not ((tensor == 0) | (tensor == 1)).all().item():
            raise ValueError("Expected binary attention mask")
        result["valid_tokens"] = int(tensor.sum().item())
    return result
