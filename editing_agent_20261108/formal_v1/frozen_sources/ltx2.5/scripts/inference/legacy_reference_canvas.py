"""Restore the 2026-11-01 image preprocessing for V3 checkpoint inference only.

Source/action videos and text use current verified caches. Only reference RGB
images are re-encoded using the original contain_image + official VAE path.
"""
from pathlib import Path
import json


def validate_legacy_profile(protocol, profile):
    if protocol != "reference_positions_v3_shared_canvas":
        raise ValueError("Legacy image restoration requires the historical V3 checkpoint")
    if profile.get("reference_protocol") != "reference_positions_v5_native_images":
        raise ValueError("Legacy restoration expects the verified original-resolution cache")
    expected_frames = {"ue": 121, "coinve": 57}[profile["source_name"]]
    if (profile["fps"] != 12 or profile["num_frames"] != expected_frames
            or profile["video_canvas"] != {"landscape": [768,448], "portrait": [448,768]}
            or profile["resize"] != "contain_letterbox" or profile["interpolation"] != "PIL_LANCZOS"
            or profile["padding_rgb"] != [0,0,0]
            or profile["vae"]["spatial_factor"] != 32):
        raise ValueError("Source video preprocessing is incompatible with historical V3")


def restore_images(bundle, sample_splits, datasets, locations, weights, weights_signature, cache):
    import gc
    import numpy as np
    import torch
    from PIL import Image
    from native_training_contract import sha_file, fingerprint
    from prepare_datamake_media import contain_image, verify_media
    from encoding_helpers import atomic_json
    from checkpoint_retention import retention_lock
    from ltx_trainer.model_loader import load_video_vae_encoder
    from project_layout import resolve_path

    spec = json.loads(Path(bundle).read_text())
    rows = {}
    for source in spec["sources"]:
        for split in set(sample_splits.values()):
            for line in resolve_path(source[split + "_manifest"]).read_text().splitlines():
                row = json.loads(line)
                uid = source["name"] + ":" + str(row["sample_id"])
                if sample_splits.get(uid) == split:
                    rows[uid] = (row, resolve_path(source["dataset_root"]))
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    restored, audit, encoder = {}, [], None
    try:
        for uid, (split, index) in locations.items():
            sample = datasets[split][index]
            row, dataset_root = rows[uid]
            canvas = [sample["source"]["width"] * 32, sample["source"]["height"] * 32]
            replacements = {}
            for i, ref in enumerate(row["references"]):
                if Path(ref["path"]).suffix.lower() in (".mp4", ".mov", ".webm"):
                    continue
                image_path = verify_media(ref, dataset_root)
                identity = {"image_sha256": ref["sha256"], "canvas_wh": canvas,
                            "weights_signature": weights_signature,
                            "method": "source_canvas_contain_black_padding_before_vae_v1"}
                key = fingerprint(identity)
                target, receipt = cache / (key + ".pt"), cache / (key + ".json")
                with retention_lock(target):
                    if receipt.exists():
                        record = json.loads(receipt.read_text())
                        if record["identity"] != identity or sha_file(target) != record["sha256"]:
                            raise ValueError("Legacy reference cache changed")
                    else:
                        with Image.open(image_path) as image:
                            image, geometry = contain_image(image, canvas, {"padding_rgb": [0,0,0]})
                        if encoder is None:
                            encoder = load_video_vae_encoder(weights["vae"], device="cuda", dtype=torch.bfloat16)
                            encoder.eval().requires_grad_(False)
                        pixels = torch.from_numpy(np.asarray(image).copy()).float().div_(255).sub_(0.5).div_(0.5)
                        pixels = pixels.permute(2,0,1).unsqueeze(0).unsqueeze(2).to("cuda", torch.bfloat16)
                        with torch.inference_mode():
                            latent = encoder(pixels)
                        if list(latent.shape) != [1,128,1,canvas[1]//32,canvas[0]//32] or not torch.isfinite(latent).all():
                            raise ValueError("Invalid historical reference latent")
                        payload = {"latents": latent[0].cpu().contiguous(), "num_frames": 1,
                                   "height": canvas[1]//32, "width": canvas[0]//32, "fps": 12.0}
                        tmp = target.with_suffix(".pt.tmp"); torch.save(payload,tmp); tmp.replace(target)
                        record = {"identity": identity, "geometry": geometry, "sha256": sha_file(target)}
                        atomic_json(receipt, record)
                        del pixels, latent, payload
                    replacements[i] = torch.load(target, map_location="cpu", weights_only=True)
                audit.append({"uid": uid, "reference_index": i, "cache": str(target), **record})
            restored[uid] = replacements
    finally:
        del encoder
        gc.collect(); torch.cuda.empty_cache()
    return restored, audit
