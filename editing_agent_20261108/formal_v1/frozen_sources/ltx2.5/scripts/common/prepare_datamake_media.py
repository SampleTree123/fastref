#!/usr/bin/env python3
"""Bounded CPU preprocessing probe; originals are immutable, MP4s are previews only.

The reusable load_video_frames/contain_image helpers return RGB PIL images suitable
for the OFFICIAL video processor. Do not reimplement VAE or latent normalization.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

IMPLEMENTATION_VERSION = "datamake_media_v1.0"
ROOT = Path(__file__).resolve().parents[1]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_profile(path):
    profile = json.loads(Path(path).read_text())
    if profile["num_frames"] != profile["duration_seconds"] * profile["fps"] + 1:
        raise ValueError("num_frames must include timestamps 0 and duration")
    if profile["resize"] != "contain_letterbox" or profile["sampling"] != "nearest_timestamp_ties_up":
        raise ValueError("Unsupported preprocessing policy")
    if profile["interpolation"] != "PIL_LANCZOS" or profile["endpoint"] != "hold_last_frame_at_10_seconds":
        raise ValueError("Unsupported interpolation/endpoint policy")
    if profile["scene_canvas"] != "same_as_source_video":
        raise ValueError("Unsupported scene canvas")
    for canvas in [*profile["video_canvas"].values(), profile["reference_image_canvas"]]:
        latent_shape(profile["num_frames"], *canvas, profile)
    return profile


def temporal_indices(source_fps, source_frames, target_fps, target_frames):
    """Nearest source frame at common absolute timestamps, ties rounded up.

    Permit only the one-frame endpoint extension, never silently repeat a short
    clip for the remainder of a ten-second sample.
    """
    source_fps, target_fps = Fraction(str(source_fps)), Fraction(str(target_fps))
    if min(source_fps, target_fps) <= 0 or min(source_frames, target_frames) <= 0:
        raise ValueError("FPS and frame counts must be positive")
    if Fraction(target_frames - 1, 1) / target_fps > Fraction(source_frames, 1) / source_fps:
        raise ValueError("Source is too short; refusing long endpoint repetition")
    result = []
    for index in range(target_frames):
        position = Fraction(index, 1) * source_fps / target_fps
        nearest = (2 * position.numerator + position.denominator) // (2 * position.denominator)
        result.append(min(nearest, source_frames - 1))
    return result


def contain_geometry(width, height, canvas_width, canvas_height):
    if min(width, height, canvas_width, canvas_height) <= 0:
        raise ValueError("Dimensions must be positive")
    scale = min(Fraction(canvas_width, width), Fraction(canvas_height, height))
    new_width = max(1, min(canvas_width, round(width * scale)))
    new_height = max(1, min(canvas_height, round(height * scale)))
    left, top = (canvas_width - new_width) // 2, (canvas_height - new_height) // 2
    return {"resized_width": new_width, "resized_height": new_height,
            "padding_ltrb": [left, top, canvas_width - new_width - left, canvas_height - new_height - top]}


def contain_image(image, canvas, profile):
    from PIL import Image
    geometry = contain_geometry(*image.size, *canvas)
    resized = image.convert("RGB").resize((geometry["resized_width"], geometry["resized_height"]), Image.Resampling.LANCZOS)
    output = Image.new("RGB", tuple(canvas), tuple(profile["padding_rgb"]))
    output.paste(resized, tuple(geometry["padding_ltrb"][:2]))
    return output, geometry


def latent_shape(frames, width, height, profile):
    vae = profile["vae"]
    temporal, spatial = vae["temporal_factor"], vae["spatial_factor"]
    if frames <= 0 or width <= 0 or height <= 0 or (frames - 1) % temporal or width % spatial or height % spatial:
        raise ValueError("Shape is not VAE aligned")
    latent = [vae["channels"], (frames - 1) // temporal + 1, height // spatial, width // spatial]
    if latent[1] % vae["patch_size_t"] or latent[2] % vae["patch_size"] or latent[3] % vae["patch_size"]:
        raise ValueError("Shape is not patch aligned")
    tokens = latent[1] * latent[2] * latent[3] // (vae["patch_size_t"] * vae["patch_size"] ** 2)
    return {"shape_cfhw": latent, "tokens": tokens}


def processing_contract():
    import av
    import PIL
    return {"implementation": IMPLEMENTATION_VERSION,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "pillow_version": PIL.__version__, "pyav_version": av.__version__,
            "libav_versions": {name: list(version) for name, version in sorted(av.library_versions.items())}}


def media_cache_key(sha256, profile, canvas, *, toolchain):
    # This is a pixel-preparation key. A future VAE key MUST additionally include
    # encoder weights/revision, encoding mode, tiling, dtype, and normalization.
    if not toolchain:
        raise ValueError("Pixel cache identity requires the actual preprocessing toolchain")
    return fingerprint({"toolchain": toolchain, "media_sha256": sha256,
                        "profile": profile, "canvas": list(canvas)})


def verify_media(record, dataset_root):
    root = dataset_root.resolve()
    relative = Path(record["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
        raise ValueError("Media escapes dataset root")
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise ValueError(f"Missing/changed media: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != record["sha256"]:
        raise ValueError(f"Media hash mismatch: {path}")
    return path


def probe_video(path):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return {"width": stream.width, "height": stream.height,
                "fps": str(stream.average_rate), "frames": stream.frames}


def video_canvas(metadata, profile):
    if [metadata["width"], metadata["height"]] not in profile["source_dimensions"]:
        raise ValueError(f"Unexpected source dimensions: {metadata}")
    if Fraction(metadata["fps"]) != profile["source_fps"] or metadata["frames"] != profile["source_frames"]:
        raise ValueError(f"Unexpected source duration/FPS: {metadata}")
    return profile["video_canvas"]["landscape" if metadata["width"] > metadata["height"] else "portrait"]


def validate_video_pair(source, target, profile):
    canvas = video_canvas(source, profile)
    video_canvas(target, profile)
    if source != target:
        raise ValueError("Source and target must share dimensions, frame count, and FPS")
    return canvas


def load_video_frames(path, canvas, profile):
    import av
    metadata = probe_video(path)
    video_canvas(metadata, profile)
    indices = temporal_indices(metadata["fps"], metadata["frames"], profile["fps"], profile["num_frames"])
    selected = set(indices)
    images = {}
    count, first_pts = 0, None
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            count += 1
            if frame.pts is None:
                raise ValueError("Cannot validate frame timing without PTS")
            timestamp = Fraction(frame.pts) * frame.time_base
            if first_pts is None:
                first_pts = timestamp
            expected = Fraction(index, profile["source_fps"])
            if abs(timestamp - first_pts - expected) > Fraction(1, 10000):
                raise ValueError("Variable or irregular frame timing; require explicit timestamp handling")
            if index in selected:
                images[index], geometry = contain_image(frame.to_image(), canvas, profile)
    if count != metadata["frames"]:
        raise ValueError("Decoded frame count disagrees with container")
    return [images[index] for index in indices], {
        "original": metadata, "indices": indices,
        "timestamps_seconds": [i / profile["fps"] for i in range(profile["num_frames"])],
        "hold_last_at_endpoint": indices[-1] == metadata["frames"] - 1,
        "geometry": geometry, "vae_candidate": latent_shape(len(indices), *canvas, profile)}


def write_preview(frames, path, fps):
    import av
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = frames[0].size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "fast"}
        for image in frames:
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def prepare_sample(sample, dataset_root, output_root, profile, export_mp4=False):
    from PIL import Image
    toolchain = processing_contract()
    source_path = verify_media(sample["source"], dataset_root)
    target_path = verify_media(sample["target"], dataset_root)
    source_metadata = probe_video(source_path)
    canvas = validate_video_pair(source_metadata, probe_video(target_path), profile)
    output = output_root / str(sample["sample_id"])
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "datamake_media_probe_v1", "sample_id": sample["sample_id"],
              "profile_sha256": fingerprint(profile), "implementation": IMPLEMENTATION_VERSION,
              "toolchain": toolchain,
              "scope": "CPU pixel preprocessing and dimensions only; no VAE, training, or quality acceptance",
              "mp4_usage": "Preview only: do not train from recompressed MP4", "media": []}
    items = [("source", sample["source"]), ("target", sample["target"])]
    items.extend((f"reference_{i:02d}_{ref['type']}", ref) for i, ref in enumerate(sample["references"]))
    for role, record in items:
        path = source_path if role == "source" else target_path if role == "target" else verify_media(record, dataset_root)
        is_video = role in ("source", "target") or record.get("type") == "motion"
        target_canvas = canvas if is_video or record.get("type") == "scene" else profile["reference_image_canvas"]
        if is_video:
            if record.get("type") == "motion":
                validate_video_pair(source_metadata, probe_video(path), profile)
            frames, details = load_video_frames(path, target_canvas, profile)
        else:
            with Image.open(path) as original:
                original_size = list(original.size)
                if record.get("type") == "scene" and original.size != (source_metadata["width"], source_metadata["height"]):
                    raise ValueError("Scene reference must match source aspect/dimensions in current dataset")
                frame, geometry = contain_image(original, target_canvas, profile)
            frames = [frame]
            details = {"original_dimensions": original_size, "geometry": geometry,
                       "vae_candidate": latent_shape(1, *target_canvas, profile)}
        for index in sorted({0, len(frames) // 2, len(frames) - 1}):
            frames[index].save(output / f"{role}_{index:03d}.png")
        if export_mp4 and is_video:
            write_preview(frames, output / f"{role}_preview.mp4", profile["fps"])
        report["media"].append({"role": role, "original_path": record["path"],
                                "original_sha256_verified": record["sha256"], "canvas_wh": target_canvas,
                                "pixel_cache_key": media_cache_key(record["sha256"], profile, target_canvas, toolchain=toolchain), **details})
    report["total_candidate_tokens"] = sum(item["vae_candidate"]["tokens"] for item in report["media"])
    temporary = output / "metadata.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output / "metadata.json")
    return report


def main():
    import site
    if site.ENABLE_USER_SITE:
        raise SystemExit(
            "Refusing user-site imports: run with PYTHONNOUSERSITE=1 (or python -s), "
            "and unset PYTHONPATH to keep the dedicated experiment environment isolated."
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Path to inner dataset_v1")
    parser.add_argument("--sample-id", required=True, type=int, action="append")
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/datamake_media_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/media_probe/v1")
    parser.add_argument("--export-mp4", action="store_true")
    args = parser.parse_args()
    if len(args.sample_id) > 4 or len(set(args.sample_id)) != len(args.sample_id):
        parser.error("Select at most four distinct samples; this is a bounded probe")
    if args.output.resolve().is_relative_to(args.dataset_root.resolve().parent):
        parser.error("Output must be outside the immutable dataset package")
    rows = {}
    with args.manifest.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_id"] in args.sample_id:
                rows[row["sample_id"]] = row
    if set(rows) != set(args.sample_id):
        parser.error("A requested sample ID is absent from the selected manifest")
    profile = load_profile(args.profile)
    for sample_id in args.sample_id:
        report = prepare_sample(rows[sample_id], args.dataset_root, args.output, profile, args.export_mp4)
        print(json.dumps({"sample_id": sample_id, "tokens": report["total_candidate_tokens"], "output": str(args.output / str(sample_id))}))


if __name__ == "__main__":
    main()
