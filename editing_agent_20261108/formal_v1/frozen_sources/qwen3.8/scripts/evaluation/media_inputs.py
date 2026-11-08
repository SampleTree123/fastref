"""Input-only media preparation; official Qwen resize, RGB cache, true timestamps."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class MediaCache:
    """Content/profile-addressed lossless uint8 npz, shared across replicas/rows.

    Size pruning is approximate under concurrent writers. Original files and
    per-row conversation occurrences are never changed or deduplicated.
    """
    def __init__(self, directory, profile, max_gib=64, minimum_free_gib=30):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.max_bytes = max_gib * 1024**3
        self.minimum_free = minimum_free_gib * 1024**3
        self.digests = {}
        self.writes = 0

    def content_digest(self, path):
        path = Path(path)
        stat = path.stat()
        identity = [str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        index_key = hashlib.sha256(canonical(identity).encode()).hexdigest()
        if index_key not in self.digests:
            index = self.directory / "source_index" / f"{index_key}.json"
            if index.exists():
                self.digests[index_key] = json.loads(index.read_text())["sha256"]
            else:
                self.digests[index_key] = sha_file(path)
                atomic_json(index, {"identity": identity, "sha256": self.digests[index_key]})
        return self.digests[index_key]

    def prune(self):
        files = []
        for path in self.directory.glob("rgb/*/*.npz"):
            try:
                stat = path.stat()
                files.append((stat.st_mtime, path, stat.st_size))
            except FileNotFoundError:
                pass
        sizes = [(path, size) for _, path, size in sorted(files)]
        total = sum(size for _, size in sizes)
        for path, size in sizes:
            stats = os.statvfs(self.directory)
            if total <= self.max_bytes and stats.f_bavail * stats.f_frsize >= self.minimum_free:
                break
            try:
                path.unlink()
                total -= size
            except FileNotFoundError:
                pass

    def load(self, block):
        import av
        import numpy as np
        import torch
        from PIL import Image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as image_resize
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize as video_resize

        path = Path(block["path"])
        kind = block["type"]
        role = block.get("media_role")
        if kind == "video" and role not in ("history", "motion_reference"):
            raise ValueError(f"Video lacks explicit history/motion_reference role: {path}")
        frames = self.profile["history_frames" if role == "history" else "motion_frames"] if kind == "video" else 1
        settings = {"version": 1, "kind": kind, "frames": frames,
                    "pixel_budget": self.profile["video_frame_pixels" if kind == "video" else "image_pixels"],
                    "resize": "official_smart_resize_torchvision_bicubic_antialias_uint8", "sampling": "linspace_round"}
        digest = self.content_digest(path)
        key = hashlib.sha256(canonical([digest, settings]).encode()).hexdigest()
        cache = self.directory / "rgb" / key[:2] / f"{key}.npz"
        if cache.exists():
            try:
                with np.load(cache, allow_pickle=False) as data:
                    return data["rgb"], json.loads(str(data["metadata"].item()))
            except (OSError, ValueError, KeyError):
                pass
        if kind == "image":
            with Image.open(path) as image:
                original = np.array(image.convert("RGB"))
            h, w = image_resize(*original.shape[:2], factor=32, min_pixels=1024,
                                max_pixels=settings["pixel_budget"])
            rgb = resize(torch.from_numpy(original).permute(2, 0, 1), [h, w],
                         InterpolationMode.BICUBIC, antialias=True).permute(1, 2, 0).numpy()
            metadata = {"original_hw": list(original.shape[:2])}
        else:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                stream.thread_count = 1
                count, fps = stream.frames, float(stream.average_rate or 0)
                if count <= 0 or fps <= 0:
                    raise ValueError(f"Video requires explicit positive frame count/fps: {path}")
                indices = np.linspace(0, count - 1, frames).round().astype(int).tolist()
                selected, wanted = {}, set(indices)
                h, w = video_resize(frames, stream.height, stream.width, temporal_factor=2,
                                    factor=32, min_pixels=4096, max_pixels=frames * settings["pixel_budget"])
                for index, frame in enumerate(container.decode(stream)):
                    if index in wanted:
                        tensor = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                        selected[index] = resize(tensor, [h, w], InterpolationMode.BICUBIC,
                                                 antialias=True).permute(1, 2, 0).numpy()
                    if len(selected) == len(wanted):
                        break
                if len(selected) != len(wanted):
                    raise ValueError(f"Decoder did not produce requested indices: {path}")
                rgb = np.stack([selected[index] for index in indices])
                metadata = {"original_hw": [stream.height, stream.width], "total_num_frames": count,
                            "fps": fps, "frames_indices": indices, "duration": count / fps}
        metadata.update({"content_sha256": digest, "cache_key": key, "profile": settings,
                         "resized_hw": [h, w], "kind": kind})
        cache.parent.mkdir(parents=True, exist_ok=True)
        stats = os.statvfs(self.directory)
        if stats.f_bavail * stats.f_frsize > self.minimum_free:
            fd, temporary = tempfile.mkstemp(dir=cache.parent, prefix=".tmp-", suffix=".npz")
            try:
                with os.fdopen(fd, "wb") as stream:
                    np.savez_compressed(stream, rgb=rgb, metadata=canonical(metadata))
                os.replace(temporary, cache)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            self.writes += 1
            if self.writes % 16 == 0:
                self.prune()
        return rgb, metadata


def prepare_inputs(model_input, processor, cache):
    """Only this input object enters the template/processor; no labels accepted."""
    from evaluation_inputs import build_messages, template_kwargs
    from transformers.video_utils import VideoMetadata

    messages = build_messages(model_input)
    images, videos, video_metadata, audit = [], [], [], []
    for message in messages:
        if not isinstance(message.get("content"), list):
            continue
        for index, block in enumerate(message["content"]):
            if block.get("type") not in ("image", "video"):
                continue
            rgb, info = cache.load(block)
            audit.append({"role": block.get("media_role"), **info})
            if block["type"] == "image":
                images.append(rgb)
            else:
                videos.append(rgb)
                video_metadata.append(VideoMetadata(**{key: info[key] for key in
                    ("total_num_frames", "fps", "frames_indices", "duration")}))
            message["content"][index] = {"type": block["type"]}
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **template_kwargs(model_input))
    kwargs = {"text": text, "return_tensors": "pt", "truncation": False}
    if images:
        kwargs.update(images=images, images_kwargs={"do_resize": False})
    if videos:
        kwargs.update(videos=videos, videos_kwargs={"do_resize": False, "do_sample_frames": False,
                                                   "video_metadata": video_metadata})
    outputs = processor(**kwargs)
    if "mm_token_type_ids" not in outputs:
        raise ValueError("Official processor omitted mm_token_type_ids")
    return outputs, {"media": audit, "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "input_tokens": outputs["input_ids"].numel(),
                     "image_grid_thw": outputs["image_grid_thw"].tolist() if images else [],
                     "video_grid_thw": outputs["video_grid_thw"].tolist() if videos else []}
