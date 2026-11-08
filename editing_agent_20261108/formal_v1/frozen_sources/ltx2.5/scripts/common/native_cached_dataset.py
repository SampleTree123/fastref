#!/usr/bin/env python3
"""Verified native cached datasets: bounded old probes and complete manifest splits.

No encoding, no filtering to a directory intersection, no hidden reference loss.
This dataset emits video-only pre-connector text features. Its trainer MUST use a
video-only embeddings processor (audio_connector=None).
"""
from __future__ import annotations

import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if (p / "common/project_layout.py").is_file()) / "common"))
from project_layout import ROOT as _PROJECT_ROOT, resolve_path, script_path, add_script_paths, resolve_cli_paths
add_script_paths()

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import stat as stat_module

import torch
from torch.utils.data import Dataset, default_collate

from compact_text_cache import make_contract, unpack_text_features
from prepare_datamake_media import fingerprint, latent_shape
from contact_sheet_contract import (runtime_sheet_metadata, sheet_provenance,
                                    validate_contact_sheet)

ROOT = _PROJECT_ROOT


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    sha = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def weight_signature(record):
    files = record["files"]
    require(len({item["path"] for item in files}) == len(files), "duplicate native weight file")
    identity = {"repo_id": record["repo_id"], "revision": record["revision"],
                "files": sorted(({k: item[k] for k in ("path", "bytes", "sha256")} for item in files),
                                key=lambda item: item["path"])}
    return fingerprint(identity)


def checked_artifact(root, artifact):
    relative = Path(artifact["path"])
    path = (root / relative).resolve()
    require(not relative.is_absolute() and ".." not in relative.parts and path.is_relative_to(root.resolve()),
            "cache artifact escapes encoding root")
    require(path.is_file() and path.stat().st_size == artifact["bytes"], f"missing/changed cache artifact: {path}")
    require(sha_file(path) == artifact["sha256"], f"cache artifact SHA mismatch: {path}")
    return path


def collate_reference_batch(samples):
    from native_grouped_sampling import sample_structure
    require(bool(samples), "Empty reference batch")
    signatures = [sample_structure(sample) for sample in samples]
    require(all(key == signatures[0] for key in signatures),
            "Incompatible frame count/canvas/reference structure/text shape in one microbatch")
    return default_collate(samples)


def collate_single_sample(samples):
    require(len(samples) == 1, "native cache supports batch size 1 only")
    return default_collate(samples)


class NativeCachedDataset(Dataset):
    def __init__(self, manifest, encoding_roots, *, sample_ids,
                 expected_profile=None, expected_weights=None):
        require(isinstance(sample_ids, (list, tuple)) and 0 < len(sample_ids) <= 4,
                "explicit one-to-four probe sample_ids required; no implicit cache intersection")
        require(all(type(x) is int for x in sample_ids) and len(set(sample_ids)) == len(sample_ids),
                "sample_ids must be unique integers")
        self.sample_ids = list(sample_ids)
        self.manifest = resolve_path(manifest).resolve()
        self.manifest_sha256 = sha_file(self.manifest)
        self.profile = expected_profile or json.loads((ROOT / "configs/data/archive/synthetic_640/media.json").read_text())
        self.weights = expected_weights or json.loads((resolve_path('runtime/models/native_model_revision.json')).read_text())
        self.profile_sha256 = fingerprint(self.profile)
        self.weights_signature = weight_signature(self.weights)
        self.rows = {}
        with self.manifest.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["sample_id"] in self.sample_ids:
                    require(row["sample_id"] not in self.rows, "duplicate selected sample in manifest")
                    require(row.get("split") == "train", "experimental loader accepts train samples only")
                    self.rows[row["sample_id"]] = row
        require(set(self.rows) == set(self.sample_ids), "requested sample missing from manifest")
        self.entries = {}
        self.metadata_records = []
        self._encoding_signature = None
        require(bool(encoding_roots), "at least one encoding root is required")
        for encoding_root in encoding_roots:
            root = resolve_path(encoding_root).resolve()
            metadata_path = root / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            require(metadata.get("schema") == "datamake_native_encoding_smoke_v1" and metadata.get("status") == "passed",
                    "encoding root must have passed native probe metadata")
            require(metadata["manifest_sha256"] == self.manifest_sha256, "manifest SHA disagrees with encoded cache")
            require(metadata["profile_sha256"] == fingerprint(metadata["profile"]) == self.profile_sha256,
                    "media preprocessing profile mismatch")
            require(weight_signature(metadata["weights"]) == self.weights_signature, "native weight signatures mismatch")
            encoding_signature = fingerprint({k: metadata[k] for k in ("seed", "torch_version", "pixel_toolchain", "smoke_script_sha256")})
            if self._encoding_signature is None:
                self._encoding_signature = encoding_signature
            require(encoding_signature == self._encoding_signature, "mixed encoding implementation/toolchain/seed")
            declared_ids = metadata["sample_ids"]
            require(len(declared_ids) == len(set(declared_ids)), "duplicate sample_ids in encoding metadata")
            video_records, text_records = defaultdict(dict), {}
            for item in metadata["video"]:
                sid, role = item["sample_id"], item["role"]
                require(sid in declared_ids and role not in video_records[sid], "undeclared sample or duplicate encoded role")
                video_records[sid][role] = item
            for item in metadata["text"]:
                sid = item["sample_id"]
                require(sid in declared_ids and sid not in text_records, "undeclared sample or duplicate text record")
                text_records[sid] = item
            for sid in set(self.sample_ids) & set(declared_ids):
                require(sid not in self.entries, "sample appears in multiple encoding roots; select one authoritative cache")
                require(sid in text_records, f"missing encoded text: {sid}")
                row = self.rows[sid]
                expected = {"source": row["source"], "target": row["target"]}
                expected.update({f"reference_{i:02d}_{ref['type']}": ref for i, ref in enumerate(row["references"])})
                require(set(video_records[sid]) == set(expected), f"missing/extra reference or source/target cache: sample {sid}")
                for role, record in expected.items():
                    item = video_records[sid][role]
                    require(item["media_path"] == record["path"] and item["media_sha256_verified"] == record["sha256"],
                            f"media identity mismatch for sample {sid}/{role}")
                    checked_artifact(root, item["artifact"])
                text = text_records[sid]
                require(text["prompt_en"] == row["prompt"]["en"] and text["caption_key"] == row["caption_key"],
                        f"text identity mismatch: {sid}")
                compact = text["compact_roundtrip"]
                require(compact["status"] == "passed" and compact["bitwise_equal_video_and_mask"] is True
                        and compact["max_abs_video_error"] == 0, "real connector compact-cache gate not passed")
                expected_contract = self._text_contract(row)
                require(compact["contract"] == expected_contract, "compact text model/input contract mismatch")
                checked_artifact(root, compact["artifact"])
                self.entries[sid] = {"root": root, "video": video_records[sid], "text": compact,
                                     "text_contract": expected_contract}
            self.metadata_records.append({"path": str(metadata_path), "sha256": sha_file(metadata_path)})
        require(set(self.entries) == set(self.sample_ids), "requested sample lacks complete encoding; refusing to filter it out")

    def _text_contract(self, row):
        packs = {item["path"].split("/")[0]: item["sha256"] for item in self.weights["files"]}
        return make_contract(sequence_length=1024, feature_dim=4096, num_learnable_registers=128,
            encoder_identity=fingerprint({"pack_sha256": packs["text_encoders"], "padding_side": "left", "enhancement": False}),
            connector_identity=fingerprint({"transformer_sha256": packs["diffusion_models"], "text_pack_sha256": packs["text_encoders"]}),
            text_identity=fingerprint({"prompt_en": row["prompt"]["en"], "caption_key": row["caption_key"]}))

    def sampling_records(self, *, source='dataset', repeat=1):
        records = []
        for sid in self.sample_ids:
            row = self.rows[sid]
            video = self.entries[sid]['video']
            shape = lambda role: video[role]['tensor']['shape'][1:]
            refs = [(ref['type'], ref.get('view', ''), shape(f"reference_{i:02d}_{ref['type']}"))
                    for i, ref in enumerate(row['references'])]
            refs = [(*description, ref['reference_representation'])
                    if ref.get('reference_representation') else description
                    for ref, description in zip(row['references'], refs)]
            group = {'target': shape('target'), 'source': shape('source'), 'references': refs,
                     'text': [1024,4096], 'fps': self.profile['fps']}
            tokens = sum(int(v['tensor']['shape'][2]) * int(v['tensor']['shape'][3]) * int(v['tensor']['shape'][4])
                         for v in video.values())
            records.append({'uid': f'{source}:{sid}', 'source': source, 'repeat': repeat, 'group': group, 'tokens': tokens})
        return records

    def __len__(self):
        return len(self.sample_ids)

    def _artifact(self, root, artifact):
        return checked_artifact(root, artifact)

    def _shape_contract(self, sid, role):
        aspect = self.rows[sid]["source"]["attributes"]["aspect_id"]
        require(aspect in (1, 2), "unsupported aspect_id")
        canvas = self.profile["video_canvas"]["landscape" if aspect == 1 else "portrait"]
        is_video = role in ("source", "target") or role.endswith("_motion")
        if not is_video and self.profile["reference_image_canvas"] == "original_minimal_edge_padding_32":
            reference_index = int(role.split('_')[1])
            dimensions = self.rows[sid]['references'][reference_index].get('original_dimensions')
            require(isinstance(dimensions, list) and len(dimensions) == 2
                    and all(type(x) is int and x > 0 for x in dimensions),
                    'Native-size reference requires original width/height in manifest')
            canvas = [((x + 31) // 32) * 32 for x in dimensions]
        elif (not is_video and not role.endswith("_scene")
                and self.profile["reference_image_canvas"] != "same_as_source_video"):
            canvas = self.profile["reference_image_canvas"]
        frames = self.profile["num_frames"] if is_video else 1
        return canvas, frames, latent_shape(frames, *canvas, self.profile)["shape_cfhw"]

    def _latent(self, sid, role):
        entry, row = self.entries[sid], self.rows[sid]
        info = entry["video"][role]
        path = self._artifact(entry["root"], info["artifact"])
        cached = torch.load(path, map_location="cpu", weights_only=True)
        value = cached["latents"]
        canvas, _, expected_shape = self._shape_contract(sid, role)
        require(info["canvas_wh"] == canvas and info["tensor"]["shape"] == [1, *expected_shape],
                f"recorded latent shape/canvas mismatch: {sid}/{role}")
        require(isinstance(value, torch.Tensor) and list(value.shape) == expected_shape,
                f"cached latent shape mismatch: {sid}/{role}")
        require(value.dtype == torch.bfloat16 and info["tensor"]["dtype"] == str(value.dtype), "expected native BF16 latent")
        require(bool(torch.isfinite(value).all()), "non-finite cached latent")
        require([cached[k] for k in ("num_frames", "height", "width")] == expected_shape[1:], "latent metadata mismatch")
        require(cached["fps"] == self.profile["fps"], "latent FPS mismatch")
        # Explicit projection: do not forward unknown cache metadata or target attrs.
        return {"latents": value, "num_frames": expected_shape[1], "height": expected_shape[2],
                "width": expected_shape[3], "fps": float(self.profile["fps"])}

    def __getitem__(self, index):
        sid = self.sample_ids[index]
        entry, row = self.entries[sid], self.rows[sid]
        compact_path = self._artifact(entry["root"], entry["text"]["artifact"])
        compact = torch.load(compact_path, map_location="cpu", weights_only=True)
        conditions = unpack_text_features(compact, expected_contract=entry["text_contract"])
        require(conditions["video_prompt_embeds"].dtype == torch.bfloat16, "expected native BF16 text features")
        references = [{"type": ref["type"], "view": ref.get("view", ""), "index": i,
                       **runtime_sheet_metadata(ref),
                       "latent": self._latent(sid, f"reference_{i:02d}_{ref['type']}")}
                      for i, ref in enumerate(row["references"])]
        return {"sample_id": sid, "video_latents": self._latent(sid, "target"),
                "source": self._latent(sid, "source"), "references": references, "conditions": conditions}


class NativeProductionCachedDataset(NativeCachedDataset):
    """Complete train/test split; no sample subset or incomplete cache intersection.

    Encoding must have finalized the requested full or explicitly scoped split first. Unique artifacts are checked
    once at construction, reusing recorded SHA checks for unchanged files, then stat-checked on access.
    Changed files are rehashed; cache directories remain fixed during a run.
    Tensors remain CPU/lazy-loaded, so the complete cache is never held in RAM.
    """
    def __init__(self, manifest, encoding_root, *, expected_profile, expected_weights, split="train", completion_metadata=None):
        require(split in {"train", "test", "development"}, "Unknown split")
        require(isinstance(expected_profile, dict) and isinstance(expected_weights, dict),
                "production cache requires explicit profile and native weights")
        self.manifest = resolve_path(manifest).resolve()
        self.manifest_sha256 = sha_file(self.manifest)
        self.profile, self.weights = expected_profile, expected_weights
        if self.profile.get('reference_protocol') == 'reference_positions_v3_shared_canvas':
            from reference_canvas_contract import validate_profile
            validate_profile(self.profile)
        self.profile_sha256 = fingerprint(self.profile)
        self.weights_signature = weight_signature(self.weights)
        self._verified_artifacts = {}
        self._canonical_artifact_paths = {}
        self._canonical_encoding_roots = {}
        self._initial_artifact_checks = set()
        self._initializing_artifacts = True
        self.rows, self.sample_ids = {}, []
        with self.manifest.open() as handle:
            for line in handle:
                row = json.loads(line)
                sid = row["sample_id"]
                require(type(sid) is int and sid not in self.rows, "duplicate/noninteger manifest sample_id")
                require(row.get("split") == ("test" if split == "development" else split), "manifest contains wrong original split")
                self.rows[sid] = row
                self.sample_ids.append(sid)
        require(bool(self.sample_ids), "empty production manifest")
        caption_prompts = {}
        for row in self.rows.values():
            key, prompt = row['caption_key'], row['prompt']['en']
            require(key not in caption_prompts or caption_prompts[key] == prompt,
                    'same caption key cannot identify different English prompts')
            caption_prompts[key] = prompt
        root = resolve_path(encoding_root).resolve()
        metadata_path = resolve_path(completion_metadata).resolve() if completion_metadata else root / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        scoped = metadata.get("schema") == "datamake_native_encoding_scope_v1"
        require((scoped or metadata.get("schema") == "datamake_native_encoding_v2") and metadata.get("status") == "passed",
                "production cache requires a completed full or explicitly scoped receipt")
        if scoped:
            expected_scope = 'independent_test' if split == 'test' else 'train_and_development'
            expected_splits = {'test'} if split == 'test' else {'train', 'development'}
            require(metadata.get('scope') == expected_scope
                    and set(metadata['scope_manifests']) == expected_splits, 'Wrong completion scope')
            require(split in metadata['scope_manifests'], 'Requested split outside completed scope')
        require(set(metadata['manifest_sha256']) == {'train', 'test'}, 'completed cache must identify both frozen splits')
        require((metadata["scope_manifests"][split]["sha256"] if scoped else metadata["manifest_sha256"][split])
                == self.manifest_sha256, "manifest SHA disagrees with encoded cache")
        require(metadata["profile_sha256"] == fingerprint(metadata["profile"]) == self.profile_sha256,
                "media preprocessing profile mismatch")
        require(weight_signature(metadata["weights"]) == self.weights_signature, "native weight signatures mismatch")
        identity = metadata["encoding_identity"]
        if identity.get('schema') == 'datamake_shared_canvas_encoding_identity_v3':
            from reference_canvas_contract import validate_identity
            validate_identity(identity)
        if identity.get('schema') == 'datamake_native_resolution_encoding_identity_v1':
            from native_image_cache_contract import validate_identity
            validate_identity(identity)
        require(metadata["identity_sha256"] == fingerprint(identity), "encoding identity signature mismatch")
        require(identity["profile_sha256"] == self.profile_sha256 and fingerprint(identity["profile"]) == self.profile_sha256
                and weight_signature(identity["weights"]) == self.weights_signature
                and identity["manifest_sha256"] == metadata["manifest_sha256"], "encoding identity disagrees with metadata")
        vae_sha = next(x["sha256"] for x in self.weights["files"] if x["path"].split('/')[0] == 'vae')
        require(identity["vae_sha256"] == vae_sha, "encoding identity VAE mismatch")
        samples = metadata["samples"]
        require(len({item["sample_id"] for item in samples}) == len(samples), "duplicate encoded sample_id")
        require(all(item["split"] in {"train", "test"} for item in samples), "unknown encoded split")
        if scoped:
            require(all(item.get('use_split') in expected_splits for item in samples), 'Unknown logical split')
            require(all(item['split'] == ('train' if item['use_split'] == 'train' else 'test') for item in samples),
                    'Scope changes original split')
            for name, info in metadata['scope_manifests'].items():
                require(info['count'] == sum(item['use_split'] == name for item in samples), 'Scope completion count mismatch')
        else:
            require(set(metadata['manifests']) == {'train', 'test'}, 'missing split completion counts')
            for name, info in metadata['manifests'].items():
                require(info['sha256'] == metadata['manifest_sha256'][name]
                        and info['count'] == sum(item['split'] == name for item in samples), 'split completion count/SHA mismatch')
        require(any(item.get('compact_validation') == 'bitwise_equal_real_video_connector_and_mask'
                    for item in metadata['text'].values()), 'completed cache lacks real compact connector verification')
        selected = {item["sample_id"]: item for item in samples if item["use_split" if scoped else "split"] == split}
        require(set(selected) == set(self.sample_ids), "incomplete/extra encoded split; refusing to filter manifest")
        self.entries = {}
        self.metadata_records = [{"path": str(metadata_path), "sha256": sha_file(metadata_path)}]
        for sid in self.sample_ids:
            row, sample = self.rows[sid], selected[sid]
            refs = sample["references"]
            require(len(refs) == len(row["references"]), f"missing/extra references: {sid}")
            aspect = row['source']['attributes']['aspect_id']
            canvas = self.profile['video_canvas']['landscape' if aspect == 1 else 'portrait']
            roles = [("source", row["source"], sample["source"]), ("target", row["target"], sample["target"])]
            for i, (ref, mapped) in enumerate(zip(row["references"], refs)):
                require(mapped["index"] == i and mapped["type"] == ref["type"]
                        and mapped.get("view", "") == ref.get("view", ""), f"reference order/type/view mismatch: {sid}/{i}")
                validate_contact_sheet(ref, canvas_wh=canvas,
                                       protocol=self.profile.get('reference_protocol'), reference_count=len(refs))
                if ref.get('reference_representation') or mapped.get('reference_representation'):
                    require(sheet_provenance(mapped) == sheet_provenance(ref),
                            f'Contact-sheet encoded provenance differs from manifest: {sid}/{i}')
                roles.append((f"reference_{i:02d}_{ref['type']}", ref, mapped["media_key"]))
            video = {}
            for role, media, key in roles:
                require(key in metadata["media"], f"missing media cache: {sid}/{role}")
                item = metadata["media"][key]
                require(item.get('status') == 'passed' and item.get('stage') == 'media' and item.get('key') == key
                        and item.get('identity_sha256') == metadata['identity_sha256'], f"media completion receipt mismatch: {sid}/{role}")
                require(item["media_sha256_verified"] == media["sha256"], f"media identity mismatch: {sid}/{role}")
                canvas, frames, shape = self._shape_contract(sid, role)
                if identity['schema'] == 'datamake_native_encoding_identity_v2':
                    expected_key = fingerprint({'media_sha256': media['sha256'], 'canvas_wh': canvas, 'num_frames': frames,
                        'profile_sha256': self.profile_sha256, 'vae_sha256': vae_sha, 'encoder_contract': identity['encoder_contract']})
                elif identity['schema'] == 'datamake_coinve_encoding_identity_v1':
                    from coinve_cache_contract import media_key as coinve_media_key
                    expected_key = coinve_media_key(media, canvas, frames, identity, row['temporal_window'])
                elif identity['schema'] == 'datamake_shared_canvas_encoding_identity_v3':
                    from reference_canvas_contract import media_key as shared_canvas_media_key
                    window = row['temporal_window'] if identity['source_name'] == 'coinve' else None
                    expected_key = shared_canvas_media_key(media, canvas, frames, identity, window)
                elif identity['schema'] == 'datamake_native_resolution_encoding_identity_v1':
                    from native_image_cache_contract import media_key as native_image_media_key
                    window = row['temporal_window'] if identity['source_name'] == 'coinve' else None
                    expected_key = native_image_media_key(media, canvas, frames, identity, window)
                else:
                    raise ValueError('Unknown encoding identity schema')
                require(key == expected_key, f"media cache key/processing identity mismatch: {sid}/{role}")
                require(item["canvas_wh"] == canvas and item["tensor"]["shape"] == [1, *shape]
                        and item["tensor"]["dtype"] == "torch.bfloat16", f"recorded latent shape/canvas mismatch: {sid}/{role}")
                self._artifact(root, item["artifact"])
                video[role] = item
            text_key = sample["text"]
            require(text_key == row["caption_key"] and text_key in metadata["text"], f"missing/wrong text cache: {sid}")
            text = metadata["text"][text_key]
            require(text.get('status') == 'passed' and text.get('stage') == 'text' and text.get('key') == text_key
                    and text.get('identity_sha256') == metadata['identity_sha256'], f"text completion receipt mismatch: {sid}")
            require(text.get('compact_validation') in {'valid_rows_exact_full_length_restored', 'bitwise_equal_real_video_connector_and_mask'},
                    f"compact text validation absent: {sid}")
            require(text["caption_key"] == text_key and text["prompt_en"] == row["prompt"]["en"], f"text identity mismatch: {sid}")
            contract = self._text_contract(row)
            require(text["contract"] == contract, f"compact text contract mismatch: {sid}")
            self._artifact(root, text["artifact"])
            self.entries[sid] = {"root": root, "video": video, "text": text, "text_contract": contract}
        self._initializing_artifacts = False
        self._initial_artifact_checks.clear()

    def _artifact(self, root, artifact):
        relative = Path(artifact["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "cache artifact escapes encoding root")
        location = (root, relative)
        if location not in self._canonical_artifact_paths:
            if root not in self._canonical_encoding_roots:
                self._canonical_encoding_roots[root] = root.resolve()
            path = (root / relative).resolve()
            require(path.is_relative_to(self._canonical_encoding_roots[root]), "cache artifact escapes encoding root")
            self._canonical_artifact_paths[location] = path
        path = self._canonical_artifact_paths[location]
        initial_key = (location, artifact["bytes"], artifact["sha256"])
        if self._initializing_artifacts and initial_key in self._initial_artifact_checks:
            return path
        stat = path.stat()
        require(stat_module.S_ISREG(stat.st_mode) and stat.st_size == artifact["bytes"],
                f"missing/changed cache artifact: {path}")
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, artifact["sha256"])
        previous_signature = self._verified_artifacts.get(path)
        if previous_signature != signature:
            # Runtime content replacement is allowed to trigger rechecking, but
            # it must not turn the cached file location into an escaping symlink.
            if previous_signature is not None:
                require(path.resolve().is_relative_to(self._canonical_encoding_roots[root]),
                        "cache artifact escapes encoding root")
            from cache_integrity import verify_payload
            verify_payload(path, artifact)
            self._verified_artifacts[path] = signature
        if self._initializing_artifacts:
            self._initial_artifact_checks.add(initial_key)
        return path


class MixedNativeCachedDataset(Dataset):
    """Read-only concatenation; repeat factors belong to the sampler, never loss."""
    def __init__(self, sources_config, *, expected_weights, split='train'):
        require(split in {'train', 'development', 'test'}, 'Mixed reader accepts train, development or independent test')
        self.sources_config = resolve_path(sources_config).resolve()
        self.spec = json.loads(self.sources_config.read_text())
        require(self.spec['schema'] == 'datamake_mixed_sources_v1', 'Unknown mixed sources schema')
        sources = self.spec['sources']
        require({s['name'] for s in sources} == {'ue','coinve'} and len(sources) == 2, 'Require UE and CoinVE exactly once')
        repeats = {s['name']: s['repeat'] for s in sources}
        require(all(type(s['repeat']) is int for s in sources) and
                repeats in ({'ue': 2, 'coinve': 4}, {'ue': 3, 'coinve': 6}),
                'Approved mixtures are UE2/CoinVE4 or UE3/CoinVE6')
        self.datasets, self.lookup, self.sample_ids, self.records, self.metadata_records = [], [], [], [], []
        path = resolve_path
        for source in sources:
            profile = json.loads(path(source['media_profile']).read_text())
            require(profile['fps'] == 12 and profile['num_frames'] == (121 if source['name'] == 'ue' else 57),
                    'Mixed source duration/FPS changed')
            dataset = NativeProductionCachedDataset(path(source[f'{split}_manifest']), path(source['encoding_root']),
                expected_profile=profile, expected_weights=expected_weights, split=split,
                completion_metadata=(path(source['test_completion_metadata']) if 'test_completion_metadata' in source
                                     else path(source['encoding_root']) / 'scope_independent_test.json')
                                     if split == 'test' else path(source['completion_metadata']))
            index = len(self.datasets)
            self.datasets.append(dataset)
            records = dataset.sampling_records(source=source['name'], repeat=source['repeat'] if split == 'train' else 1)
            for i, record in enumerate(records):
                self.lookup.append((index, i))
                self.sample_ids.append(record['uid'])
                self.records.append(record)
            self.metadata_records.extend(dataset.metadata_records)
        require(len(set(self.sample_ids)) == len(self.sample_ids), 'Mixed UID collision')
        expected_counts = {'train': {'ue':26326,'coinve':3200}, 'development': {'ue':50,'coinve':50},
                           'test': {'ue':4840,'coinve':750}}[split]
        require({s['name']: len(d) for s, d in zip(sources, self.datasets)} == expected_counts,
                'Approved train/development/independent-test sample counts changed')

    def __len__(self):
        return len(self.lookup)

    def sampling_records(self):
        return self.records

    def profile_for(self, index):
        return self.datasets[self.lookup[index][0]].profile

    def __getitem__(self, index):
        source, local = self.lookup[index]
        item = self.datasets[source][local]
        item['sample_uid'] = self.sample_ids[index]
        item['source_name'] = self.spec['sources'][source]['name']
        return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=resolve_path('data/synthetic/original/manifests/train.jsonl'))
    parser.add_argument("--encoding-root", type=Path, action="append", required=True)
    parser.add_argument("--sample-id", type=int, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(); resolve_cli_paths(args)
    dataset = NativeCachedDataset(args.manifest, args.encoding_root, sample_ids=args.sample_id)
    rows = []
    for i in range(len(dataset)):
        item = dataset[i]
        batch = collate_single_sample([item])
        rows.append({"sample_id": item["sample_id"], "reference_count": len(item["references"]),
                     "reference_roles": [{k: ref[k] for k in ("type", "view", "index")} for ref in item["references"]],
                     "target_shape": list(batch["video_latents"]["latents"].shape),
                     "reference_shapes": [list(ref["latent"]["latents"].shape) for ref in batch["references"]],
                     "text_shape": list(batch["conditions"]["video_prompt_embeds"].shape),
                     "text_valid_tokens": int(batch["conditions"]["prompt_attention_mask"].sum())})
    report = {"status": "passed", "scope": "four-or-fewer native cached probe samples; CPU load and batch1 only, not training acceptance",
              "manifest_sha256": dataset.manifest_sha256, "profile_sha256": dataset.profile_sha256,
              "weights_signature": dataset.weights_signature, "encoding_metadata": dataset.metadata_records,
              "samples": rows}
    require(not args.report.exists(), "report already exists; use a new path")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "passed", "sample_count": len(dataset), "report": str(args.report)}))


if __name__ == "__main__":
    main()
