#!/usr/bin/env python3
"""Read-only DataMake package audit and lossless, reference-based single-turn index.

Standard library only. Media bytes are NOT copied or fully SHA-verified here.
Paths point to dataset_v1; recorded media SHA values come from integrity.json.
Cache identities in this index are inputs to cache keys, not complete encoder keys.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

SCHEMA = "datamake_ltx_single_turn_v1"
CATEGORIES = ("garment", "hair", "identity", "motion", "scene")
AXIS_MODES = {("camera", "attribute"), ("action", "attribute"),
              ("identity", "attribute"), ("site", "attribute"),
              ("hair_color", "attribute"), ("hair", "attribute"),
              ("wardrobe", "single_piece"), ("wardrobe", "whole_outfit")}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(root, relative):
    require(isinstance(relative, str) and relative, "empty/non-string path")
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and ".." not in pure.parts and "\\" not in relative,
            f"unsafe path: {relative}")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"path escapes root: {relative}")
    return path


class Package:
    def __init__(self, package):
        self.root = Path(package).resolve()
        self.data = self.root / "dataset_v1"
        raw = (self.root / "integrity.json").read_bytes()
        integrity = json.loads(raw)
        require(integrity["algorithm"] == "sha256", "unsupported integrity algorithm")
        self.integrity_sha256 = hashlib.sha256(raw).hexdigest()
        self.expected = {}
        for entry in integrity["files"]:
            path = entry["path"]
            safe_path(self.root, path)
            require(path not in self.expected, f"duplicate integrity path: {path}")
            require(isinstance(entry["bytes"], int) and entry["bytes"] >= 0,
                    f"invalid byte count: {path}")
            require(len(entry["sha256"]) == 64 and all(c in "0123456789abcdef" for c in entry["sha256"]),
                    f"invalid sha256: {path}")
            self.expected[path] = entry
        self.json_verified = set()
        self.media = {}

    def expected_file(self, relative):
        key = "dataset_v1/" + relative
        path = safe_path(self.data, relative)
        require(key in self.expected, f"missing integrity record: {key}")
        entry = self.expected[key]
        require(path.is_file(), f"missing file: {key}")
        require(path.stat().st_size == entry["bytes"], f"size mismatch: {key}")
        return path, entry

    def read_json(self, relative):
        path, expected = self.expected_file(relative)
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected["sha256"],
                f"JSON sha256 mismatch: {relative}")
        self.json_verified.add(relative)
        return json.loads(raw)

    def media_record(self, relative):
        if relative not in self.media:
            path, expected = self.expected_file(relative)
            require(path.suffix.lower() in (".mp4", ".png", ".jpg", ".jpeg"),
                    f"unexpected media type: {relative}")
            self.media[relative] = {
                "path": relative, "absolute_path": str(path), "bytes": expected["bytes"],
                "sha256": expected["sha256"], "media_key": "sha256:" + expected["sha256"],
            }
        return dict(self.media[relative])


def validate_references(sample, refs):
    axis, mode = sample["axis"], sample["mode"]
    require((axis, mode) in AXIS_MODES,
            f"unsupported axis/mode: sample {sample['id']}: {axis}/{mode}")
    types = [r["type"] for r in refs]
    if axis in ("camera", "hair_color"):
        require(not refs, f"unexpected references: sample {sample['id']}")
    elif axis in ("identity", "site", "action"):
        expected = {"identity": "identity", "site": "scene", "action": "motion"}[axis]
        require(types == [expected], f"wrong reference kind/count: sample {sample['id']}")
    elif axis in ("hair", "wardrobe"):
        expected = "hair" if axis == "hair" else "garment"
        require(refs and all(t == expected for t in types) and len(refs) % 2 == 0,
                f"invalid paired references: sample {sample['id']}")
        if axis == "hair" or mode == "single_piece":
            require(len(refs) == 2, f"expected one reference pair: sample {sample['id']}")
        # Pairs are identified by their registered attributes, not by an assumed
        # front/back ordering. Preserve the published ordering unchanged.
        groups = defaultdict(list)
        for ref in refs:
            groups[digest(ref["attributes"])].append(ref.get("view"))
        require(all(sorted(views) == ["back", "front"] for views in groups.values()),
                f"unpaired or mismatched reference attributes: sample {sample['id']}")
    else:
        raise ValueError(f"unsupported axis: {axis}")


def build(package):
    pkg = Package(package)
    nodes = {}
    for node in pkg.read_json("nodes/index.json")["nodes"]:
        require(node["id"] not in nodes, f"duplicate node ID: {node['id']}")
        require(node["split"] in ("train", "test"), "unsupported node split")
        nodes[node["id"]] = node
        pkg.media_record(node["path"])
    references = {}
    category_counts = {}
    for category in CATEGORIES:
        registry = pkg.read_json(f"conditions/{category}/index.json")
        require(registry["category"] == category, "reference category mismatch")
        seen_ids = set()
        category_counts[category] = len(registry["references"])
        for ref in registry["references"]:
            require(ref["id"] not in seen_ids, f"duplicate {category} ID")
            seen_ids.add(ref["id"])
            require(ref["path"] not in references, "duplicate reference path")
            references[ref["path"]] = {**ref, "type": category}
            pkg.media_record(ref["path"])
    index = pkg.read_json("chains/single_turn/index.json")
    require(index["count"] == len(index["samples"]), "sample index count mismatch")
    rows, ids, pairs, used_paths = [], set(), set(), set()
    targets = defaultdict(list)
    split_nodes = defaultdict(set)
    for item in index["samples"]:
        require(item["path"] not in used_paths, "duplicate sample path")
        used_paths.add(item["path"])
        sample = pkg.read_json(item["path"])
        sid = sample["id"]
        require(isinstance(sid, int) and sid not in ids, f"duplicate/invalid sample ID: {sid}")
        ids.add(sid)
        for key in ("id", "split", "source_node_id"):
            require(item[key] == sample[key], f"index/sample mismatch: {sid}/{key}")
        split = sample["split"]
        require(split in ("train", "test"), f"invalid sample split: {sid}")
        pair = (sample["source_node_id"], sample["target_node_id"])
        require(pair[0] != pair[1] and pair not in pairs, f"duplicate/self edge: {sid}")
        pairs.add(pair)
        require(all(n in nodes for n in pair), f"unknown node: {sid}")
        require(all(nodes[n]["split"] == split for n in pair), f"cross-split edge: {sid}")
        split_nodes[split].update(pair)
        prompt = sample["prompt"]
        require(all(isinstance(prompt.get(lang), str) and prompt[lang].strip() for lang in ("en", "zh")),
                f"missing prompt: {sid}")
        refs = []
        ref_paths = set()
        for ref in sample["references"]:
            path = ref["path"]
            require(path in references and path not in ref_paths, f"unknown/repeated reference: {sid}/{path}")
            ref_paths.add(path)
            registered = references[path]
            require(registered["type"] == ref["type"], f"reference type mismatch: {sid}")
            if "view" in registered:
                require(registered["view"] == ref.get("view"), f"reference view mismatch: {sid}")
            refs.append({**ref, **pkg.media_record(path), "reference_id": registered["id"],
                         "attributes": registered["attributes"]})
        validate_references(sample, refs)
        def node_record(nid):
            node = nodes[nid]
            return {**pkg.media_record(node["path"]), "node_id": nid,
                    "attributes": node["attributes"], "expression_id": node.get("expression_id")}
        row = {"schema": SCHEMA, "sample_id": sid, "split": split,
               "source_node_id": pair[0], "target_node_id": pair[1],
               "axis": sample["axis"], "mode": sample["mode"], "prompt": prompt,
               "source": node_record(pair[0]), "target": node_record(pair[1]),
               "references": refs, "sample_path": item["path"],
               "sample_metadata": {k: v for k, v in sample.items() if k not in {
                   "id", "split", "source_node_id", "target_node_id", "axis", "mode", "prompt", "references"}},
               "caption_key": digest({"sample_id": sid, "prompt": prompt})}
        rows.append(row)
        targets[pair[1]].append(row)
    rows.sort(key=lambda row: row["sample_id"])
    require(not (split_nodes["train"] & split_nodes["test"]), "train/test node overlap")
    representatives = {}
    for row in rows:
        key = (row["split"], row["axis"], row["mode"], row["source"]["attributes"]["aspect_id"])
        representatives.setdefault(key, row)
    selected = {row["sample_id"]: row for row in representatives.values()}
    for split in ("train", "test"):
        candidates = [r for r in rows if r["split"] == split]
        if candidates:
            row = max(candidates, key=lambda r: (len(r["references"]), -r["sample_id"]))
            selected[row["sample_id"]] = row
    differing = {tid: items for tid, items in targets.items() if len({digest(r["prompt"]) for r in items}) > 1}
    audit = {
        "schema": SCHEMA, "status": "passed", "package_root": str(pkg.root),
        "dataset_root": str(pkg.data), "integrity_sha256": pkg.integrity_sha256,
        "manifest_content_sha256": digest(rows), "sample_count": len(rows),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "axis_mode_split_counts": dict(sorted(Counter(f"{r['split']}/{r['axis']}/{r['mode']}" for r in rows).items())),
        "reference_count_histogram": dict(sorted(Counter(len(r["references"]) for r in rows).items())),
        "maximum_references": max((len(r["references"]) for r in rows), default=0),
        "node_count": len(nodes), "reference_registry_counts": category_counts,
        "unique_media_paths": len(pkg.media), "media_bytes": sum(m["bytes"] for m in pkg.media.values()),
        "json_sha256_verified_count": len(pkg.json_verified), "media_size_verified_count": len(pkg.media),
        "media_sha256_recomputed_count": 0, "media_decode_checked_count": 0,
        "verification_scope": "All indexed single-turn samples, node/reference registries: JSON SHA256. All registered media: existence and bytes only. Media SHA256 copied from package integrity, NOT recomputed. No visual/temporal/decode acceptance.",
        "train_test_node_overlap": 0,
        "targets_with_multiple_edges": sum(len(v) > 1 for v in targets.values()),
        "targets_with_different_prompts": len(differing),
        "samples_affected_by_target_only_caption_cache": sum(len(v) for v in differing.values()),
        "caption_collision_examples": [{"target_node_id": tid, "sample_ids": [r["sample_id"] for r in items[:8]]}
                                       for tid, items in sorted(differing.items())[:5]],
        "cache_contract": "media_key is expected content identity only; add VAE revision, dtype, normalization, sampling/resize/profile hash and encode policy. caption_key includes edge and original bilingual prompt; add exact rendered text, template, encoder/connectors revisions and max length before caching tensors.",
        "split_policy": "Published train/test preserved. Node-disjoint does not mean identity/reference-asset-disjoint. Test representatives are evaluation only.",
        "representative_count": len(selected),
    }
    return rows, sorted(pkg.media.values(), key=lambda m: m["path"]), [selected[k] for k in sorted(selected)], audit


def atomic_text(path, text):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, path)


def write_outputs(output, rows, media, representatives, audit):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # audit.json is the completion marker. Remove an old success before replacing
    # artifacts so interrupted reruns cannot advertise a mixed output as valid.
    (output / "audit.json").unlink(missing_ok=True)
    for split in ("train", "test"):
        atomic_text(output / f"{split}.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r["split"] == split))
    for name, value in (("media_registry.json", {"schema": SCHEMA, "media": media}),
                        ("representatives.json", {"schema": SCHEMA, "selection": "first sample by split/axis/mode/aspect; plus max-reference example per split", "samples": representatives})):
        atomic_text(output / name, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    audit["artifact_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                  for name in ("train.jsonl", "test.jsonl", "media_registry.json", "representatives.json")}
    atomic_text(output / "audit.json", json.dumps(audit, ensure_ascii=False, indent=2) + "\n")


def main():
    experiment = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=experiment.parents[2] / "00_engine/outputs/dataset_package_v1")
    parser.add_argument("--output", type=Path, default=experiment / "outputs/data_prep/v1")
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.package.resolve()), "output must not modify the original package")
    result = build(args.package)
    write_outputs(args.output, *result)
    print(json.dumps({"output": str(args.output), "samples": len(result[0]), "audit": result[3]["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
