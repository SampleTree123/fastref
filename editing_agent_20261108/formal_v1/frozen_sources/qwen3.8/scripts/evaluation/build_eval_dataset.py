#!/usr/bin/env python3
"""Prepare every round of all fork chains; CPU-only, no inference/training.

Owner: ../../datasets/history_selection_original/README.md
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from evaluation_inputs import ROOT, TASK_PROMPT, assemble_model_input, assistant_tool_call, parse_tool_output, score_prediction

WORKSPACE = ROOT.parents[2]
DEFAULT_CHAINS = ROOT.parent / "ltx2.5/data/uedata/outfit_camera_first_frame/chains/multi_turn/chains.jsonl"
DEFAULT_DATASET = WORKSPACE / "00_engine/outputs/dataset_package_v1"
DEFAULT_OUTPUT = ROOT / "datasets/history_selection_original"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_row(chain, round_number, selection, dataset_root, prompt):
    turn = chain["turns"][round_number - 1]
    history = chain["history_node_ids"][:round_number]
    parent = turn["source_history_index"]
    assert turn["edit_round"] == round_number
    assert 0 <= parent < len(history) and history[parent] == turn["source_node_id"]
    assert selection["matching_history_indices"] == [parent]
    assert chain["history_node_ids"][round_number] == turn["target_node_id"]
    kind = selection["kind"]
    assert kind in {"initial", "latest", "numbered", "edit_event", "single_feature"}, kind
    label = -1 if kind in {"initial", "latest"} else parent
    if label == -1:
        assert parent == len(history) - 1
    for index, past in enumerate(chain["turns"][:round_number - 1], 1):
        assert past["target_node_id"] == history[index]
        assert history[past["source_history_index"]] == past["source_node_id"]
    model_input = assemble_model_input(chain, round_number, dataset_root, prompt)
    digest = hashlib.sha256(compact(model_input).encode()).hexdigest()
    current_target_path = str((dataset_root / "dataset_v1/nodes/nodes" /
                               f"node_{turn['target_node_id']}.mp4").resolve())
    legal_history_paths = [b["path"] for b in model_input["messages"][1]["content"]
                           if b.get("media_role") == "history"]
    assert len(legal_history_paths) == round_number
    return {
        "sample_id": f"chain_{chain['chain_id']:05d}_round_{round_number}",
        "model_input": model_input,
        "supervision": {"source_video_id": label, "resolved_parent_video_id": parent,
                        "assistant_message": assistant_tool_call(label)},
        "audit": {"chain_id": chain["chain_id"], "edit_round": round_number,
                  "original_split": chain["split"], "chain_kind": chain["kind"],
                  "chain_length": len(chain["turns"]), "fork_round": chain["fork_round"],
                  "selection_kind": kind, "selection_category": "visual" if kind == "single_feature" else kind,
                  "source_selection": selection, "current_axis": turn["axis"],
                  "current_mode": turn["mode"], "history_node_ids": history,
                  "current_source_node_id": turn["source_node_id"],
                  "current_target_node_id": turn["target_node_id"],
                  "current_camera_reference_is_authorized_target_first_frame": turn["axis"] == "camera",
                  "current_target_aliases_legal_history_media": current_target_path in legal_history_paths,
                  "current_target_video_injected": False, "history_policy": "all_completed_gt_branches",
                  "input_sha256": digest},
    }


def check_label_isolation(chain, round_number, dataset_root, prompt, expected_input):
    # Current source/target and future data are audit/label facts. They must have
    # no effect on input construction; previous edits' source indices stay legal.
    changed = deepcopy(chain)
    current = changed["turns"][round_number - 1]
    for key in ("source_history_index", "source_node_id", "target_node_id", "edge_id", "axis", "mode"):
        current[key] = "DO_NOT_READ_CURRENT_LABEL"
    changed["turns"] = changed["turns"][:round_number]
    changed["history_node_ids"] = changed["history_node_ids"][:round_number]
    assert assemble_model_input(changed, round_number, dataset_root, prompt) == expected_input


def check_parser():
    def call(value):
        return f"<tool_call>\n<function=select_source_video>\n<parameter=source_video_id>\n{value}\n</parameter>\n</function>\n</tool_call>"
    for value in (-1, 0, 1, 4):
        assert parse_tool_output(call(value), 5)["source_video_id"] == value
        assert parse_tool_output(call(value) + "<|im_end|>", 5)["valid"]
    for raw in (call(5), call(-2), call("1.0"), call("true"), call("01"), call("+1"),
                call(1) + call(1), "Reasoning: " + call(1), '{"source_video_id":1}',
                call(1).replace("source_video_id", "wrong_argument"),
                call(1).replace("select_source_video", "wrong_function")):
        assert not parse_tool_output(raw, 5)["valid"], raw
    row = {"audit": {"edit_round": 2},
           "supervision": {"source_video_id": -1, "resolved_parent_video_id": 1}}
    assert score_prediction(call(-1), row)["strict_correct"]
    assert not score_prediction(call(1), row)["strict_correct"]
    assert score_prediction(call(1), row)["resolved_correct"]
    assert not score_prediction("bad", row)["resolved_correct"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", type=Path, default=DEFAULT_CHAINS)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-selection", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    selection_path = args.source_selection or args.dataset_root / "source_selection.json"
    selections = {c["chain_id"]: {t["edit_round"]: t["source_selection"] for t in c["turns"]}
                  for c in json.loads(selection_path.read_text())["chains"]}
    prompt = TASK_PROMPT.read_text().strip()
    check_parser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "eval.jsonl"
    temporary = args.output_dir / "eval.jsonl.tmp"
    counts = defaultdict(Counter)
    split_counts = defaultdict(lambda: defaultdict(Counter))
    unique_media = set()
    input_hashes = Counter()
    chain_ids = set()
    examples = {}
    aliases = 0
    round_count = 0
    with temporary.open("w") as target, args.chains.open() as source:
        for line in source:
            chain = json.loads(line)["chain"]
            if chain["kind"] != "fork":
                continue
            assert chain["chain_id"] not in chain_ids
            chain_ids.add(chain["chain_id"])
            counts["chains_by_split"][chain["split"]] += 1
            for number in range(1, len(chain["turns"]) + 1):
                row = build_row(chain, number, selections[chain["chain_id"]][number], args.dataset_root, prompt)
                check_label_isolation(chain, number, args.dataset_root, prompt, row["model_input"])
                blocks = row["model_input"]["messages"][1]["content"]
                for block in blocks:
                    if block["type"] in {"image", "video"}:
                        unique_media.add(block["path"])
                text = "\n".join(block["text"] for block in blocks if block["type"] == "text")
                assert all(key not in text for key in ("source_node_id", "target_node_id", "source_history_index", "/scratch/"))
                audit = row["audit"]
                split = audit["original_split"]
                for name, value in (("rounds_by_split", split), ("selection_kind", audit["selection_kind"]),
                                    ("edit_round", number), ("current_axis", audit["current_axis"]),
                                    ("label", row["supervision"]["source_video_id"])):
                    counts[name][str(value)] += 1
                    split_counts[split][name][str(value)] += 1
                counts["media_occurrences"].update(block["media_role"] for block in blocks if "media_role" in block)
                aliases += audit["current_target_aliases_legal_history_media"]
                input_hashes[audit["input_sha256"]] += 1
                examples.setdefault(audit["selection_kind"], row["sample_id"])
                target.write(compact(row) + "\n")
                round_count += 1
    missing = [path for path in sorted(unique_media) if not Path(path).is_file()]
    assert not missing, missing[:10]
    assert len(chain_ids) == 4800, len(chain_ids)
    assert round_count == 18960, round_count
    temporary.replace(output)
    inventory = {
        "schema": "qwen_history_selection_original_eval_v1", "date": "2026-11-06",
        "status": "prepared_cpu_checks_passed_no_model_inference",
        "scope": "All 4,800 original fork chains, every round; 18,960 independent GT-history decisions.",
        "chain_count": len(chain_ids), "round_count": round_count,
        "counts": {key: dict(value) for key, value in counts.items()},
        "counts_by_split": {split: {key: dict(value) for key, value in groups.items()}
                            for split, groups in split_counts.items()},
        "unique_media_files": len(unique_media), "missing_media_files": len(missing),
        "unique_model_inputs": len(input_hashes), "duplicate_model_input_rows": round_count - len(input_hashes),
        "deduplication": "none; every original chain round retained with sample_id",
        "current_target_aliases_legal_history_media_rows": aliases,
        "checks": {"all_current_label_and_future_mutations_leave_input_unchanged": True,
                   "all_history_slots_and_past_parents_match_gt": True,
                   "all_media_files_exist": True, "parser_invalid_output_never_receives_credit": True,
                   "all_references_preserved_in_occurrence_order": True},
        "category_example_sample_ids": examples,
        "inputs": {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
                   for path in (args.chains, selection_path, TASK_PROMPT, Path(__file__),
                                Path(__file__).with_name("evaluation_inputs.py"))},
        "output": {"file": output.name, "sha256": sha256(output), "bytes": output.stat().st_size},
        "media_policy": "References only, no media copied/decoded/cached; front/back separate; camera first-frame authorized.",
        "evaluation_policy": "No training, model loading, GPU use or predicted-history propagation; train/test labels preserved.",
    }
    (args.output_dir / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"chain_count": len(chain_ids), "round_count": round_count,
                      "counts": inventory["counts"], "unique_media_files": len(unique_media),
                      "output_bytes": output.stat().st_size}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
