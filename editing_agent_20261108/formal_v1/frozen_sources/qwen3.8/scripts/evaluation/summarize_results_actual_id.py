#!/usr/bin/env python3
"""Summarize one frozen evaluation identity; incompatible records never get credit."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import statistics

from evaluation_inputs_actual_id import score_prediction, validate_row
from media_inputs import atomic_json, sha_file
from run_model_eval_actual_id import safe_id, digest

ROOT = Path(__file__).resolve().parents[2]
COUNT_KEYS = ("expected", "answered", "runtime_error", "missing", "incompatible",
              "valid", "strict_correct", "resolved_correct", "invalid_format_or_range")


def row_scopes(row, review_flags=None):
    audit = row["audit"]
    scopes = ["all", "split/" + audit["original_split"],
              "kind/" + audit["selection_kind"], "round/" + str(audit["edit_round"]),
              "split_kind/" + audit["original_split"] + "/" + audit["selection_kind"]]
    for field in ("dataset_origin", "question_type", "variant", "sample_family", "selection_axis",
                  "selection_reference_purpose", "history_topology", "repeated_selection_reference",
                  "has_explicit_history_text_clue", "selection_reference_repeated_in_history",
                  "explicit_history_text_clue"):
        if field in audit:
            scopes.append(field + "/" + str(audit[field]))
    diagnostic = row.get("audit_only", {})
    for field in ("exact_selection_reference_in_completed_history", "singleton_only_candidate", "explicit_history_textual_clue"):
        if field in diagnostic:
            scopes.append(field + "/" + str(diagnostic[field]))
    current_axis = audit.get("current_axis", diagnostic.get("current_axis"))
    if current_axis is not None:
        scopes.append("current_edit_axis/" + str(current_axis))
    if "selection_axis" in audit and "history_topology" in audit:
        scopes.append("selection_axis_topology_split/" + str(audit["selection_axis"]) + "/" +
                      str(audit["history_topology"]) + "/" + audit["original_split"])
    if review_flags is not None:
        review = {**review_flags.get("default", {}), **review_flags.get("samples", {}).get(row["sample_id"], {})}
        scopes.append("review/rgb_reviewed/" + str(review.get("rgb_reviewed", False)))
        scopes.append("review/rgb_review_status/" + str(review.get("rgb_review_status", "not_reviewed")))
        if review.get("rgb_reviewed"):
            scopes.append("review/initial_blind_match_gt/" + str(review.get("initial_blind_match_gt")))
        if review.get("followup_rgb_status") is not None:
            scopes.append("review/followup_rgb_status/" + str(review["followup_rgb_status"]))
        risks = review.get("risks", [])
        scopes.append("review/has_risk/" + str(bool(risks)))
        for risk in sorted(set(risks)):
            scopes.append("review/risk/" + str(risk))
    return scopes


def paired_counts(rows, paired, review_flags=None):
    shared = paired["low"].keys() & paired["high"].keys()
    result = {"answered_in_both": len(shared),
              "both_correct": sum(paired["low"][key] and paired["high"][key] for key in shared),
              "both_wrong": sum(not paired["low"][key] and not paired["high"][key] for key in shared),
              "high_only_correct": sum(not paired["low"][key] and paired["high"][key] for key in shared),
              "low_only_correct": sum(paired["low"][key] and not paired["high"][key] for key in shared)}
    groups = defaultdict(lambda: Counter({key: 0 for key in result}))
    for row in rows:
        key = row["sample_id"]
        if key not in shared:
            continue
        low, high = paired["low"][key], paired["high"][key]
        for scope in row_scopes(row, review_flags):
            groups[scope]["answered_in_both"] += 1
            groups[scope]["both_correct"] += bool(low and high)
            groups[scope]["both_wrong"] += bool(not low and not high)
            groups[scope]["high_only_correct"] += bool(not low and high)
            groups[scope]["low_only_correct"] += bool(low and not high)
    result["groups"] = {scope: dict(count) for scope, count in groups.items()}
    return result


def expected_manifest(profile_root, profile, dataset_sha256):
    path = profile_root / "expected_run.json"
    try:
        manifest = json.loads(path.read_text())
        identity = manifest["identity"]
        fingerprint = manifest["run_fingerprint"]
        if not isinstance(identity, dict) or not isinstance(fingerprint, str):
            return None, "invalid_manifest_types"
        if digest(identity) != fingerprint:
            return None, "manifest_identity_hash_mismatch"
        if identity.get("dataset_sha256") != dataset_sha256:
            return None, "manifest_dataset_mismatch"
        config = identity.get("config")
        if not isinstance(config, dict) or config.get("name") != profile:
            return None, "manifest_profile_mismatch"
        if config.get("selection_contract") != "actual_id_v1":
            return None, "manifest_selection_contract_mismatch"
        if identity.get("prepare_only") is not False:
            return None, "manifest_is_not_model_inference"
        return manifest, None
    except (OSError, ValueError, KeyError, TypeError):
        return None, "missing_or_invalid_manifest"


def validate_record(record, row, profile, manifest):
    if not isinstance(record, dict):
        return "invalid_record_type", None
    if record.get("sample_id") != row["sample_id"]:
        return "sample_id_mismatch", None
    if record.get("profile") != profile:
        return "record_profile_mismatch", None
    if record.get("row_sha256") != digest(row):
        return "row_hash_mismatch", None
    if record.get("run_fingerprint") != manifest["run_fingerprint"]:
        return "run_fingerprint_mismatch", None
    if record.get("status") == "error":
        return None, None
    if record.get("status") != "ok":
        return "not_a_terminal_model_record", None
    if not isinstance(record.get("raw_output"), str):
        return "missing_raw_output", None
    scores = score_prediction(record["raw_output"], row)
    stored = record.get("scores")
    if not isinstance(stored, dict) or any(
            type(stored.get(key)) is not bool or stored[key] != scores[key]
            for key in ("valid", "strict_correct", "resolved_correct")):
        return "stored_scores_disagree_with_raw_output", None
    return None, scores


def summarize(dataset, output, review_flags_path=None):
    with dataset.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for row in rows:
        validate_row(row)
    identifiers = [row["sample_id"] for row in rows]
    if not rows or len(set(identifiers)) != len(rows):
        raise ValueError("Expected dataset must be nonempty with unique sample IDs")
    if review_flags_path is None:
        candidate = dataset.parent / "independent_review_flags.json"
        review_flags_path = candidate if candidate.exists() else None
    review_flags = None
    review_identity = None
    if review_flags_path is not None:
        review_flags = json.loads(review_flags_path.read_text())
        if review_flags.get("schema") != "qwen_selection_v3_independent_review_flags_v1":
            raise ValueError("Unknown independent-review sidecar schema")
        if not isinstance(review_flags.get("samples"), dict):
            raise ValueError("Independent-review samples must be a mapping")
        for flags in [review_flags.get("default", {}), *review_flags["samples"].values()]:
            if type(flags.get("rgb_reviewed")) is not bool:
                raise ValueError("Every independent-review entry must declare rgb_reviewed bool")
            if not isinstance(flags.get("risks", []), list) or any(not isinstance(r, str) for r in flags.get("risks", [])):
                raise ValueError("Independent-review risks must be a string list")
        review_identity = {"path": str(review_flags_path.resolve()), "sha256": sha_file(review_flags_path),
                           "explicit_sample_flags": len(review_flags["samples"])}
    dataset_sha256 = sha_file(dataset)
    report = {"updated_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
              "expected_per_profile": len(rows), "dataset_sha256": dataset_sha256,
              "independent_review_flags": review_identity, "profiles": {}}
    paired = {}
    for profile in ("low", "high"):
        profile_root = output / profile
        manifest, manifest_error = expected_manifest(profile_root, profile, dataset_sha256)
        groups = defaultdict(lambda: Counter({key: 0 for key in COUNT_KEYS}))
        times, generation, preparation, peaks = [], [], [], []
        fingerprints, incompatible_reasons = Counter(), Counter()
        correct = {}
        for row in rows:
            audit = row["audit"]
            scopes = row_scopes(row, review_flags)
            path = profile_root / "records" / (safe_id(row["sample_id"]) + ".json")
            record = scores = None
            reason = None
            if path.exists():
                try:
                    record = json.loads(path.read_text())
                    if isinstance(record, dict):
                        fingerprints[str(record.get("run_fingerprint"))] += 1
                    reason = manifest_error
                    if reason is None:
                        reason, scores = validate_record(record, row, profile, manifest)
                except (OSError, ValueError, KeyError, TypeError):
                    reason = "unreadable_or_invalid_record"
            if reason:
                incompatible_reasons[reason] += 1
            for scope in scopes:
                count = groups[scope]
                count["expected"] += 1
                if reason:
                    count["incompatible"] += 1
                elif record is None:
                    count["missing"] += 1
                elif record["status"] == "error":
                    count["runtime_error"] += 1
                else:
                    count["answered"] += 1
                    for key in ("valid", "strict_correct", "resolved_correct"):
                        count[key] += scores[key]
                    count["invalid_format_or_range"] += not scores["valid"]
            if record is not None and reason is None and record["status"] == "ok":
                for target, key in ((times, "total_seconds"), (generation, "generation_seconds"),
                                    (preparation, "prepare_seconds"), (peaks, "gpu_peak_allocated_bytes")):
                    value = record.get(key)
                    if isinstance(value, (float, int)) and not isinstance(value, bool):
                        target.append(value)
                correct[row["sample_id"]] = scores["strict_correct"]
        stats = {}
        for scope, count in groups.items():
            stats[scope] = dict(count)
            denominator = count["answered"] + count["runtime_error"]
            stats[scope]["strict_accuracy_observed_including_errors"] = count["strict_correct"] / denominator if denominator else None
            stats[scope]["resolved_accuracy_observed_including_errors"] = count["resolved_correct"] / denominator if denominator else None
            stats[scope]["complete"] = not count["missing"] and not count["incompatible"]
        report["profiles"][profile] = {
            "groups": stats, "expected_run_fingerprint": manifest["run_fingerprint"] if manifest else None,
            "manifest_error": manifest_error, "run_fingerprints": dict(fingerprints),
            "incompatible_reasons": dict(incompatible_reasons),
            "mean_seconds_per_answer": statistics.mean(times) if times else None,
            "mean_generation_seconds": statistics.mean(generation) if generation else None,
            "mean_preparation_seconds": statistics.mean(preparation) if preparation else None,
            "peak_allocated_gib": max(peaks) / 1024**3 if peaks else None}
        paired[profile] = correct
    report["paired"] = paired_counts(rows, paired, review_flags)
    report["complete"] = all(p["groups"]["all"]["complete"] for p in report["profiles"].values())
    report["all_inference_succeeded"] = report["complete"] and all(
        not p["groups"]["all"]["runtime_error"] for p in report["profiles"].values())
    report["mixed_fingerprints"] = any(len(p["run_fingerprints"]) > 1 for p in report["profiles"].values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/history_selection_v3/eval.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/history_selection_actual_id_20261107")
    parser.add_argument("--review-flags", type=Path, help="Frozen independent RGB-review sidecar; default: eval.jsonl sibling")
    args = parser.parse_args()
    report = summarize(args.dataset, args.output, args.review_flags)
    atomic_json(args.output / "summary.json", report)
    lines = ["# Qwen真实编号评估结果", "", report["updated_at"], "",
             "全部样本已有兼容记录。" if report["complete"] else "当前为部分结果或含不兼容记录，不能当作全量准确率。", "",
             "| 档位 | 已回答 | 运行错误 | 未完成 | 不兼容 | 严格答对 | 选对实际视频 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for profile, result in report["profiles"].items():
        count = result["groups"]["all"]
        lines.append(f"| {profile} | {count['answered']} | {count['runtime_error']} | {count['missing']} | {count['incompatible']} | {count['strict_correct']} | {count['resolved_correct']} |")
    lines += ["", "只接受官方XML工具格式和真实非负编号；-1始终非法且不回退计分。严格答对与选对实际视频在本合同下相同。运行失败计入已完成记录的评估分母；缺失及不兼容记录保持未完成，不进入已观测准确率。仅接受固定expected_run.json中的运行身份，并从原始输出重新核对得分。"]
    if report["independent_review_flags"] is not None:
        lines += ["", "画面审查分层来自固定sidecar：`" + report["independent_review_flags"]["sha256"] + "`。未看过的题保持not_reviewed；首次盲判和后续复核分别保留，画面困难不等于标签已通过RGB验收。"]
    lines += ["", "同题两档对照按相同样本比较；以下分型中缺失/不兼容题不进入paired计数。", "",
              "| 分型 | 两档均回答 | 两档均对 | 两档均错 | 仅低档对 | 仅高档对 |",
              "|---|---:|---:|---:|---:|---:|"]
    for scope, count in sorted(report["paired"]["groups"].items()):
        lines.append(f"| {scope} | {count['answered_in_both']} | {count['both_correct']} | {count['both_wrong']} | {count['low_only_correct']} | {count['high_only_correct']} |")
    (args.output / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({profile: result["groups"]["all"] for profile, result in report["profiles"].items()}))


if __name__ == "__main__":
    main()
