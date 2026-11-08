#!/usr/bin/env python3
"""CPU-only supplementary report; never changes frozen inference or summary.json.

Experimental lifecycle and Technical/output owner: scripts/evaluation/README.md.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import re
import statistics
from zoneinfo import ZoneInfo

from media_inputs import atomic_json, sha_file
from run_model_eval import safe_id
from summarize_results import expected_manifest, validate_record

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("expected", "answered", "runtime_error", "missing", "incompatible", "valid",
          "format_error", "out_of_range", "other_protocol_error", "strict_correct", "resolved_correct")
# Same official XML envelope; expand the integer grammar only for error diagnosis.
# Correctness always comes from the unchanged, frozen score_prediction function.
INTEGER_TOOL = re.compile(
    r"\s*<tool_call>\s*<function=select_source_video>\s*<parameter=source_video_id>\s*"
    r"(-?[1-9][0-9]*|0)\s*</parameter>\s*</function>\s*</tool_call>\s*(?:<\|im_end\|>\s*)?")


def scopes(row):
    audit = row["audit"]
    result = ["all", "split/" + audit["original_split"], "kind/" + audit["selection_kind"],
              "split_kind/" + audit["original_split"] + "/" + audit["selection_kind"],
              "round/" + str(audit["edit_round"])]
    if "current_axis" in audit:
        result.append("axis/" + audit["current_axis"])
    return result


def invalid_category(raw, count, scores):
    if scores["valid"]:
        return None
    match = INTEGER_TOOL.fullmatch(raw)
    if match is None:
        return "format_error"
    if not -1 <= int(match.group(1)) < count:
        return "out_of_range"
    return "other_protocol_error"


def numeric_stats(values):
    values = sorted(value for value in values if isinstance(value, (float, int))
                    and not isinstance(value, bool) and math.isfinite(value) and value >= 0)
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None, "sum": 0}
    return {"n": len(values), "mean": statistics.mean(values), "p50": statistics.median(values),
            "p95": values[round((len(values)-1)*.95)], "max": values[-1], "sum": sum(values)}


def finish_counts(count):
    result = dict(count)
    denominator = count["answered"] + count["runtime_error"]
    result["observed_denominator_including_runtime_errors"] = denominator
    result["complete"] = not count["missing"] and not count["incompatible"]
    for metric in ("strict", "resolved"):
        result[metric + "_accuracy_observed_including_runtime_errors"] = (
            count[metric + "_correct"] / denominator if denominator else None)
        result[metric + "_accuracy_expected_lower_bound"] = (
            count[metric + "_correct"] / count["expected"] if count["expected"] else None)
    assert sum(count[key] for key in ("answered", "runtime_error", "missing", "incompatible")) == count["expected"]
    assert count["valid"] + sum(count[key] for key in ("format_error", "out_of_range", "other_protocol_error")) == count["answered"]
    return result


def pair_stats(identifiers, low, high, metric):
    pairs = Counter({key: 0 for key in ("both_correct", "high_only_correct", "low_only_correct", "both_wrong")})
    for sample_id in identifiers:
        a, b = low[sample_id][metric], high[sample_id][metric]
        pairs["both_correct" if a and b else "high_only_correct" if b else "low_only_correct" if a else "both_wrong"] += 1
    n = len(identifiers)
    low_correct = pairs["both_correct"] + pairs["low_only_correct"]
    high_correct = pairs["both_correct"] + pairs["high_only_correct"]
    return {"n": n, **pairs, "low_correct": low_correct, "high_correct": high_correct,
            "low_accuracy": low_correct/n if n else None, "high_accuracy": high_correct/n if n else None,
            "high_minus_low_percentage_points": 100*(high_correct-low_correct)/n if n else None}


def summarize_details(dataset, output):
    with dataset.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    identifiers = [row["sample_id"] for row in rows]
    if not rows or len(set(identifiers)) != len(rows):
        raise ValueError("Expected dataset must be nonempty with unique sample IDs")
    dataset_hash = sha_file(dataset)
    report = {"schema": "qwen_selection_details_v1",
              "updated_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
              "dataset_path": str(dataset.resolve()), "dataset_sha256": dataset_hash,
              "expected_per_profile": len(rows), "expected_total": 2*len(rows), "profiles": {},
              "semantics": {
                  "strict": "Official tool format and canonical parameter must match GT; -1 differs from an explicit latest slot.",
                  "resolved": "Expand -1 to latest history slot, then compare GT parent slot; repeated node/SHA aliases are not merged.",
                  "errors": "Compatible runtime errors get zero credit; missing/incompatible stay visible. Observed accuracy excludes unfinished rows; expected lower bound includes them as zero.",
                  "invalid_categories": "Malformed envelope/noncanonical integer = format_error; a canonical integer outside [-1,history_count-1], including -2, = out_of_range. Frozen scoring is unchanged.",
                  "performance": "Compatible terminal records including runtime errors where numeric data exists. Summed row times overlap across GPUs and are not elapsed wall time."}}
    terminal, answered = {}, {}
    for profile in ("low", "high"):
        profile_root = output/profile
        manifest, manifest_error = expected_manifest(profile_root, profile, dataset_hash)
        counts = defaultdict(lambda: Counter({key: 0 for key in FIELDS}))
        reasons, scorer_reasons = Counter(), Counter()
        records = []
        terminal[profile], answered[profile] = {}, {}
        for row in rows:
            sample_id = row["sample_id"]
            record_path = profile_root/"records"/(safe_id(sample_id)+".json")
            record = scores = None
            reason = None
            if record_path.exists():
                try:
                    record = json.loads(record_path.read_text())
                    reason = manifest_error
                    if reason is None:
                        reason, scores = validate_record(record, row, profile, manifest)
                except (OSError, ValueError, KeyError, TypeError):
                    reason = "unreadable_or_invalid_record"
            status = "incompatible" if reason else "missing" if record is None else "runtime_error" if record["status"] == "error" else "answered"
            if reason:
                reasons[reason] += 1
            category = None
            if status in ("answered", "runtime_error"):
                records.append(record)
                terminal[profile][sample_id] = {"strict": False, "resolved": False}
            if status == "answered":
                category = invalid_category(record["raw_output"], row["audit"]["edit_round"], scores)
                if not scores["valid"]:
                    scorer_reasons[scores["error"]] += 1
                terminal[profile][sample_id] = {metric: scores[metric+"_correct"] for metric in ("strict", "resolved")}
                answered[profile][sample_id] = terminal[profile][sample_id]
            for scope in scopes(row):
                count = counts[scope]
                count["expected"] += 1
                count[status] += 1
                if status == "answered":
                    for key in ("valid", "strict_correct", "resolved_correct"):
                        count[key] += scores[key]
                    if category:
                        count[category] += 1
        performance = {key: numeric_stats([record.get(key) for record in records]) for key in
                       ("total_seconds", "prepare_seconds", "generation_seconds", "input_tokens", "generated_tokens")}
        performance["gpu_peak_allocated_gib"] = numeric_stats([record["gpu_peak_allocated_bytes"]/1024**3 for record in records if isinstance(record.get("gpu_peak_allocated_bytes"), (float,int))])
        starts, ends = [], []
        for record in records:
            try:
                start = datetime.fromisoformat(record["created_at"])
                starts.append(start)
                ends.append(start+timedelta(seconds=record["total_seconds"]))
            except (KeyError, ValueError, TypeError):
                pass
        performance["observed_record_wall_span_seconds"] = (max(ends)-min(starts)).total_seconds() if starts else None
        report["profiles"][profile] = {"expected_run_path": str((profile_root/"expected_run.json").resolve()),
            "records_path": str((profile_root/"records").resolve()), "manifest_error": manifest_error,
            "expected_run_fingerprint": manifest["run_fingerprint"] if manifest else None,
            "groups": {scope: finish_counts(count) for scope,count in counts.items()},
            "incompatible_reasons": dict(reasons), "frozen_scorer_invalid_reasons": dict(scorer_reasons),
            "performance": performance}
    paired = {}
    for name, collections in (("answered_in_both", answered), ("terminal_in_both_including_errors", terminal)):
        shared = collections["low"].keys() & collections["high"].keys()
        by_scope = defaultdict(list)
        for row in rows:
            if row["sample_id"] in shared:
                for scope in scopes(row):
                    by_scope[scope].append(row["sample_id"])
        paired[name] = {"n": len(shared), "not_in_pair": len(rows)-len(shared),
                       "groups": {scope: {metric: pair_stats(ids, collections["low"], collections["high"], metric)
                                          for metric in ("strict", "resolved")} for scope,ids in by_scope.items()}}
    report["paired"] = paired
    report["complete"] = all(result["groups"]["all"]["complete"] for result in report["profiles"].values())
    report["all_inference_succeeded"] = report["complete"] and all(not result["groups"]["all"]["runtime_error"] for result in report["profiles"].values())
    report["status"] = "complete" if report["complete"] else "partial_do_not_report_as_full_evaluation"
    return report


def percent(value):
    return "—" if value is None else f"{100*value:.2f}%"


def number(value):
    return "—" if value is None else f"{value:.2f}"


def render_markdown(report):
    profile_labels = {"low": "低档", "high": "高档"}
    kind_labels = {"initial": "首轮", "latest": "常规继续", "numbered": "指定编号",
                   "edit_event": "历史编辑事件", "single_feature": "单属性特征"}
    lines = ["# 原题选择评估补充统计", "", report["updated_at"], "",
             "两档完整覆盖。" if report["complete"] else "**当前仅部分结果，不能作为完整评估准确率。**", "",
             f"每档应有{report['expected_per_profile']:,}题，共{report['expected_total']:,}次回答。", "",
             "| 档位 | 已回答/应有 | 运行失败 | 缺失 | 不兼容 | 格式错误 | 越界 | 严格准确率 | 起点选择准确率 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for profile,result in report["profiles"].items():
        count = result["groups"]["all"]
        lines.append(f"| {profile_labels[profile]} | {count['answered']}/{count['expected']} | {count['runtime_error']} | {count['missing']} | {count['incompatible']} | {count['format_error']} | {count['out_of_range']} | {percent(count['strict_accuracy_observed_including_runtime_errors'])} | {percent(count['resolved_accuracy_observed_including_runtime_errors'])} |")
    lines += ["", "表中准确率分母为已回答＋运行失败；缺失及不兼容均单列。JSON另给按全部应有题数计算的准确率下界，未完成不能当全量结果。严格评分要求-1约定正确；起点选择评分将-1展开为最新编号。", "",
              "| 档位/分组 | 已观测/应有 | 严格准确率 | 起点选择准确率 |", "|---|---:|---:|---:|"]
    for profile,result in report["profiles"].items():
        for scope,count in result["groups"].items():
            if not scope.startswith(("split/", "kind/")):
                continue
            group_label = {"split/train": "训练集", "split/test": "测试集"}.get(scope)
            if group_label is None:
                group_label = kind_labels.get(scope.removeprefix("kind/"), scope)
            lines.append(f"| {profile_labels[profile]}/{group_label} | {count['observed_denominator_including_runtime_errors']}/{count['expected']} | {percent(count['strict_accuracy_observed_including_runtime_errors'])} | {percent(count['resolved_accuracy_observed_including_runtime_errors'])} |")
    lines += ["",
              "| 成对范围 | 题数 | 严格：两档均对/仅高对/仅低对/均错 | 起点选择：两档均对/仅高对/仅低对/均错 |", "|---|---:|---|---|"]
    for name,pairs in report["paired"].items():
        values = []
        for metric in ("strict", "resolved"):
            count = pairs["groups"].get("all", {}).get(metric, {})
            values.append("/".join(str(count.get(key,0)) for key in ("both_correct","high_only_correct","low_only_correct","both_wrong")))
        label = "两档均返回回答" if name == "answered_in_both" else "两档均有终态（运行失败计错）"
        lines.append(f"| {label} | {pairs['n']} | {values[0]} | {values[1]} |")
    lines += ["", "JSON的profiles.<档>.groups保留split/train、split/test、kind/<选择类型>、split_kind、round和axis分组；paired下两种范围也保留相同分组。", "",
              "| 档位 | 每题均值/P95(s) | 准备均值(s) | 生成均值(s) | 峰值显存(GiB) | 已观测时间窗(s) |", "|---|---:|---:|---:|---:|---:|"]
    for profile,result in report["profiles"].items():
        perf = result["performance"]
        lines.append(f"| {profile_labels[profile]} | {number(perf['total_seconds']['mean'])}/{number(perf['total_seconds']['p95'])} | {number(perf['prepare_seconds']['mean'])} | {number(perf['generation_seconds']['mean'])} | {number(perf['gpu_peak_allocated_gib']['max'])} | {number(perf['observed_record_wall_span_seconds'])} |")
    lines += ["",
              "性能见profiles.<档>.performance：total_seconds、prepare_seconds、generation_seconds均含n/mean/p50/p95/max/sum，另含input_tokens、generated_tokens、gpu_peak_allocated_gib及已观测记录时间窗。跨GPU的sum不代表实际墙钟用时。", "",
              "错误分类只用于补充诊断：XML中合规整数越出[-1,历史槽位数−1]记越界，包括-2；其他结构或参数格式错误记格式错误。原始冻结评分不变，frozen_scorer_invalid_reasons保留原解析器分类。"]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT/"datasets/history_selection_original/eval.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT/"experiments/history_selection_20261106")
    args = parser.parse_args()
    report = summarize_details(args.dataset, args.output)
    atomic_json(args.output/"selection_details.json", report)
    (args.output/"selection_details.md").write_text(render_markdown(report))
    print(json.dumps({"status": report["status"], "expected_per_profile": report["expected_per_profile"],
                      "profiles": {key: value["groups"]["all"] for key,value in report["profiles"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
