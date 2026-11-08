#!/usr/bin/env python3
"""Read stored calls and write supplementary metrics outside the frozen run.

Example: python analysis/summarize_runtime.py --run development_v3 \
  --dataset /absolute/chains_debug.jsonl --labels /absolute/private/chains_debug_labels.jsonl \
  --output analysis/development_v3
No model imports, GPU access, quality judging, or modifications to source runs.
"""
from collections import Counter, defaultdict
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics

ANALYSIS = Path(__file__).resolve().parent
CORRECTION = 'The previous tool call was not executed. Error: '


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def identifier(row, index):
    return str(row.get('chain_id', row.get('sample_id', row.get('question_id', index))))


def times(values):
    values = sorted(v for v in values if type(v) in (int, float))
    if not values:
        return {'count': 0, 'sum_seconds': 0}
    at = (len(values) - 1) * .95; lo = int(at); hi = min(lo + 1, len(values) - 1)
    return {'count': len(values), 'sum_seconds': sum(values), 'mean_seconds': statistics.mean(values),
            'median_seconds': statistics.median(values), 'p95_seconds': values[lo] + (values[hi] - values[lo]) * (at - lo)}


def summarize(run, public, labels=None):
    run = Path(run).resolve(); public_by_id = {identifier(r, i): r for i, r in enumerate(public)}
    private = {identifier(r, i): r for i, r in enumerate(labels or [])}
    stages = defaultdict(Counter); durations = defaultdict(lambda: defaultdict(list)); groups = defaultdict(list)
    selection_types = Counter(); selection_locations = Counter(); selection_value_types = Counter()
    inspections, decisions, invalid_calls, failed_cases, read_errors = [], [], [], [], []
    results, cards = {}, []

    def safe(path):
        try:
            return read(path)
        except (ValueError, OSError) as exc:
            read_errors.append({'path': str(path), 'error': str(exc)})
            return None

    for cid in public_by_id:
        path = run / 'cases' / cid / 'worker_result.json'
        if path.is_file():
            result = safe(path)
            if result:
                results[cid] = result
                if result.get('status') == 'error':
                    message = result.get('error', '')
                    match = re.search(r'\b(observe|select|edit|review): invalid output after one correction:', message)
                    failed_cases.append({'id': cid, 'error': message,
                                         'confirmed_terminal_tool_stage': match.group(1) if match else None,
                                         'error_type': result.get('error_type'), 'path': str(path.relative_to(run))})
        observed = run / 'cases' / cid / 'observed_history.json'
        state = run / 'cases' / cid / 'state.json'
        if observed.is_file():
            value = safe(observed)
            if isinstance(value, list):
                cards.extend({'case_id': cid, **v} for v in value)
        elif state.is_file():
            value = safe(state)
            if value:
                cards.extend({'case_id': cid, **v} for v in value.get('history', {}).get('videos', []))

    expected_rounds = sum(1 if 'history' in r else len(r['turns']) for r in public)
    for path in sorted(run.rglob('*_call_*.json')):
        rel = path.relative_to(run)
        # Only actual model call outputs, not frozen copies, reports or logs.
        if rel.parts[0] not in ('cases', 'observed_cards'):
            continue
        cid = rel.parts[1] if rel.parts[0] == 'cases' else None
        if cid is not None and cid not in public_by_id:
            continue
        call = safe(path)
        if not call:
            continue
        raw_stage = call.get('stage', 'unknown')
        stage = raw_stage
        if raw_stage == 'observe':
            stage = ('shared_history_observe' if rel.parts[0] == 'observed_cards' else
                     'output_only_observe' if 'output_observation' in rel.parts else 'initial_observe')
        count = stages[stage]; count['calls'] += 1
        count['valid_calls'] += call.get('status') == 'valid'
        count['invalid_calls'] += call.get('status') == 'invalid'
        count['unclassified_status_calls'] += call.get('status') not in ('valid', 'invalid')
        messages = call.get('messages', [])
        retry = bool(messages and messages[-1].get('role') == 'user' and
                     isinstance(messages[-1].get('content'), str) and messages[-1]['content'].startswith(CORRECTION))
        count['correction_calls'] += retry
        count['correction_calls_valid'] += retry and call.get('status') == 'valid'
        count['generation_limit_hits'] += bool(call.get('hit_generation_limit'))
        for name in ('prepare_seconds', 'generation_seconds', 'total_seconds'):
            if type(call.get(name)) in (int, float):
                durations[stage][name].append(call[name])
        number = int(re.search(r'_call_(\d+)\.json$', path.name).group(1))
        group_key = (str(path.parent.relative_to(run)), stage)
        groups[group_key].append((number, call, cid, raw_stage))
        if call.get('status') == 'invalid':
            invalid_calls.append({'path': str(rel), 'stage': stage, 'error': call.get('validation_error'), 'correction': retry})
        parsed = call.get('parsed_call', {}) if call.get('status') == 'valid' else {}
        turn_part = next((x for x in rel.parts if re.fullmatch(r'turn_\d+', x)), None)
        tid = int(turn_part.split('_')[1]) if turn_part else None
        label = private.get(cid, {})
        if 'turns' in label:
            label = next((t for t in label['turns'] if t['turn_id'] == tid), {})
        selection_kind = label.get('selection_kind', 'not_provided')
        if parsed.get('name') == 'inspect_history':
            ids = parsed['arguments']['video_ids']
            inspections.append({'case_id': cid, 'turn_id': tid, 'video_ids': ids,
                                'selection_kind': selection_kind, 'path': str(rel)})
        elif parsed.get('name') == 'select_source_video':
            selected = parsed['arguments']['source_video_id']; allowed = []
            for tool in call.get('tools', []):
                function = tool.get('function', {})
                if function.get('name') == 'select_source_video':
                    allowed = function.get('parameters', {}).get('properties', {}).get('source_video_id', {}).get('enum', [])
            location = 'unknown_available_ids'
            if allowed and type(selected) is int:
                location = ('only_root_available' if len(allowed) == 1 and selected == 0 else
                            'latest' if selected == max(allowed) else 'root' if selected == 0 else 'earlier_nonroot')
            selection_locations[location] += 1; selection_value_types[type(selected).__name__] += 1
            selection_types[selection_kind] += 1
            decisions.append({'case_id': cid, 'turn_id': tid, 'selected_id': selected, 'available_ids': allowed,
                              'location': location, 'selection_kind': selection_kind, 'path': str(rel)})

    for (directory, stage), entries in groups.items():
        _, last, cid, raw_stage = sorted(entries, key=lambda x: x[0])[-1]
        count = stages[stage]; count['call_groups'] += 1
        count['groups_ending_valid'] += last.get('status') == 'valid'
        if last.get('status') == 'invalid':
            error = results.get(cid, {}).get('error', '') if cid else ''
            terminal = results.get(cid, {}).get('status') == 'error' and f'{raw_stage}: invalid output after one correction:' in error
            count['groups_with_confirmed_terminal_tool_failure'] += terminal
            count['groups_ending_invalid_without_confirmed_terminal_failure'] += not terminal

    ids = [c.get('observation_identity') for c in cards if c.get('observation_identity')]
    unique = set(ids); cached = []
    for path in sorted((run / 'observed_cards').glob('*/cached_card.json')):
        record = safe(path)
        if record:
            cached.append({'path': str(path.relative_to(run)), 'observation_identity': record.get('observation_identity')})
    case_status = Counter(r.get('status', 'unknown') for r in results.values())
    case_status['pending'] = len(public) - len(results)
    timing = {stage: {name: times(values) for name, values in columns.items()} for stage, columns in durations.items()}
    return {'schema': 'editing_agent_supplementary_usage_v1', 'snapshot_at': datetime.now(timezone.utc).isoformat(),
            'run': str(run), 'expected_cases': len(public), 'expected_rounds': expected_rounds,
            'case_status': dict(case_status), 'live_snapshot_not_atomic_across_all_cases': True,
            'recorded_model_calls': sum(v['calls'] for v in stages.values()), 'stages': dict(stages),
            'history_inspection': {'accepted_tool_calls': len(inspections),
                'rounds_with_inspection': len({(i['case_id'], i['turn_id']) for i in inspections}),
                'viewed_video_slots': sum(len(i['video_ids']) for i in inspections),
                'unique_case_video_pairs': len({(i['case_id'], vid) for i in inspections for vid in i['video_ids']}),
                'per_selection_kind': dict(Counter(i['selection_kind'] for i in inspections)), 'events': inspections},
            'source_selection': {'accepted_calls': len(decisions), 'location_counts': dict(selection_locations),
                'value_type_counts': dict(selection_value_types), 'selection_kind_counts': dict(selection_types), 'events': decisions},
            'observed_card_references': {'persisted_history_slots': len(cards), 'slots_with_observation_identity': len(ids),
                'unique_observation_identities': len(unique), 'repeated_identity_assignments': len(ids) - len(unique),
                'stored_shared_card_files': len(cached), 'shared_card_files_used_by_persisted_history':
                    sum(c['observation_identity'] in unique for c in cached),
                'meaning': 'Exact duplicate references among saved histories, NOT instrumented cache-hit counts or saved GPU calls.'},
            'durations_by_stage': timing, 'completed_case_elapsed': times([r.get('seconds') for r in results.values()]),
            'invalid_calls': invalid_calls, 'terminal_cases': failed_cases, 'read_errors': read_errors,
            'metric_scope': {'exact_from_saved_evidence': ['accepted inspect_history calls and requested video IDs',
                'valid/invalid call status', 'correction requests whose final user message is the harness correction',
                'accepted source IDs and allowed-ID enums', 'stored card identity references', 'recorded numeric timings'],
                'limited_inferences': ['Terminal tool stage only when recorded worker error explicitly names it; other runtime failures remain unattributed.',
                    'Last invalid call without a terminal worker error may be a running correction, not a failed stage.'],
                'not_measured': ['actual observation-cache hit events', 'avoided GPU work', 'multi-GPU wall-clock duration',
                    'unrecorded model failures', 'editing quality', 'independent human acceptance'],
                'timing_note': 'Summed call/case seconds overlap under concurrency; they are not job duration. Cached-history lock waits may be in case time.'}}


def markdown(data):
    inspect = data['history_inspection']; cache = data['observed_card_references']
    text = ['# 工具调用与历史查看补充统计', '', '时间：' + data['snapshot_at'], '',
        f"范围：{data['expected_cases']}个案例；当前记录状态：{data['case_status']}。运行中截取的结果不是整批最终结论。", '',
        f"模型实际调用 inspect_history {inspect['accepted_tool_calls']}次，涉及{inspect['rounds_with_inspection']}轮；一共查看{inspect['viewed_video_slots']}个视频槽位。",
        f"成功输出的起点编号分布：{data['source_selection']['location_counts']}。题型按已有答案文件分类，不靠关键词猜测。", '',
        '| 阶段 | 已记录调用 | 校验失败 | 纠正调用 | 纠正成功 | 明确终止失败组 |', '|---|---:|---:|---:|---:|---:|']
    names = {'select': '选择编辑起点', 'edit': '整理编辑指令', 'review': '生成后比较', 'initial_observe': '查看原始视频', 'output_only_observe': '单独查看生成视频', 'shared_history_observe': '查看历史视频（共享缓存）'}
    for stage, value in sorted(data['stages'].items()):
        text.append(f"| {names.get(stage, stage)} | {value['calls']} | {value['invalid_calls']} | {value['correction_calls']} | {value['correction_calls_valid']} | {value.get('groups_with_confirmed_terminal_tool_failure', 0)} |")
    text.extend(['', f"已保存历史中有{cache['persisted_history_slots']}个卡片槽位，其中{cache['slots_with_observation_identity']}个保存了观察结果标识；这些标识对应{cache['unique_observation_identities']}份不同观察结果，重复引用{cache['repeated_identity_assignments']}次。未保存标识的卡片不参与此统计，不能由此推断它们没有被观察。这也不等于精确缓存命中次数。",
                 '', '耗时按已保存的模型调用记录统计，详细分阶段总和、中位数和95分位数见summary.json。多卡并行时不能把这些秒数相加当作任务总时长。',
                 '', '工具调用校验失败可能是JSON/字段/参数问题，不能全部称为语法错误；模型选择与答案不一致、画面不好不属于这里的程序失败。本报告不自动评价编辑质量。'])
    return '\n'.join(text) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True); parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    if not output.is_relative_to(ANALYSIS):
        raise SystemExit('Write supplementary outputs only under EXP/analysis, never into frozen input/code directories')
    result = summarize(args.run, rows(args.dataset), rows(args.labels) if args.labels else None)
    result['analysis_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['dataset_sha256'] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if args.labels:
        result['labels_sha256'] = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    for name, content in [('summary.json', json.dumps(result, ensure_ascii=False, indent=2) + '\n'), ('README.md', markdown(result))]:
        temp = output / ('.' + name + '.tmp'); temp.write_text(content); temp.replace(output / name)
    print(json.dumps({'output': str(output), 'cases': result['case_status'], 'inspect_calls': result['history_inspection']['accepted_tool_calls'],
                      'recorded_calls': result['recorded_model_calls'], 'read_errors': len(result['read_errors'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
