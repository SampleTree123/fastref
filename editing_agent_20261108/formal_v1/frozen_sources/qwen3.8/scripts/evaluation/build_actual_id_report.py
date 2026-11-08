#!/usr/bin/env python3
"""Read-only Chinese results/case reports for a pinned actual-ID evaluation.

No media decoding, model imports, evaluation, scheduler actions, or frozen writes.
Counts are rebuilt from verified records so a stale saved summary cannot hide gaps.
Owner/lifecycle: scripts/evaluation/README.md, experimental.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo
from evaluation_inputs_actual_id import validate_row
from media_inputs import atomic_json, sha_file
from run_model_eval_actual_id import safe_id
from summarize_results_actual_id import expected_manifest, validate_record, row_scopes, paired_counts

ROOT = Path(__file__).resolve().parents[2]
AXES = {'identity': '身份', 'hair': '发型', 'hair_color': '发色', 'wardrobe': '服装',
        'action': '动作', 'camera': '镜头', 'site': '场景'}
KINDS = {'content_reference_match': '按选择参考匹配', 'content_description_match': '按内容描述匹配',
         'event_before': '某次编辑之前', 'event_after': '某次编辑之后',
         'normal_continue': '普通继续编辑', 'explicit_latest': '明确选择最新'}
REVIEW = {'not_reviewed': '这道题还没有人工看过画面', 'reviewed_clear': '抽查画面较清楚',
          'reviewed_with_difficulty': '已查看，仍属困难题', 'unresolved_visual': '画面尚无法可靠判定'}
RISKS = {'action_reference_upper_body_only': '动作选择参考只包含上半身',
         'strong_color_cast': '强烈色偏', 'small_face_or_person': '人物或面部较小',
         'camera_name_vs_animated_body_orientation': '镜头命名与人物动态朝向容易混淆',
         'initial_blind_mismatch': '首次盲判与标准答案不一致', 'temporal_ambiguity': '时序线索不够清楚'}
COUNTS = ('expected', 'answered', 'runtime_error', 'missing', 'incompatible', 'valid', 'strict_correct')


def read_flags(path):
    if path is None:
        return None
    flags = json.loads(path.read_text())
    if flags.get('schema') != 'qwen_selection_v3_independent_review_flags_v1' or not isinstance(flags.get('samples'), dict):
        raise ValueError('Unknown review sidecar schema')
    for entry in [flags.get('default', {}), *flags['samples'].values()]:
        if type(entry.get('rgb_reviewed')) is not bool:
            raise ValueError('Review flags must explicitly declare rgb_reviewed bool')
    return flags


def review_for(row, flags):
    if flags is None:
        return {'rgb_reviewed': False, 'rgb_review_status': 'not_reviewed', 'risks': []}
    return {**flags.get('default', {}), **flags['samples'].get(row['sample_id'], {})}


def load_snapshot(dataset, results, review_flags=None, saved_summary=None):
    with dataset.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Require nonempty dataset with unique sample IDs')
    for row in rows:
        validate_row(row)
    flags = read_flags(review_flags)
    dataset_sha = sha_file(dataset)
    snapshot = {'rows': rows, 'records': {}, 'flags': flags, 'warnings': [], 'profiles': {},
                'dataset_sha256': dataset_sha, 'review_flags_sha256': sha_file(review_flags) if review_flags else None}
    paired = {}
    for profile in ('low', 'high'):
        manifest, manifest_error = expected_manifest(results / profile, profile, dataset_sha)
        groups = defaultdict(lambda: Counter({key: 0 for key in COUNTS}))
        answers = {}
        times, preparation, generation, peaks = [], [], [], []
        by_id = {}
        for row in rows:
            path = results / profile / 'records' / (safe_id(row['sample_id']) + '.json')
            state, reason, record, scores = 'missing', None, None, None
            if path.exists():
                try:
                    record = json.loads(path.read_text())
                    if manifest_error:
                        reason = manifest_error
                    else:
                        reason, scores = validate_record(record, row, profile, manifest)
                    state = 'incompatible' if reason else 'runtime_error' if record['status'] == 'error' else 'answered'
                except (ValueError, OSError, KeyError, TypeError):
                    state, reason = 'incompatible', 'unreadable_or_invalid_record'
            by_id[row['sample_id']] = {'state': state, 'reason': reason, 'record': record,
                                       'scores': scores, 'path': str(path.resolve())}
            for scope in row_scopes(row, flags):
                groups[scope]['expected'] += 1
                groups[scope][state] += 1
                if state == 'answered':
                    groups[scope]['valid'] += scores['valid']
                    groups[scope]['strict_correct'] += scores['strict_correct']
            if state == 'answered':
                answers[row['sample_id']] = scores['strict_correct']
                for target, key in ((times, 'total_seconds'), (preparation, 'prepare_seconds'),
                                    (generation, 'generation_seconds'), (peaks, 'gpu_peak_allocated_bytes')):
                    value = record.get(key)
                    if isinstance(value, (float, int)) and not isinstance(value, bool):
                        target.append(value)
        snapshot['records'][profile] = by_id
        snapshot['profiles'][profile] = {'groups': {key: dict(value) for key, value in groups.items()},
            'run_fingerprint': manifest['run_fingerprint'] if manifest else None, 'manifest_error': manifest_error,
            'mean_total_seconds': sum(times) / len(times) if times else None,
            'mean_preparation_seconds': sum(preparation) / len(preparation) if preparation else None,
            'mean_generation_seconds': sum(generation) / len(generation) if generation else None,
            'peak_allocated_gib': max(peaks) / 1024**3 if peaks else None}
        paired[profile] = answers
    snapshot['paired'] = paired_counts(rows, paired, flags)
    snapshot['all_terminal_compatible'] = all(not p['groups']['all']['missing'] and not p['groups']['all']['incompatible'] for p in snapshot['profiles'].values())
    snapshot['all_inference_succeeded'] = snapshot['all_terminal_compatible'] and all(not p['groups']['all']['runtime_error'] for p in snapshot['profiles'].values())
    snapshot['saved_summary'] = None
    if saved_summary is not None and saved_summary.exists():
        snapshot['saved_summary'] = {'path': str(saved_summary.resolve()), 'sha256': sha_file(saved_summary)}
        try:
            saved = json.loads(saved_summary.read_text())
            if saved.get('dataset_sha256') != dataset_sha:
                snapshot['warnings'].append('已保存统计对应的题目清单不同；本报告只使用本次清单和核对通过的记录。')
            for profile in ('low', 'high'):
                saved_profile = saved.get('profiles', {}).get(profile, {})
                current = snapshot['profiles'][profile]
                if saved_profile.get('expected_run_fingerprint') != current['run_fingerprint']:
                    snapshot['warnings'].append(profile + '已保存统计的结果版本不同，未采用旧统计。')
                elif any(saved_profile.get('groups', {}).get('all', {}).get(key) != current['groups']['all'][key] for key in COUNTS):
                    snapshot['warnings'].append(profile + '已保存统计尚未反映当前记录；本页已重新核对。')
        except (ValueError, OSError, TypeError):
            snapshot['warnings'].append('已保存统计无法读取；本页已逐题核对现有记录。')
    else:
        snapshot['warnings'].append('尚无已保存统计；本页只汇总当前已核对的逐题记录。')
    return snapshot


def cell(text):
    return str(text).replace('|', '\\|').replace('\n', '<br>')


def link(label, path):
    return '[' + str(label).replace('[', '\\[').replace(']', '\\]') + '](<' + str(path) + '>)'


def accuracy(count):
    denominator = count['answered'] + count['runtime_error']
    return f"{count['strict_correct']}/{denominator} ({count['strict_correct']/denominator:.1%})" if denominator else '—（尚无回答）'


def group_table(snapshot, specs):
    lines = ['| 题目类型 | 应有题数 | 低档：正确/已处理 | 高档：正确/已处理 | 同题对照：仅低对/仅高对/均与标准答案不一致 |',
             '|---|---:|---:|---:|---|']
    zero = {key: 0 for key in COUNTS}
    for label, scope in specs:
        low = snapshot['profiles']['low']['groups'].get(scope, zero)
        high = snapshot['profiles']['high']['groups'].get(scope, zero)
        pair = snapshot['paired']['groups'].get(scope, {})
        lines.append(f"| {cell(label)} | {low['expected']} | {accuracy(low)} | {accuracy(high)} | {pair.get('low_only_correct', 0)}/{pair.get('high_only_correct', 0)}/{pair.get('both_wrong', 0)}（共{pair.get('answered_in_both', 0)}题） |")
    return lines


def case_priority(row, snapshot):
    entries = [snapshot['records'][profile][row['sample_id']] for profile in ('low', 'high')]
    ok = [entry['scores']['strict_correct'] for entry in entries if entry['state'] == 'answered']
    if len(ok) == 2:
        tier = 0 if ok[0] != ok[1] else 1 if not ok[0] else 3
    elif ok:
        tier = 2 if not ok[0] else 4
    else:
        tier = 5
    reviewed = review_for(row, snapshot['flags']).get('rgb_reviewed', False)
    return tier, not reviewed, row['sample_id']


def choose_cases(snapshot, per_axis=2):
    selected = []
    for axis in AXES:
        candidates = sorted((row for row in snapshot['rows'] if row['audit'].get('selection_axis') == axis),
                            key=lambda row: case_priority(row, snapshot))
        selected.extend(candidates[:per_axis])
    controls = sorted((row for row in snapshot['rows'] if row['audit'].get('selection_axis') is None),
                      key=lambda row: case_priority(row, snapshot))
    if controls:
        selected.append(controls[0])
        alternate = next((row for row in controls[1:] if row['audit']['selection_kind'] != controls[0]['audit']['selection_kind']), None)
        if alternate:
            selected.append(alternate)
    return selected


def media_and_request(row):
    history, selection, current_refs, descriptions = [], [], [], {}
    current = ''
    after_current = False
    last_text = ''
    for message in row['model_input']['messages']:
        if not isinstance(message['content'], list):
            continue
        for block in message['content']:
            if block['type'] == 'text':
                last_text = block['text']
                if last_text.startswith('Current request: '):
                    current = last_text.removeprefix('Current request: ')
                    after_current = True
                match = re.match(r'Completed edit ([0-9]+): started from video ([0-9]+)\. The user instruction was: (.*)', last_text, re.S)
                if match:
                    descriptions[int(match[1])] = (int(match[2]), match[3])
            elif block['type'] in ('image', 'video'):
                if block.get('media_role') == 'history':
                    history.append(block['path'])
                elif block.get('reference_purpose') == 'selection':
                    selection.append((last_text, block['path']))
                elif after_current:
                    current_refs.append((last_text, block['path']))
    return current, history, selection, current_refs, descriptions


def raw_fence(raw):
    if raw is None:
        return '未产生兼容的模型工具输出。'
    runs = [len(match.group()) for match in re.finditer(r'`+', raw)]
    fence = '`' * max(3, max(runs, default=0) + 1)
    return fence + 'xml\n' + raw + ('' if raw.endswith('\n') else '\n') + fence


def render_cases(snapshot, selected, preview):
    lines = ['# Qwen真实编号选择案例' + ('（小测预览）' if preview else ''), '',
             '案例优先展示两档分歧或两档均与标准答案不一致的题；没有此类题时展示正确例。每个属性最多两题，不代表该类总体比例。', '',
             '有些题的画面仍难辨，会在案例中标记；答案不一致不一定是模型问题。首次人工判断和后续复核分别保留。']
    if preview:
        lines += ['', '**仅为报告功能的小测预览，不能替代正式全量结论。**']
    if not snapshot['all_inference_succeeded']:
        lines += ['', '**当前记录尚未全部完成：缺失、运行错误、版本不符的记录会分别标出。**']
    for number, row in enumerate(selected, 1):
        sample = row['sample_id']
        axis = AXES.get(row['audit'].get('selection_axis'), '事件与继续')
        flags = review_for(row, snapshot['flags'])
        current, history, refs, current_refs, descriptions = media_and_request(row)
        topology = {'linear': '直链', 'fork': '分叉'}.get(row['audit']['history_topology'], row['audit']['history_topology'])
        lines += ['', f"## {number}. {axis}：{sample}", '',
                  f"原{row['audit']['original_split']}组；{topology}；{KINDS.get(row['audit']['selection_kind'], row['audit']['selection_kind'])}。标准答案起点：**视频{row['supervision']['source_video_id']}**。", '',
                  '**当前用户请求：**', '', current, '',
                  '| 已有编号 | 原媒体 | 历史编辑关系与当时指令 |', '|---|---|---|']
        for index, path in enumerate(history):
            parent, instruction = descriptions.get(index, (None, '原始视频'))
            description = instruction if index == 0 else f'由视频{parent}编辑得到；{instruction}'
            lines.append(f'| 视频{index} | {link("打开原视频", path)} | {cell(description)} |')
        lines += ['', '**选择参考（用于找起点）：**', '']
        if refs:
            lines += [f'- {link(label, path)}' for label, path in refs]
        else:
            lines += ['本题无选择参考媒体；按当前请求中的内容描述或历史事件选择。']
        if current_refs:
            lines += ['', '**本轮编辑参考（用于后续修改）：**', '']
            lines += [f'- {link(label, path)}' for label, path in current_refs]
        review_status = flags.get('rgb_review_status', 'not_reviewed')
        tags = [REVIEW.get(review_status, review_status)]
        audit = row.get('audit_only', {})
        for field, label in [('explicit_history_textual_clue', '历史文字有明确属性线索'),
                             ('exact_selection_reference_in_completed_history', '选择参考在历史参考中重复'),
                             ('singleton_only_candidate', '只有一个属性状态是单独候选')]:
            if audit.get(field):
                tags.append(label)
        tags += [RISKS.get(risk, risk) for risk in flags.get('risks', [])]
        lines += ['', '**审阅与线索：** ' + '；'.join(tags) + '。']
        if flags.get('rgb_reviewed'):
            blind = flags.get('initial_blind_match_gt')
            blind_text = '一致' if blind is True else '不一致' if blind is False else '首次不确定'
            lines += ['', '首次人工判断与标准答案：' + blind_text + '；后续复核：' + str(flags.get('followup_rgb_status', '未另记')) + '。']
        lines += ['', '| 档位 | 状态 | 解析编号 | 按标准答案判分 | 原始记录 |', '|---|---|---:|---|---|']
        for profile, label in [('low', '低档'), ('high', '高档')]:
            entry = snapshot['records'][profile][sample]
            score = entry['scores']
            if entry['state'] == 'answered':
                value = str(score['source_video_id']) if score['source_video_id'] is not None else '无法解析'
                status = '有效工具调用' if score['valid'] else '非法格式或编号'
                verdict = '一致' if score['strict_correct'] else '不一致'
            else:
                status = {'missing': '未完成', 'runtime_error': '运行错误', 'incompatible': '版本不符，已隔离'}[entry['state']]
                value, verdict = '—', '不作模型能力判断'
            lines.append(f"| {label} | {status} | {value} | {verdict} | {link('JSON', entry['path']) if entry['state'] != 'missing' else '尚无'} |")
        for profile, label in [('low', '低档'), ('high', '高档')]:
            entry = snapshot['records'][profile][sample]
            lines += ['', f'**{label}原始输出：**', '']
            if entry['state'] == 'answered':
                lines.append(raw_fence(entry['record']['raw_output']))
            elif entry['state'] == 'runtime_error':
                lines.append('运行错误：' + str(entry['record'].get('error', '未记录原因'))[:500])
            else:
                lines.append('未完成。' if entry['state'] == 'missing' else '记录版本不符，未将原文当作本轮回答：' + str(entry['reason']))
    return '\n'.join(lines) + '\n'


def render_report(snapshot, preview, dataset, results, selected):
    lines = ['# Qwen实际编号评估' + ('：小测报告预览' if preview else '结果报告'), '',
             datetime.now(ZoneInfo('Asia/Singapore')).isoformat(), '']
    if preview:
        lines += ['**这是小测预览，用于验证报告生成；不能当作正式全量评估结论。**', '']
    if snapshot['all_inference_succeeded']:
        lines += [f"所选清单的{len(snapshot['rows']):,}题已完成两档回答并通过记录核对，没有运行错误。", '']
    else:
        lines += ['**当前为部分结果或含运行错误、版本不符记录，不能作为全量结论。** 尚未完成和版本不符的记录不进入正确率；运行错误计入已处理题数，但不直接解释为模型能力问题。', '']
    lines += ['严格评分要求官方XML工具调用和真实非负编号；`-1`非法且没有兜底得分。内容题保留历史文字和参考，不能将总体正确率解释为纯视觉能力。', '',
              '| 档位 | 应有 | 已回答 | 有效格式 | 正确 | 运行错误 | 缺失 | 版本不符 | 已处理严格正确率 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for profile, label in [('low', '低档'), ('high', '高档')]:
        count = snapshot['profiles'][profile]['groups']['all']
        lines.append(f"| {label} | {count['expected']} | {count['answered']} | {count['valid']} | {count['strict_correct']} | {count['runtime_error']} | {count['missing']} | {count['incompatible']} | {accuracy(count)} |")
    pair = snapshot['paired']
    lines += ['', f"同题两档均有回答{pair['answered_in_both']:,}题：均与标准答案一致{pair['both_correct']:,}；仅低档一致{pair['low_only_correct']:,}；仅高档一致{pair['high_only_correct']:,}；均与标准答案不一致{pair['both_wrong']:,}。", '',
              '以下表格的“正确/已处理”分母为已回答加运行错误；应有题数同时给出，避免部分结果看起来像全量。高低档差异只比较两档均有回答的同一道题。', '', '## 按选择依据', '']
    lines += group_table(snapshot, [(label, 'selection_axis/'+axis) for axis, label in AXES.items()] + [('事件与继续', 'selection_axis/None')])
    lines += ['', '## 直链、分叉与原数据分组', '']
    lines += group_table(snapshot, [('直链', 'history_topology/linear'), ('分叉', 'history_topology/fork'),
                                     ('原训练组', 'split/train'), ('原测试组', 'split/test')])
    lines += ['', '## 历史文字与重复参考', '']
    lines += group_table(snapshot, [(label + ('：有' if value else '：无'), field+'/'+str(value))
        for field, label in [('explicit_history_textual_clue', '明确历史文字线索'),
                             ('exact_selection_reference_in_completed_history', '选择参考在历史重复'),
                             ('singleton_only_candidate', '单独属性候选状态仅一个')]
        for value in (True, False)])
    lines += ['', '## 独立画面审查与风险', '',
              '审查记录区分已经看清、仍难辨和没有人工查看的题。画面仍有疑问时，只报告与标准答案是否一致。', '']
    if snapshot['flags'] is None:
        lines += ['当前没有提供固定审查记录，不能据此声称任何题已通过人工画面验收。']
    else:
        specs = [(label, 'review/rgb_review_status/'+status) for status, label in REVIEW.items()]
        scopes = snapshot['profiles']['low']['groups']
        specs += [(RISKS.get(scope.removeprefix('review/risk/'), scope.removeprefix('review/risk/')), scope)
                  for scope in sorted(scopes) if scope.startswith('review/risk/')]
        lines += group_table(snapshot, specs)
    lines += ['', '## 耗时与证据', '', '| 档位 | 每题平均准备/生成/合计秒 | 显存峰值GiB |', '|---|---|---:|']
    for profile, label in [('low', '低档'), ('high', '高档')]:
        values = snapshot['profiles'][profile]
        fmt = lambda key: f"{values[key]:.2f}" if values[key] is not None else '—'
        lines.append(f"| {label} | {fmt('mean_preparation_seconds')}/{fmt('mean_generation_seconds')}/{fmt('mean_total_seconds')} | {fmt('peak_allocated_gib')} |")
    lines += ['', '逐题耗时包含缓存及编译状态，跨GPU相加不等于墙钟时间；部分记录或小测均值不能保证正式运行完成时间。', '',
              f"案例见[cases.md](cases.md)，共{len(selected)}题。完整分类统计和核对依据保存在[report_audit.json](report_audit.json)。", '',
              '清单：' + link('eval.jsonl', dataset.resolve()) + '；结果目录：' + str(results.resolve()) + '。', '',
              '本报告只读取现有回答，不重新运行模型。']
    if snapshot['warnings']:
        lines += ['', '读取说明：', ''] + ['- '+warning for warning in snapshot['warnings']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True, help='Contains low/ and high/ records')
    parser.add_argument('--output', type=Path, required=True, help='Text reports only; cannot be under frozen/')
    parser.add_argument('--review-flags', type=Path)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--preview', action='store_true', help='Prominently label small-sample previews')
    parser.add_argument('--cases-per-axis', type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    if 'frozen' in args.output.resolve().parts:
        parser.error('Report output cannot be inside frozen/')
    flags = args.review_flags
    if flags is None:
        candidate = args.dataset.parent / 'independent_review_flags.json'
        flags = candidate if candidate.exists() else None
    snapshot = load_snapshot(args.dataset, args.results, flags, args.summary or args.results/'summary.json')
    selected = choose_cases(snapshot, args.cases_per_axis)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'final_report.md').write_text(render_report(snapshot, args.preview, args.dataset, args.results, selected))
    (args.output/'cases.md').write_text(render_cases(snapshot, selected, args.preview))
    audit = {key: value for key, value in snapshot.items() if key not in ('rows', 'records', 'flags')}
    audit.update(schema='qwen_actual_id_text_report_v1', preview=args.preview,
                 dataset=str(args.dataset.resolve()), results=str(args.results.resolve()),
                 report_script_sha256=sha_file(Path(__file__)), selected_case_ids=[row['sample_id'] for row in selected],
                 record_checks='expected_run fingerprint + row hash + raw-output rescoring; incompatible records excluded',
                 media_copied=False, model_inference_performed=False)
    atomic_json(args.output/'report_audit.json', audit)
    print(json.dumps({'output':str(args.output),'rows':len(snapshot['rows']),'cases':len(selected),
                      'all_inference_succeeded':snapshot['all_inference_succeeded'],'preview':args.preview}))


if __name__ == '__main__':
    main()
