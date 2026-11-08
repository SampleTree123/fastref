#!/usr/bin/env python3
"""Offline closed-loop progress/timings. Writes only under this analysis folder."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ANALYSIS = Path(__file__).resolve().parent


def stats(values):
    values = [v for v in values if type(v) in (int, float)]
    return {'count': len(values), 'median': statistics.median(values) if values else None,
            'max': max(values) if values else None, 'sum': sum(values)}


def summarize(run, dataset, limit):
    run = Path(run).resolve(); selected = dataset[:limit]; issues = []; chains = []; events = []
    counts = Counter(); source_counts = Counter(); generated_source_counts = Counter(); transitions = Counter(); generated_transitions = Counter()
    timing = defaultdict(list); peaks = defaultdict(list); seen_executions = set()

    def read(path):
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            issues.append({'path': str(path), 'error': str(exc)}); return {}

    def source_kind(path, root, earlier_outputs):
        if not path:
            return 'unknown'
        value = Path(path).resolve()
        if value == Path(root).resolve():
            return 'root_video'
        if value in earlier_outputs:
            return 'actual_generated_history'
        return 'unmatched_path'

    for row in selected:
        cid = row['chain_id']; case = run / 'cases' / cid
        state = read(case / 'state.json'); worker = read(case / 'worker_result.json')
        edits = {e['output_video_id']: e for e in state.get('history', {}).get('edits', [])}
        videos = {v['video_id']: v for v in state.get('history', {}).get('videos', [])}
        earlier_outputs = set(); available = [0]; turns = []; recorded = 0
        for idx, turn in enumerate(row['turns']):
            tid = turn['turn_id']; directory = case / f'turn_{tid:02d}'
            progress = read(directory / 'progress.json'); request = read(directory / 'edit_request.json') or progress.get('request', {})
            receipt = read(directory / 'generation/editing_generation.json')
            chosen = request.get('source_video_id'); selection_evidence = 'edit_request_or_progress' if chosen is not None else None
            # Selection can have succeeded before prompt enhancement or generation starts.
            if chosen is None:
                for p in sorted(directory.glob('select_call_*.json')):
                    call = read(p); parsed = call.get('parsed_call') or {}
                    if call.get('status') == 'valid' and parsed.get('name') == 'select_source_video':
                        chosen = parsed.get('arguments', {}).get('source_video_id'); selection_evidence = str(p.relative_to(run))
            has_selection = type(chosen) is int and chosen in available
            if chosen is not None and not has_selection:
                issues.append({'case_id': cid, 'turn_id': tid, 'error': 'source ID unavailable in earlier successful outputs', 'source_video_id': chosen})
            transition = ('first_edit' if idx == 0 else 'continue_latest' if chosen == max(available) else 'rollback') if has_selection else None
            requested_source = receipt.get('request', {}).get('source_video')
            if not requested_source and has_selection:
                requested_source = videos.get(chosen, {}).get('video_path') or (row['root_video'] if chosen == 0 else None)
            source = source_kind(requested_source, row['root_video'], earlier_outputs) if has_selection else None
            if has_selection:
                counts['selected_turns'] += 1; source_counts[source] += 1; transitions[transition] += 1
            path = receipt.get('video_path'); successful = receipt.get('status') == 'ok' and bool(path) and Path(path).is_file()
            if receipt.get('status') == 'ok' and not successful:
                issues.append({'case_id': cid, 'turn_id': tid, 'error': 'successful receipt output file missing'})
            # Dedicated audit_runtime.py checks media hashes/content. This is a light count, not a substitute.
            done = (successful and tid in edits and tid in videos and bool(progress.get('review'))
                    and idx < state.get('completed_turns', 0))
            counts['generated_turns'] += successful; counts['recorded_completed_turns'] += done; recorded += done
            counts['generated_not_recorded_turns'] += successful and not done
            counts['selected_not_generated_turns'] += has_selection and not successful
            if successful:
                generated_source_counts[source or 'unknown'] += 1
                generated_transitions[transition or 'unknown'] += 1
                execution = receipt.get('execution_sha256') or str(directory.resolve())
                if execution not in seen_executions:
                    seen_executions.add(execution)
                    for key in ('total_seconds', 'encoding_seconds', 'denoise_seconds', 'decode_seconds'):
                        timing[key].append(receipt.get('timings', {}).get(key))
                    for key in ('peak_allocated_bytes', 'denoise_peak_allocated_bytes'):
                        peaks[key].append(receipt.get(key))
                available.append(tid); earlier_outputs.add(Path(path).resolve())
            event = {'chain_id': cid, 'turn_id': tid, 'source_video_id': chosen, 'source_kind': source,
                     'transition': transition, 'selection_evidence': selection_evidence,
                     'selected': has_selection, 'generated': successful, 'recorded_completed': done,
                     'receipt_status': receipt.get('status'), 'receipt': str(directory.relative_to(run) / 'generation/editing_generation.json') if receipt else None,
                     'execution_sha256': receipt.get('execution_sha256')}
            turns.append(event); events.append(event)
        complete = recorded == len(row['turns']) and (state.get('status') == 'completed' or worker.get('status') == 'ok')
        # Do not infer recovery from file mtimes: a later failed state is still a failure.
        failed = worker.get('status') == 'error' or state.get('status') in ('failed', 'error')
        conflict = ((worker.get('status') == 'error' and state.get('status') in ('running', 'completed'))
                    or (worker.get('status') == 'ok' and state.get('status') in ('failed', 'error')))
        if conflict:
            issues.append({'case_id': cid, 'error': 'conflicting worker/state status; no automatic recovery inferred',
                           'worker_status': worker.get('status'), 'state_status': state.get('status')})
        started = bool(state or worker or any(t['selected'] or t['receipt_status'] for t in turns) or (case/'initial').exists())
        status = 'completed' if complete else 'failed' if failed else 'in_progress' if started else 'not_started'
        counts['chains_' + status] += 1
        chains.append({'chain_id': cid, 'status': status, 'expected_turns': len(row['turns']), 'recorded_turns': recorded,
                       'state_status': state.get('status'), 'last_worker_status': worker.get('status'), 'worker_state_status_conflict': conflict})
    for key in ('chains_completed', 'chains_failed', 'chains_in_progress', 'chains_not_started', 'selected_turns', 'generated_turns', 'recorded_completed_turns', 'generated_not_recorded_turns', 'selected_not_generated_turns'):
        counts.setdefault(key, 0)
    return {'schema': 'editing_generation_summary_v1', 'snapshot_at': datetime.now(timezone.utc).isoformat(),
            'run': str(run), 'limit': limit, 'expected_chains': len(selected), 'expected_turns': sum(len(r['turns']) for r in selected),
            'counts': dict(counts), 'source_counts_selected_turns': dict(source_counts), 'source_counts_generated_turns': dict(generated_source_counts),
            'transition_counts_selected_turns': dict(transitions), 'transition_counts_generated_turns': dict(generated_transitions),
            'distinct_successful_executions': len(seen_executions),
            'native_timings_seconds': {k: stats(v) for k,v in timing.items()},
            'native_peak_allocated_bytes': {k: stats(v) for k,v in peaks.items()},
            'chains': chains, 'turns': events, 'issues': issues,
            'scope': ['Successful generation requires native status=ok and existing output; SHA/provenance validation belongs to audit_runtime.py.',
                      'Recorded completion requires state history edit+video, state completed count and a saved review, not merely generated MP4.',
                      'first_edit is separate; continue_latest uses latest already generated ID; rollback means choosing an older ID.',
                      'Counts aggregate persisted turn evidence across handoffs, not worker_result.seconds. in_progress means started but unfinished, not proof of a live process.',
                      'Complete turn evidence takes precedence; otherwise any worker error or failed/error state is failed. Conflicting statuses are reported, never treated as recovery from mtimes.',
                      'Each execution digest contributes timings once; sums overlap under GPU concurrency and are NOT wall-clock elapsed time.',
                      'encoding_seconds includes media/cache preparation and dynamic text encoding, not pure VAE compute.',
                      'peak_allocated_bytes starts at pre-denoising reset, covering denoising+decode, NOT earlier encoding peak or total GPU reserved memory.',
                      'No review-based quality scores or automatic editing-quality judgments. Live snapshot is not globally atomic.']}


def markdown(d):
    source_names={'root_video':'原视频','actual_generated_history':'实际生成历史','unmatched_path':'路径未对应','unknown':'来源待确认'}
    transition_names={'first_edit':'首次编辑','continue_latest':'继续最新','rollback':'回溯','unknown':'方式待确认'}
    c=d['counts']; lines=['# 完整生成进度与耗时', '', '快照时间：'+d['snapshot_at'], '',
      f"本次只统计公开清单前{d['expected_chains']}条链、共{d['expected_turns']}轮。",
      f"链：完成{c.get('chains_completed',0)}、失败{c.get('chains_failed',0)}、开始但未完成{c.get('chains_in_progress',0)}、未开始{c.get('chains_not_started',0)}。",
      f"轮：已选源{c.get('selected_turns',0)}、已成功生成{c.get('generated_turns',0)}、完成观察并记入历史{c.get('recorded_completed_turns',0)}。",
      f"其中已生成但尚未记完历史{c.get('generated_not_recorded_turns',0)}轮；已选源但未成功生成{c.get('selected_not_generated_turns',0)}轮。", '',
      '以下源类型和回溯数量只计算已成功生成轮；完整选择计数另存JSON。',
      '源类型：'+ '、'.join(f"{source_names.get(k,k)}{v}轮" for k,v in d['source_counts_generated_turns'].items()),
      '编辑方式：'+ '、'.join(f"{transition_names.get(k,k)}{v}轮" for k,v in d['transition_counts_generated_turns'].items()), '',
      '| LTX实际执行记录 | 次数 | 中位数 | 最大值 |', '|---|---:|---:|---:|']
    names={'total_seconds':'生成总耗时','encoding_seconds':'媒体准备和文字编码','denoise_seconds':'去噪','decode_seconds':'视频解码与保存'}
    for k,v in d['native_timings_seconds'].items():
        if v['count']:lines.append(f"| {names[k]} | {v['count']} | {v['median']:.2f}秒 | {v['max']:.2f}秒 |")
    peak=d['native_peak_allocated_bytes'].get('peak_allocated_bytes',{})
    if peak.get('count'):lines+=['',f"去噪到解码阶段的已分配显存峰值：中位数{peak['median']/2**30:.2f}GiB，最大{peak['max']/2**30:.2f}GiB。去噪前才重置统计，所以这里不包含编码阶段峰值，也不是整卡显存占用。"]
    lines+=['','这些耗时来自每次实际LTX生成收据；续跑不会将同一轮重复计数，也不使用worker本次启动的seconds作为整链耗时。多卡并行的时间总和不能当作墙钟时间。','', '“开始但未完成”只代表已有执行记录，不保证当前进程仍活跃。已完成也只表示程序走完，不代表编辑效果好；这里不把模型自评当独立质量评分。源文件SHA与内容正确性请看独立程序审查。', '',f"读取或一致性问题{len(d['issues'])}项，详见summary.json。"]
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--limit',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.limit < 1: p.error('--limit must be positive')
    if not a.output.resolve().is_relative_to(ANALYSIS):p.error('--output must be under EXP/analysis')
    rows=[json.loads(s) for s in a.dataset.read_text().splitlines() if s.strip()];d=summarize(a.run,rows,a.limit)
    d['dataset_sha256']=hashlib.sha256(a.dataset.read_bytes()).hexdigest();d['analysis_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.mkdir(parents=True,exist_ok=True)
    for name,text in [('summary.json',json.dumps(d,ensure_ascii=False,indent=2)+'\n'),('README.md',markdown(d))]:
        tmp=a.output/('.'+name+'.tmp');tmp.write_text(text);tmp.replace(a.output/name)
    print(json.dumps({'counts':d['counts'],'issues':len(d['issues']),'output':str(a.output)},ensure_ascii=False))

if __name__=='__main__':main()
