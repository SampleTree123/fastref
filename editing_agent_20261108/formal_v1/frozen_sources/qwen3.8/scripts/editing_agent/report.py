#!/usr/bin/env python3
"""Offline scoring and portable gallery. Private answers never enter model workers."""
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import shutil


def read_rows(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def key(row, i):
    return str(row.get('chain_id', row.get('sample_id', row.get('question_id', i))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--labels', required=True, type=Path)
    p.add_argument('--limit', type=int)
    p.add_argument('--gallery', action='store_true')
    a = p.parse_args()
    rows = read_rows(a.dataset)
    if a.limit is not None:
        rows = rows[:a.limit]
    labels = {key(r, i): r for i, r in enumerate(read_rows(a.labels))}
    report = a.run / 'report'
    report.mkdir(parents=True, exist_ok=True)
    summary = {'expected_cases': len(rows), 'cases_ok': 0, 'cases_error': 0,
               'cases_pending': 0, 'route_correct': 0, 'route_attempted': 0,
               'route_total_expected': 0, 'format_calls': 0, 'format_invalid_calls': 0,
               'interpretation': 'Understanding: exact source-ID accuracy. Generated chains: intended-route agreement only; inspect actual outputs when attributes drift.'}
    grouped = defaultdict(Counter)
    details, sections = [], []
    copied = {}
    counted_calls = set()

    def count_calls(paths):
        for callpath in paths:
            resolved = callpath.resolve()
            if resolved in counted_calls:
                continue
            counted_calls.add(resolved)
            call = json.loads(callpath.read_text())
            summary['format_calls'] += 1
            summary['format_invalid_calls'] += call.get('status') == 'invalid'

    # Understanding histories share observations across questions. These model
    # calls live outside cases; count each actual call once, not once per reuse.
    count_calls((a.run / 'observed_cards').rglob('*_call_*.json'))

    def asset(path):
        # Dereference original media links so the exported folder contains only
        # ordinary files, including when the dataset itself uses symlinks.
        source = Path(path).resolve()
        if not source.is_file():
            return None
        st = source.stat()
        identity = (str(source.resolve()), st.st_size, st.st_mtime_ns)
        if identity not in copied:
            name = hashlib.sha256(repr(identity).encode()).hexdigest()[:20] + source.suffix.lower()
            dest = report / 'media' / name
            dest.parent.mkdir(exist_ok=True)
            if not dest.exists():
                try:
                    os.link(source, dest)
                except OSError:
                    shutil.copy2(source, dest)
            copied[identity] = 'media/' + name
        return copied[identity]

    def video(path, title):
        ref = asset(path)
        if ref is None:
            return '<div>' + html.escape(title) + '：视频缺失</div>'
        return '<div class="video"><b>' + html.escape(title) + '</b><video controls preload="none" src="' + ref + '"></video></div>'

    def parent_badge(tid, parent, *, planned=False):
        if parent is None:
            return '<p class="parent pending">尚未选择起点</p>'
        branch = parent != tid - 1
        label = '原计划' if planned else '实际来源'
        suffix = ' · 回溯编辑' if branch else ' · 继续编辑'
        return ('<p class="parent ' + ('branch' if branch else 'continue') + '">' + label +
                '：#' + html.escape(str(parent)) + ' → #' + str(tid) + suffix + '</p>')

    for i, row in enumerate(rows):
        cid = key(row, i)
        label = labels[cid]
        directory = a.run / 'cases' / cid
        rpath = directory / 'worker_result.json'
        result = json.loads(rpath.read_text()) if rpath.is_file() else None
        status = result['status'] if result else 'pending'
        summary['cases_' + status] += 1
        item = {'id': cid, 'status': status, 'turns': []}
        if result and status == 'error':
            item['error'] = result.get('error')
        count_calls(directory.rglob('*_call_*.json'))
        if 'history' in row:
            expected = label['expected_source_video_id']
            predicted = result['result']['source_video_id'] if result and status == 'ok' else None
            comparisons = [(label.get('selection_kind', 'unknown'), expected, predicted)]
        else:
            comparisons = []
            for turn in label['turns']:
                progress = directory / ('turn_%02d' % turn['turn_id']) / 'progress.json'
                saved = json.loads(progress.read_text()) if progress.exists() else {}
                predicted = saved.get('request', {}).get('source_video_id')
                comparisons.append((turn.get('selection_kind', 'unknown'), turn['expected_source_video_id'], predicted))
        for kind, expected, predicted in comparisons:
            summary['route_total_expected'] += 1
            summary['route_attempted'] += predicted is not None
            summary['route_correct'] += predicted == expected
            grouped[kind]['total'] += 1
            grouped[kind]['attempted'] += predicted is not None
            grouped[kind]['correct'] += predicted == expected
            item['turns'].append({'kind': kind, 'expected': expected, 'predicted': predicted, 'match': expected == predicted})
        details.append(item)
        if a.gallery and 'root_video' in row:
            block = ['<section><h2>' + html.escape(cid) + ' · ' + status + '</h2>',
                     '<p class="hint">按生成编号从左到右；每格注明实际起点。回溯时读取标注编号，不表示从左边紧邻的视频继续。</p>',
                     '<div class="chain-scroll"><div class="chain-grid" style="--columns:' + str(len(row['turns']) + 1) + '">',
                     '<h3 class="row-label">实际编辑结果</h3><article class="chain-cell" data-video-id="0">',
                     '<p class="parent">起点 · #0</p>', video(row['root_video'], '原始视频'), '</article>']
            gt = ['<h3 class="row-label">原数据真值对照 · 与上面各轮对应，不回填实际历史</h3>',
                  '<article class="chain-cell gt-cell" data-video-id="0"><p class="parent">起点 · #0</p>',
                  video(row['root_video'], '同一原始视频'), '</article>']
            for turn, comparison, private in zip(row['turns'], item['turns'], label['turns']):
                tid = turn['turn_id']
                progress_path = directory / ('turn_%02d' % tid) / 'progress.json'
                progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
                request = progress.get('request', {})
                parent = comparison['predicted']
                block.append('<article class="chain-cell" data-video-id="%d" data-parent-id="%s">' %
                             (tid, html.escape(str(parent)) if parent is not None else ''))
                block.append(parent_badge(tid, parent))
                generated = progress.get('generation', {}).get('video_path')
                # A generation may have finished before Qwen's post-generation
                # review, or before its progress write. Show that real output.
                generation_receipt = directory / ('turn_%02d' % tid) / 'generation/editing_generation.json'
                if not generated and generation_receipt.is_file():
                    receipt = json.loads(generation_receipt.read_text())
                    if receipt.get('status') == 'ok':
                        generated = receipt.get('video_path')
                if generated:
                    block.append(video(generated, '实际生成 · 编号%d' % tid))
                else:
                    block.append('<div class="pending-video">第%d轮 · 尚无生成结果</div>' % tid)
                block.append('<details><summary>本轮指令：原文与整理后</summary><p><b>用户：</b>' +
                             html.escape(turn['user_prompt']) + '</p><p><b>交给编辑模型：</b>' +
                             html.escape(request.get('prompt', '尚未完成')) + '</p></details>')
                block.append('<details><summary>参考、内容卡与检查记录</summary><div class="reference-row">')
                for ref in turn['references']:
                    link = asset(ref['path'])
                    if link:
                        if Path(ref['path']).suffix.lower() in ('.mp4', '.mov', '.mkv'):
                            block.append(video(ref['path'], ref['reference_id']))
                        else:
                            block.append('<img loading="lazy" src="' + link + '" alt="' + html.escape(ref['reference_id']) + '">')
                block.append('</div><p>选中编号：%s；原计划编号：%s。</p><pre>' %
                             (html.escape(str(parent)), html.escape(str(comparison['expected']))) +
                             html.escape(json.dumps(progress.get('review', {}), ensure_ascii=False, indent=2)) +
                             '</pre></details></article>')
                gt.extend(['<article class="chain-cell gt-cell" data-video-id="%d">' % tid,
                           parent_badge(tid, comparison['expected'], planned=True),
                           video(private['target_video_path'], '真值对照 · 第%d轮' % tid), '</article>'])
            block.extend(gt)
            block.append('</div></div></section>')
            sections.append(''.join(block))
    summary['groups'] = dict(grouped)
    summary['accuracy_over_all_expected'] = summary['route_correct'] / summary['route_total_expected'] if summary['route_total_expected'] else None
    summary['agreement_over_attempted'] = summary['route_correct'] / summary['route_attempted'] if summary['route_attempted'] else None
    summary['all_cases_finished'] = summary['cases_pending'] == 0
    summary['all_cases_succeeded'] = summary['cases_ok'] == len(rows)
    summary['route_metric'] = ('source_id_accuracy' if all('history' in r for r in rows)
                               else 'planned_route_agreement')
    (report / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (report / 'cases.json').write_text(json.dumps(details, ensure_ascii=False, indent=2) + '\n')
    text = ['# 多轮编辑测试结果', '', '本报告由实际输出重新统计。', '',
            f"案例：{len(rows)}；完成：{summary['cases_ok']}；失败：{summary['cases_error']}；未完成：{summary['cases_pending']}。",
            f"按原计划选中编号：{summary['route_correct']}/{summary['route_total_expected']}；实际作出选择：{summary['route_attempted']}次。", '',
            '理解测试可按真实编号评分；完整生成链的画面可能偏离原计划，因此编号不一致需结合实际历史查看。模型自己的生成后检查不代替人工质量评价。', '',
            '| 选择方式 | 与原计划一致 | 已作出选择 | 应测试 |', '|---|---:|---:|---:|']
    for kind, count in sorted(grouped.items()):
        text.append(f"| {kind} | {count['correct']} | {count['attempted']} | {count['total']} |")
    (report / 'README.md').write_text('\n'.join(text) + '\n')
    if a.gallery:
        (report / 'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>多轮编辑结果</title><style>
body{font:16px sans-serif;max-width:1900px;margin:auto;padding:24px;background:#f5f6f8;color:#202b36}
section{background:white;padding:22px;margin:25px 0;border-radius:10px}.hint{color:#536273;font-size:14px}
.chain-scroll{overflow-x:auto;padding-bottom:14px}.chain-grid{display:grid;grid-template-columns:repeat(var(--columns),280px);gap:14px;width:max-content}
.row-label{grid-column:1/-1;margin:12px 0 0;font-size:17px}.chain-cell{min-width:0;padding:10px;background:#f6f8fb;border:1px solid #dbe1e8;border-radius:8px}.gt-cell{background:#f5f8f3}
.parent{font-size:14px;margin:0 0 10px;font-weight:600;min-height:34px;line-height:1.4}.branch{color:#8c4805}.continue{color:#225e83}.pending{color:#7c8189}
.video b{display:block;font-size:14px;margin-bottom:6px}.video{width:100%}video{display:block;width:100%;height:190px;object-fit:contain;background:#171b20;border-radius:5px}
.pending-video{height:213px;display:flex;align-items:center;justify-content:center;background:#e9edf2;color:#697586;border-radius:5px}
details{margin-top:12px;font-size:14px}summary{cursor:pointer;color:#274b6c}details p{overflow-wrap:anywhere}.reference-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.reference-row img{max-width:100%;max-height:260px;object-fit:contain}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}p{line-height:1.5}
</style><h1>真实生成历史的多轮编辑</h1><p>每个案例上排是实际生成，下排是真值对照；左右拖动可查看整条链。下一轮可以选择任意已有版本，实际起点见每格的“#来源 → #结果”。参考图只供编辑，不用于选择历史编号。</p>''' + ''.join(sections) + '</html>')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
