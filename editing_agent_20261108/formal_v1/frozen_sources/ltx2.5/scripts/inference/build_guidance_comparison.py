#!/usr/bin/env python3
"""Compare completed baseline and guided inference on the same frozen cases.

Owner: experiments/inference/notes/inference_guidance_20261102.md. No inference or training.
"""
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if (p / "common/project_layout.py").is_file()) / "common"))
from project_layout import ROOT as _PROJECT_ROOT, resolve_path, script_path, add_script_paths, resolve_cli_paths
add_script_paths()

import argparse
import html
import json
from pathlib import Path
import re

from build_fixed_case_gallery import (ROOT, read_cases, require, link_media, media_card,
                                     comparison_config_signature)
from run_fixed_cases_parallel import collect_results, sha_file, write_status
from contact_sheet_contract import resolve_manifest_media
from run_fixed_case_inference import guidance_contract


def validate_comparison_settings(baseline, guided):
    """Bind the visible labels to actual receipts, including all guidance knobs."""
    a, b = baseline['comparison_contract'], guided['comparison_contract']
    for key in ('seed', 'steps', 'bundle_sha256', 'checkpoint_sha256', 'checkpoint'):
        require((resolve_path(a[key]) == resolve_path(b[key]) if key == 'checkpoint' else a[key] == b[key]),
                f'Controlled comparison changed {key}')
    require(a['seed'] == 42 and a['steps'] == 40 and a['cfg'] == 1 and b['cfg'] == 3,
            'Expected seed42/40steps, CFG1 versus CFG3')
    match = re.fullmatch(r'lora_weights_step_(\d+)\.safetensors', Path(a['checkpoint']).name)
    require(match is not None and int(match[1]) == 682, 'This page compares step682 only')
    old_guidance = a.get('guidance')
    if old_guidance is not None:
        require(old_guidance == guidance_contract(cfg_scale=1.0), 'Baseline guidance differs from historical CFG1')
    new_guidance = b.get('guidance')
    require(isinstance(new_guidance, dict), 'Missing guided inference settings')
    require(new_guidance == guidance_contract(cfg_scale=3.0, stg_scale=1.0,
                guidance_rescale=0.7, stg_blocks=[28],
                negative_prompt=new_guidance.get('negative_prompt'),
                guidance_batch_size=new_guidance.get('guidance_batch_size', 1)),
            'Guided receipt differs from CFG3/STG1/rescale0.7/block28')
    for value in (baseline, guided):
        require(all(case['cfg_scale'] == value['comparison_contract']['cfg'] for case in value['cases']),
                'Case CFG differs from recorded worker settings')
    return a, b


def load_comparison(path, bundle, cases):
    value = json.loads(path.read_text())
    require(value.get('status') == 'completed' and value.get('schema') == 'datamake_parallel_fixed_cases_v1', 'Incomplete comparison')
    expected = value['comparison_contract']
    require(resolve_path(expected['bundle']) == resolve_path(bundle) and expected['bundle_sha256'] == sha_file(bundle), 'Wrong data bundle')
    require([(c['uid'], c['sample_id'], c['source'], c.get('split', 'development'), c['seed'], c['num_inference_steps']) for c in value['cases']]
            == [(c['uid'], c['sample_id'], c['source'], c.get('split', 'development'), c['seed'], c['num_inference_steps']) for c in cases], 'Different cases, splits or seeds/steps')
    require(value['checkpoint'] == expected['checkpoint'] and expected['checkpoint'], 'Missing/wrong parent checkpoint')
    folder = path.parent.resolve()
    require(all(resolve_path(j['output']).resolve().is_relative_to(folder) for j in value['jobs']), 'Worker escapes run')
    wanted_splits = {case['uid']: case.get('split', 'development') for case in cases}
    for job in value['jobs']:
        assigned = job.get('sample_splits', {uid:'development' for uid in job['sample_uids']})
        require(set(assigned) == set(job['sample_uids'])
                and all(uid in wanted_splits and assigned[uid] == wanted_splits[uid] for uid in assigned),
                'Worker split assignment differs from declared comparison cases')
    result = collect_results(value['jobs'], folder, expected)
    require(result == value['results'] and set(result) == {c['uid'] for c in cases}, 'Missing/changed results')
    require(all(result[uid].get('split', 'development') == split for uid, split in wanted_splits.items()),
            'Generated result split differs from displayed case split')
    signatures = set()
    for job in value['jobs']:
        worker = json.loads((resolve_path(job['output']) / 'generation.json').read_text())
        require(worker['sampler'] == 'official_LTX2Scheduler_Euler', 'Sampler changed')
        signatures.add((worker['weights_signature'], comparison_config_signature(folder, worker['config_sha256'])))
    require(len(signatures) == 1, 'Workers disagree on model/config')
    return value, signatures.pop()


def read_extra_cases(bundle_path, cases_path):
    """Extra cases with explicit split; the original fixed-12 reader is unchanged."""
    bundle = json.loads(bundle_path.read_text())
    cases = json.loads(cases_path.read_text())
    require(bundle.get('schema') == 'datamake_mixed_sources_v1', 'Unknown mixed bundle')
    require(isinstance(cases, list) and cases and len({c['uid'] for c in cases}) == len(cases),
            'Extra cases must be a nonempty list with distinct UIDs')
    sources = {source['name']: source for source in bundle['sources']}
    require(len(sources) == len(bundle['sources']), 'Duplicate source name')
    for case in cases:
        require(case['source'] in sources and case['source'] in ('ue', 'coinve')
                and case['uid'] == f"{case['source']}:{case['sample_id']}",
                'Extra case UID/source/sample ID mismatch')
        require(case.get('split', 'development') in ('train', 'development', 'test'), 'Unknown extra case split')
        require((case['seed'], case['num_inference_steps'], case['cfg_scale']) == (42, 40, 1),
                'Extra cases require baseline CFG1, seed42 and 40 steps')
    rows = {}
    for name, split in sorted({(case['source'], case.get('split', 'development')) for case in cases}):
        source = sources[name]
        wanted = {case['sample_id'] for case in cases if case['source'] == name and case.get('split', 'development') == split}
        for line in resolve_path(source[f'{split}_manifest']).read_text().splitlines():
            row = json.loads(line)
            if row['sample_id'] in wanted:
                uid = f"{name}:{row['sample_id']}"
                require(uid not in rows and row['split'] == ('train' if split == 'train' else 'test'),
                        'Duplicate or wrong split for extra case')
                rows[uid] = (row, resolve_path(source['dataset_root']).resolve())
    require(set(rows) == {case['uid'] for case in cases}, 'Extra case is outside its declared split')
    return cases, rows


def validate_groups(groups):
    """Reject duplicates and changed model/settings before writing any page media."""
    require(bool(groups), 'No comparison groups')
    seen = set(); first_identity = None; first_comparable = None
    keys = ('seed', 'steps', 'cfg', 'bundle', 'bundle_sha256', 'checkpoint', 'checkpoint_sha256')
    for group in groups:
        a, b = validate_comparison_settings(group['baseline'], group['guided'])
        require(group['baseline_signature'] == group['guided_signature'], 'Model/config differs within a comparison group')
        identity = {'model_signature': group['baseline_signature'],
                    'baseline': {**{k:a[k] for k in keys}, 'guidance': a.get('guidance') or guidance_contract()},
                    'guided': {**{k:b[k] for k in keys}, 'guidance': b['guidance']}}
        # Compare resolved locations in a disposable view; keep receipt-derived
        # identity strings untouched in the returned gallery provenance.
        comparable = {**identity}
        for section in ('baseline', 'guided'):
            comparable[section] = {**identity[section]}
            for key in ('bundle', 'checkpoint'):
                value = comparable[section][key]
                comparable[section][key] = str(resolve_path(value)) if value is not None else None
        if first_identity is None:
            first_identity = identity
            first_comparable = comparable
        else:
            require(comparable == first_comparable, 'Model/checkpoint/inference settings differ across comparison groups')
        uids = [case['uid'] for case in group['cases']]
        require(len(uids) == len(set(uids)) and not (seen & set(uids)), 'Sample UID repeats across comparison groups')
        seen.update(uids)
    return first_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--guided', type=Path, required=True)
    p.add_argument('--bundle', type=Path, default=resolve_path('data/mixed/original_images/sources.json'))
    p.add_argument('--cases', type=Path, required=True, help='Explicit case list matching this comparison; no retired default list')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extra-baseline', type=Path)
    p.add_argument('--extra-guided', type=Path)
    p.add_argument('--extra-cases', type=Path)
    args = p.parse_args(); resolve_cli_paths(args)
    extra_args = (args.extra_baseline, args.extra_guided, args.extra_cases)
    if any(value is not None for value in extra_args) and not all(value is not None for value in extra_args):
        p.error('Supply --extra-baseline, --extra-guided and --extra-cases together')
    args.bundle = args.bundle.resolve(); args.output = args.output.resolve()
    cases, rows, _ = read_cases(args.bundle, args.cases)
    baseline, old_sig = load_comparison(args.baseline.resolve(), args.bundle, cases)
    guided, new_sig = load_comparison(args.guided.resolve(), args.bundle, cases)
    require(old_sig == new_sig, 'Model/config differs between comparison runs')
    a, b = validate_comparison_settings(baseline, guided)
    groups = [{'label': '原有案例', 'cases': cases, 'rows': rows, 'baseline': baseline, 'guided': guided,
               'baseline_path': args.baseline.resolve(), 'guided_path': args.guided.resolve(),
               'baseline_signature': old_sig, 'guided_signature': new_sig, 'cases_path': args.cases.resolve()}]
    if args.extra_cases is not None:
        extra_cases, extra_rows = read_extra_cases(args.bundle, args.extra_cases)
        extra_baseline, extra_old_sig = load_comparison(args.extra_baseline.resolve(), args.bundle, extra_cases)
        extra_guided, extra_new_sig = load_comparison(args.extra_guided.resolve(), args.bundle, extra_cases)
        groups.append({'label': '补充案例', 'cases': extra_cases, 'rows': extra_rows,
                       'baseline': extra_baseline, 'guided': extra_guided,
                       'baseline_path': args.extra_baseline.resolve(), 'guided_path': args.extra_guided.resolve(),
                       'baseline_signature': extra_old_sig, 'guided_signature': extra_new_sig,
                       'cases_path': args.extra_cases.resolve()})
    validate_groups(groups)
    all_cases = [case for group in groups for case in group['cases']]
    args.output.mkdir(parents=True, exist_ok=True); (args.output / 'media').mkdir(exist_ok=True)
    sections = []
    for group in groups:
        if len(groups) > 1:
            sections.append(f'<h2>{html.escape(group["label"])}（{len(group["cases"])}个）</h2>')
        for name, label in (('ue', '合成数据'), ('coinve', '真实数据')):
            selected = [case for case in group['cases'] if case['source'] == name]
            if not selected:
                continue
            sections.append(f'<h2>{label}</h2>')
            for case in selected:
                uid = case['uid']; row, root = group['rows'][uid]
                title = case.get('display_title', row['axis'])
                sections.append(f'<article><h3>{html.escape(uid)} · {html.escape(title)}</h3><p>{html.escape(row["prompt"]["en"])}</p><p>{html.escape(row["prompt"].get("zh", ""))}</p><div class="row">')
                if case.get('split', 'development') == 'train':
                    sections[-1] = sections[-1].replace('<div class="row">', '<p><strong>训练集样本：仅展示模型对已见数据的表现，不作为未见数据的泛化评估。</strong></p><div class="row">')
                start = row.get('temporal_window', {}).get('start_seconds', 0)
                end = start + row.get('temporal_window', {}).get('span_seconds', 10)
                for role, title in (('source', '源视频'), ('target', '目标真值')):
                    item = row[role]; path = resolve_manifest_media(item, root)
                    require(path.stat().st_size == item['bytes'] and sha_file(path) == item['sha256'], 'Changed original media')
                    sections.append(media_card(title, link_media(path, args.output, item['sha256']), start=start, end=end))
                for value, receipt, title in ((group['baseline'], group['baseline_path'], '原设置：CFG 1'),
                                              (group['guided'], group['guided_path'], '新设置：CFG 3 / STG 1 / rescale 0.7')):
                    item = value['results'][uid]
                    sections.append(media_card(title, link_media(receipt.parent / item['output'], args.output, item['sha256'])))
                sections.append('</div></article>')
    settings = html.escape(json.dumps(b, ensure_ascii=False, indent=2))
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LTX step682 推理引导对比</title><style>body{font-family:system-ui;background:#111827;color:#e5e7eb;margin:24px}article{background:#1f2937;padding:18px;margin:18px 0;border-radius:12px}.row{display:grid;grid-template-columns:repeat(4,minmax(220px,1fr));gap:12px;overflow:auto}figure{margin:0}video{width:100%;max-height:420px;background:black}figcaption{padding:8px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}p{line-height:1.6}</style>
<h1>同一 step682 模型：推理参数对比</h1><p>同一批CASE_COUNT个案例、40步、seed42、相同源视频和参考内容。新设置同时启用CFG、STG、rescale及质量负面提示词，属于组合测试。源视频/真值显示原文件的对应时间窗口；生成视频从0秒开始。</p>
<details><summary>新设置的完整运行记录</summary><pre>''' + settings + '</pre></details>' + ''.join(sections) + '''
<script>document.querySelectorAll('video[data-start]').forEach(v=>{const s=+v.dataset.start,e=+v.dataset.end;v.addEventListener('loadedmetadata',()=>{v.currentTime=s});v.addEventListener('timeupdate',()=>{if(Number.isFinite(e)&&v.currentTime>=e){v.pause();v.currentTime=s}})});</script></html>'''
    page = page.replace('CASE_COUNT', str(len(all_cases)), 1)
    (args.output / 'index.html').write_text(page)
    write_status(args.output / 'comparison.json', {'status': 'completed', 'baseline': str(args.baseline.resolve()),
        'guided': str(args.guided.resolve()), 'baseline_sha256': sha_file(args.baseline), 'guided_sha256': sha_file(args.guided),
        'sample_uids': [c['uid'] for c in all_cases], 'case_count': len(all_cases),
        'groups': [{'label':g['label'], 'sample_uids':[c['uid'] for c in g['cases']],
                    'sample_splits':{c['uid']:c.get('split', 'development') for c in g['cases']},
                    'baseline':str(g['baseline_path']), 'baseline_sha256':sha_file(g['baseline_path']),
                    'guided':str(g['guided_path']), 'guided_sha256':sha_file(g['guided_path']),
                    'cases':str(g['cases_path']), 'cases_sha256':sha_file(g['cases_path'])} for g in groups],
        'html_sha256': sha_file(args.output / 'index.html')})
    print(json.dumps({'status': 'completed', 'index': str(args.output / 'index.html')}))


if __name__ == '__main__':
    main()
