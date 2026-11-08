#!/usr/bin/env python3
"""Build a CPU-only, symlink-backed fixed-case comparison page.

Owner: experiments/inference/notes/fixed_case_gallery_20261031.md. Optional comparison receipts must
be completed and verified. Missing slots remain visibly pending. Source/GT keep their full timeline. Browser-incompatible codecs receive presentation
copies; CoinVE playback uses its shared time window.
"""
from __future__ import annotations
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if (p / "common/project_layout.py").is_file()) / "common"))
from project_layout import ROOT as _PROJECT_ROOT, resolve_path, script_path, add_script_paths, resolve_cli_paths
add_script_paths()

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import yaml
from urllib.parse import quote
from run_fixed_cases_parallel import collect_results, sha_file, write_status
from contact_sheet_contract import resolve_manifest_media, validate_contact_sheet
from browser_video import browser_video

ROOT = _PROJECT_ROOT
SCHEMA = 'datamake_fixed_case_gallery_v1'
SLOTS = ('base', 'step50', 'step256', 'final')


def require(value, message):
    if not value:
        raise ValueError(message)


def comparison_config_signature(folder, digest, config_path=None):
    """Verify the recorded config, excluding only the training learning rate.

    The inference entry never uses optimization.learning_rate. Keep all other
    fields in the comparison so this cannot hide a model/input setting change.
    """
    snapshot = folder / 'config_snapshot.yaml'
    candidate = snapshot if snapshot.exists() else (resolve_path(config_path) if config_path else ROOT / 'configs/training/original_images_original_time/train.yaml')
    raw = candidate.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest,
            f'Cannot verify recorded generation config: {folder}')
    config = yaml.safe_load(raw)
    config['optimization'].pop('learning_rate')
    normalized = json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
    if not snapshot.exists():
        snapshot.write_bytes(raw)
    return hashlib.sha256(normalized).hexdigest()


def read_cases(bundle_path, cases_path):
    bundle = json.loads(bundle_path.read_text())
    cases = json.loads(cases_path.read_text())
    require(bundle.get('schema') == 'datamake_mixed_sources_v1', 'Unknown mixed bundle')
    require(cases and len({x['uid'] for x in cases}) == len(cases), 'Require distinct explicit cases')
    settings = {(x['seed'], x['num_inference_steps'], x['cfg_scale']) for x in cases}
    require(len(settings) == 1, 'Comparison cases must share seed, steps and CFG')
    seed, steps, cfg = next(iter(settings))
    require(seed >= 0 and steps >= 2 and cfg >= 0, 'Invalid generation settings')
    rows, identities = {}, {}
    sources = {source['name']: source for source in bundle['sources']}
    require(len(sources) == len(bundle['sources']), 'Duplicate source name')
    selections = {}
    for case in cases:
        name, split = case['source'], case.get('split', 'development')
        require(name in sources and split in ('development', 'test', 'train'), 'Unknown case source/split')
        require(case['uid'] == f"{name}:{case['sample_id']}", 'Case UID differs from source/sample ID')
        if split == 'train':
            require(bool(case.get('evaluation_note')), 'Training examples need an explicit evaluation note')
        selections.setdefault((name, split), set()).add(case['sample_id'])
    for (name, split), wanted in selections.items():
        source = sources[name]
        manifest = resolve_path(source[f'{split}_manifest']).resolve()
        identities[name if split == 'development' else name + ':' + split] = {'path': str(manifest), 'sha256': sha_file(manifest)}
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            if row['sample_id'] in wanted:
                uid = f"{name}:{row['sample_id']}"
                require(uid not in rows and row['split'] == ('train' if split == 'train' else 'test'),
                        'Duplicate case or manifest split mismatch')
                rows[uid] = (row, resolve_path(source['dataset_root']).resolve())
    require(set(rows) == {x['uid'] for x in cases}, 'Case is missing from its explicit manifest')
    return cases, rows, identities


def checked_comparison(path, slot, cases, bundle_path, final_step):
    value = json.loads(path.read_text())
    require(value.get('schema') == 'datamake_parallel_fixed_cases_v1' and value.get('status') == 'completed',
            f'{slot}: comparison is not completed')
    require(value['cases'] == cases, f'{slot}: fixed case list/settings changed')
    expected = value['comparison_contract']
    require(all(expected.get(k) == v for k, v in {'seed': cases[0]['seed'], 'steps': cases[0]['num_inference_steps'], 'cfg': cases[0]['cfg_scale'],
                'bundle_sha256': sha_file(bundle_path)}.items())
            and resolve_path(expected['bundle']) == resolve_path(bundle_path),
            f'{slot}: input/settings contract changed')
    checkpoint = expected.get('checkpoint')
    require(value.get('checkpoint') == checkpoint, 'Parent checkpoint differs from comparison contract')
    if slot == 'base':
        require(checkpoint is None and expected.get('checkpoint_sha256') is None, 'Base slot contains a trained adapter')
    else:
        step = final_step if slot == 'final' else int(slot.removeprefix('step'))
        match = re.fullmatch(r'lora_weights_step_(\d+)\.safetensors', Path(checkpoint or '').name)
        require(match is not None and int(match[1]) == step and
                re.fullmatch(r'[0-9a-f]{64}', expected.get('checkpoint_sha256') or ''), f'{slot}: wrong checkpoint step/hash')
    folder = path.parent.resolve()
    jobs = value['jobs']
    require(set(u for j in jobs for u in j['sample_uids']) == {x['uid'] for x in cases} and
            sum(len(j['sample_uids']) for j in jobs) == len(cases), 'Worker case coverage differs')
    require(all(resolve_path(j['output']).resolve().is_relative_to(folder) for j in jobs), 'Worker output escapes comparison root')
    results = collect_results(jobs, folder, expected)
    require(results == value['results'], 'Parent summary differs from verified worker receipts')
    signatures = set()
    for job in jobs:
        worker = json.loads((resolve_path(job['output'])/'generation.json').read_text())
        require(worker['sampler'] == 'official_LTX2Scheduler_Euler', 'Unexpected sampler')
        signatures.add((worker['weights_signature'], worker['config_sha256']))
    require(len(signatures) == 1, 'Workers used different weights/configurations')
    weights_signature, config_digest = signatures.pop()
    signature = (weights_signature, comparison_config_signature(folder, config_digest, value.get('config')),
                 json.dumps(expected.get('guidance', {}), sort_keys=True))
    return results, signature, {'path': str(path), 'sha256': sha_file(path), 'checkpoint': checkpoint,
                                     'checkpoint_sha256': expected.get('checkpoint_sha256')}


def link_media(path, output, digest):
    path = path.resolve()
    require(path.is_file(), f'Missing media: {path}')
    name = digest + path.suffix.lower()
    link = output/'media'/name
    if link.is_symlink() or link.is_file():
        # Identical outputs from different checkpoints may live at different
        # paths. Content identity, rather than pathname, is the shared key.
        existing = link.resolve()
        require(existing.is_file() and sha_file(existing) == digest,
                f'Existing media link is missing or its content changed: {link}')
    else:
        require(not link.exists(), f'Refusing to overwrite media: {link}')
        link.symlink_to(path)
    return quote('media/' + name)


def media_card(label, url=None, *, image=False, start=0, end=None):
    label = html.escape(label)
    if url is None:
        body = '<div class="pending">待生成</div>'
    elif image:
        body = f'<img loading="lazy" src="{html.escape(url, quote=True)}" alt="{label}">'
    else:
        window = f' data-start="{start}"' + (f' data-end="{end}"' if end is not None else '')
        body = f'<video controls muted playsinline preload="none"{window} src="{html.escape(url, quote=True)}"></video>'
    return f'<figure><figcaption>{label}</figcaption>{body}</figure>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=resolve_path('data/mixed/original_images/sources.json'))
    parser.add_argument('--cases', type=Path, required=True, help='Explicit case list matching this comparison; no retired default list')
    parser.add_argument('--output', type=Path, required=True)
    for slot in SLOTS:
        parser.add_argument('--'+slot, type=Path, help='Completed parallel comparison.json; omit for pending')
    parser.add_argument('--final-step', type=int, default=512)
    args = parser.parse_args(); resolve_cli_paths(args)
    require(args.final_step > 256, 'Final step must follow step256')
    bundle_path, cases_path, output = args.bundle.resolve(), args.cases.resolve(), args.output.resolve()
    cases, rows, manifests = read_cases(bundle_path, cases_path)
    metadata = {'schema': SCHEMA, 'bundle_sha256': sha_file(bundle_path), 'cases_sha256': sha_file(cases_path),
                'development_manifests': manifests, 'final_step': args.final_step, 'comparisons': {}}
    results, signatures = {}, set()
    for slot in SLOTS:
        path = getattr(args, slot)
        if path is not None:
            path = path.resolve()
            found, signature, receipt = checked_comparison(path, slot, cases, bundle_path, args.final_step)
            results[slot] = (found, path.parent)
            signatures.add(signature)
            metadata['comparisons'][slot] = receipt
    require(len(signatures) <= 1, 'Comparison stages used different model/configuration identities')
    if output.exists():
        require((output/'gallery.json').is_file(), 'Existing output is not this gallery; choose a new directory')
        previous = json.loads((output/'gallery.json').read_text())
        require(all(previous.get(k) == metadata[k] for k in ('schema', 'bundle_sha256', 'cases_sha256', 'development_manifests', 'final_step')),
                'Existing gallery has different case/input identities')
    output.mkdir(parents=True, exist_ok=True)
    (output/'media').mkdir(exist_ok=True)
    verified = {}
    def original(item, dataset_root):
        path = resolve_manifest_media(item, dataset_root)
        if path not in verified:
            require(path.is_file() and path.stat().st_size == item['bytes'] and sha_file(path) == item['sha256'], f'Original media identity changed: {path}')
            verified[path] = item['sha256']
        require(verified[path] == item['sha256'], 'Conflicting original media hashes')
        display_path, display_sha = browser_video(path, item['sha256'])
        return link_media(display_path, output, display_sha)
    sections = []
    for source, title in [('ue', '合成数据 UE'), ('coinve', '真实数据 CoinVE')]:
        sections.append(f'<h2>{title}</h2>')
        for case in cases:
            if case['source'] != source:
                continue
            uid = case['uid']; row, dataset_root = rows[uid]
            window = row.get('temporal_window', {})
            start = window.get('start_seconds', 0)
            end = start + window['span_seconds'] if window else 10
            prompt = row['prompt']['en']
            chinese = row['prompt'].get('zh', '')
            sections.append(f'<article><h3>{html.escape(uid)} · {html.escape(row["axis"])}</h3><pre>{html.escape(prompt)}</pre>')
            note = case.get('evaluation_note', '')
            if case.get('split') == 'train':
                note = '训练集示例，不作为独立验证。' + note
            if note:
                sections.append(f'<p class="note">{html.escape(note)}</p>')
            if chinese:
                sections.append(f'<details><summary>中文指令</summary><p>{html.escape(chinese)}</p></details>')
            sections.append(f'<p class="note">源／真值为原文件，浏览器播放窗口 {start:.3f}–{end:.3f} 秒；未转码成训练帧率和画幅。生成结果从 0 秒开始。</p><div class="row">')
            sections.append(media_card('源视频', original(row['source'], dataset_root), start=start, end=end))
            sections.append(media_card('目标真值', original(row['target'], dataset_root), start=start, end=end))
            sections.append('</div><details open><summary>当前轮参考内容</summary><div class="refs">')
            for i, ref in enumerate(row['references']):
                is_sheet = validate_contact_sheet(ref, reference_count=len(row['references']))
                view = ('正面在左 / 背面在右' if ref['layout'] == 'left_front_right_back' else '正面在上 / 背面在下') if is_sheet else ref.get('view', '')
                sections.append(media_card(f'{i+1}. {ref["type"]} / {view}', original(ref, dataset_root),
                                           image=Path(ref['path']).suffix.lower() not in ('.mp4', '.mov', '.webm')))
            if not row['references']:
                sections.append('<p>无额外参考内容</p>')
            sections.append('</div></details><div class="row outputs">')
            for slot in SLOTS:
                url = None
                if slot in results:
                    found, parent = results[slot]; entry = found[uid]
                    url = link_media(parent/entry['output'], output, entry['sha256'])
                label = {'base':'官方基座', 'step50':'Step 50', 'step256':'Step 256', 'final':f'最终 Step {args.final_step}'}[slot]
                sections.append(media_card(label, url))
            sections.append('</div></article>')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>LTX 固定案例对照</title><style>
body{font:16px system-ui;margin:24px;background:#101724;color:#eef2f7}h1,h2{color:#b8d7ff}article{padding:20px;margin:20px 0;background:#1b2638;border-radius:12px}pre{white-space:pre-wrap;line-height:1.5;font:inherit}.row,.refs{display:flex;gap:12px;overflow-x:auto;padding:12px 0}.row figure{flex:0 0 360px}.refs{flex-wrap:wrap}.refs figure{width:180px}figure{margin:0}figcaption{padding:8px;font-weight:600}video,img,.pending{width:100%;height:225px;object-fit:contain;background:#080c12}.refs img,.refs video{height:180px}.pending{display:grid;place-items:center;color:#a7b4c9}.note{font-size:14px;color:#a7b4c9}summary{cursor:pointer}details p{white-space:pre-wrap}
</style><h1>LTX 固定案例对照</h1><p>__CASE_SETTINGS__ · 官方 Euler。各阶段使用相同案例和参考内容；空位明确等待生成。</p><p class="note">完整展示需要index.html和media目录；导出为实际媒体文件后可以下载查看或上传。拖动页面横向查看全部输出。</p>'''+''.join(sections)+'''<script>
for(const v of document.querySelectorAll('video')){const start=Number(v.dataset.start||0),end=Number(v.dataset.end||Infinity);v.addEventListener('loadedmetadata',()=>{v.currentTime=start});v.addEventListener('play',()=>{if(v.currentTime<start||v.currentTime>=end)v.currentTime=start});v.addEventListener('timeupdate',()=>{if(v.currentTime>=end){v.pause();v.currentTime=start}})}
</script></html>'''
    settings = cases[0]
    document = document.replace('__CASE_SETTINGS__', f"{len(cases)}例 · Seed {settings['seed']} · {settings['num_inference_steps']}步 · CFG {settings['cfg_scale']}")
    temporary = output/'index.html.tmp'
    temporary.write_text(document)
    temporary.replace(output/'index.html')
    metadata.update(status='page_built', sample_uids=[c['uid'] for c in cases],
                    completed_slots=list(results), pending_slots=[s for s in SLOTS if s not in results],
                    original_files_verified=len(verified), html_sha256=sha_file(output/'index.html'))
    write_status(output/'gallery.json', metadata)
    print(json.dumps({'status': 'page_built', 'index': str(output/'index.html'), 'completed_slots': list(results)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
