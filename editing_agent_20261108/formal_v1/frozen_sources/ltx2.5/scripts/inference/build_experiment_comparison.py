#!/usr/bin/env python3
"""Compare explicit experiments on the same cases, allowing different reference images."""
from browser_video import browser_video

import argparse
import html
import json
from pathlib import Path

from build_fixed_case_gallery import (
    read_cases, checked_comparison, link_media, media_card, require, sha_file,
    resolve_manifest_media, write_status,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'media').mkdir(exist_ok=True)
    groups, completed, cases = [], [], None
    for run in manifest['runs']:
        current_cases, rows, _ = read_cases(Path(run['bundle']), Path(manifest['cases']))
        require(cases is None or current_cases == cases, 'Experiment cases differ')
        cases = current_cases
        receipt_path = Path(run['comparison'])
        results = {}
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get('status') == 'completed':
                require(Path(receipt['checkpoint']).resolve() == Path(run['checkpoint']).resolve(),
                        'Result checkpoint differs from the selected experiment')
                results, _, _ = checked_comparison(receipt_path, f"step{run['step']}", cases,
                                                   Path(run['bundle']), run['step'])
                require(receipt['guidance'] == run.get('guidance', manifest['guidance']), 'Inference guidance differs')
                completed.append(run['name'])
        groups.append((run, rows, results))
    require(groups and cases, 'No experiments or cases')
    verified = {}

    def original(item, root):
        path = resolve_manifest_media(item, root)
        if path not in verified:
            require(path.stat().st_size == item['bytes'] and sha_file(path) == item['sha256'],
                    f'Media changed: {path}')
            verified[path] = item['sha256']
        require(verified[path] == item['sha256'], 'Conflicting media identity')
        display_path, display_sha = browser_video(path, item['sha256'])
        return link_media(display_path, output, display_sha)

    sections = []
    for case in cases:
        uid = case['uid']
        first, root = groups[0][1][uid]
        window = first.get('temporal_window', {})
        start = window.get('start_seconds', 0)
        end = start + window['span_seconds'] if window else 10
        sections.append(f'<article><h2>{html.escape(uid)} · {html.escape(case.get("label", first["axis"]))}</h2>')
        sections.append(f'<p>{html.escape(case.get("evaluation_note", ""))}</p>')
        sections.append('<div class="row">' + media_card('源视频', original(first['source'], root), start=start, end=end)
                        + media_card('目标真值', original(first['target'], root), start=start, end=end) + '</div>')
        sections.append('<p class="note">源与真值展示原文件的对应时间窗口；生成结果从0秒开始。</p><div class="row results">')
        for run, rows, results in groups:
            row, dataset_root = rows[uid]
            require(all(row[k]['sha256'] == first[k]['sha256'] for k in ('source', 'target'))
                    and row.get('temporal_window') == first.get('temporal_window'),
                    'Experiments do not share source/target/time window')
            entry = results.get(uid)
            url = link_media(Path(run['comparison']).parent / entry['output'], output, entry['sha256']) if entry else None
            sections.append('<section class="experiment">' + media_card(run['label'], url))
            prompt = row['prompt'].get('zh') or row['prompt']['en']
            sections.append(f'<details><summary>本组指令与参考内容</summary><p>{html.escape(prompt)}</p><div class="refs">')
            for ref in row['references']:
                path = resolve_manifest_media(ref, dataset_root)
                sections.append(media_card(f"{ref['type']} / {ref.get('view', '')}", original(ref, dataset_root),
                                           image=path.suffix.lower() not in ('.mp4', '.mov', '.webm')))
            if not row['references']:
                sections.append('<p>无额外参考内容</p>')
            sections.append('</div></details></section>')
        sections.append('</div></article>')
    settings = manifest['guidance']
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LTX 训练实验对比</title><style>
body{font:16px system-ui;background:#101724;color:#eef2f7;margin:24px}article{background:#1b2638;padding:20px;margin:20px 0;border-radius:12px}
h1,h2{color:#b8d7ff}.row,.refs{display:flex;gap:18px;overflow-x:auto;padding:12px 0}.row>figure,.experiment{flex:0 0 430px}figure{margin:0}figcaption{padding:8px;font-weight:600}
video,img,.pending{width:100%;height:260px;object-fit:contain;background:#080c12}.pending{display:grid;place-items:center}.refs{flex-wrap:wrap}.refs figure{width:160px}.refs img,.refs video{height:160px}
.row.results{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));overflow:visible}.row.results .experiment{min-width:0}.row.results figcaption{font-size:14px;line-height:1.5}.row.results video{height:230px}@media(max-width:900px){.row.results{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:520px){.row.results{grid-template-columns:1fr}}
details p{white-space:pre-wrap;line-height:1.6}.note{color:#b0bdce;font-size:14px}summary{cursor:pointer}
</style><h1>LTX 训练实验对比</h1>'''
    document += '<p>' + html.escape(manifest['description']) + '</p>'
    document += f'<p>{len(cases)}例 · {cases[0]["num_inference_steps"]}步生成 · Seed {cases[0]["seed"]} · 文本CFG {settings["cfg_scale"]} · STG {settings["stg_scale"]} · rescale {settings["guidance_rescale"]}</p>'
    document += '<p class="note">每行最多四组；视觉参考CFG如有变化会在列标题标注，其余设置以各组记录为准。</p>'
    document += ''.join(sections) + '''<script>
for(const v of document.querySelectorAll('video')){const start=Number(v.dataset.start||0),end=Number(v.dataset.end||Infinity);v.addEventListener('loadedmetadata',()=>{v.currentTime=start});v.addEventListener('play',()=>{if(v.currentTime<start||v.currentTime>=end)v.currentTime=start});v.addEventListener('timeupdate',()=>{if(v.currentTime>=end){v.pause();v.currentTime=start}})}
</script></html>'''
    tmp = output / 'index.html.tmp'; tmp.write_text(document); tmp.replace(output / 'index.html')
    write_status(output / 'comparison.json', {'status': 'completed' if len(completed) == len(groups) else 'partial',
                 'completed_experiments': completed, 'manifest': str(args.manifest.resolve()),
                 'html_sha256': sha_file(output / 'index.html'), 'sample_uids': [c['uid'] for c in cases]})
    print(json.dumps({'page': str(output / 'index.html'), 'completed': completed}, ensure_ascii=False))


if __name__ == '__main__':
    main()
