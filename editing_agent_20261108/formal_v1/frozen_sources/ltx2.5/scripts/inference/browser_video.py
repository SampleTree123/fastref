#!/usr/bin/env python3
"""Prepare browser-playable presentation copies; never change dataset media."""
import argparse
from functools import lru_cache
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'experiments/internal/inference/browser_media'


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def command(tool, *args):
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(ROOT / 'runtime/envs/trainer/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    return subprocess.run([str(ROOT / 'runtime/envs/trainer/bin' / tool), *map(str, args)],
                          env=env, check=True, capture_output=True, text=True).stdout


def probe(path):
    info = json.loads(command('ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=codec_name,pix_fmt,width,height:format=duration',
                            '-of', 'json', path))
    return info['streams'][0], float(info['format']['duration'])


@lru_cache(maxsize=1024)
def browser_video(path, source_sha):
    """Keep already compatible files; transcode other videos with their full timeline."""
    path = Path(path).resolve()
    if path.suffix.lower() not in ('.mp4', '.mov', '.webm', '.avi'):
        return path, source_sha
    receipt = CACHE / (source_sha + '.json')
    if receipt.exists():
        data = json.loads(receipt.read_text())
        dest = CACHE / data['output']
        if dest.is_file() and sha_file(dest) == data['output_sha256']:
            return dest, data['output_sha256']
    stream, duration = probe(path)
    if path.suffix.lower() == '.mp4' and stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p':
        return path, source_sha
    if sha_file(path) != source_sha:
        raise ValueError(f'Source content differs from manifest: {path}')
    CACHE.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(suffix='.mp4', dir=CACHE)
    os.close(handle)
    tmp = Path(name)
    try:
        # Preserve full duration and frame size, so existing shared-window controls remain valid.
        command('ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', path,
                '-map', '0:v:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                '-pix_fmt', 'yuv420p', '-threads', '4', '-movflags', '+faststart', tmp)
        output_stream, output_duration = probe(tmp)
        if (output_stream['codec_name'] != 'h264' or output_stream['pix_fmt'] != 'yuv420p'
                or any(output_stream[k] != stream[k] for k in ('width', 'height'))
                or abs(output_duration - duration) > .15):
            raise ValueError(f'Presentation transcode changed dimensions/duration: {path}')
        digest = sha_file(tmp)
        dest = CACHE / (digest + '.mp4')
        tmp.replace(dest)
        receipt.write_text(json.dumps({'source': str(path), 'source_sha256': source_sha,
            'source_stream': stream, 'output': dest.name, 'output_sha256': digest,
            'output_stream': output_stream, 'duration_seconds': output_duration,
            'purpose': 'Browser presentation only; full timeline, no resizing; audio omitted.'}, indent=2) + '\n')
        return dest, digest
    finally:
        tmp.unlink(missing_ok=True)


def repair_page(folder, sources):
    """Repair only source URLs in an existing exported page, keeping output videos untouched."""
    page = folder / 'index.html'
    text = page.read_text()
    changed = 0
    retired = []
    legacy = {}
    for url in re.findall(r'src="([^"]+)"', text):
        if url.startswith('cases/coinve_') and url.endswith('/source.mp4'):
            legacy.setdefault(sha_file(folder / url), []).append(url)
    for path, digest in sources:
        old_urls = [url for url in ['media/' + digest + path.suffix.lower(), *legacy.get(digest, [])] if url in text]
        if not old_urls:
            continue
        new_path, new_digest = browser_video(path, digest)
        if new_digest == digest:
            continue
        new = 'media/' + new_digest + '.mp4'
        dest = folder / new
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            shutil.copyfile(new_path, dest)
        if dest.is_symlink() or sha_file(dest) != new_digest:
            raise ValueError(f'Invalid presentation copy: {dest}')
        for old in old_urls:
            text = text.replace(old, new)
            retired.append(old)
        changed += 1
    if changed:
        tmp = page.with_suffix('.html.tmp'); tmp.write_text(text); tmp.replace(page)
        for name in ('comparison.json', 'gallery.json'):
            p = folder / name
            if p.exists():
                value = json.loads(p.read_text())
                if 'html_sha256' in value:
                    value['html_sha256'] = sha_file(page)
                p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        (folder / 'playback_fix.json').write_text(json.dumps({
            'changed_sources': changed, 'html_sha256': sha_file(page),
            'note': 'CoinVE源视频已增加H.264展示副本；原数据和模型输出未改。'}, ensure_ascii=False, indent=2) + '\n')
        # These are redundant copies in a self-contained display folder, not dataset sources.
        for old in retired:
            if old not in text:
                (folder / old).unlink(missing_ok=True)
    return {'page': str(page), 'replaced_sources': changed}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coinve-manifest', type=Path, required=True)
    p.add_argument('--display', type=Path, nargs='+', required=True)
    args = p.parse_args()
    sources = []
    for line in args.coinve_manifest.read_text().splitlines():
        row = json.loads(line)
        item = row['source']
        sources.append((Path(item['absolute_path']), item['sha256']))
    for folder in args.display:
        print(json.dumps(repair_page(folder, sources), ensure_ascii=False), flush=True)
