#!/usr/bin/env python3
"""Compare raw video, exact training pixels and official VAE reconstruction."""
import argparse
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/common'))
from encoding_helpers import atomic_json, weight_paths, sha_file, shape_check


def worker(args):
    import numpy as np
    import torch
    from PIL import Image
    import prepare_datamake_media as media
    from ltx_trainer.model_loader import load_video_vae_encoder
    from ltx_pipelines.utils.blocks.video_decoder import VideoDecoder
    from ltx_pipelines.utils.media_io import encode_video
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    source = json.loads(args.cases.read_text())
    profile = json.loads((ROOT / 'configs/data/synthetic_original_resolution/media.json').read_text())
    record, weights = weight_paths(ROOT / 'runtime/models/native_model_revision.json', ROOT / 'runtime/verification/native_weights_verified.json')
    state = {'status': 'loading', 'worker': args.worker_index, 'results': [], 'started_unix': time.time(),
             'weights_revision': record['revision'], 'profile': profile,
             'vae': next(x for x in record['files'] if x['path'].split('/')[0] == 'vae')}
    receipt = args.output / f'worker_{args.worker_index:02d}.json'
    assert not receipt.exists(), 'Use a new output directory for a repeated experiment'
    atomic_json(receipt, state)
    try:
        encoder = load_video_vae_encoder(weights['vae'], device='cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
        decoder = VideoDecoder(str(weights['vae']), dtype=torch.bfloat16, device=torch.device('cuda'))
        for case in source['cases'][args.worker_index::args.workers]:
            item = case['media']; node = item['node_id']
            state.update(status='running', current_node=node); atomic_json(receipt, state)
            path = Path(item['absolute_path'])
            assert path.stat().st_size == item['bytes'] and sha_file(path) == item['sha256']
            canvas = media.video_canvas(media.probe_video(path), profile)
            frames, sampling = media.load_video_frames(path, canvas, profile)
            rgb = torch.from_numpy(np.stack([np.asarray(frame) for frame in frames])).float().div_(255)
            del frames
            pixels = rgb.mul(2).sub(1).permute(3, 0, 1, 2).unsqueeze(0).to('cuda', torch.bfloat16)
            torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
            with torch.inference_mode():
                latent = encoder(pixels)
                shape_check(latent.shape, len(rgb), canvas, profile)
                assert torch.isfinite(latent).all()
                shape = list(latent.shape)
                del pixels
                restored = torch.cat([chunk.detach().float().cpu() for chunk in decoder(
                    latent, generator=torch.Generator(device='cuda').manual_seed(42))], dim=0)
                del latent
            torch.cuda.synchronize()
            assert restored.shape == rgb.shape and torch.isfinite(restored).all()
            seconds = time.monotonic() - start
            peak = torch.cuda.max_memory_allocated()
            # Compute before MP4 conversion; exclude black letterbox pixels from a second measurement.
            mask = (rgb.amax(dim=(0, 3)) > 0)
            sse = active_sse = 0.0
            for i in range(0, len(rgb), 8):
                error = (restored[i:i+8] - rgb[i:i+8]).square()
                sse += error.sum().item()
                active_sse += error[:, mask, :].sum().item()
            mse = sse / rgb.numel()
            active_mse = active_sse / (len(rgb) * mask.sum().item() * 3)
            case_dir = args.output / 'display/cases' / f'node_{node}'
            case_dir.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(path, case_dir / 'original.mp4')
            snapshots = [0, 60, 120]
            for i in snapshots:
                for name, tensor in [('input', rgb), ('restored', restored)]:
                    Image.fromarray(tensor[i].clamp(0, 1).mul(255).round().byte().numpy()).save(case_dir / f'{name}_{i:03d}.png')
            for name, tensor in [('input', rgb), ('restored', restored)]:
                encode_video(tensor, fps=profile['fps'], audio=None, output_path=str(case_dir / f'{name}.mp4'),
                             video_chunks_number=1, crf=16, preset='fast', thread_count=2)
            result = {'node_id': node, 'source_sha256': item['sha256'], 'canvas_wh': canvas,
                      'frames': len(rgb), 'latent_shape': shape, 'seconds': seconds,
                      'peak_allocated_bytes': peak, 'rgb_mse_before_mp4': mse,
                      'rgb_psnr_before_mp4_db': -10 * math.log10(mse),
                      'nonblack_rgb_psnr_before_mp4_db': -10 * math.log10(active_mse),
                      'sampling': sampling, 'directory': str(case_dir.relative_to(args.output))}
            state['results'].append(result); atomic_json(receipt, state)
            print(json.dumps({k: v for k,v in result.items() if k != 'sampling'}), flush=True)
            del restored, rgb
        state['status'] = 'completed'
    except BaseException as e:
        state.update(status='failed', error=repr(e)); raise
    finally:
        state['finished_unix'] = time.time(); atomic_json(receipt, state)


def gallery(args, results):
    by_id = {x['node_id']: x for x in results}
    spec = json.loads(args.cases.read_text())
    parts = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>10个正面近景：VAE还原</title><style>
body{background:#121a28;color:#eee;font:16px system-ui;margin:24px}article{background:#202d40;padding:18px;margin:20px 0;border-radius:10px}.row{display:flex;gap:12px;flex-wrap:wrap}figure{margin:8px 0;flex:1;min-width:260px}video{width:100%;height:320px;background:#000}img{max-width:100%;height:auto}figcaption{padding:8px}button{padding:9px;cursor:pointer}a{color:#9ecaff}</style><h1>10个正面近景：原视频与VAE还原</h1><p>从测试集96个面部近景候选视频随机抽样，seed 42；人工发现3例实际为侧脸/背面，继续随机补抽正面案例，补抽seed 43。最终10例，只运行官方VAE，不加载LoRA或生成模型。</p><p>左→中：训练前的缩放与12fps采样；中→右：VAE编码和解码。横屏768×448、竖屏448×768，121帧。视频使用同样的H.264设置；下方PNG不经过视频再次压缩，适合看脸部细节。</p>''']
    for case in spec['cases']:
        node = case['media']['node_id']; row = by_id[node]; base = f'cases/node_{node}'
        parts.append(f'<article><h2>视频 {node} · 人物 {case["media"]["attributes"]["identity_id"]}</h2><button onclick="playGroup(this)">从头同时播放 / 暂停</button><div class="row">')
        for name, label in [('original','原始视频'),('input','训练尺寸：VAE输入'),('restored','VAE还原')]:
            parts.append(f'<figure><figcaption>{label}</figcaption><video controls muted playsinline preload="metadata" src="{base}/{name}.mp4"></video></figure>')
        parts.append(f'</div><p>压缩前后逐像素比较PSNR：{row["rgb_psnr_before_mp4_db"]:.2f} dB（全画面）。这是像素误差指标，不直接代表人脸身份保持或主观清晰度。</p><details><summary>展开第0、5、10秒的PNG对照（点击原图放大）</summary>')
        for i in [0, 60, 120]:
            parts.append(f'<p>第{i/12:g}秒</p><div class="row">')
            for name, label in [('input','VAE输入'),('restored','VAE还原')]:
                url = f'{base}/{name}_{i:03d}.png'
                parts.append(f'<figure><figcaption>{label}</figcaption><a href="{url}" target="_blank"><img loading="lazy" src="{url}"></a></figure>')
            parts.append('</div>')
        parts.append('</details></article>')
    parts.append('''<script>function playGroup(b){const vs=[...b.closest('article').querySelectorAll('video')];if(vs.some(v=>!v.paused)){vs.forEach(v=>v.pause());return}vs.forEach(v=>{v.currentTime=0;v.play()})}</script></html>''')
    (args.output / 'display/index.html').write_text(''.join(parts))
    (args.output / 'display/.nojekyll').write_text('')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4); p.add_argument('--worker-index', type=int)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.worker_index is not None:
        worker(args); return
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    assert os.environ.get('SLURM_JOB_ID') and len(devices) >= args.workers and all(devices)
    state = {'status': 'running', 'allocation': os.environ['SLURM_JOB_ID'], 'workers': args.workers,
             'started_unix': time.time(), 'cases': str(args.cases.resolve())}
    atomic_json(args.output / 'state.json', state)
    children = []
    try:
        for i in range(args.workers):
            command = [sys.executable, str(Path(__file__).resolve()), '--cases', str(args.cases.resolve()),
                       '--output', str(args.output.resolve()), '--workers', str(args.workers), '--worker-index', str(i)]
            with (args.output / f'worker_{i:02d}.log').open('w') as log:
                children.append(subprocess.Popen(command, env={**os.environ,'CUDA_VISIBLE_DEVICES':devices[i]},stdout=log,stderr=subprocess.STDOUT))
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('VAE worker failed; inspect worker logs')
            time.sleep(2)
        assert all(c.returncode==0 for c in children)
        results=[]
        for i in range(args.workers):
            receipt=json.loads((args.output/f'worker_{i:02d}.json').read_text());assert receipt['status']=='completed';results.extend(receipt['results'])
        assert len(results)==len(json.loads(args.cases.read_text())['cases'])==10
        gallery(args, results)
        state.update(status='completed',results=results)
    except BaseException as e:
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:c.wait()
        state.update(status='failed',error=repr(e));raise
    finally:
        state['finished_unix']=time.time();atomic_json(args.output/'state.json',state)


if __name__ == '__main__':main()
