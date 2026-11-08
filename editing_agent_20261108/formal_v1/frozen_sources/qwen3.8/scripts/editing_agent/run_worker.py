#!/usr/bin/env python3
"""One Qwen replica, independent chains/questions; never reads private labels."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['development', 'understanding', 'closed_loop'], required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queue', type=Path)
    parser.add_argument('--reuse-generations', type=Path,
                        help='Development only: reuse identical verified LTX outputs; Qwen still runs.')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.reuse_generations and args.mode != 'development':
        parser.error('--reuse-generations is restricted to development')
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError('Require 0 <= rank < world-size')
    if args.limit is not None and args.limit <= 0:
        raise ValueError('limit must be positive')
    args.output = args.output.resolve()
    if args.mode != 'understanding' and args.queue is None:
        raise ValueError('Real generation requires --queue')
    from queue_client import atomic_json, LTXQueueClient
    from qwen_backend import QwenBackend
    from harness import run_chain, observe_video, choose_and_edit_request
    config = json.loads(args.config.read_text())
    high = json.loads(Path(config['qwen_profile']).read_text())
    high['generation']['max_new_tokens'] = config['qwen_max_new_tokens']
    high['cache_max_gib'] = 96
    rows = [json.loads(l) for l in args.dataset.open() if l.strip()]
    if args.limit is not None:
        rows = rows[:args.limit]
    selected = [(i, row) for i, row in enumerate(rows) if i % args.world_size == args.rank]
    args.output.mkdir(parents=True, exist_ok=True)
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}
    prompt_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT / 'configs/editing_agent/prompts').rglob('*') if p.is_file()}
    reuse_source = str(args.reuse_generations.resolve()) if args.reuse_generations else None
    version = digest({'config': config, 'high': high, 'code': code, 'prompts': prompt_hashes,
                      'mode': args.mode, 'development_generation_reuse': reuse_source})
    run_identity = {'version': version, 'config': config, 'code': code, 'prompts': prompt_hashes,
                    'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(), 'mode': args.mode,
                    'development_generation_reuse': reuse_source}
    # Same identity across all ranks; rank receipt is atomic and disjoint.
    atomic_json(args.output / 'workers' / f'rank_{args.rank}_identity.json', run_identity)
    backend = QwenBackend(config=high, rank=args.rank)
    editor = LTXQueueClient(args.queue) if args.queue else None
    if args.reuse_generations:
        if editor is None:
            raise ValueError('Generation reuse requires a live LTX queue for exact identity and fallback')
        from reuse_editor import StrictReuseEditor
        editor = StrictReuseEditor(editor, previous_run=args.reuse_generations, wait_ready_seconds=120)
    start = time.monotonic()
    n_ok = 0
    errors = []
    for index, row in selected:
        key = str(row.get('chain_id', row.get('sample_id', row.get('question_id', index))))
        if not key or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in key):
            raise ValueError('Unsafe public case ID')
        directory = args.output / 'cases' / key
        directory.mkdir(parents=True, exist_ok=True)
        identity = {'version': version, 'public_input_sha256': digest(row)}
        ipath = directory / 'input_identity.json'
        if ipath.exists() and json.loads(ipath.read_text()) != identity:
            raise ValueError('Use separate output for changed harness/config/case')
        atomic_json(ipath, identity)
        final = directory / 'worker_result.json'
        if final.exists() and json.loads(final.read_text()).get('status') == 'ok':
            n_ok += 1
            continue
        began = time.monotonic()
        print(json.dumps({'event': 'case_start', 'id': key, 'mode': args.mode}), flush=True)
        try:
            if args.mode == 'understanding':
                history = []
                for h in row['history']:
                    video = Path(h['video_path'])
                    stat = video.stat()
                    cache_key = digest({'path': str(video), 'bytes': stat.st_size,
                                        'mtime_ns': stat.st_mtime_ns, 'observation_version': version})
                    cache_dir = args.output / 'observed_cards' / cache_key
                    card_path = cache_dir / 'cached_card.json'
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    with (cache_dir / '.observe.lock').open('a') as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        if card_path.exists():
                            card = json.loads(card_path.read_text())
                        else:
                            card = observe_video(str(video), h['video_id'], backend, cache_dir, config)
                            atomic_json(card_path, card)
                    # Copy the same observation into its current conversation slot.
                    item = {**h, **card, 'video_id': h['video_id'], 'video_path': str(video),
                            'parent_id': h.get('parent_id')}
                    history.append(item)
                atomic_json(directory / 'observed_history.json', history)
                result = choose_and_edit_request(history, row['turn'], backend,
                                                 directory / 'decision', config)
            else:
                if editor is None:
                    raise ValueError('Real generation requires an LTX queue')
                result = run_chain(row, directory, backend, editor, config)
                if result.get('status') != 'completed':
                    raise RuntimeError('Chain failed: ' + result.get('error', 'unknown failure'))
            record = {'status': 'ok', 'id': key, 'dataset_index': index, 'version': version,
                      'mode': args.mode, 'seconds': time.monotonic() - began, 'result': result}
            n_ok += 1
        except Exception as exc:
            record = {'status': 'error', 'id': key, 'dataset_index': index, 'version': version,
                      'mode': args.mode, 'seconds': time.monotonic() - began,
                      'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()}
            errors.append({'id': key, 'error': str(exc)})
        atomic_json(final, record)
        print(json.dumps({k: record[k] for k in ('id', 'status', 'seconds')}, ensure_ascii=False), flush=True)
        atomic_json(args.output / 'workers' / f'rank_{args.rank}_progress.json',
                    {'selected': len(selected), 'ok': n_ok, 'errors': errors, 'last_id': key})
    atomic_json(args.output / 'workers' / f'rank_{args.rank}_done.json',
                {'selected': len(selected), 'ok': n_ok, 'errors': errors,
                 'seconds': time.monotonic() - start, 'version': version})


if __name__ == '__main__':
    main()
