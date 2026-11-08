#!/usr/bin/env python3
"""Run the fixed comparison cases on idle GPUs within an existing allocation.

Owner: experiments/training/notes/mixed_training_vlm_plan_20261031.md. This never allocates GPUs,
cancels an allocation, or starts training. Explicit presets/CLI options may override case settings.
Each GPU loads the model once and handles its assigned cases sequentially.
The caller must verify that exposed GPUs are idle and belong to its allocation;
this launcher does not inspect or terminate other GPU processes.
"""
from __future__ import annotations

import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if (p / "common/project_layout.py").is_file()) / "common"))
from project_layout import ROOT as _PROJECT_ROOT, resolve_path, script_path, add_script_paths, resolve_cli_paths
add_script_paths()

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = _PROJECT_ROOT

from run_fixed_case_inference import guidance_contract, case_split_mapping


def validate_devices(devices):
    uuid = r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}'
    patterns = (r'(?:0|[1-9][0-9]*)', rf'GPU-{uuid}', rf'MIG-{uuid}',
                rf'MIG-GPU-{uuid}/(?:0|[1-9][0-9]*)/(?:0|[1-9][0-9]*)')
    kinds = []
    for device in devices:
        matches = [i for i, pattern in enumerate(patterns) if isinstance(device, str)
                   and re.fullmatch(pattern, device)]
        if not matches:
            raise ValueError('Use canonical nonnegative GPU indices or full GPU/MIG UUIDs')
        kinds.append(matches[0])
    if not devices or len(set(x.lower() for x in devices)) != len(devices) or len(set(kinds)) != 1:
        raise ValueError('Require distinct devices using one identifier format; mixed aliases are forbidden')


def sha_file(path):
    digest = hashlib.sha256()
    with resolve_path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def make_jobs(cases, devices, *, bundle, output, checkpoint=None, guidance=None, config=None):
    validate_devices(devices)
    if not cases or not devices or len({x['uid'] for x in cases}) != len(cases):
        raise ValueError('Require nonempty cases/devices and distinct case UIDs')
    sample_splits = case_split_mapping([x['uid'] for x in cases], [x.get('split', 'development') for x in cases])
    settings = {(x['seed'], x['num_inference_steps'], x['cfg_scale']) for x in cases}
    if len(settings) != 1:
        raise ValueError('Comparison cases must share seed, step count and CFG')
    seed, steps, cfg = settings.pop()
    if steps < 2:
        raise ValueError('This native comparison requires at least two steps')
    if guidance is None:
        guidance = guidance_contract(cfg_scale=cfg)
    else:
        normalized = guidance_contract(cfg, guidance['stg_scale'], guidance['guidance_rescale'],
            guidance['stg_blocks'], guidance['negative_prompt'], guidance['guidance_batch_size'], guidance.get('ref_cfg', 1.0))
        if guidance != normalized:
            raise ValueError('Guidance settings disagree with comparison cases')
    groups = [cases[i::len(devices)] for i in range(len(devices))]
    jobs = []
    for i, group in enumerate(groups):
        if not group:
            continue
        folder = output / f'gpu_{i:02d}'
        command = ['bash', str(script_path('run_trainer_python.sh')),
                   str(script_path('run_fixed_case_inference.py')),
                   '--bundle', str(bundle), '--output', str(folder),
                   '--seed', str(seed), '--steps', str(steps), '--cfg-scale', str(cfg),
                   '--stg-scale', str(guidance['stg_scale']),
                   '--guidance-rescale', str(guidance['guidance_rescale']),
                   '--guidance-batch-size', str(guidance['guidance_batch_size']),
                   '--ref-cfg', str(guidance.get('ref_cfg', 1.0))]
        if guidance['stg_blocks']:
            command += ['--stg-blocks', *map(str, guidance['stg_blocks'])]
        if guidance['negative_prompt'] is not None:
            command += ['--negative-prompt', guidance['negative_prompt']]
        if checkpoint is not None:
            command += ['--checkpoint', str(checkpoint)]
        if config is not None:
            command += ['--config', str(config)]
        for case in group:
            command += ['--uid', case['uid'], '--split', sample_splits[case['uid']]]
        jobs.append({'gpu': devices[i], 'sample_uids': [x['uid'] for x in group],
                     'sample_splits': {x['uid']: sample_splits[x['uid']] for x in group},
                     'output': str(folder), 'command': command})
    return jobs


def collect_results(jobs, output, expected):
    results = {}
    for job in jobs:
        folder = resolve_path(job['output'])
        result = json.loads((folder/'generation.json').read_text())
        if result.get('schema') != 'datamake_fixed_case_generation_v1' or result.get('status') != 'completed':
            raise ValueError('Generation receipt is incomplete or has the wrong schema')
        if result.get('sample_uids') != job['sample_uids']:
            raise ValueError('Assigned cases changed')
        default_splits = case_split_mapping(job['sample_uids'])
        assigned_splits = job.get('sample_splits', default_splits)
        if not isinstance(assigned_splits, dict) or set(assigned_splits) != set(job['sample_uids']):
            raise ValueError('Invalid assigned case split mapping')
        case_split_mapping(list(assigned_splits), list(assigned_splits.values()))
        if result.get('sample_splits', default_splits) != assigned_splits:
            raise ValueError('Generation receipt sample split assignment changed')
        for key, value in expected.items():
            if key not in result or result[key] != value:
                raise ValueError(f'Generation receipt comparison contract mismatch: {key}')
        entries = result.get('results', [])
        if [entry['sample_uid'] for entry in entries] != job['sample_uids']:
            raise ValueError('Missing, repeated or unexpected generated case')
        for entry in entries:
            if entry.get('split', 'development') != assigned_splits[entry['sample_uid']]:
                raise ValueError('Generated case split differs from assigned split')
            video = (folder/entry['output']).resolve()
            if not video.is_relative_to(folder.resolve()) or not video.is_file():
                raise ValueError(f'Generated video is missing or escapes worker output: {video}')
            if video.stat().st_size != entry['bytes'] or sha_file(video) != entry['sha256']:
                raise ValueError(f'Changed generated video: {video}')
            if entry['sample_uid'] in results:
                raise ValueError('Case repeated across workers')
            results[entry['sample_uid']] = {**entry, 'output': str(video.relative_to(output.resolve()))}
    return results


def write_status(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def stop_children(children):
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic()+15
    while any(c.poll() is None for c in children) and time.monotonic() < deadline:
        time.sleep(.1)
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        child.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, help='Defaults to the sources_config in the selected training config')
    p.add_argument('--config', type=Path, default=ROOT/'configs/training/original_images_original_time/train.yaml',
                   help='Training/reference configuration forwarded unchanged to every worker')
    p.add_argument('--cases', type=Path, required=True, help='Explicit case list matching this comparison; no retired default list')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-count', type=int, default=6)
    p.add_argument('--steps', type=int, default=None, help='Override generation steps in the case list')
    p.add_argument('--seed', type=int, default=None, help='Override seed in the case list')
    p.add_argument('--cfg-scale', type=float, default=None, help='Override the common CFG in the fixed-case list')
    p.add_argument('--stg-scale', type=float, default=0.0)
    p.add_argument('--guidance-rescale', type=float, default=0.0)
    p.add_argument('--stg-blocks', type=int, nargs='+', default=[28])
    p.add_argument('--negative-prompt', default=None)
    p.add_argument('--guidance-batch-size', type=int, default=1)
    p.add_argument('--ref-cfg', type=float, default=1.0)
    p.add_argument('--legacy-reference-cache', type=Path, help='Historical V3 image restoration cache; inference only')
    p.add_argument('--dry-run', action='store_true')
    from inference_settings import parse_with_inference_settings
    args = parse_with_inference_settings(p); resolve_cli_paths(args)
    if args.gpu_count < 1:
        p.error('GPU count must be positive')
    args.output = args.output.resolve()
    args.config = args.config.resolve()
    if not args.config.is_file():
        p.error('Configuration must be an existing file')
    if args.bundle is None:
        from native_training_contract import load_config
        bundle = load_config(args.config).get('data', {}).get('sources_config')
        if not bundle:
            p.error('Selected config has no sources_config; supply --bundle explicitly')
        args.bundle = resolve_path(bundle)
    args.bundle = args.bundle.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if args.output.exists():
        p.error('Use a new output directory')
    if checkpoint is not None and not checkpoint.is_file():
        p.error('Checkpoint must be an existing exact file')
    cases = json.loads(args.cases.read_text())
    if not cases:
        p.error('Fixed case list is empty')
    if args.steps is not None and args.steps < 2:
        p.error('steps must be at least 2')
    if args.seed is not None and args.seed < 0:
        p.error('seed must be nonnegative')
    overrides = {key: value for key, value in (('num_inference_steps', args.steps),
                 ('seed', args.seed), ('cfg_scale', args.cfg_scale)) if value is not None}
    cases = [{**case, **overrides} for case in cases]
    try:
        guidance = guidance_contract(cases[0]['cfg_scale'], args.stg_scale,
            args.guidance_rescale, args.stg_blocks, args.negative_prompt, args.guidance_batch_size, args.ref_cfg)
    except ValueError as error:
        p.error(str(error))
    devices = [x.strip() for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')]
    if args.dry_run:
        devices = [str(i) for i in range(args.gpu_count)]
    if len(devices) < args.gpu_count:
        p.error('Expose the requested idle GPUs from an existing allocation')
    validate_devices(devices)
    jobs = make_jobs(cases, devices[:args.gpu_count], bundle=args.bundle,
                     output=args.output, checkpoint=checkpoint, guidance=guidance, config=args.config)
    if args.legacy_reference_cache is not None:
        for job in jobs:
            job['command'] += ['--legacy-reference-cache', str(args.legacy_reference_cache.resolve())]
    status = {'schema': 'datamake_parallel_fixed_cases_v1', 'status': 'dry_run' if args.dry_run else 'running',
              'cases': cases, 'jobs': jobs, 'guidance': guidance, 'config': str(args.config),
              'checkpoint': str(checkpoint) if checkpoint else None,
              'started_unix': time.time()}
    if args.dry_run:
        print(json.dumps(status, ensure_ascii=False))
        return
    if checkpoint is not None:
        from checkpoint_retention import pin_for_validation
        pin_for_validation(checkpoint, args.output)
    expected = {'seed': cases[0]['seed'], 'steps': cases[0]['num_inference_steps'],
                'cfg': cases[0]['cfg_scale'], 'guidance': guidance, 'bundle': str(args.bundle),
                'bundle_sha256': sha_file(args.bundle),
                'config_sha256': sha_file(args.config),
                'checkpoint': str(checkpoint) if checkpoint else None,
                'checkpoint_sha256': sha_file(checkpoint) if checkpoint else None}
    status['comparison_contract'] = expected
    args.output.mkdir(parents=True)
    status_path = args.output/'comparison.json'
    write_status(status_path, status)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    children = []
    try:
        for i, job in enumerate(jobs):
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': job['gpu']}
            logfile = args.output/f'gpu_{i:02d}.log'
            with logfile.open('w') as handle:
                children.append(subprocess.Popen(job['command'], env=env, cwd=ROOT,
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
        while True:
            codes = [child.poll() for child in children]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f'Generation worker failed: {codes}; inspect per-GPU logs')
            if all(code == 0 for code in codes):
                break
            time.sleep(1)
        if sha_file(args.config) != expected['config_sha256']:
            raise ValueError('Configuration changed while workers were running')
        if sha_file(args.bundle) != expected['bundle_sha256'] or (
                checkpoint is not None and sha_file(checkpoint) != expected['checkpoint_sha256']):
            raise ValueError('Bundle or checkpoint changed while workers were running')
        results = collect_results(jobs, args.output, expected)
        status.update(status='completed', results=results)
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        stop_children(children)
        status['finished_unix'] = time.time()
        write_status(status_path, status)


if __name__ == '__main__':
    main()
