#!/usr/bin/env python3
"""Run on an existing allocation: Qwen/LTX pairs, or Qwen-only understanding."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LTX = ROOT.parent / 'ltx2.5'
sys.path.insert(0, str(Path(__file__).parent))
from queue_client import atomic_json


def ltx_command(config, queue):
    """Match recorded settings to the persistent worker's explicit interface."""
    fixed = {'cfg_scale': 3.0, 'stg_scale': 1.0, 'guidance_rescale': 0.7, 'stg_blocks': [28]}
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError(f'LTX worker uses fixed {key}={value}; supplied config differs')
    def native_path(value):
        path = Path(value)
        path = path if path.is_absolute() else LTX / path
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path.resolve())
    return ['bash', str(LTX / 'scripts/common/run_trainer_python.sh'),
            str(LTX / 'scripts/inference/editing_tool_worker.py'), '--queue', str(queue),
            '--config', native_path(config['config']), '--checkpoint', native_path(config['checkpoint']),
            '--steps', str(config['steps']), '--ref-cfg', str(config['ref_cfg'])]


def stop_children(children):
    for p in children:
        if p.poll() is None:
            p.terminate()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and any(p.poll() is None for p in children):
        time.sleep(0.2)
    for p in children:
        if p.poll() is None:
            p.kill()
        p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['development', 'understanding', 'closed_loop'], required=True)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gpus', type=int, choices=[2, 4, 8], required=True)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--reuse-generations', type=Path)
    a = ap.parse_args()
    if a.reuse_generations and a.mode != 'development':
        ap.error('--reuse-generations is restricted to development')
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != a.gpus or not all(devices) or len(set(devices)) != a.gpus:
        raise ValueError('Expose exactly the requested allocated GPUs; got ' + repr(devices))
    cfg = json.loads(a.config.read_text())
    if a.limit is not None and a.limit <= 0:
        raise ValueError('limit must be positive')
    a.output.mkdir(parents=True, exist_ok=True)
    allocation_lock = (a.output / '.allocation.lock').open('a')
    fcntl.flock(allocation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logs = a.output / 'logs'
    logs.mkdir(exist_ok=True)
    job = os.environ.get('SLURM_JOB_ID', 'local')
    status = {'mode': a.mode, 'job': job, 'started_unix': time.time(), 'status': 'starting',
              'dataset': str(a.dataset.resolve()), 'config': str(a.config.resolve()),
              'gpus': a.gpus, 'limit': a.limit, 'children': []}
    atomic_json(a.output / 'allocation.json', status)
    children, streams, ltx_workers, qwen_workers, queues = [], [], [], [], []
    interrupted = []
    def interrupted_handler(sig, frame):
        interrupted.append(sig)
        for proc in children:
            if proc.poll() is None:
                proc.terminate()
    signal.signal(signal.SIGTERM, interrupted_handler)
    signal.signal(signal.SIGINT, interrupted_handler)
    def start(command, device, kind, rank):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        stream = (logs / f'job_{job}_{kind}_{rank}.log').open('a', buffering=1)
        streams.append(stream)
        proc = subprocess.Popen(command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        children.append(proc)
        status['children'].append({'kind': kind, 'rank': rank, 'device': device, 'pid': proc.pid, 'command': command})
        atomic_json(a.output / 'allocation.json', status)
        return proc
    try:
        workers = a.gpus if a.mode == 'understanding' else a.gpus // 2
        if a.mode != 'understanding':
            for rank in range(workers):
                queue = (a.output / 'queues' / f'job_{job}_pair_{rank}').resolve()
                queue.mkdir(parents=True, exist_ok=True)
                # Previous stopped worker may be restarted with identical requests.
                for marker in ['STOP', 'ready.json', 'fatal.json']:
                    (queue / marker).unlink(missing_ok=True)
                queues.append(queue)
                command = ltx_command(cfg['ltx'], queue)
                ltx_workers.append(start(command, devices[rank * 2 + 1], 'ltx', rank))
        # Qwen loading/initial observation overlaps with LTX startup.
        for rank in range(workers):
            command = ['bash', str(ROOT / 'scripts/run_gpu_python.sh'),
                       str(ROOT / 'scripts/editing_agent/run_worker.py'), '--mode', a.mode,
                       '--dataset', str(a.dataset.resolve()), '--config', str(a.config.resolve()),
                       '--output', str(a.output.resolve()), '--rank', str(rank), '--world-size', str(workers)]
            if a.limit is not None:
                command.extend(['--limit', str(a.limit)])
            if a.reuse_generations:
                command.extend(['--reuse-generations', str(a.reuse_generations.resolve())])
            if queues:
                command.extend(['--queue', str(queues[rank])])
            qwen_workers.append(start(command, devices[rank if a.mode == 'understanding' else rank * 2], 'qwen', rank))
        status['status'] = 'running'
        atomic_json(a.output / 'allocation.json', status)
        while any(p.poll() is None for p in qwen_workers):
            if interrupted:
                raise RuntimeError('Allocation interrupted')
            if any(p.poll() is not None for p in ltx_workers):
                raise RuntimeError('An LTX worker exited before Qwen finished')
            if any(p.poll() not in (None, 0) for p in qwen_workers):
                raise RuntimeError('A Qwen worker exited unsuccessfully')
            time.sleep(1)
        codes = [p.wait() for p in qwen_workers]
        if any(codes):
            raise RuntimeError('Qwen worker exit codes: ' + repr(codes))
        status['status'] = 'finished_workers'
        status['qwen_exit_codes'] = codes
        summaries = [json.loads((a.output / 'workers' / f'rank_{rank}_done.json').read_text())
                     for rank in range(workers)]
        status['case_counts'] = {'selected': sum(r['selected'] for r in summaries),
                                 'ok': sum(r['ok'] for r in summaries),
                                 'errors': sum(len(r['errors']) for r in summaries)}
        if status['case_counts']['errors']:
            status['status'] = 'finished_with_case_errors'
        for q in queues:
            (q / 'STOP').touch()
        deadline = time.monotonic() + 30
        while any(p.poll() is None for p in ltx_workers) and time.monotonic() < deadline:
            time.sleep(0.5)
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        stop_children(children)
        for stream in streams:
            stream.close()
        status['ended_unix'] = time.time()
        atomic_json(a.output / 'allocation.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
