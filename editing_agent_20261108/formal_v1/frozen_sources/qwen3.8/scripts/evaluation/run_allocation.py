#!/usr/bin/env python3
"""Run eight independent Qwen replicas inside an existing eight-GPU allocation.

Does not submit/cancel Slurm jobs. Each record is resumable by run_model_eval.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def terminate_children(children, grace_seconds=10):
    """Terminate all live replicas concurrently, then reap or kill each one."""
    for child in children:
        if child.poll() is None:
            child.terminate()
    deadline = time.monotonic() + grace_seconds
    for child in children:
        if child.poll() is None:
            try:
                child.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, choices=['low', 'high'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    devices = [item.strip() for item in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')]
    if len(devices) != 8 or any(not item for item in devices) or len(set(devices)) != 8:
        raise ValueError('Scheduler must expose exactly eight distinct GPUs: ' + repr(devices))
    logs = args.output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    children, streams = [], []
    stopped_signal = None

    def stop(signum, frame):
        nonlocal stopped_signal
        stopped_signal = signum or signal.SIGTERM
        for child in children:
            if child.poll() is None:
                child.terminate()

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for rank, device in enumerate(devices):
            if stopped_signal is not None:
                break
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            path = logs / f"job_{os.environ.get('SLURM_JOB_ID', 'unknown')}_rank_{rank}.log"
            stream = path.open('a', buffering=1)
            streams.append(stream)
            command = ['bash', str(ROOT / 'scripts/run_gpu_python.sh'),
                       str(ROOT / 'scripts/evaluation/run_model_eval.py'),
                       '--dataset', str(ROOT / 'datasets/history_selection_original/eval.jsonl'),
                       '--profile', str(ROOT / 'configs/evaluation' / f'{args.profile}.json'),
                       '--output', str(args.output.resolve()), '--rank', str(rank), '--world-size', '8']
            children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT))
            if stopped_signal is not None:
                break
            # Avoid eight simultaneous tokenizer and weight metadata opens.
            time.sleep(2)
        while stopped_signal is None and any(child.poll() is None for child in children):
            time.sleep(0.2)
        if stopped_signal is not None:
            return 128 + stopped_signal
        codes = [child.wait() for child in children]
        print(json.dumps({'profile': args.profile, 'exit_codes': codes}), flush=True)
        return 0 if len(codes) == 8 and all(code == 0 for code in codes) else 2
    finally:
        try:
            terminate_children(children)
        finally:
            for stream in streams:
                stream.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == '__main__':
    sys.exit(main())
