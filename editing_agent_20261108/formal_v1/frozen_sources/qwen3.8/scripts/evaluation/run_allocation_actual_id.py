#!/usr/bin/env python3
"""Run one deterministic Qwen replica per scheduler-exposed GPU, with strict IDs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_allocation import terminate_children
ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, choices=(4, 8), required=True)
    parser.add_argument('--profiles', nargs='+', choices=('low', 'high'), default=['low', 'high'])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    devices = [x.strip() for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')]
    if len(devices) != args.gpus or any(not x for x in devices) or len(set(devices)) != args.gpus:
        raise ValueError('Scheduler must expose exactly the requested distinct GPUs: ' + repr(devices))
    for profile in args.profiles:
        logs = args.output / profile / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        children, streams = [], []
        stopped = []

        def stop(signum, frame):
            stopped.append(signum)
            for child in children:
                if child.poll() is None:
                    child.terminate()

        previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            for rank, device in enumerate(devices):
                if stopped:
                    break
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                stream = (logs / f"job_{os.environ.get('SLURM_JOB_ID', 'unknown')}_rank_{rank}.log").open('a', buffering=1)
                streams.append(stream)
                command = ['bash', str(ROOT / 'scripts/run_gpu_python.sh'),
                           str(ROOT / 'scripts/evaluation/run_model_eval_actual_id.py'),
                           '--dataset', str(args.dataset.resolve()),
                           '--profile', str(ROOT / f'configs/evaluation/actual_id/{profile}.json'),
                           '--output', str((args.output / profile).resolve()),
                           '--rank', str(rank), '--world-size', str(args.gpus)]
                if args.prepare_only:
                    command.append('--prepare-only')
                children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT))
                time.sleep(2)
            while not stopped and any(child.poll() is None for child in children):
                time.sleep(0.2)
            codes = [child.wait() for child in children] if not stopped else []
            print(json.dumps({'profile': profile, 'exit_codes': codes, 'prepare_only': args.prepare_only}), flush=True)
            if stopped:
                return 128 + stopped[0]
            if len(codes) != args.gpus or any(code for code in codes):
                return 2
        finally:
            terminate_children(children)
            for stream in streams:
                stream.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if not args.prepare_only and profile == 'high' and 'low' in args.profiles:
            subprocess.run([sys.executable, str(ROOT / 'scripts/evaluation/summarize_results_actual_id.py'),
                            '--dataset', str(args.dataset.resolve()), '--output', str(args.output.resolve())], check=True)
            summary = json.loads((args.output / 'summary.json').read_text())
            if not summary['all_inference_succeeded'] or summary['mixed_fingerprints']:
                raise RuntimeError('Completion audit failed; inspect summary.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
