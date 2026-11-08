#!/usr/bin/env python3
"""Use authorized existing allocations, resume records after allocation expiry.

No sbatch/scancel, no model training, no periodic Codex wake-up messages.
"""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import subprocess
import time

from media_inputs import atomic_json

ROOT = Path(__file__).resolve().parents[2]


def job_state(job):
    try:
        result = subprocess.run(['squeue', '-h', '-j', str(job), '-o', '%T'], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return 'UNKNOWN'
    if result.returncode:
        return 'UNKNOWN'
    return result.stdout.strip() or 'FINISHED'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/history_selection_20261106')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Only this workflow's lock; do not disturb user allocations or LTX.
    import fcntl
    with (args.output / '.workflow.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_workflow(args)


def run_workflow(args):
    state = {'status': 'starting', 'profiles_completed': [], 'attempts': [],
             'authorized_allocations': [193061, 193287], 'training': False}

    def save():
        state['updated_at'] = datetime.now(ZoneInfo('Asia/Singapore')).isoformat()
        atomic_json(args.output / 'state.json', state)

    exhausted = set()
    for profile in ['low', 'high']:
        state['profile'] = profile
        while True:
            allocation = None
            states = {job: job_state(job) for job in [193061, 193287]}
            state['allocation_states'] = states
            for job in [193061, 193287]:
                if states[job] == 'RUNNING' and job not in exhausted:
                    allocation = job
                    break
            if allocation is None:
                if all(value in {'FINISHED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'COMPLETED'} for value in states.values()):
                    state['status'] = 'needs_new_allocation'
                    save()
                    return 3
                state['status'] = 'waiting_for_authorized_allocation'
                save()
                time.sleep(30)
                continue
            profile_root = args.output / profile
            command = ['srun', f'--jobid={allocation}', '--overlap', '--exact', '--nodes=1',
                       '--ntasks=1', '--cpus-per-task=32', '--gres=gpu:h200:8', '--mem=1000G',
                       '--cpu-bind=none', 'bash', str(ROOT / 'scripts/run_gpu_python.sh'),
                       str(ROOT / 'scripts/evaluation/run_allocation.py'), '--profile', profile,
                       '--output', str(profile_root)]
            attempt = {'job_id': allocation, 'profile': profile, 'command': command,
                       'started_at': datetime.now(ZoneInfo('Asia/Singapore')).isoformat()}
            state['attempts'].append(attempt)
            state.update(status='running', allocation=allocation)
            save()
            with (args.output / f'job_{allocation}_{profile}_launcher.log').open('a') as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    state['completed_record_files'] = len(list((profile_root / 'records').glob('*.json')))
                    save()
                    time.sleep(30)
                attempt['exit_code'] = process.returncode
            if process.returncode == 0:
                subprocess.run(['python', str(ROOT / 'scripts/evaluation/summarize_results.py'), '--output', str(args.output)], cwd=ROOT, check=True)
                summary = json.loads((args.output / 'summary.json').read_text())['profiles'][profile]['groups']['all']
                if not summary['complete'] or summary.get('runtime_error', 0) or summary.get('incompatible', 0):
                    state['status'] = 'needs_result_review'
                    state['result_check'] = summary
                    save()
                    return 2
                state['profiles_completed'].append(profile)
                save()
                break
            if job_state(allocation) not in {'RUNNING', 'UNKNOWN'}:
                exhausted.add(allocation)
                state['status'] = 'allocation_ended_resuming_remaining_records'
                save()
                continue
            state['status'] = 'needs_error_review'
            save()
            return 2
    state['status'] = 'model_runs_finished_pending_summary'
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
