#!/usr/bin/env python3
"""One-shot handoff from our authorized 194849 four-GPU step to allocated eight GPUs.

Never cancels an allocation. It signals only a verified numeric Slurm step,
waits for the shared campaign lock, then exits so the batch script can resume.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

AUTHORIZED_SOURCE_JOB = '194849'
HERE = Path(__file__).resolve().parent


def lock_available(output):
    with (output / '.campaign.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle, fcntl.LOCK_UN)
        return True


def verify_owner(output, owner, target_job):
    if not target_job or not target_job.isdigit() or target_job == AUTHORIZED_SOURCE_JOB:
        raise ValueError('Handoff must run inside a different, already allocated target Slurm job')
    if owner.get('output') != str(output.resolve()):
        raise ValueError('Owner output differs from requested formal output')
    if str(owner.get('job_id')) != AUTHORIZED_SOURCE_JOB or owner.get('gpus') != 4:
        raise ValueError('Only the authorized 194849 four-GPU campaign may be handed off')
    step = str(owner.get('step_id', ''))
    if not step.isdigit():
        raise ValueError('Only numeric srun step IDs are permitted; never batch/extern/allocation')
    manifest = output / 'freeze_manifest.json'
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if owner.get('freeze_manifest_sha256') != actual:
        raise ValueError('Running campaign and current freeze manifest differ')
    return AUTHORIZED_SOURCE_JOB + '.' + step


def verify_step(text, expected_step, uid):
    match = re.search(r'\bStepId=(\S+)', text)
    if not match or match.group(1) != expected_step:
        raise ValueError('Slurm returned a different step identity')
    user = re.search(r'\bUserId=([^\s]+)', text)
    if not user:
        raise ValueError('Cannot verify Slurm step owner')
    value = user.group(1)
    found = re.search(r'\((\d+)\)$', value)
    reported_uid = int(found.group(1)) if found else int(value) if value.isdigit() else None
    if reported_uid != uid:
        raise ValueError('Slurm step belongs to a different user')


def handoff(output, target_job, timeout=120, run=subprocess.run, available=lock_available,
            clock=time.monotonic, sleep=time.sleep):
    output = output.resolve()
    if not (output / 'freeze_manifest.json').is_file():
        raise ValueError('Formal output is not frozen')
    run([sys.executable, str(HERE / 'freeze.py'), '--output', str(output), '--verify'], check=True)
    if available(output):
        return {'status': 'no_active_campaign', 'output': str(output)}
    owner_path = output / 'campaign_owner.json'
    owner = json.loads(owner_path.read_text())
    step = verify_owner(output, owner, target_job)
    query = run(['scontrol', 'show', 'step', step, '-o'], check=True, capture_output=True, text=True)
    verify_step(query.stdout, step, os.getuid())
    # Recheck just before sending the signal; a different owner must not be
    # stopped because it acquired the lock while scontrol was running.
    if available(output):
        return {'status': 'previous_campaign_finished', 'output': str(output)}
    if json.loads(owner_path.read_text()) != owner:
        raise ValueError('Campaign owner changed during handoff; refusing to signal')
    run(['scancel', '--signal=TERM', step], check=True)
    deadline = clock() + timeout
    while clock() < deadline:
        if available(output):
            return {'status': 'handoff_ready', 'stopped_step': step, 'output': str(output),
                    'source_allocation_cancelled': False}
        current = json.loads(owner_path.read_text())
        if any(current.get(k) != owner.get(k) for k in ('output', 'job_id', 'step_id', 'pid', 'started_unix')):
            raise ValueError('Another campaign acquired output while awaiting handoff')
        sleep(1)
    raise TimeoutError(f'Campaign lock was not released after TERM to {step}; allocation was not cancelled')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    result = handoff(args.output, os.environ.get('SLURM_JOB_ID'), args.timeout)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
