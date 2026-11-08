#!/usr/bin/env python3
"""Frozen formal evaluation: understanding, thirty generated chains, then measured expansion."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from queue_client import atomic_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CLEANUP_RESERVE = 180
EXPANSION_RESERVE = 1800


def run_command(args, timeout):
    """Terminate the whole launcher/worker group on timeout, including children."""
    if timeout <= 0:
        raise subprocess.TimeoutExpired(args, timeout)
    proc = subprocess.Popen([sys.executable] + [str(x) for x in args], cwd=ROOT, start_new_session=True)
    try:
        code = proc.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, args)
    except BaseException:
        # The launcher handles TERM and closes both model workers. A hard group
        # kill is the final fallback; terminating just the launcher leaks GPUs.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        raise


def successful_cases(output, public_rows):
    count = 0
    for i, row in enumerate(public_rows):
        cid = str(row.get('chain_id', row.get('sample_id', row.get('question_id', i))))
        path = output / 'cases' / cid / 'worker_result.json'
        if path.is_file() and json.loads(path.read_text()).get('status') == 'ok':
            count += 1
    return count


def unattempted_cases(output, public_rows):
    """No retry of committed successes or errors; interrupted partial work resumes."""
    pending = []
    for i, row in enumerate(public_rows):
        cid = str(row.get('chain_id', row.get('sample_id', row.get('question_id', i))))
        path = output / 'cases' / cid / 'worker_result.json'
        if path.is_file():
            if json.loads(path.read_text()).get('status') not in ('ok', 'error'):
                raise ValueError('Unexpected committed worker result status: ' + str(path))
        else:
            pending.append(row)
    return pending


def measured_seconds_per_chain(stages):
    values = [s['seconds_per_new_case'] for s in stages
              if s.get('mode') == 'closed_loop' and (s.get('seconds_per_new_case') or 0) > 0]
    return max(values) if values else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--gpus', type=int, choices=[4, 8], required=True)
    p.add_argument('--hours', type=float, default=17)
    a = p.parse_args()
    if a.hours <= 0:
        p.error('--hours must be positive')
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    campaign_lock = (a.output / '.campaign.lock').open('a')
    try:
        fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        campaign_lock.close()
        raise RuntimeError('Another campaign already owns this output; use the verified allocation handoff.')
    manifest = a.output / 'freeze_manifest.json'
    owner = {'output': str(a.output), 'job_id': os.environ.get('SLURM_JOB_ID'),
             'step_id': os.environ.get('SLURM_STEP_ID'), 'gpus': a.gpus,
             'pid': os.getpid(), 'host': socket.gethostname(), 'started_unix': time.time(),
             'freeze_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None}
    atomic_json(a.output / 'campaign_owner.json', owner)
    available = a.hours * 3600
    # A resume within an allocation must not reset its scheduler lifetime.
    if os.environ.get('SLURM_JOB_END_TIME', '').isdigit():
        available = min(available, int(os.environ['SLURM_JOB_END_TIME']) - time.time())
    deadline = time.monotonic() + max(0, available)
    dataset = ROOT / 'datasets/editing_agent_v1'
    config = ROOT / 'configs/editing_agent/config.json'
    state_path = a.output / 'campaign.json'
    state = json.loads(state_path.read_text()) if state_path.is_file() else {'stages': [], 'started_unix': time.time()}
    state.update(status='running', job=os.environ.get('SLURM_JOB_ID'), attempt_started_unix=time.time())
    state.update(minimum_target=30, minimum_pending=True, minimum_completed=0)
    state.pop('error', None)
    state.pop('expansion_stopped_reason', None)

    def run(args):
        run_command(args, deadline - time.monotonic() - CLEANUP_RESERVE)

    def stage(mode, filename, label, limit):
        run([HERE / 'freeze.py', '--output', a.output, '--verify'])
        output = a.output / mode
        public_rows = [json.loads(line) for line in (dataset / filename).read_text().splitlines() if line.strip()][:limit]
        if len(public_rows) != limit:
            raise ValueError('Dataset has fewer cases than requested')
        before_ok = successful_cases(output, public_rows)
        pending_rows = unattempted_cases(output, public_rows)
        began = time.monotonic()
        state.update(active_stage=mode, active_limit=limit)
        atomic_json(state_path, state)
        report_args = [HERE / 'report.py', '--run', output, '--dataset', dataset / filename,
                       '--labels', dataset / 'private' / label, '--limit', limit]
        if mode == 'closed_loop':
            report_args += ['--gallery']
        timed_out = None
        try:
            if pending_rows:
                # Derived public-only dispatch list; reporting keeps the full
                # original prefix so failed cases stay in the denominator.
                text = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in pending_rows)
                dispatch = a.output / 'dispatch' / f'{mode}_{limit:04d}_{hashlib.sha256(text.encode()).hexdigest()[:16]}.jsonl'
                dispatch.parent.mkdir(exist_ok=True)
                dispatch.write_text(text)
                run([HERE / 'launch_allocation.py', '--mode', mode, '--dataset', dispatch,
                     '--config', config, '--output', output, '--gpus', a.gpus, '--limit', len(pending_rows)])
        except subprocess.TimeoutExpired as exc:
            timed_out = exc
        # On timeout the model workers have stopped. Use a portion of the reserved
        # cleanup window to retain a report of completed/partial/pending cases.
        run_command(report_args, min(90, max(1, deadline - time.monotonic())))
        summary = json.loads((output / 'report/summary.json').read_text())
        elapsed = time.monotonic() - began
        new_ok = max(0, summary['cases_ok'] - before_ok)
        record = {'mode': mode, 'limit': limit, 'seconds': elapsed, 'before_ok': before_ok,
                  'new_ok': new_ok, 'reused_all': before_ok == limit, 'summary': summary,
                  'dispatched_cases': len(pending_rows),
                  'seconds_per_new_case': elapsed / new_ok if new_ok else None}
        state['stages'].append(record)
        if mode == 'closed_loop':
            state.update(minimum_completed=summary['cases_ok'], minimum_pending=summary['cases_ok'] < 30,
                         closed_loop_requested=limit)
        atomic_json(state_path, state)
        if timed_out:
            raise timed_out
        if summary['cases_pending'] or summary['cases_error'] > max(2, int(limit * .10)):
            raise RuntimeError('Execution failures exceed expansion gate; inspect retained per-case errors.')
        return record

    def interrupted(sig, frame):
        raise KeyboardInterrupt(f'Received signal {sig}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        run([HERE / 'freeze.py', '--output', a.output, '--verify'])
        stage('understanding', 'understanding_test.jsonl', 'understanding_test_labels.jsonl', 1000)
        if deadline - time.monotonic() < EXPANSION_RESERVE:
            state.update(status='paused_time_limit', expansion_stopped_reason='Not enough allocation time to start generated-chain evaluation.')
            return
        stage('closed_loop', 'chains_test.jsonl', 'chains_test_labels.jsonl', 30)
        completed = 30
        chain_rows = [json.loads(line) for line in (dataset / 'chains_test.jsonl').read_text().splitlines() if line.strip()]
        # Thirty requested chains is not thirty completed chains. Retain errors
        # and append new ordered candidates until the actual completion minimum.
        while state['minimum_pending']:
            target = min(len(chain_rows), completed + 30 - state['minimum_completed'])
            if target <= completed:
                state.update(status='minimum_pending', expansion_stopped_reason='Candidate pool exhausted before 30 completed chains.')
                return
            remaining = len(unattempted_cases(a.output / 'closed_loop', chain_rows[:target]))
            per_chain = measured_seconds_per_chain(state['stages'])
            required = remaining * (per_chain or 0) * 1.35 + 900 + EXPANSION_RESERVE
            if remaining and (per_chain is None or deadline - time.monotonic() < required):
                state.update(status='paused_time_limit', expansion_stopped_reason='30 completed chains not yet reached; retain failures and wait for time to append new cases.')
                return
            stage('closed_loop', 'chains_test.jsonl', 'chains_test_labels.jsonl', target)
            completed = target
        for target in [60, 100, 128]:
            if target <= completed:
                continue
            per_chain = measured_seconds_per_chain(state['stages'])
            remaining = len(unattempted_cases(a.output / 'closed_loop', chain_rows[:target]))
            if remaining and per_chain is None:
                state['expansion_stopped_reason'] = 'No measured generation throughput yet; do not estimate speed from cached cases.'
                break
            required = remaining * (per_chain or 0) * 1.35 + 900
            if remaining and deadline - time.monotonic() < required + EXPANSION_RESERVE:
                state['expansion_stopped_reason'] = 'Insufficient estimated allocation time for the next batch plus reserve.'
                break
            stage('closed_loop', 'chains_test.jsonl', 'chains_test_labels.jsonl', target)
            completed = target
        state.update(status='completed', closed_loop_requested=completed)
    except subprocess.TimeoutExpired as exc:
        state.update(status='paused_time_limit', error=str(exc))
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        state['ended_unix'] = time.time()
        atomic_json(state_path, state)
        owner['ended_unix'] = time.time()
        atomic_json(a.output / 'campaign_owner.json', owner)
        fcntl.flock(campaign_lock, fcntl.LOCK_UN)
        campaign_lock.close()


if __name__ == '__main__':
    main()
