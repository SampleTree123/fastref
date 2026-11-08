#!/usr/bin/env python3
"""Run the final four-group comparison through the existing fixed-case launcher.

Experimental owner: experiments/inference/final_camera_comparison_20261108/README.md.
Requires eight already allocated, idle GPUs. Does not allocate or cancel resources.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.campaign.read_text())
    root = args.campaign.resolve().parent
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 8 or len(set(devices)) != 8 or any(not d for d in devices):
        raise ValueError('Expose exactly eight distinct allocated GPUs')
    runs = {r['name']: r for r in plan['runs']}
    state = {'status': 'running', 'started_unix': time.time(), 'groups': {}}
    lock = threading.Lock()
    children = {}
    stop = threading.Event()

    def save():
        tmp = root / 'execution_state.json.tmp'
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(root / 'execution_state.json')

    def lane(index, names):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(devices[index*4:(index+1)*4])}
        for name in names:
            if stop.is_set():
                return
            run = runs[name]
            if Path(run['comparison']).exists():
                raise ValueError('Campaign output already exists: ' + name)
            with (root / 'logs' / f'{name}.log').open('w') as log:
                with lock:
                    child = subprocess.Popen(run['command'], env=env, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    children[name] = child
                    state['groups'][name] = {'status': 'running', 'pid': child.pid,
                                            'devices': env['CUDA_VISIBLE_DEVICES'], 'started_unix': time.time()}
                    save()
                code = child.wait()
                with lock:
                    state['groups'][name].update(status='completed' if code == 0 else 'failed',
                                                 exit_code=code, finished_unix=time.time())
                    save()
                if code:
                    raise RuntimeError(f'{name} failed with exit {code}; inspect its log')

    def interrupt(signum, frame):
        stop.set()
        for child in list(children.values()):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        raise KeyboardInterrupt(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    save()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(lane, i, names) for i, names in enumerate(plan['lanes'])]
            for future in futures:
                future.result()
        state['status'] = 'generation_completed_pending_reuse_merge_and_gallery'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['finished_unix'] = time.time()
        save()


if __name__ == '__main__':
    main()
