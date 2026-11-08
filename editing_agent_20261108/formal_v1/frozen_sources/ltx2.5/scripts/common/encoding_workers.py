"""Run allocated GPU workers with separate CPU sets and stop children on failure."""
from contextlib import contextmanager
import fcntl
import json
import os
import signal
import subprocess
import time
from project_layout import ROOT


def stop(children):
    for child in children:
        if child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    deadline=time.monotonic()+15
    while time.monotonic()<deadline and any(c.poll() is None for c in children):time.sleep(.1)
    for child in children:
        if child.poll() is None:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        child.wait()


def execute_jobs(items,args,logdir):
    children=[]
    try:
        for item in items:
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=item['gpu'],OMP_NUM_THREADS=str(args.threads_per_worker),
                       MKL_NUM_THREADS=str(args.threads_per_worker))
            path=logdir/f'{item["source"]}_{item["stage"]}_{item["shard"]:03d}.log'
            with path.open('w') as handle:
                child=subprocess.Popen(item['command'],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,
                    start_new_session=True,preexec_fn=lambda affinity=item['cpu_affinity']:os.sched_setaffinity(0,affinity))
            children.append(child)
            print(json.dumps({**item,'pid':child.pid,'log':str(path)}),flush=True)
        while True:
            statuses=[x.poll() for x in children]
            if any(x is not None and x!=0 for x in statuses):raise RuntimeError(f'Encoding failed: {statuses}; see {logdir}')
            if all(x==0 for x in statuses):break
            time.sleep(1)
    finally:stop(children)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another mixed-cache controller is active') from None
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)

