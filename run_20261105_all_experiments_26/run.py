"""Generate only missing experiments, then export a self-contained eight-way page."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'scripts/common'))
from checkpoint_retention import pin_for_validation
from encoding_helpers import atomic_json
p=argparse.ArgumentParser();p.add_argument('--gpu-count',type=int,default=8);args=p.parse_args()
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Run inside an allocated Slurm GPU job')
if (HERE/'display/index.html').exists():
    raise RuntimeError('This comparison is already completed')
manifest=HERE/'manifest.json';spec=json.loads(manifest.read_text())
state={'status':'starting','allocation':os.environ['SLURM_JOB_ID'],'started_unix':time.time(),'stages':[]}
child=None

def stop(sig,frame):
    if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    raise SystemExit(128+sig)
for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)

def call(name,command):
    global child
    state['status']=name;atomic_json(HERE/'state.json',state)
    (HERE/'logs').mkdir(exist_ok=True)
    with (HERE/'logs'/f'{name}.log').open('a') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        code=child.wait();child=None
    state['stages'].append({'name':name,'exit_code':code,'finished_unix':time.time()})
    if code:raise RuntimeError(f'{name} failed; see logs/{name}.log')

def command(script,*args):
    return ['bash',str(ROOT/'scripts/common/run_trainer_python.sh'),str(ROOT/'scripts/inference'/script),*map(str,args)]

def page():call('gallery',command('build_experiment_comparison.py','--manifest',manifest,'--output',HERE/'gallery'))

try:
    for run in spec['runs']:
        result=Path(run['comparison'])
        if run.get('reuse_completed'):
            if not result.exists() or json.loads(result.read_text()).get('status')!='completed':
                raise RuntimeError('Expected reused result missing: '+run['name'])
            continue
        if run.get('select_latest_at_launch') and not result.exists():
            candidates=sorted(p for p in Path(run['checkpoint_directory']).glob('lora_weights_step_*.safetensors')
                              if p.with_suffix('.contract.json').is_file())
            chosen=candidates[-1]
            pin_for_validation(chosen,HERE)
            run.update(checkpoint=str(chosen),step=int(chosen.stem.split('_')[-1]))
            run['label']=f"镜头参考＋套装正背＋连续时间 · {run['step']}步 · 视觉CFG=1"
        pin_for_validation(run['checkpoint'],HERE)
    atomic_json(manifest,spec)
    page()
    for run in spec['runs']:
        result=Path(run['comparison'])
        if result.exists() and json.loads(result.read_text()).get('status')=='completed':
            continue  # The gallery verifies full receipt and media hashes before reuse.
        argv=command('run_fixed_cases_parallel.py','--config',run['config'],'--bundle',run['bundle'],
                     '--cases',spec['cases'],'--checkpoint',run['checkpoint'],
                     '--inference-config',spec['inference_config'],'--gpu-count',args.gpu_count,
                     '--output',result.parent,'--ref-cfg',run.get('ref_cfg',1))
        if run.get('legacy_reference_cache'):
            argv+=['--legacy-reference-cache',run['legacy_reference_cache']]
        call(run['name'],argv);page()
    call('export',command('export_guidance_gallery.py','--gallery',HERE/'gallery','--output',HERE/'display'))
    state.update(status='completed',finished_unix=time.time())
except BaseException as error:
    state.update(status='failed',error=str(error),finished_unix=time.time())
    raise
finally:
    atomic_json(HERE/'state.json',state)
