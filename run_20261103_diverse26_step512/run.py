"""Generate 26 fixed cases with two final checkpoints inside a user-provided GPU allocation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
MANIFEST=HERE/'manifest.json'
GALLERY_MANIFEST=MANIFEST
PYTHON=ROOT/'scripts/common/run_trainer_python.sh'
spec=json.loads(MANIFEST.read_text())
state={'status':'starting','allocation':os.environ.get('SLURM_JOB_ID'),'gpu_count':4,'started_unix':time.time(),'stages':[]}
active=None

def save():
 p=HERE/'state.json.tmp';p.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');p.replace(HERE/'state.json')

def execute(name,argv):
 global active
 row={'name':name,'status':'running','started_unix':time.time()};state['stages'].append(row);state['status']=name;save()
 (HERE/'logs').mkdir(exist_ok=True)
 with (HERE/'logs'/f'{name}.log').open('w') as f:
  active=subprocess.Popen(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  code=active.wait();active=None
 row.update(status='completed' if code==0 else 'failed',exit_code=code,seconds=time.time()-row['started_unix']);save()
 if code:raise RuntimeError(f'{name} exited {code}; see logs/{name}.log')

def python(script,*args):return ['bash',str(PYTHON),str(ROOT/'scripts'/script),*map(str,args)]
def gallery(name):execute(name,python('inference/build_experiment_comparison.py','--manifest',GALLERY_MANIFEST,'--output',HERE/'gallery'))
def stop(signum,frame):
 if active is not None and active.poll() is None:
  os.killpg(active.pid,signal.SIGTERM)
 raise SystemExit(128+signum)

signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run inside a user-provided Slurm GPU allocation')
 if (HERE/'display').exists():raise RuntimeError('Comparison is already exported; do not repeat')
 # Pin both complete checkpoints before either model is loaded.
 code="import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from checkpoint_retention import pin_for_validation; m=json.loads(Path(sys.argv[2]).read_text()); [pin_for_validation(r['checkpoint'],str(Path(sys.argv[2]).parent)) for r in m['runs']]; print('Both checkpoints retained')"
 execute('pin_checkpoints',['bash',str(PYTHON),'-c',code,str(ROOT/'scripts/common'),str(MANIFEST)])
 if not (HERE/'gallery/index.html').exists():gallery('initial_page')
 for run in spec['runs']:
  execute(run['name'],python('inference/run_fixed_cases_parallel.py','--config',run['config'],'--bundle',run['bundle'],'--cases',spec['cases'],'--checkpoint',run['checkpoint'],'--inference-config',spec['inference_config'],'--gpu-count',4,'--output',Path(run['comparison']).parent))
  gallery('page_after_'+run['name'])
 execute('export',python('inference/export_guidance_gallery.py','--gallery',HERE/'gallery','--output',HERE/'display'))
 state['status']='completed';state['finished_unix']=time.time();save()
except BaseException as e:
 state['status']='failed';state['error']=str(e);state['finished_unix']=time.time();save()
 if active is not None and active.poll() is None:
  os.killpg(active.pid,signal.SIGTERM)
 raise
