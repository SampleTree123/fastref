"""One official T2V followed by three official single-turn edits.
Owner: README.md in this experiment. No model implementation is modified.
"""
import hashlib, html, json, os, subprocess, time, sys
import cv2
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CODE=ROOT/'runtime/official_4baeee0'
MODEL=ROOT/'models/Lance'
PROMPTS=[
 ('t2v', 'A realistic cinematic medium shot of an adult woman with shoulder-length straight black hair, wearing a plain white long-sleeved blouse and dark trousers, walking slowly along a sunlit garden path. Green trees and flowers are visible behind her. Her face, hair and upper body remain clearly visible. A steady frontal tracking camera follows her smooth natural walking motion. Soft daylight, clear facial features, realistic textures.'),
 ('video_edit', "Change the woman's white blouse to a solid royal blue blouse. Keep her face, black hair, trousers, walking motion, camera framing and garden background unchanged."),
 ('video_edit', "Change the woman's hair color from black to blonde. Keep the same hairstyle, face, royal blue blouse, trousers, walking motion, camera framing and garden background unchanged."),
 ('video_edit', "Replace the garden background with a modern city street lined with buildings in daylight. Keep the same woman, blonde hair, royal blue blouse, trousers, walking motion and camera framing unchanged.")
]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def save(state):
 temp=HERE/'state.tmp';temp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');temp.replace(HERE/'state.json')
def gallery(state):
 cards=[]
 for r in state['rounds']:
  cards.append(f'<article><h2>{r["round"]}: {html.escape(r["task"])}</h2><p>{html.escape(r["prompt"])}</p><video controls preload="metadata" src="{html.escape(r["video_relative"])}"></video><p>源：{r["parent_round"] if r["parent_round"] is not None else "文字生成"}；{r["frames"]}帧；{r["width"]}×{r["height"]}；{r["seconds"] if r["seconds"] is not None else "复用已生成视频"}</p></article>')
 text='<!doctype html><html lang="zh"><meta charset="utf-8"><title>Lance官方基座连续编辑</title><style>body{font-family:system-ui;max-width:1800px;margin:24px auto;padding:12px;background:#f4f5f7;color:#182230}.grid{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:16px}article{background:white;border-radius:12px;padding:18px}video{width:100%}p{line-height:1.6}</style><h1>Lance官方基座：文生视频＋连续3次编辑</h1><p>没有加载DataMake LoRA；每次编辑只输入上一轮实际输出和当前编辑文字。30步、文字CFG 4、seed42。生成失败也如实保留；此页面不表示质量已达标。</p><div class="grid">'+''.join(cards)+'</div></html>'
 (HERE/'index.html').write_text(text)
def main():
 import torch
 assert torch.cuda.is_available()
 free,total=torch.cuda.mem_get_info()
 if total-free>2*1024**3: raise RuntimeError(f'Allocated GPU is not idle: {total-free} bytes in use')
 assert (HERE/'model_revision.json').is_file(), 'Wait for complete model download'
 if (HERE/'state.json').exists() and '--resume' not in sys.argv: raise RuntimeError('Results already exist; use explicit --resume after inspecting the failure')
 tracked=subprocess.check_output(['git','-C',str(CODE),'diff','--name-only','HEAD'],text=True).strip()
 assert not tracked,tracked
 env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1';env['TOKENIZERS_PARALLELISM']='false';env['OMP_NUM_THREADS']='8'
 for key in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'): env.pop(key,None)
 state={'status':'running','code_commit':subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip(),'weights':json.loads((HERE/'model_revision.json').read_text()),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'gpu':torch.cuda.get_device_name(0),'settings':{'steps':30,'cfg_text':4,'timestep_shift':3.5,'seed':42,'num_frames_requested':61,'resolution':'video_480p','height_requested':480,'width_requested':848,'lora':False,'enhance_prompt':False},'rounds':[]}
 if (HERE/'state.json').exists():
  old=json.loads((HERE/'state.json').read_text())
  assert old['code_commit']==state['code_commit'] and old['weights']==state['weights'] and old['settings']==state['settings']
  state['rounds']=old['rounds'];state['recovered_from']=old.get('error')
 save(state)
 try:
  for i,(task,prompt) in enumerate(PROMPTS):
   if i<len(state['rounds']):
    assert state['rounds'][i]['prompt']==prompt and digest(Path(state['rounds'][i]['video']))==state['rounds'][i]['sha256'];continue
   folder=HERE/f'round_{i:02d}';folder.mkdir(exist_ok='--resume' in sys.argv)
   parent=Path(state['rounds'][-1]['video']) if i else None
   payload={'000000.mp4':prompt} if not i else {'000000':{'interleave_array':[prompt,str(parent)],'element_dtype_array':['text','video'],'istarget_in_interleave':[0,0]}}
   inp=folder/'input.json'
   if inp.exists(): assert json.loads(inp.read_text())==payload
   else: inp.write_text(json.dumps(payload,indent=2)+'\n')
   cmd=['bash','inference_lance.sh','--TASK_NAME',task,'--NUM_GPUS','1','--MODEL_PATH',str(MODEL/'Lance_3B_Video'),'--CONFIG_PATH',str(inp),'--RESOLUTION','video_480p','--NUM_FRAMES','61','--VIDEO_HEIGHT','480','--VIDEO_WIDTH','848','--VALIDATION_NUM_TIMESTEPS','30','--CFG_TEXT_SCALE','4.0','--VALIDATION_TIMESTEP_SHIFT','3.5','--VALIDATION_DATA_SEED','42','--ENHANCE_PROMPT','false','--SAVE_PATH_GEN',str(folder/'generated')]
   (folder/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
   videos=list((folder/'generated').glob('*.mp4'))
   reused=bool(videos) and '--resume' in sys.argv
   start=time.monotonic();print(f'START round={i} task={task} reuse_existing={reused}',flush=True)
   if not reused:
    with (folder/'inference.log').open('w') as log: completed=subprocess.run(cmd,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
    videos=list((folder/'generated').glob('*.mp4'))
    if completed.returncode or len(videos)!=1: raise RuntimeError(f'Round {i}: returncode={completed.returncode}, videos={len(videos)}; inspect inference.log (official shell may mask Python failures)')
   assert len(videos)==1
   video=videos[0];reader=cv2.VideoCapture(str(video));assert reader.isOpened()
   fps=reader.get(cv2.CAP_PROP_FPS);width=int(reader.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT));expected=int(reader.get(cv2.CAP_PROP_FRAME_COUNT));frames=0
   while True:
    ok,frame=reader.read()
    if not ok: break
    assert frame.shape[:2]==(height,width);frames+=1
   reader.release();assert frames==expected and frames>1 and width>0 and height>0
   row={'round':i,'task':task,'prompt':prompt,'parent_round':i-1 if i else None,'source_video':str(parent) if parent else None,'source_sha256':digest(parent) if parent else None,'video':str(video),'video_relative':str(video.relative_to(HERE)),'sha256':digest(video),'frames':frames,'fps':fps,'width':width,'height':height,'seconds':round(time.monotonic()-start,2) if not reused else None,'reused_completed_video':reused}
   if parent: assert row['source_sha256']==state['rounds'][-1]['sha256']
   state['rounds'].append(row);save(state);gallery(state);print(f'DONE round={i} seconds={row["seconds"]}',flush=True)
  state['status']='completed';save(state)
 except BaseException as exc:
  state['status']='failed';state['error']=str(exc);save(state);raise
if __name__=='__main__':main()
