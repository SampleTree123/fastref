import json
import sys
from pathlib import Path
ROOT=Path('/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/qwen3.8')
sys.path.insert(0,str(ROOT/'scripts/editing_agent'))
from qwen_backend import QwenBackend
from harness import review_output
cfg=json.loads((ROOT/'configs/editing_agent/config.json').read_text())
high=json.loads(Path(cfg['qwen_profile']).read_text());high['generation']['max_new_tokens']=cfg['qwen_max_new_tokens'];high['cache_max_gib']=96
b=QwenBackend(config=high)
old=ROOT/'experiments/editing_agent_20261108/development_v1_smoke/cases/debug_chain_04581'
state=json.loads((old/'state.json').read_text());turn=state['identity']['chain']['turns'][0]
p=json.loads((old/'turn_01/progress.json').read_text())
result=review_output(state['identity']['chain']['root_video'],p['generation']['video_path'],turn,p['request']['prompt'],b,Path(__file__).parent,cfg)
print(json.dumps(result,ensure_ascii=False),flush=True)
