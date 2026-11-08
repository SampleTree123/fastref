from pathlib import Path
import json
from huggingface_hub import HfApi, snapshot_download
root=Path(__file__).resolve().parents[2]
repo='bytedance-research/Lance'
revision='7395315758865e6f56ab87ad06a88c7ac172f056'
patterns=['Lance_3B_Video/*','Qwen2.5-VL-ViT/*','Wan2.2_VAE.pth','config.json','README.md']
info=HfApi().model_info(repo,revision=revision,files_metadata=True)
path=snapshot_download(repo,revision=revision,allow_patterns=patterns,local_dir=root/'models/Lance',max_workers=4)
import fnmatch
files=[{'name':f.rfilename,'size':f.size} for f in info.siblings if any(fnmatch.fnmatch(f.rfilename,p) for p in patterns)]
for item in files:
 assert (Path(path)/item['name']).stat().st_size==item['size'],item
(Path(__file__).parent/'model_revision.json').write_text(json.dumps({'repo':repo,'revision':revision,'path':str(path),'files':files},indent=2)+'\n')
print('Download and size check complete',path,flush=True)
