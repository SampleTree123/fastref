"""Encode both CoinVE videos using their shared absolute time window."""
import coinve_cache_contract as coinve
import prepare_datamake_media as media
from encoding_helpers import shape_check, tensor_info


def encode_coinve_media(job,args,identity,vae):
    import numpy as np
    import torch
    path=media.verify_media(job['media'],args.dataset_root)
    frames,pixel_metadata=coinve.load_video_frames(path,job['canvas_wh'],identity['profile'],job['temporal_window'])
    pixels=torch.from_numpy(np.stack([np.asarray(frame) for frame in frames]))
    pixels=pixels.float().div_(255).sub_(0.5).div_(0.5).permute(3,0,1,2).unsqueeze(0).to('cuda',torch.bfloat16)
    latent=vae(pixels)
    shape_check(latent.shape,len(frames),job['canvas_wh'],identity['profile'])
    info=tensor_info(latent)
    payload={'latents':latent[0].cpu().contiguous(),'num_frames':latent.shape[2],
             'height':latent.shape[3],'width':latent.shape[4],'fps':12}
    return payload,{'media_path':job['media']['path'],'media_sha256_verified':job['media']['sha256'],
                   'canvas_wh':job['canvas_wh'],'num_frames':57,'pixel_metadata':pixel_metadata,'tensor':info}

