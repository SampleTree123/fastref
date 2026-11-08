"""V3 shared-canvas profile, cache keys and explicit immutable-payload reuse rules.

Owner: experiments/data_preparation/notes/reference_canvas_migration_20261101.md. Pure metadata operations;
original pixel/encoder implementations remain pinned and unchanged.
"""
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if (p / "common/project_layout.py").is_file()) / "common"))
from project_layout import ROOT as _PROJECT_ROOT, resolve_path, script_path, add_script_paths, resolve_cli_paths
add_script_paths()

from copy import deepcopy
import json
from pathlib import Path
from prepare_datamake_media import fingerprint, latent_shape

PROTOCOL = 'reference_positions_v3_shared_canvas'
IDENTITY_SCHEMA = 'datamake_shared_canvas_encoding_identity_v3'
PROFILE_SCHEMA = 'datamake_shared_canvas_media_profile_v1'
CANVAS_CONTRACT = 'source_canvas_contain_black_padding_before_vae_v1'
TIME_CONTRACT = 'official_local_time_zero_origin_no_reference_offsets_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_profile(path):
    p = json.loads(resolve_path(path).read_text())
    validate_profile(p)
    return p


def validate_profile(p):
    require(p['schema'] == PROFILE_SCHEMA and p['reference_protocol'] == PROTOCOL, 'Wrong shared-canvas profile/version')
    require(p['reference_image_canvas'] == p['scene_canvas'] == 'same_as_source_video', 'All images must use source canvas')
    require(p['video_canvas'] == {'landscape': [768,448], 'portrait': [448,768]}, 'Unapproved video canvas')
    require(p['reference_canvas_contract'] == CANVAS_CONTRACT and p['reference_time_contract'] == TIME_CONTRACT, 'Reference contract changed')
    require(p['resize'] == 'contain_letterbox' and p['interpolation'] == 'PIL_LANCZOS'
            and p['padding_rgb'] == [0,0,0] and p['fps'] == 12, 'Pixel resizing/FPS changed')
    name = p['source_name']
    require(name in ('ue','coinve') and p['num_frames'] == (121 if name == 'ue' else 57), 'Unapproved duration')
    require(p['sampling'] == ('nearest_timestamp_ties_up' if name == 'ue' else 'nearest_presentation_timestamp_ties_later'), 'Sampling changed')
    require(p['endpoint'] == ('hold_last_frame_at_10_seconds' if name == 'ue' else 'reject_out_of_decoded_range'), 'Endpoint changed')
    require(p['vae'] == {'channels':128,'temporal_factor':8,'spatial_factor':32,'patch_size':1,'patch_size_t':1}, 'VAE geometry changed')
    for canvas in p['video_canvas'].values():
        latent_shape(p['num_frames'], *canvas, p)
        latent_shape(1, *canvas, p)


def media_key(record, canvas, frames, identity, temporal_window=None):
    require(identity['schema'] == IDENTITY_SCHEMA, 'Wrong shared-canvas encoding identity')
    return fingerprint({'schema': 'shared_canvas_media_key_v3', 'media_sha256': record['sha256'],
        'canvas_wh': list(canvas), 'num_frames': frames, 'profile_sha256': identity['profile_sha256'],
        'vae_sha256': identity['vae_sha256'], 'encoder_contract': identity['encoder_contract'],
        'temporal_window': temporal_window})


def pixel_semantics(job, identity):
    """Only facts affecting actual pixels/latent encoding; no positional labels."""
    profile = identity['profile']
    encoder = deepcopy(identity['encoder_contract'])
    encoder.pop('shared_canvas_migration', None)
    fields = ('resize','interpolation','padding_rgb','vae')
    pixel_profile = {key: profile[key] for key in fields}
    if job['is_video']:
        for key in ('fps','num_frames','duration_seconds','source_fps','source_frames',
                    'source_dimensions','sampling','endpoint'):
            if key in profile:
                pixel_profile[key] = profile[key]
    return {'media_sha256':job['media']['sha256'],'bytes':job['media']['bytes'],
            'canvas_wh':job['canvas_wh'],'num_frames':job['num_frames'],'is_video':job['is_video'],
            'temporal_window':job.get('temporal_window'),'pixel_profile':pixel_profile,
            'vae_sha256':identity['vae_sha256'],'encoder_contract':encoder}


def can_reuse_media(old_job, old_identity, new_job, new_identity):
    return pixel_semantics(old_job, old_identity) == pixel_semantics(new_job, new_identity)


def can_reuse_text(old_job, old_identity, new_job, new_identity):
    old = deepcopy(old_identity['encoder_contract']);new = deepcopy(new_identity['encoder_contract'])
    old.pop('shared_canvas_migration', None);new.pop('shared_canvas_migration', None)
    return old_job == new_job and old_identity['weights'] == new_identity['weights'] and old == new


def validate_identity(identity):
    require(identity['schema'] == IDENTITY_SCHEMA, 'Wrong encoding identity')
    validate_profile(identity['profile'])
    require(identity['profile_sha256'] == fingerprint(identity['profile']), 'Shared-canvas profile SHA mismatch')
    require(identity['source_name'] == identity['profile']['source_name'], 'Source/profile name mismatch')
    migration = identity['encoder_contract']['shared_canvas_migration']
    require(migration['canvas_contract'] == CANVAS_CONTRACT and migration['time_contract'] == TIME_CONTRACT,
            'Shared-canvas encoding contract mismatch')
