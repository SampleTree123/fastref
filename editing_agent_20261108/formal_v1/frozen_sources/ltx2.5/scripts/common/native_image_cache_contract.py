"""Original-size image preprocessing and shared immutable encoding identities.

Images retain every original pixel. Only incomplete 32-pixel cells are extended
with edge pixels; video preprocessing remains the separately recorded recipe.
"""
from copy import deepcopy
from prepare_datamake_media import fingerprint

IDENTITY_SCHEMA = 'datamake_native_resolution_encoding_identity_v1'
PROFILE_SCHEMA = 'datamake_native_resolution_media_profile_v1'
IMAGE_POLICY = 'original_minimal_edge_padding_32'


def aligned_canvas(dimensions):
    if (len(dimensions) != 2 or any(type(x) is not int or x <= 0 for x in dimensions)):
        raise ValueError('Image dimensions must be two positive integers')
    return [((x + 31) // 32) * 32 for x in dimensions]


def prepare_image(image):
    import numpy as np
    from PIL import Image
    width, height = image.size
    canvas = aligned_canvas([width, height])
    dx, dy = canvas[0] - width, canvas[1] - height
    left, top = dx // 2, dy // 2
    rgb = np.asarray(image.convert('RGB'))
    result = np.pad(rgb, ((top, dy - top), (left, dx - left), (0, 0)), mode='edge')
    return Image.fromarray(result), {
        'original_dimensions': [width, height], 'canvas_wh': canvas,
        'padding_ltrb': [left, top, dx - left, dy - top],
        'padding_mode': 'edge', 'resize': False, 'crop': False,
    }


def validate_identity(identity):
    profile = identity['profile']
    if identity['schema'] != IDENTITY_SCHEMA or profile['schema'] != PROFILE_SCHEMA:
        raise ValueError('Wrong original-resolution encoding schema')
    if identity['profile_sha256'] != fingerprint(profile):
        raise ValueError('Original-resolution profile changed')
    if profile['reference_image_canvas'] != IMAGE_POLICY or profile['scene_canvas'] != IMAGE_POLICY:
        raise ValueError('Images must retain original resolution with minimal edge padding')
    if profile['vae'] != {'channels': 128, 'temporal_factor': 8, 'spatial_factor': 32,
                          'patch_size': 1, 'patch_size_t': 1}:
        raise ValueError('Unexpected VAE geometry')
    if profile['num_frames'] != (121 if identity['source_name'] == 'ue' else 57) or profile['fps'] != 12:
        raise ValueError('Video sampling changed')


def video_semantics(profile):
    return {k: deepcopy(profile[k]) for k in (
        'fps', 'num_frames', 'duration_seconds', 'source_fps', 'source_frames',
        'source_dimensions', 'sampling', 'endpoint', 'resize', 'interpolation',
        'padding_rgb', 'vae') if k in profile}


def media_key(record, canvas, frames, identity, temporal_window=None):
    """No prompt, manifest, or RoPE scheme in a VAE payload key."""
    image = frames == 1
    return fingerprint({
        'schema': 'native_resolution_media_key_v1', 'media_sha256': record['sha256'],
        'canvas_wh': list(canvas), 'num_frames': frames,
        'pixels': IMAGE_POLICY if image else video_semantics(identity['profile']),
        'vae_sha256': identity['vae_sha256'],
        'encoder': identity['payload_encoder'], 'temporal_window': temporal_window,
    })
