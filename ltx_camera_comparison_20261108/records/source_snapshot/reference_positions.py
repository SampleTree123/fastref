"""Versioned reference coordinates shared by training and inference.

Versioned protocols; v1 defaults preserve historical probes. Build local coordinates with official
patchifier/get_pixel_coords(causal_fix=True)/fps FIRST, then call this helper.
Do not use first_latent_frame as a replacement: it changes a standalone first
frame's interval width when applied before the official causal correction.
V3 uses the source canvas and official zero-origin coordinates for every reference;
image slots preserve input ordering but intentionally do not create RoPE offsets.
V5/V6/V7 keep each image's own spatial extent; 2048 is a RoPE scale, not a hard bound.
V4 shares V3 local/spatial bounds, then shifts each segment along one sequential
time axis: source, ordered optional references, target. Physical packing stays target-first.
V7 uses ordered optional references, source, target for both coordinates and packing.
V8 uses the same packing while retaining V5 shared local time/spatial coordinates.
"""
from __future__ import annotations

import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if (p / "common/project_layout.py").is_file()) / "common"))
from project_layout import ROOT as _PROJECT_ROOT, resolve_path, script_path, add_script_paths, resolve_cli_paths
add_script_paths()

import torch

PROTOCOL_VERSION = 'reference_positions_v1_candidate'
PROTOCOL_VERSION_V2 = 'reference_positions_v2'
PROTOCOL_VERSION_V3 = 'reference_positions_v3_shared_canvas'
PROTOCOL_VERSION_V4 = 'reference_positions_v4_sequential_time'
PROTOCOL_VERSION_V5 = 'reference_positions_v5_native_images'
PROTOCOL_VERSION_V6 = 'reference_positions_v6_native_images_sequential_time'
PROTOCOL_VERSION_V7 = 'reference_positions_v7_refs_source_target'
PROTOCOL_VERSION_V8 = 'reference_positions_v8_shared_time_refs_source_target'
REFS_FIRST_PROTOCOLS = (PROTOCOL_VERSION_V7, PROTOCOL_VERSION_V8)
SHARED_CANVAS_PROTOCOLS = (PROTOCOL_VERSION_V3, PROTOCOL_VERSION_V4)
NATIVE_IMAGE_PROTOCOLS = (PROTOCOL_VERSION_V5, PROTOCOL_VERSION_V6, PROTOCOL_VERSION_V7, PROTOCOL_VERSION_V8)
ZERO_LOCAL_OFFSET_PROTOCOLS = (*SHARED_CANVAS_PROTOCOLS, *NATIVE_IMAGE_PROTOCOLS)
SEQUENTIAL_TIME_PROTOCOLS = (PROTOCOL_VERSION_V4, PROTOCOL_VERSION_V6, PROTOCOL_VERSION_V7)
NATIVE_IMAGE_CONTRACT = 'native_image_resolution_minimal_edge_padding_32_v1'
SEQUENTIAL_TIME_CONTRACT = 'source_ordered_refs_target_native_seconds_v1'
REFS_FIRST_TIME_CONTRACT = 'ordered_refs_source_target_native_seconds_v1'
REFS_FIRST_LAYOUT = 'ordered_optional_refs_then_source_then_target'
MAX_SEQUENCE_SECONDS = 64.0
PROTOCOLS = {
    PROTOCOL_VERSION: {'max_spatial_extent': 640.0, 'motion_width_offset': 704.0},
    PROTOCOL_VERSION_V2: {'max_spatial_extent': 768.0, 'motion_width_offset': 832.0},
    PROTOCOL_VERSION_V3: {'max_spatial_extent': 768.0, 'motion_width_offset': 0.0},
    PROTOCOL_VERSION_V4: {'max_spatial_extent': 768.0, 'motion_width_offset': 0.0},
    PROTOCOL_VERSION_V5: {'max_spatial_extent': None, 'motion_width_offset': 0.0},
    PROTOCOL_VERSION_V6: {'max_spatial_extent': None, 'motion_width_offset': 0.0},
    PROTOCOL_VERSION_V7: {'max_spatial_extent': None, 'motion_width_offset': 0.0},
    PROTOCOL_VERSION_V8: {'max_spatial_extent': None, 'motion_width_offset': 0.0},
}


def sequential_time_contract(protocol):
    return REFS_FIRST_TIME_CONTRACT if protocol == PROTOCOL_VERSION_V7 else SEQUENTIAL_TIME_CONTRACT


def rotate_state_prefix_to_end(state, count):
    """Move a complete token prefix, including all token-indexed fields.

    V7 uses this once after official target-first construction and reverses it
    only after denoising, before official target-first output extraction.
    """
    from dataclasses import replace
    if state.generated_keyframe_layout is not None or state.generated_keyframes is not None:
        raise ValueError('Reference editing does not use generated keyframe slots')
    length = state.latent.shape[1]
    if not 0 < count < length:
        raise ValueError('Expected nonempty target and reference segments')
    def rotate(value, axis):
        if value is None:
            return None
        return torch.cat((value.narrow(axis, count, length-count), value.narrow(axis, 0, count)), dim=axis)
    attention = state.attention_mask
    if attention is not None:
        attention = rotate(rotate(attention, 1), 2)
    return replace(state, latent=rotate(state.latent, 1),
                   positions=rotate(state.positions, 2),
                   clean_latent=rotate(state.clean_latent, 1),
                   denoise_mask=state.denoise_mask.with_same_values(rotate(state.denoise_mask.tensor, 1)),
                   keyframes_mask=rotate(state.keyframes_mask, 1), attention_mask=attention)


def protocol_spec(protocol_version: str) -> dict:
    if protocol_version not in PROTOCOLS:
        raise ValueError(f'Unknown reference coordinate protocol: {protocol_version!r}')
    return dict(PROTOCOLS[protocol_version])

FPS = 12.0
MAX_FRAMES = 121
MAX_LOCAL_SPATIAL_EXTENT = 640.0
MOTION_WIDTH_OFFSET = 704.0
IMAGE_TIME_ORIGIN = 12.0
IMAGE_SLOT_SPACING = 1.0 / 3.0
MAX_IMAGES = 20
ROPE_TIME_SCALE = 20.0
ROPE_SPATIAL_SCALE = 2048.0


def reference_position_offset(role: str, image_slot: int | None = None, *, protocol_version: str = PROTOCOL_VERSION) -> tuple[float, float, float]:
    """Return (seconds, pixel-height, pixel-width); slots are explicit, not array indices."""
    spec = protocol_spec(protocol_version)
    if role not in {'source', 'motion', 'image'}:
        raise ValueError(f'Unsupported reference role: {role!r}')
    if role != 'image':
        if image_slot is not None:
            raise ValueError('Only image roles may specify image_slot')
        return (0.0, 0.0, spec['motion_width_offset'] if role == 'motion' else 0.0)
    if type(image_slot) is not int or not 0 <= image_slot < MAX_IMAGES:
        raise ValueError(f'image_slot must be an integer in [0, {MAX_IMAGES - 1}]')
    if protocol_version in ZERO_LOCAL_OFFSET_PROTOCOLS:
        return (0.0, 0.0, 0.0)
    return (IMAGE_TIME_ORIGIN + image_slot * IMAGE_SLOT_SPACING, 0.0, 0.0)


def apply_reference_position_protocol(
    positions: torch.Tensor,
    *,
    role: str,
    image_slot: int | None = None,
    protocol_version: str = PROTOCOL_VERSION,
) -> torch.Tensor:
    """Return a COPY of official [B,3,N,2] local pixel/second bounds.

    Input must be float32, unshifted, causal-corrected coordinates for 12 FPS media
    within the selected protocol spatial bound / <=121 frame profile. Static image reference
    time intervals must be exactly [0,1/12]. Offset is applied equally to both ends.
    No VAE values, token ordering, noise timestep, marker, or mask are changed here.
    V1/V2 translate reference coordinates; V3/V4 use zero local offsets.
    V4 callers subsequently assemble the full sequential timeline.
    Caller owns these fields and must permute them with the corresponding segment.
    """
    spec = protocol_spec(protocol_version)
    offsets = reference_position_offset(role, image_slot, protocol_version=protocol_version)
    if not isinstance(positions, torch.Tensor):
        raise TypeError('positions must be a torch.Tensor')
    if positions.ndim != 4 or positions.shape[1] != 3 or positions.shape[-1] != 2:
        raise ValueError('Expected positions shape [B, 3, N, 2]')
    if positions.shape[0] < 1 or positions.shape[2] < 1:
        raise ValueError('Empty coordinate segments are not allowed')
    if positions.dtype != torch.float32:
        raise ValueError('Official coordinate bounds must remain float32')
    if positions.requires_grad:
        raise ValueError('This fixed protocol does not support trainable coordinates')
    if not bool(torch.isfinite(positions).all()):
        raise ValueError('Non-finite coordinates')
    if bool((positions < 0).any()) or bool((positions[..., 1] <= positions[..., 0]).any()):
        raise ValueError('Expected nonnegative local coordinates with positive interval widths')
    if spec['max_spatial_extent'] is not None and bool((positions[:, 1:] > spec['max_spatial_extent']).any()):
        raise ValueError('Local spatial extent exceeds this candidate profile')
    if bool((positions[:, 0] > MAX_FRAMES / FPS + 1e-5).any()):
        raise ValueError('Local time extent exceeds this candidate profile')
    if role == 'image':
        time = positions[:, 0]
        if not (torch.allclose(time[..., 0], torch.zeros_like(time[..., 0]), rtol=0, atol=1e-6)
                and torch.allclose(time[..., 1], torch.full_like(time[..., 1], 1.0 / FPS), rtol=0, atol=1e-6)):
            raise ValueError('Image reference must encode one standalone frame at [0, 1/12]')
    translated = positions + positions.new_tensor(offsets).view(1, 3, 1, 1)
    # These checks are protocol limits, not claims that RoPE cannot extrapolate.
    if bool((translated[:, 0] >= ROPE_TIME_SCALE).any()):
        raise ValueError('Candidate time namespace exceeded')
    if protocol_version not in NATIVE_IMAGE_PROTOCOLS and bool((translated[:, 1:] >= ROPE_SPATIAL_SCALE).any()):
        raise ValueError('Candidate spatial namespace exceeded')
    return translated


def validate_reference_slots(segments: list[dict], *, protocol_version: str = PROTOCOL_VERSION) -> None:
    """Validate one request's explicit roles without adding placeholders or truncating.

    A current dataset request has one source plus images OR a motion reference.
    Image slot holes are allowed for future explicit mappings; slot identity must
    survive list reordering. Pair/view semantics belong to the dataset adapter.
    """
    protocol_spec(protocol_version)
    source_count = 0
    motion_count = 0
    slots = set()
    for segment in segments:
        role = segment['role']
        slot = segment.get('image_slot')
        reference_position_offset(role, slot, protocol_version=protocol_version)
        source_count += role == 'source'
        motion_count += role == 'motion'
        if role == 'image':
            if slot in slots:
                raise ValueError(f'Duplicate image_slot: {slot}')
            slots.add(slot)
    if source_count != 1 or motion_count > 1:
        raise ValueError('Expected exactly one source and at most one motion reference')
    if motion_count and slots:
        raise ValueError('Mixed motion and image references require a new approved data contract')


def native_temporal_span(latent_frames, fps, temporal_scale):
    """Seconds covered by official causal LTX bounds; one image spans 1/fps."""
    import math
    if type(latent_frames) is not int or latent_frames < 1 or temporal_scale < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid native temporal geometry')
    return ((latent_frames - 1) * temporal_scale + 1) / fps


def apply_sequential_time_layout(target_positions, reference_positions, *, target_span, reference_spans):
    """Translate local native bounds; keep target-first physical packing.

    Clock order follows the provided reference list, then target. V4/V6 callers
    pass source first; V7 callers pass optional references then source. No re-scaling,
    spatial changes, padding, model max_pos changes, or GPU scalar readbacks.
    V4 explicitly permits coordinates beyond the native nominal 20-second scale.
    The 64-second bound is a local experiment guard, not a model capability claim.
    Callers must supply already validated local bounds and metadata-derived spans.
    """
    import math
    if not reference_positions or len(reference_positions) != len(reference_spans):
        raise ValueError('Sequential time needs source and matching spans')
    spans = [*reference_spans, target_span]
    if any(not math.isfinite(s) or s <= 0 for s in spans) or sum(spans) > MAX_SEQUENCE_SECONDS:
        raise ValueError('Sequential timeline exceeds explicit experiment bounds')
    result = []
    cursor = 0.0
    for positions, span in zip(reference_positions, reference_spans):
        translated = positions.clone()
        translated[:, 0] += cursor
        result.append(translated)
        cursor += span
    target = target_positions.clone()
    target[:, 0] += cursor
    return target, result
