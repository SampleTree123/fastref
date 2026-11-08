"""CoinVE 57-frame native-cache contract; no model loading or media mutation."""
from __future__ import annotations
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if (p / "common/project_layout.py").is_file()) / "common"))
from project_layout import ROOT as _PROJECT_ROOT, resolve_path, script_path, add_script_paths, resolve_cli_paths
add_script_paths()

from fractions import Fraction
from pathlib import Path
import hashlib
import json

from prepare_datamake_media import fingerprint, contain_image, latent_shape


def load_profile(path):
    p = json.loads(resolve_path(path).read_text())
    if (p['schema'] != 'datamake_coinve_media_profile_v1' or p['num_frames'] != 57
            or p['fps'] != 12 or p['reference_protocol'] != 'reference_positions_v2'
            or p['sampling'] != 'nearest_presentation_timestamp_ties_later'
            or p['endpoint'] != 'reject_out_of_decoded_range'):
        raise ValueError('Unsupported CoinVE profile')
    for canvas in p['video_canvas'].values():
        latent_shape(p['num_frames'], *canvas, p)
    return p


def choose_window(row, seed=42):
    """Stable per-edge window. Reserve half a frame at both ends for FPS rounding.

    Historical probe is planning evidence only. Actual decoding validates chosen
    times against real normalized PTS and never silently clamps/extrapolates.
    """
    probes = [row[role]['historical_technical_probe'] for role in ('source', 'target')]
    final_pts = min((x['frames'] - 1) / x['fps'] for x in probes)
    margin = max(1 / (2 * x['fps']) for x in probes)
    span = 56 / 12
    available = final_pts - margin - span
    if available < margin:
        raise ValueError(f'CoinVE clip too short: {row["sample_id"]}')
    digest = hashlib.sha256(f'coinve-window-v1:{seed}:{row["sample_id"]}'.encode()).digest()
    fraction = int.from_bytes(digest[:8], 'big') / (2**64 - 1)
    start = round(margin + fraction * (available - margin), 9)
    return {'schema': 'coinve_shared_window_v1', 'seed': seed, 'start_seconds': start,
            'fps': 12, 'num_frames': 57, 'span_seconds': span,
            'timestamps_seconds': [start + index / 12 for index in range(57)],
            'selection': 'sha256_sample_id_uniform_safe_common_range',
            'planning_last_common_pts_seconds': final_pts,
            'planning_endpoint_margin_seconds': margin}


def media_key(record, canvas, frames, identity, temporal_window):
    return fingerprint({'media_sha256': record['sha256'], 'canvas_wh': list(canvas),
        'num_frames': frames, 'profile_sha256': identity['profile_sha256'],
        'vae_sha256': identity['vae_sha256'], 'encoder_contract': identity['encoder_contract'],
        'temporal_window': temporal_window})


def load_video_frames(path, canvas, profile, window):
    """Decode in presentation order; match identical absolute requested times.

    Normalize each stream's PTS origin, not its duration. Different FPS and VFR
    therefore do not rescale motion. Only neighboring frames around 57 requested
    times are resized. Decode stops after the last requested time is bracketed.
    """
    import av
    times = [Fraction(str(x)) for x in window['timestamps_seconds']]
    if len(times) != 57 or window['fps'] != 12 or window['num_frames'] != 57:
        raise ValueError('Wrong CoinVE time grid')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Time grid must be strictly increasing')
    result, chosen, errors = [], [], []
    first_pts = previous_t = previous = None
    index = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        actual_canvas = profile['video_canvas']['landscape' if stream.width >= stream.height else 'portrait']
        if list(actual_canvas) != list(canvas):
            raise ValueError('Actual media orientation differs from planned canvas')
        original = {'width': stream.width, 'height': stream.height,
                    'fps': str(stream.average_rate), 'frames': stream.frames}
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError('CoinVE decoding requires presentation timestamps')
            absolute_t = Fraction(frame.pts) * frame.time_base
            if first_pts is None:
                first_pts = absolute_t
            t = absolute_t - first_pts
            if previous_t is not None and t <= previous_t:
                raise ValueError('Non-increasing decoded presentation timestamps')
            while index < len(times) and times[index] <= t:
                requested = times[index]
                if previous is None and requested < t:
                    raise ValueError('Requested time precedes the decoded video')
                pick = frame
                picked_t = t
                if previous is not None and requested - previous_t < t - requested:
                    pick, picked_t = previous, previous_t
                image, geometry = contain_image(pick.to_image(), canvas, profile)
                result.append(image)
                chosen.append(float(picked_t))
                errors.append(float(abs(picked_t - requested)))
                index += 1
            if index == len(times):
                break
            previous, previous_t = frame, t
    if index != len(times):
        raise ValueError(f'Actual decoded clip too short: {path}; selected {index}/57')
    # An isolated large PTS gap is not accepted as ordinary nearest-frame sampling.
    if max(errors) > 1 / profile['fps']:
        raise ValueError(f'CoinVE temporal sampling error exceeds one output frame: {max(errors)}')
    return result, {'original': original, 'geometry': geometry,
                    'temporal_window': window, 'selected_pts_seconds': chosen,
                    'maximum_nearest_timestamp_error_seconds': max(errors),
                    'decoded_pts_origin_seconds': float(first_pts),
                    'vae_candidate': latent_shape(57, *canvas, profile)}
