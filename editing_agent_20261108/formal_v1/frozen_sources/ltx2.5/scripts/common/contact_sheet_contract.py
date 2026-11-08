"""Contact-sheet metadata shared by manifest, cache and runtime readers.

Owner: experiments/data_preparation/notes/dataset_contact_sheet_handoff_questions_20261101.md.
This validates the agreed single-image input; it does not alter latent positions.
Original separate front/back references retain their existing runtime checks.
"""
from __future__ import annotations

import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if (p / "common/project_layout.py").is_file()) / "common"))
from project_layout import ROOT as _PROJECT_ROOT, resolve_path, script_path, add_script_paths, resolve_cli_paths
add_script_paths()

import copy
import json
from pathlib import Path
import re

REPRESENTATION = 'contact_sheet_v1'
NATIVE_REPRESENTATION = 'original_size_contact_sheet_v2'
NATIVE_PROTOCOLS = ('reference_positions_v5_native_images', 'reference_positions_v6_native_images_sequential_time', 'reference_positions_v7_refs_source_target', 'reference_positions_v8_shared_time_refs_source_target')
NATIVE_LAYOUTS = {(2048, 1024): 'left_front_right_back', (1024, 2048): 'top_front_bottom_back'}
PROTOCOL = 'reference_positions_v3_shared_canvas'
SCOPE_BY_TYPE = {'outfit': 'complete_outfit', 'garment': 'single_piece', 'hair': 'hairstyle'}
LAYOUTS = {
    (768, 448): ('left_front_right_back', [0, 32, 0, 32]),
    (448, 768): ('top_front_bottom_back', [32, 0, 32, 0]),
}
PROVENANCE_FIELDS = (
    'reference_representation', 'reference_scope', 'sources', 'layout', 'tile_size',
    'canvas_size', 'padding_ltrb', 'padding_rgb', 'resize_filter', 'pillow_version',
    'outfit_id', 'attributes', 'original_dimensions',
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sheet_provenance(ref):
    """Stable source/layout identity; excludes cache keys and mutable array indices."""
    return copy.deepcopy({key: ref[key] for key in PROVENANCE_FIELDS if key in ref})


def validate_contact_sheet(ref, *, canvas_wh=None, protocol=None, reference_count=None):
    """Return whether this reference is a sheet, rejecting disguised/new formats."""
    representation = ref.get('reference_representation')
    if representation is None:
        require(ref.get('view') != 'front_back', 'Combined front/back reference requires contact-sheet metadata')
        if ref.get('type') == 'outfit':
            require(ref.get('view') in {'front', 'back'} and (protocol is None or protocol in NATIVE_PROTOCOLS),
                    'Separate outfit references require native-image protocol and front/back views')
        return False
    require(representation in (REPRESENTATION, NATIVE_REPRESENTATION), 'Unknown image reference representation')
    kind = ref.get('type')
    require(kind in SCOPE_BY_TYPE and ref.get('reference_scope') == SCOPE_BY_TYPE[kind],
            'Contact-sheet type and editing scope disagree')
    require(ref.get('view') == 'front_back', 'Contact sheet must declare both views')
    if reference_count is not None:
        require(reference_count == 1, 'Contact sheet must be the only optional reference')
    canvas = ref.get('canvas_size')
    if representation == NATIVE_REPRESENTATION:
        require(protocol is None or protocol in NATIVE_PROTOCOLS, 'Original-size sheets require the native-image protocol')
        require(isinstance(canvas, list) and tuple(canvas) in NATIVE_LAYOUTS, 'Unsupported original-size contact-sheet canvas')
        require(ref.get('layout') == NATIVE_LAYOUTS[tuple(canvas)]
                and ref.get('tile_size') == [1024, 1024] and ref.get('padding_ltrb') == [0, 0, 0, 0]
                and ref.get('resize_filter') == 'none', 'Original-size sheet must preserve both complete 1024-square views')
        require(ref.get('original_dimensions') == canvas, 'Original-size sheet dimensions disagree with manifest')
        if canvas_wh is not None:
            require((canvas_wh[0] > canvas_wh[1]) == (canvas[0] > canvas[1]), 'Contact-sheet layout orientation differs from video')
    else:
        if protocol is not None:
            require(protocol == PROTOCOL, 'Historical contact sheet requires the shared-canvas position protocol')
        require(isinstance(canvas, list) and tuple(canvas) in LAYOUTS, 'Unsupported contact-sheet canvas')
        if canvas_wh is not None:
            require(list(canvas_wh) == canvas, 'Contact-sheet canvas differs from source video')
        layout, padding = LAYOUTS[tuple(canvas)]
        require(ref.get('layout') == layout and ref.get('padding_ltrb') == padding
                and ref.get('tile_size') == [384, 384] and ref.get('padding_rgb') == [0, 0, 0],
                'Contact-sheet view placement, tile size or padding changed')
        require(ref.get('resize_filter') == 'PIL.Image.Resampling.LANCZOS'
                and isinstance(ref.get('pillow_version'), str) and bool(ref['pillow_version']),
                'Contact-sheet resize implementation must be recorded')
    sources = ref.get('sources')
    require(isinstance(sources, list) and len(sources) == 2
            and [item.get('view') for item in sources] == ['front', 'back'],
            'Contact-sheet sources must retain ordered front/back provenance')
    for item in sources:
        require(isinstance(item.get('path'), str) and bool(item['path'])
                and '..' not in Path(item['path']).parts
                and isinstance(item.get('sha256'), str)
                and re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None
                and type(item.get('bytes')) is int and item['bytes'] > 0,
                'Invalid contact-sheet source identity')
    if kind == 'outfit':
        require(isinstance(ref.get('outfit_id'), str) and bool(ref['outfit_id']),
                'Complete-outfit sheet must retain its outfit ID')
    return True


def runtime_sheet_metadata(ref):
    """Keep structured provenance as one string through PyTorch default_collate.

    Source fields differ between outfit and original garment/hair references.
    Serializing metadata avoids batching optional nested fields as model tensors.
    No contents of this field are fed to the transformer or text encoder.
    """
    if not validate_contact_sheet(ref):
        return {}
    return {'reference_representation': ref['reference_representation'],
            'contact_sheet_json': json.dumps(sheet_provenance(ref), sort_keys=True,
                                            separators=(',', ':'), ensure_ascii=False)}


def resolve_manifest_media(record, dataset_root):
    """Resolve an explicit per-reference root without allowing relative traversal."""
    if 'media_root' in record:
        require(Path(record['media_root']).is_absolute(), 'Explicit media_root must be absolute')
    root = resolve_path(record.get('media_root', dataset_root))
    relative = Path(record['path'])
    path = (root / relative).resolve()
    require(not relative.is_absolute() and '..' not in relative.parts
            and path.is_relative_to(root), 'Media path escapes its declared root')
    if 'absolute_path' in record:
        require(path == resolve_path(record['absolute_path']).resolve(), 'Media path/root mismatch')
    return path
