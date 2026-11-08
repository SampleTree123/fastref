#!/usr/bin/env python3
"""Generate explicit cases for base/first/middle/final checkpoints.

Owner: experiments/training/notes/mixed_training_vlm_plan_20261031.md. Run on a free allocated GPU,
never inside an occupied training rank. Default: native Euler 40 steps, CFG=1,
seed 42, no audio, unchanged source canvas/duration and all optional references.
"""
from __future__ import annotations
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if (p / "common/project_layout.py").is_file()) / "common"))
from project_layout import ROOT as _PROJECT_ROOT, resolve_path, script_path, add_script_paths, resolve_cli_paths
add_script_paths()

import argparse
import json
from pathlib import Path
import time

ROOT = _PROJECT_ROOT
CASE_SPLITS = ('train', 'development', 'test')

# Quality-only selection inspired by native examples, not their exact preset.
# Do not reject 3D/CGI/cartoon styles, garment logos or scene text in UE data.
DEFAULT_NEGATIVE_PROMPT = (
    'blurry, out of focus, excessive noise, compression artifacts, flickering, '
    'jittery movement, distorted proportions, deformed facial features, extra limbs, '
    'disfigured hands, unnatural transitions, inconsistent lighting'
)


def guidance_contract(cfg_scale=1.0, stg_scale=0.0, guidance_rescale=0.0,
                      stg_blocks=(28,), negative_prompt=None, guidance_batch_size=1, ref_cfg=1.0):
    """JSON settings shared by worker and launcher; no GPU imports needed."""
    import math
    if not math.isfinite(ref_cfg) or ref_cfg < 1:
        raise ValueError('Reference CFG must be finite and >=1')
    if (not all(math.isfinite(x) for x in (cfg_scale, stg_scale, guidance_rescale))
            or cfg_scale < 1 or stg_scale < 0 or not 0 <= guidance_rescale <= 1):
        raise ValueError('Require finite CFG>=1, STG>=0, and rescale in [0,1]')
    if (type(guidance_batch_size) is not int or guidance_batch_size < 1
            or not isinstance(stg_blocks, (list, tuple)) or len(set(stg_blocks)) != len(stg_blocks)
            or any(type(i) is not int or i < 0 for i in stg_blocks)
            or (stg_scale != 0 and not stg_blocks)):
        raise ValueError('Invalid STG blocks or guidance batch size')
    if negative_prompt is not None and not isinstance(negative_prompt, str):
        raise ValueError('Negative prompt must be text')
    if cfg_scale == 1.0 and negative_prompt is not None:
        raise ValueError('A negative prompt is unused with CFG=1; omit it')
    negative = (DEFAULT_NEGATIVE_PROMPT if negative_prompt is None else negative_prompt) if cfg_scale != 1 else None
    result = {'cfg_scale': float(cfg_scale), 'stg_scale': float(stg_scale),
            'guidance_rescale': float(guidance_rescale), 'stg_blocks': list(stg_blocks),
            'negative_prompt': negative, 'guidance_batch_size': guidance_batch_size,
            'implementation': 'official_GuidedDenoiser_MultiModalGuider_BatchSplitAdapter'
                if (cfg_scale != 1 or stg_scale != 0 or guidance_rescale != 0) else 'official_SimpleDenoiser',
            'audio_guidance': False, 'rescale_scope': 'official_full_packed_latent',
            'references': 'same_clean_source_and_references_in_all_passes'}
    if ref_cfg != 1:
        result.update(ref_cfg=float(ref_cfg),
                      visual_guidance='independent_positive_text_optional_ref_delta_before_official_rescale',
                      references='source_always_present_extra_refs_removed_in_visual_negative_pass')
    return result


def case_split_mapping(uids, splits=None):
    """CLI alignment is explicit; historical requests default to development."""
    if not uids or len(set(uids)) != len(uids):
        raise ValueError('Require distinct, nonempty case UIDs')
    if splits is None:
        splits = ['development'] * len(uids)
    if len(splits) != len(uids) or any(split not in CASE_SPLITS for split in splits):
        raise ValueError('Provide one train/development/test --split per --uid, or omit all splits')
    return dict(zip(uids, splits))


def load_case_datasets(bundle, weights_record, sample_splits, dataset_type=None):
    """Load each requested complete split once; never relabel training as dev."""
    if dataset_type is None:
        from native_cached_dataset import MixedNativeCachedDataset
        dataset_type = MixedNativeCachedDataset
    normalized = case_split_mapping(list(sample_splits), list(sample_splits.values()))
    datasets, locations = {}, {}
    for split in dict.fromkeys(normalized.values()):
        dataset = dataset_type(bundle, expected_weights=weights_record, split=split)
        positions = {uid: i for i, uid in enumerate(dataset.sample_ids)}
        if len(positions) != len(dataset.sample_ids):
            raise ValueError(f'Duplicate UIDs in {split} dataset')
        requested = {uid for uid, assigned in normalized.items() if assigned == split}
        if not requested <= positions.keys():
            raise ValueError(f'Requested case is absent from its declared {split} split: {sorted(requested - positions.keys())}')
        datasets[split] = dataset
        locations.update({uid: (split, positions[uid]) for uid in requested})
    return datasets, locations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT/'configs/training/original_images_original_time/train.yaml')
    p.add_argument('--uid', action='append', required=True, help='Exact sample UID; repeat for each case')
    p.add_argument('--split', action='append', choices=CASE_SPLITS,
                   help='One split for each --uid in the same order; omit all for historical development cases')
    p.add_argument('--checkpoint', type=Path, help='Omit for official base plus zero-effect fresh adapter')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cfg-scale', type=float, default=1.0)
    p.add_argument('--ref-cfg', type=float, default=1.0)
    p.add_argument('--legacy-reference-cache', type=Path, help='Explicitly restore V3 image canvas into this inference-only cache')
    p.add_argument('--stg-scale', type=float, default=0.0)
    p.add_argument('--guidance-rescale', type=float, default=0.0)
    p.add_argument('--stg-blocks', type=int, nargs='+', default=[28])
    p.add_argument('--negative-prompt', default=None)
    p.add_argument('--guidance-batch-size', type=int, default=1)
    from inference_settings import parse_with_inference_settings
    args = parse_with_inference_settings(p); resolve_cli_paths(args)
    if not args.uid or len(set(args.uid)) != len(args.uid) or args.steps < 2:
        p.error('Require distinct sample UIDs and at least two inference steps for the native scheduler')
    if args.output.exists():
        p.error('Output must be a new directory; completed comparisons are never overwritten')
    try:
        sample_splits = case_split_mapping(args.uid, args.split)
        guidance = guidance_contract(args.cfg_scale, args.stg_scale, args.guidance_rescale,
            args.stg_blocks, args.negative_prompt, args.guidance_batch_size, args.ref_cfg)
    except ValueError as error:
        p.error(str(error))

    if args.checkpoint is not None:
        from checkpoint_retention import pin_for_validation
        pin_for_validation(args.checkpoint, args.output)

    import torch
    import yaml
    from accelerate.utils import send_to_device
    from ltx_pipelines.utils.blocks.video_decoder import VideoDecoder
    from ltx_pipelines.utils.media_io import encode_video
    from native_cached_dataset import collate_single_sample
    from native_reference_inference import load_inference_models, generate_latent, encode_negative_context
    from native_training_contract import fingerprint, require, sha_file
    from encoding_helpers import weight_paths, atomic_json

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Expose exactly one unused, allocated GPU')
    weights_record, weights = weight_paths(resolve_path('runtime/models/native_model_revision.json'),
                                          resolve_path('runtime/verification/native_weights_verified.json'))
    cfg = yaml.safe_load(args.config.read_text())
    source_spec = json.loads(args.bundle.read_text())
    for source in source_spec['sources']:
        profile_path = resolve_path(source['media_profile'])
        if not profile_path.is_absolute():
            profile_path = ROOT / profile_path
        from native_training_contract import validate_media_protocol
        profile = json.loads(profile_path.read_text())
        if args.legacy_reference_cache is not None:
            from legacy_reference_canvas import validate_legacy_profile
            validate_legacy_profile(cfg['reference_protocol'], profile)
        else:
            validate_media_protocol(cfg['reference_protocol'], profile['reference_protocol'])
    checkpoint_sha = None
    if args.checkpoint:
        receipt_path = args.checkpoint.with_suffix('.contract.json')
        require(receipt_path.is_file(), 'Checkpoint has not committed its completion receipt')
        receipt = json.loads(receipt_path.read_text())
        expected = receipt['files'][args.checkpoint.name]
        checkpoint_sha = sha_file(args.checkpoint)
        require(args.checkpoint.stat().st_size == expected['bytes'] and checkpoint_sha == expected['sha256'],
                'Checkpoint file differs from committed receipt')
        identity = receipt['identity']
        require(identity['record_sha256']['models/native_model_revision.json'] ==
                sha_file(resolve_path('runtime/models/native_model_revision.json')),
                'Checkpoint base weights differ from current native weight record')
        require(identity['config']['lora'] == cfg['lora'] and
                identity['config']['reference_protocol'] == cfg['reference_protocol'],
                'Checkpoint and evaluation LoRA/reference configurations differ')
    if args.legacy_reference_cache is not None:
        require(args.checkpoint is not None, 'Legacy restoration requires an explicit historical checkpoint')
    datasets, locations = load_case_datasets(args.bundle, weights_record, sample_splits)
    args.output.mkdir(parents=True)
    status = {'schema': 'datamake_fixed_case_generation_v1', 'status': 'running',
              'bundle': str(args.bundle.resolve()), 'bundle_sha256': sha_file(args.bundle),
              'weights_signature': fingerprint(weights_record), 'config_sha256': sha_file(args.config),
              'checkpoint': str(args.checkpoint.resolve()) if args.checkpoint else None,
              'checkpoint_sha256': checkpoint_sha, 'seed': args.seed, 'steps': args.steps,
              'cfg': args.cfg_scale, 'guidance': guidance,
              'sampler': 'official_LTX2Scheduler_Euler', 'sample_uids': args.uid, 'sample_splits': sample_splits,
              'started_unix': time.time(), 'results': []}
    atomic_json(args.output/'generation.json', status)
    device = torch.device('cuda:0')
    try:
        restored_images = {}
        if args.legacy_reference_cache is not None:
            from legacy_reference_canvas import restore_images
            restored_images, audit = restore_images(args.bundle, sample_splits, datasets, locations,
                weights, fingerprint(weights_record), args.legacy_reference_cache)
            status['legacy_reference_restoration'] = audit
            atomic_json(args.output/'generation.json', status)
        negative_context = None
        if guidance['negative_prompt'] is not None:
            text_start = time.monotonic()
            negative_context = encode_negative_context(weights, guidance['negative_prompt'], device=device)
            status['negative_text_encoding'] = {'seconds': time.monotonic()-text_start,
                'context_shape': list(negative_context.shape), 'context_dtype': str(negative_context.dtype),
                'method': 'official_Gemma_left_padding_feature_extractor_video_connector_no_enhancement'}
            atomic_json(args.output/'generation.json', status)
        model, processor, scale = load_inference_models(
            weights, cfg['lora'], checkpoint=args.checkpoint, protocol=cfg['reference_protocol'], device=device, ref_cfg=args.ref_cfg)
        if negative_context is not None:
            negative_context = negative_context.to(device)
        decoder = VideoDecoder(str(weights['vae']), dtype=torch.bfloat16, device=device)
        for i, uid in enumerate(args.uid):
            split, index = locations[uid]
            sample = datasets[split][index]
            # Strip supervised target before the sample ever reaches the GPU.
            sample.pop('video_latents')
            for ref_index, payload in restored_images.get(uid, {}).items():
                sample['references'][ref_index]['latent'] = payload
            batch = send_to_device(collate_single_sample([sample]), device)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.monotonic()
            latent = generate_latent(model, processor, source=batch['source'], references=batch['references'],
                                     conditions=batch['conditions'], scale_factors=scale,
                                     protocol=cfg['reference_protocol'], seed=args.seed, steps=args.steps,
                                     negative_context=negative_context, cfg_scale=args.cfg_scale,
                                     stg_scale=args.stg_scale, guidance_rescale=args.guidance_rescale,
                                     stg_blocks=args.stg_blocks, guidance_batch_size=args.guidance_batch_size, ref_cfg=args.ref_cfg)
            torch.cuda.synchronize(device)
            denoise_peak = torch.cuda.max_memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            output = args.output/f'case_{i:02d}.mp4'
            with torch.inference_mode():
                frames = decoder(latent, generator=torch.Generator(device=device).manual_seed(args.seed))
                encode_video(frames, fps=12, audio=None, output_path=str(output),
                             video_chunks_number=1, thread_count=2)
            torch.cuda.synchronize(device)
            result = {'sample_uid': uid, 'split': split, 'output': output.name, 'seconds': time.monotonic()-start,
                      'denoise_peak_allocated_bytes': denoise_peak,
                      'decode_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                      'bytes': output.stat().st_size, 'sha256': sha_file(output)}
            status['results'].append(result)
            atomic_json(args.output/'generation.json', status)
            print(json.dumps(result), flush=True)
            del latent, batch
        status['status'] = 'completed'
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        status['finished_unix'] = time.time()
        atomic_json(args.output/'generation.json', status)


if __name__ == '__main__':
    main()
