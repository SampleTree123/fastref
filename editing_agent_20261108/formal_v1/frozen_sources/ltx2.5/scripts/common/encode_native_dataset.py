#!/usr/bin/env python3
"""Resumable native VAE/compact-text conversion of complete train+test manifests.

No training, resizing changes, filtering, or copied/recompressed source videos.
Each cache item has an atomic completion receipt. Final metadata is published only
when every source/target/reference/text item exists and passes identity/SHA checks.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import re
import site
import time
import uuid

import prepare_datamake_media as media
from smoke_native_encoding import ROOT, atomic_json, sha_file, shape_check, tensor_info, weight_paths


def require(value, message):
    if not value:
        raise ValueError(message)


def media_key(record, canvas, frames, identity):
    return media.fingerprint({'media_sha256': record['sha256'], 'canvas_wh': list(canvas), 'num_frames': frames,
        'profile_sha256': identity['profile_sha256'], 'vae_sha256': identity['vae_sha256'],
        'encoder_contract': identity['encoder_contract']})


def item_paths(root, stage, key):
    require(re.fullmatch('[0-9a-f]{64}', key) is not None, 'Cache key must be a SHA256 hex string')
    base = root / stage / key[:2] / key
    return base.with_suffix('.pt'), base.with_suffix('.json')


@contextmanager
def lock(path, *, blocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise RuntimeError(f'Another encoder owns {path}; use disjoint shard assignments') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def build_plan(manifest_paths, profile, identity):
    """Pure metadata indexing; media keys include dimensions/frame count/weights."""
    samples, media_jobs, text_jobs, seen_ids = [], {}, {}, set()
    for split, manifest in manifest_paths.items():
        with Path(manifest).open() as handle:
            for line in handle:
                row = json.loads(line)
                sid = row['sample_id']
                require(row['schema'] == 'datamake_ltx_single_turn_v1' and row['split'] == split, 'Manifest schema/split mismatch')
                require(type(sid) is int and sid not in seen_ids, 'Duplicate/invalid sample ID across train/test')
                seen_ids.add(sid)
                aspect = row['source']['attributes']['aspect_id']
                require(aspect in (1, 2), 'Unsupported source aspect')
                canvas = profile['video_canvas']['landscape' if aspect == 1 else 'portrait']
                require(row['target']['attributes']['aspect_id'] == aspect, 'Source/target aspect mismatch')
                source_dimensions = [1920, 1080] if aspect == 1 else [1080, 1920]
                refs = row['references']
                require(len(refs) <= 20, 'Too many references')
                require(not any(x['type'] == 'motion' for x in refs) or len(refs) == 1, 'Motion must be the only optional reference')
                sample = {'sample_id': sid, 'split': split, 'references': [], 'text': row['caption_key']}
                items = [('source', row['source']), ('target', row['target'])]
                items += [(f'reference_{i}', item) for i, item in enumerate(refs)]
                for role, record in items:
                    relative = Path(record['path'])
                    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe media path')
                    require(type(record['bytes']) is int and record['bytes'] > 0 and re.fullmatch('[0-9a-f]{64}', record['sha256']), 'Invalid media identity')
                    kind = record.get('type') if role.startswith('reference_') else 'video'
                    require(kind in {'video', 'motion', 'identity', 'hair', 'garment', 'scene'}, 'Unknown reference kind')
                    is_video = kind in ('video', 'motion')
                    item_canvas = canvas if is_video or kind == 'scene' else profile['reference_image_canvas']
                    frames = profile['num_frames'] if is_video else 1
                    key = media_key(record, item_canvas, frames, identity)
                    job = {'key': key, 'media': {k: record[k] for k in ('path', 'bytes', 'sha256')},
                           'canvas_wh': item_canvas, 'num_frames': frames, 'is_video': is_video,
                           'scene_original_dimensions': source_dimensions if kind == 'scene' else None}
                    if key in media_jobs:
                        previous = media_jobs[key]
                        require(previous['media']['bytes'] == record['bytes'] and previous['is_video'] == is_video,
                                'Same media content key has incompatible metadata')
                        # Content-identical aliases share one verified representative.
                        if kind == 'scene':
                            require(previous['scene_original_dimensions'] in (None, source_dimensions), 'Scene aspect conflict')
                            previous['scene_original_dimensions'] = source_dimensions
                    else:
                        media_jobs[key] = job
                    if role in ('source', 'target'):
                        sample[role] = key
                    else:
                        index = int(role.split('_')[1])
                        if kind in ('hair', 'garment'):
                            require(len(refs) % 2 == 0 and record.get('view') == ('front' if index % 2 == 0 else 'back'), 'Malformed front/back reference pair')
                        sample['references'].append({'index': index, 'type': kind, 'view': record.get('view', ''), 'media_key': key})
                text_key = row['caption_key']
                require(re.fullmatch('[0-9a-f]{64}', text_key) is not None and isinstance(row['prompt']['en'], str)
                        and bool(row['prompt']['en'].strip()), 'Missing prompt/caption key')
                text_job = {'key': text_key, 'caption_key': text_key, 'prompt_en': row['prompt']['en']}
                require(text_key not in text_jobs or text_jobs[text_key] == text_job, 'Caption key collision with different text')
                text_jobs[text_key] = text_job
                samples.append(sample)
    require(samples, 'Empty manifests')
    return {'samples': samples, 'media': media_jobs, 'text': text_jobs}


def artifact_info(path, root):
    return {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha_file(path)}


def verify_receipt(root, stage, key, identity_sha):
    artifact, receipt_path = item_paths(root, stage, key)
    if not receipt_path.exists():
        return None  # Orphan artifact from interrupted write is not complete.
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get('status') == 'passed' and receipt.get('key') == key
            and receipt.get('identity_sha256') == identity_sha and receipt.get('stage') == stage,
            f'Wrong cache identity: {receipt_path}')
    evidence = receipt['artifact']
    require(evidence['path'] == str(artifact.relative_to(root)), 'Receipt artifact path mismatch')
    require(artifact.is_file() and artifact.stat().st_size == evidence['bytes'] and sha_file(artifact) == evidence['sha256'],
            f'Missing/changed completed cache: {artifact}; refusing silent overwrite')
    return receipt


def validate_item_semantics(receipt, job, identity, stage):
    if stage == 'media':
        expected_shape = [1, *media.latent_shape(job['num_frames'], *job['canvas_wh'], identity['profile'])['shape_cfhw']]
        require(receipt['media_sha256_verified'] == job['media']['sha256']
                and receipt['media_path'] == job['media']['path']
                and receipt['canvas_wh'] == job['canvas_wh'] and receipt['num_frames'] == job['num_frames'],
                'Completed media receipt metadata disagrees with planned item')
        require(receipt['tensor']['shape'] == expected_shape and receipt['tensor']['dtype'] == 'torch.bfloat16'
                and receipt['tensor']['finite'] is True, 'Completed media shape/dtype/finite contract mismatch')
    else:
        from compact_text_cache import make_contract
        files = {x['path'].split('/')[0]: x['sha256'] for x in identity['weights']['files']}
        expected = make_contract(sequence_length=1024, feature_dim=4096, num_learnable_registers=128,
            encoder_identity=media.fingerprint({'pack_sha256': files['text_encoders'], 'padding_side': 'left', 'enhancement': False}),
            connector_identity=media.fingerprint({'transformer_sha256': files['diffusion_models'], 'text_pack_sha256': files['text_encoders']}),
            text_identity=media.fingerprint({'prompt_en': job['prompt_en'], 'caption_key': job['caption_key']}))
        require(receipt['prompt_en'] == job['prompt_en'] and receipt['caption_key'] == job['caption_key']
                and receipt['contract'] == expected, 'Completed text receipt/input/model contract mismatch')
        require(receipt['pre_connector']['shape'] == [1,1024,4096]
                and receipt['pre_connector']['dtype'] == 'torch.bfloat16' and receipt['pre_connector']['finite'] is True,
                'Completed text shape/dtype/finite contract mismatch')
        require(receipt['compact_validation'] in {'valid_rows_exact_full_length_restored', 'bitwise_equal_real_video_connector_and_mask'}
                and type(receipt['valid_rows']) is int and 0 < receipt['valid_rows'] <= 1024, 'Missing compact text validation')


def save_item(root, stage, key, identity_sha, payload, metadata):
    import torch
    artifact, receipt_path = item_paths(root, stage, key)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix(f'.{uuid.uuid4().hex}.tmp')
    try:
        torch.save(payload, temporary)
        temporary.replace(artifact)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {'status': 'passed', 'stage': stage, 'key': key, 'identity_sha256': identity_sha,
               'artifact': artifact_info(artifact, root), **metadata}
    atomic_json(receipt_path, receipt)
    return receipt


def make_identity(args, profile, weights):
    native = ROOT / 'repos/LTX-2/packages'
    upstream = ['ltx-trainer/src/ltx_trainer/model_loader.py',
                'ltx-core/src/ltx_core/model/video_vae/video_vae.py',
                'ltx-core/src/ltx_core/model/video_vae/normalization.py',
                'ltx-core/src/ltx_core/text_encoders/gemma/embeddings_processor.py']
    file_hashes = {x['path'].split('/')[0]: x['sha256'] for x in weights['files']}
    contract = {'dtype': 'torch.bfloat16', 'normalization': 'official_video_encoder_forward_normalized_mean',
                'tiling': False, 'pixel_toolchain': media.processing_contract(), 'seed': args.seed,
                'encoder_script_sha256': sha_file(__file__), 'compact_script_sha256': sha_file(ROOT / 'scripts/compact_text_cache.py'),
                'torch_version': importlib.metadata.version('torch'),
                'upstream_sha256': {p: sha_file(native / p) for p in upstream}}
    return {'schema': 'datamake_native_encoding_identity_v2', 'profile': profile, 'profile_sha256': media.fingerprint(profile),
            'weights': weights, 'vae_sha256': file_hashes['vae'], 'encoder_contract': contract,
            'manifest_sha256': {split: sha_file(path) for split, path in args.manifests.items()},
            'manifest_paths': {split: str(path.resolve()) for split, path in args.manifests.items()},
            'dataset_root': str(args.dataset_root.resolve())}


def ensure_plan(args):
    require(args.output.resolve().is_relative_to(ROOT / 'outputs/native_dataset'), 'Output must stay under outputs/native_dataset')
    profile = media.load_profile(args.profile)
    require(profile.get('reference_protocol') == 'reference_positions_v2', 'Production conversion requires explicit v2 profile')
    weights, paths = weight_paths(args.weights, args.verified_weights)
    identity = make_identity(args, profile, weights)
    identity_sha = media.fingerprint(identity)
    args.output.mkdir(parents=True, exist_ok=True)
    with lock(args.output / '.locks/plan.lock', blocking=True):
        plan_path = args.output / 'plan.json'
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            require(plan['identity'] == identity and plan['identity_sha256'] == identity_sha, 'Existing conversion identity differs; use a new output root')
            require(plan['plan_sha256'] == media.fingerprint(plan['index']), 'Conversion plan index changed')
        else:
            index = build_plan(args.manifests, profile, identity)
            plan = {'identity': identity, 'identity_sha256': identity_sha, 'plan_sha256': media.fingerprint(index), 'index': index}
            atomic_json(plan_path, plan)
    return plan, paths


def encode_media(job, args, identity, vae):
    import numpy as np
    import torch
    from PIL import Image
    profile = identity['profile']
    path = media.verify_media(job['media'], args.dataset_root)
    canvas = job['canvas_wh']
    if job['is_video']:
        require(media.video_canvas(media.probe_video(path), profile) == canvas, 'Video dimensions/aspect disagree with sample source')
        frames, pixel_metadata = media.load_video_frames(path, canvas, profile)
    else:
        with Image.open(path) as original:
            expected = job['scene_original_dimensions']
            require(expected is None or list(original.size) == expected, 'Scene reference does not match source dimensions')
            image, geometry = media.contain_image(original, canvas, profile)
            pixel_metadata = {'original_dimensions': list(original.size), 'geometry': geometry}
        frames = [image]
    pixels = torch.from_numpy(np.stack([np.asarray(frame) for frame in frames]))
    pixels = pixels.float().div_(255).sub_(0.5).div_(0.5).permute(3, 0, 1, 2).unsqueeze(0).to('cuda', torch.bfloat16)
    latent = vae(pixels)
    shape_check(latent.shape, len(frames), canvas, profile)
    info = tensor_info(latent)
    payload = {'latents': latent[0].cpu().contiguous(), 'num_frames': latent.shape[2],
               'height': latent.shape[3], 'width': latent.shape[4], 'fps': profile['fps']}
    details = {'media_path': job['media']['path'], 'media_sha256_verified': job['media']['sha256'],
               'canvas_wh': canvas, 'num_frames': len(frames), 'pixel_metadata': pixel_metadata, 'tensor': info}
    return payload, details


def text_contract(job, identity, processor, features):
    from compact_text_cache import make_contract
    files = {x['path'].split('/')[0]: x['sha256'] for x in identity['weights']['files']}
    return make_contract(sequence_length=features.shape[1], feature_dim=features.shape[2],
        num_learnable_registers=processor.video_connector.num_learnable_registers,
        encoder_identity=media.fingerprint({'pack_sha256': files['text_encoders'], 'padding_side': 'left', 'enhancement': False}),
        connector_identity=media.fingerprint({'transformer_sha256': files['diffusion_models'], 'text_pack_sha256': files['text_encoders']}),
        text_identity=media.fingerprint({'prompt_en': job['prompt_en'], 'caption_key': job['caption_key']}))


def encode_text(job, identity, encoder, processor, verify_connector=False):
    import torch
    from compact_text_cache import pack_text_features, unpack_text_features, validate_connector
    hidden, mask = encoder.encode([job['prompt_en']])[0]
    features, _ = processor.feature_extractor(hidden, mask, 'left')
    info = tensor_info(features)
    contract = text_contract(job, identity, processor, features)
    validate_connector(processor.video_connector, contract)
    packed = pack_text_features(features[0], mask[0], contract=contract)
    restored = unpack_text_features(packed, expected_contract=contract)
    require(torch.equal(restored['video_prompt_embeds'][mask[0].bool().cpu()], features[0][mask[0].bool()].cpu()), 'Compact valid text rows changed')
    verification = 'valid_rows_exact_full_length_restored'
    if verify_connector:
        from ltx_core.text_encoders.gemma.embeddings_processor import convert_to_additive_mask
        original, _, original_mask = processor.create_embeddings(features, None, convert_to_additive_mask(mask, features.dtype))
        recovered, _, recovered_mask = processor.create_embeddings(restored['video_prompt_embeds'].unsqueeze(0).to('cuda'), None,
            convert_to_additive_mask(restored['prompt_attention_mask'].unsqueeze(0).to('cuda'), features.dtype))
        require(torch.equal(original, recovered) and torch.equal(original_mask, recovered_mask), 'Compact cache real connector equality failed')
        verification = 'bitwise_equal_real_video_connector_and_mask'
    return packed, {'prompt_en': job['prompt_en'], 'caption_key': job['caption_key'], 'contract': contract,
                    'pre_connector': info, 'compact_validation': verification,
                    'valid_rows': packed['metadata']['valid_rows']}


def run_stage(args, plan, weights):
    import torch
    from ltx_trainer.model_loader import load_video_vae_encoder, load_text_encoder, load_embeddings_processor, embedding_weight_paths
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose one allocated GPU per encoding worker')
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    identity, identity_sha = plan['identity'], plan['identity_sha256']
    jobs = sorted(plan['index'][args.stage].values(), key=lambda x: x['key'])
    jobs = jobs[args.shard_index::args.num_shards]
    models = None
    progress = {'schema': 'datamake_encoding_progress_v2', 'stage': args.stage, 'status': 'running',
                'shard_index': args.shard_index, 'num_shards': args.num_shards, 'assigned': len(jobs),
                'completed': 0, 'reused': 0, 'identity_sha256': identity_sha, 'limit_new_items': args.limit,
                'started_unix': time.time()}
    progress_dir = args.output / 'progress'
    progress_dir.mkdir(exist_ok=True)
    progress_path = progress_dir / f'{args.stage}_{args.shard_index:04d}_of_{args.num_shards:04d}.json'
    atomic_json(progress_path, progress)
    print(json.dumps(progress), flush=True)
    try:
        with lock(args.output / f'.locks/{args.stage}_{args.shard_index}_of_{args.num_shards}.lock'), torch.inference_mode():
            for job in jobs:
                progress['current_key'] = job['key']
                with lock(args.output / '.locks/items' / f'{args.stage}_{job["key"]}.lock'):
                    cached = verify_receipt(args.output, args.stage, job['key'], identity_sha)
                    if cached is not None:
                        validate_item_semantics(cached, job, identity, args.stage)
                        progress['reused'] += 1
                        if progress['reused'] % 100 == 0:
                            progress['updated_unix'] = time.time()
                            atomic_json(progress_path, progress)
                            print(json.dumps(progress), flush=True)
                        continue
                    if args.limit is not None and progress['completed'] >= args.limit:
                        break
                    if models is None:
                        print(json.dumps({'event': 'loading_native_encoder', 'stage': args.stage}), flush=True)
                        if args.stage == 'media':
                            models = load_video_vae_encoder(weights['vae'], device='cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
                        else:
                            encoder = load_text_encoder(weights['text_encoders'], device='cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
                            processor = load_embeddings_processor(embedding_weight_paths(weights['diffusion_models'], weights['text_encoders']),
                                gemma_model_path=weights['text_encoders'], device='cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
                            processor.audio_connector = None
                            models = (encoder, processor)
                    started = time.monotonic()
                    if args.stage == 'media':
                        payload, details = encode_media(job, args, identity, models)
                    else:
                        payload, details = encode_text(job, identity, *models, verify_connector=progress['completed'] == 0)
                    details['seconds'] = time.monotonic() - started
                    details['gpu'] = torch.cuda.get_device_name(0)
                    validate_item_semantics(details, job, identity, args.stage)
                    save_item(args.output, args.stage, job['key'], identity_sha, payload, details)
                    del payload
                    progress['completed'] += 1
                    progress['updated_unix'] = time.time()
                    atomic_json(progress_path, progress)
                    print(json.dumps({**progress, 'item_seconds': details['seconds']}), flush=True)
            progress['status'] = 'complete' if progress['completed'] + progress['reused'] == len(jobs) else 'partial_limit'
    except BaseException as error:
        progress.update(status='failed', error=repr(error))
        raise
    finally:
        progress['finished_unix'] = time.time()
        atomic_json(progress_path, progress)
        del models
        gc.collect()
        torch.cuda.empty_cache()
    return progress


def finalize(args, plan):
    identity, identity_sha = plan['identity'], plan['identity_sha256']
    receipts = {'media': {}, 'text': {}}
    missing = {stage: 0 for stage in receipts}
    for stage in receipts:
        for key in plan['index'][stage]:
            receipt = verify_receipt(args.output, stage, key, identity_sha)
            if receipt is None:
                missing[stage] += 1
            else:
                validate_item_semantics(receipt, plan['index'][stage][key], identity, stage)
                receipts[stage][key] = receipt
    require(not any(missing.values()), f'Incomplete conversion; refusing passed metadata: {missing}')
    require(any(item['compact_validation'] == 'bitwise_equal_real_video_connector_and_mask' for item in receipts['text'].values()),
            'Final dataset lacks a real connector compact-cache equality verification')
    result = {'schema': 'datamake_native_encoding_v2', 'status': 'passed', 'encoding_identity': identity,
              'identity_sha256': identity_sha, 'manifest_sha256': identity['manifest_sha256'],
              'manifest_paths': identity['manifest_paths'], 'profile': identity['profile'], 'profile_sha256': identity['profile_sha256'],
              'weights': identity['weights'], 'seed': args.seed, 'torch_version': identity['encoder_contract']['torch_version'],
              'pixel_toolchain': identity['encoder_contract']['pixel_toolchain'],
              'encoder_sha256': identity['encoder_contract']['encoder_script_sha256'],
              'media': receipts['media'], 'text': receipts['text'], 'samples': plan['index']['samples']}
    result['manifests'] = {split: {'path': identity['manifest_paths'][split], 'sha256': sha,
                          'count': sum(row['split'] == split for row in result['samples'])}
                           for split, sha in identity['manifest_sha256'].items()}
    with lock(args.output / '.locks/finalize.lock'):
        atomic_json(args.output / 'metadata.json', result)
    return {'status': 'passed', 'media': len(receipts['media']), 'text': len(receipts['text']), 'samples': len(result['samples'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'media', 'text', 'finalize'), required=True)
    parser.add_argument('--train-manifest', type=Path, default=ROOT / 'outputs/data_prep/v1/train.jsonl')
    parser.add_argument('--test-manifest', type=Path, default=ROOT / 'outputs/data_prep/v1/test.jsonl')
    parser.add_argument('--dataset-root', type=Path, default=ROOT.parents[2] / '00_engine/outputs/dataset_package_v1/dataset_v1')
    parser.add_argument('--profile', type=Path, default=ROOT / 'configs/datamake_media_768_v2.json')
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/native_model_revision.json')
    parser.add_argument('--verified-weights', type=Path, default=ROOT / 'outputs/install_smoke/native_weights_verified.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/native_dataset/768_v2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, help='Maximum newly encoded items this invocation; completed caches are verified/reused')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    require(not site.ENABLE_USER_SITE and not os.environ.get('PYTHONPATH'), 'Use isolated trainer Python with no user-site/PYTHONPATH')
    require(args.num_shards > 0 and 0 <= args.shard_index < args.num_shards, 'Invalid shard assignment')
    require(args.limit is None or args.limit > 0, 'Limit must be positive')
    args.manifests = {'train': args.train_manifest, 'test': args.test_manifest}
    plan, weights = ensure_plan(args)
    if args.stage == 'plan':
        result = {'status': 'planned', 'samples': len(plan['index']['samples']),
                  'unique_media': len(plan['index']['media']), 'text': len(plan['index']['text']),
                  'identity_sha256': plan['identity_sha256'], 'scope': 'metadata-only; no original media decoded or models loaded'}
    elif args.stage == 'finalize':
        result = finalize(args, plan)
    else:
        result = run_stage(args, plan, weights)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
