"""CPU-only launch/identity contracts. No torch/model imports or cache encoding."""
from __future__ import annotations
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if (p / "common/project_layout.py").is_file()) / "common"))
from project_layout import ROOT as _PROJECT_ROOT, resolve_path, script_path, add_script_paths, resolve_cli_paths
add_script_paths()

import copy
import hashlib
import json
import math
import importlib.metadata
import platform
from pathlib import Path
import re
import subprocess
import yaml

ROOT = _PROJECT_ROOT
VIDEO_MODULES = [f"{a}.{p}" for a in ("attn1", "attn2") for p in ("to_q", "to_k", "to_v", "to_out.0")]
VIDEO_MODULES += ["ff.net.0.proj", "ff.net.2"]
PROTOCOL = "reference_positions_v1_candidate"
NATIVE_IMAGE_PROTOCOLS = {'reference_positions_v5_native_images', 'reference_positions_v6_native_images_sequential_time', 'reference_positions_v7_refs_source_target', 'reference_positions_v8_shared_time_refs_source_target'}
MODALITY = "native_compact_fp32_time_bf16_latent_all_visible_context_v1"
ADAPTER = "unfused_peft_bf16_autocast"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    h = hashlib.sha256()
    with resolve_path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def resolve_path(value):
    from project_layout import resolve_path as relocated_path
    return relocated_path(value)


def validate_config(config):
    required = {'schema', 'seed', 'lora', 'optimization', 'checkpoints', 'acceleration',
                'reference_protocol', 'data', 'wandb'}
    require(set(config) in (required, required | {'condition_dropout'}), 'Unknown/missing config fields')
    if 'condition_dropout' in config:
        drop = config['condition_dropout']
        require(isinstance(drop, dict) and set(drop) == {'visual_reference_probability'},
                'Unknown/missing condition dropout fields')
        probability = drop['visual_reference_probability']
        require(type(probability) in (int, float) and math.isfinite(probability) and 0 <= probability <= 1,
                'Visual reference dropout probability must be finite and in [0,1]')
    require(config['schema'] == 'datamake_native_lora_train_v1', 'Unknown training schema')
    require(type(config['seed']) is int and config['seed'] >= 0, 'Seed must be nonnegative integer')
    lo = config['lora']
    require(set(lo) == {'rank', 'alpha', 'dropout', 'target_modules'}, 'Unknown/missing LoRA fields')
    require(lo['rank'] == 64 and lo['alpha'] == 64 and lo['dropout'] == 0, 'This entry creates rank64/alpha64/dropout0 LoRA')
    require(len(lo['target_modules']) == len(VIDEO_MODULES) and set(lo['target_modules']) == set(VIDEO_MODULES),
            'Use exact video attention/FFN LoRA allowlist; generic to_q or audio modules forbidden')
    op = config['optimization']
    require(set(op) == {'steps', 'batch_size', 'gradient_accumulation_steps', 'learning_rate', 'optimizer_type',
                       'scheduler_type', 'max_grad_norm', 'enable_gradient_checkpointing'}, 'Unknown/missing optimization fields')
    for key in ('batch_size', 'gradient_accumulation_steps'):
        require(type(op[key]) is int and op[key] > 0, f'{key} must be positive integer')
    require(op['steps'] is None or (type(op['steps']) is int and op['steps'] > 0), 'Steps must be positive or derived from plan')
    require(op['batch_size'] in (1,4) and op['enable_gradient_checkpointing'] is True, 'Require batch1/4 and checkpointing')
    require(op['optimizer_type'] == 'adamw' and op['scheduler_type'] in {'constant', 'linear'},
            'Supported optimization uses AdamW and official constant/linear scheduling')
    require(math.isfinite(op['learning_rate']) and op['learning_rate'] > 0, 'Invalid learning rate')
    require(math.isfinite(op['max_grad_norm']) and op['max_grad_norm'] > 0, 'Require finite positive gradient clipping')
    ck = config['checkpoints']
    required_ck = {'interval', 'keep_last_n', 'precision', 'save_training_state'}
    require(set(ck) in (required_ck, required_ck | {'validation_steps'}), 'Unknown/missing checkpoint fields')
    selected = ck.get('validation_steps', [])
    require(isinstance(selected, list) and all(type(step) is int and step > 0 for step in selected)
            and selected == sorted(set(selected)), 'Validation checkpoint steps must be sorted distinct positive integers')
    require(type(ck['interval']) is int and ck['interval'] > 0 and type(ck['keep_last_n']) is int and ck['keep_last_n'] > 0,
            'Require positive checkpoint interval/retention')
    require(ck['precision'] == 'float32' and ck['save_training_state'] == 'full', 'Full FP32 adapter/optimizer resume required')
    require(config['acceleration'] == {'mixed_precision_mode': 'bf16', 'quantization': None}, 'Require unquantized BF16 AMP')
    require(config['reference_protocol'] in (PROTOCOL, 'reference_positions_v2_pending', 'reference_positions_v2',
                                           'reference_positions_v3_shared_canvas', 'reference_positions_v4_sequential_time', *NATIVE_IMAGE_PROTOCOLS), 'Unknown reference coordinate protocol')
    data = config['data']
    require(set(data) in ({'backend', 'media_profile', 'manifest', 'encoding_roots', 'sample_ids', 'order', 'num_workers'},
                          {'backend', 'media_profile', 'manifest', 'encoding_roots', 'sample_ids', 'order', 'num_workers', 'sources_config'}), 'Unknown/missing data fields')
    require(data['backend'] in {'pending', 'validated_probe_cache', 'validated_manifest_cache', 'validated_mixed_cache'}, 'Unknown dataset backend')
    require(data['order'] in {'sequential_repeat_v1', 'deterministic_shuffle_v1', 'finite_grouped_mixture_v1'} and data['num_workers'] == 0,
            'Use supported deterministic order and workers0')
    if data['backend'] == 'validated_mixed_cache':
        require(isinstance(data.get('sources_config'), str) and data['sources_config'], 'Mixed source config required')
        require(data['order'] == 'finite_grouped_mixture_v1' and data['sample_ids'] == [], 'Mixed training requires full finite sampling plan')
        require(op['batch_size'] == 4 and op['gradient_accumulation_steps'] == 4, 'Approved mixed recipe requires batch4/accumulation4')
        require(config['reference_protocol'] in {'reference_positions_v2', 'reference_positions_v3_shared_canvas', 'reference_positions_v4_sequential_time', *NATIVE_IMAGE_PROTOCOLS},
                'Mixed training requires a versioned reference protocol')
    else:
        require(isinstance(data['media_profile'], str) and data['media_profile'], 'Explicit media profile path required')
        require(op['steps'] is not None, 'Legacy entry requires explicit steps')
    if data['backend'] == 'validated_probe_cache':
        require(data['order'] == 'sequential_repeat_v1', 'Historical probes retain explicit sequential order')
        require(config['reference_protocol'] == PROTOCOL, 'Historical probe backend requires explicit v1 protocol; 768 v2 not implemented yet')
        ids = data['sample_ids']
        require(1 <= len(ids) <= 4 and all(type(i) is int for i in ids) and len(set(ids)) == len(ids), 'Probe backend requires 1..4 distinct explicit sample IDs')
        require(data['manifest'] and data['encoding_roots'], 'Probe backend requires manifest and encoding roots')
    if data['backend'] == 'validated_manifest_cache':
        require(data['order'] == 'deterministic_shuffle_v1', 'Complete manifest training requires deterministic shuffle')
        require(config['reference_protocol'] in {'reference_positions_v2', 'reference_positions_v3_shared_canvas', 'reference_positions_v4_sequential_time', *NATIVE_IMAGE_PROTOCOLS},
                'Complete manifest backend requires a versioned reference protocol')
        require(data['sample_ids'] == [], 'Complete manifest backend forbids sample selection')
        require(data['manifest'] and len(data['encoding_roots']) == 1, 'Complete manifest backend requires manifest and one finalized encoding root')
    wb = config['wandb']
    require(set(wb) == {'enabled', 'project', 'entity', 'tags'} and type(wb['enabled']) is bool,
            'Unknown/missing W&B fields')
    return config


def load_config(path):
    return validate_config(yaml.safe_load(resolve_path(path).read_text()))


def build_identity(config, world_size):
    require(type(world_size) is int and world_size > 0, 'Invalid world size')
    data = config['data']
    base = copy.deepcopy(config)
    # Segment stops/save frequency may change; mixed full duration is locked by
    # sampling_plan_sha256. A non-mixed linear horizon cannot be extended.
    scheduler_horizon = (base['optimization']['steps']
                         if data['backend'] != 'validated_mixed_cache' and base['optimization']['scheduler_type'] == 'linear'
                         else None)
    base['optimization'].pop('steps')
    base.pop('checkpoints')
    base.pop('wandb')
    records = {}
    for path in ('models/native_model_revision.json', 'outputs/install_smoke/native_weights_verified.json'):
        p = ROOT / path
        records[path] = sha_file(p)
    metadata = []
    if data['backend'] == 'validated_mixed_cache':
        sources_path = resolve_path(data['sources_config'])
        sources = json.loads(sources_path.read_text())
        records['sources_config'] = sha_file(sources_path)
        for source in sources['sources']:
            for name in ('train_manifest', 'development_manifest', 'media_profile'):
                records[f"{source['name']}/{name}"] = sha_file(resolve_path(source[name]))
            p = resolve_path(source['completion_metadata'])
            metadata.append({'path': str(p), 'sha256': sha_file(p)})
    elif data['backend'] != 'pending':
        records['dataset_manifest'] = sha_file(resolve_path(data['manifest']))
        records['media_profile'] = sha_file(resolve_path(data['media_profile']))
        metadata = [{'path': str(resolve_path(p) / 'metadata.json'),
                     'sha256': sha_file(resolve_path(p) / 'metadata.json')} for p in data['encoding_roots']]
    scripts = ('train_native_lora.py', 'native_training_contract.py', 'native_training_runtime.py',
               'native_reference_strategy.py', 'reference_positions.py', 'native_cached_dataset.py',
               'compact_text_cache.py', 'native_grouped_sampling.py', 'contact_sheet_contract.py')
    if config['reference_protocol'] in {'reference_positions_v3_shared_canvas', 'reference_positions_v4_sequential_time'}:
        scripts += ('reference_canvas_contract.py', 'migrate_reference_canvas.py')
    if config['reference_protocol'] in NATIVE_IMAGE_PROTOCOLS:
        scripts += ('native_image_cache_contract.py',)
    upstream = ('ltx-trainer/src/ltx_trainer/trainer.py', 'ltx-trainer/src/ltx_trainer/training_state.py',
                'ltx-trainer/src/ltx_trainer/training_strategies/flexible.py',
                'ltx-core/src/ltx_core/model/transformer/model.py', 'ltx-core/src/ltx_core/model/transformer/transformer.py',
                'ltx-core/src/ltx_core/model/transformer/modality.py', 'ltx-core/src/ltx_core/types.py',
                'ltx-pipelines/src/ltx_pipelines/utils/helpers.py')
    upstream_paths = [resolve_path(ROOT / 'repos/LTX-2/packages' / p) for p in upstream]
    require(all(p.is_file() for p in upstream_paths), 'Pinned upstream source files missing')
    import project_layout
    upstream_revision = getattr(project_layout, 'FROZEN_UPSTREAM_REVISION', None)
    if upstream_revision is None:
        upstream_revision = subprocess.check_output(['git', '-C', str(resolve_path('runtime/repos/LTX-2')), 'rev-parse', 'HEAD'], text=True).strip()
    dependencies = {name: importlib.metadata.version(name) for name in ('torch', 'accelerate', 'peft', 'transformers', 'safetensors')}
    return {'schema': 'datamake_native_training_identity_v2', 'config': base, 'world_size': world_size,
            'scheduler_step_contract': 'one_step_per_global_optimizer_update_v1',
            'nonmixed_scheduler_horizon_steps': scheduler_horizon,
            'record_sha256': records, 'encoding_metadata': metadata,
            'upstream_sha256': {str(p.relative_to(ROOT)): sha_file(p) for p in upstream_paths},
            'dependencies': dependencies, 'python': platform.python_version(), 'upstream_revision': upstream_revision,
            'code_sha256': {n: sha_file(script_path(n)) for n in scripts},
            'modality_contract': MODALITY, 'adapter_compute_contract': ADAPTER,
            'resume_data_contract': ('finite_grouped_global_microbatch_v1' if data['order'] == 'finite_grouped_mixture_v1' else 'global_cursor_epoch_seeded_permutation_v1' if data['order'] == 'deterministic_shuffle_v1'
                                     else 'global_microstep_sequential_modulo_dataset_v1')}


def resume_paths(checkpoint):
    checkpoint = resolve_path(checkpoint).resolve()
    match = re.fullmatch(r'lora_weights_step_(\d+)\.safetensors', checkpoint.name)
    require(match is not None and checkpoint.is_file(), 'Resume must name an existing exact LoRA safetensors file')
    step = match.group(1)
    return checkpoint, checkpoint.with_name(f'training_state_step_{step}.pt'), checkpoint.with_suffix('.contract.json'), int(step)


def validate_resume(checkpoint, identity, total_steps):
    checkpoint, state_path, receipt_path, step = resume_paths(checkpoint)
    require(receipt_path.is_file(), 'Missing atomic checkpoint completion receipt; do not resume partial/legacy checkpoint')
    receipt = json.loads(receipt_path.read_text())
    require(receipt['identity'] == identity, 'Resume model/data/config/code/world identity mismatch')
    require(receipt['step'] == step and 0 < step < total_steps, 'Resume step invalid or no remaining steps')
    if identity['resume_data_contract'] == 'finite_grouped_global_microbatch_v1':
        require(receipt.get('next_global_microbatch') == step * identity['config']['optimization']['gradient_accumulation_steps']
                and receipt.get('sampling_plan_sha256') == identity['sampling_plan_sha256'], 'Grouped resume cursor/plan mismatch')
    else:
        require(receipt['next_global_microstep'] == step * identity['config']['optimization']['gradient_accumulation_steps'] * identity['world_size'], 'Invalid resume data cursor')
    expected = {checkpoint.name, state_path.name} | {f'rng_step_{step:05d}_rank_{r:04d}.pt' for r in range(identity['world_size'])}
    require(set(receipt['files']) == expected, 'Incomplete checkpoint receipt file set')
    for name, evidence in receipt['files'].items():
        p = checkpoint.parent / name
        require(p.is_file() and p.stat().st_size == evidence['bytes'] and sha_file(p) == evidence['sha256'], 'Checkpoint file missing/changed: ' + name)
    return receipt


def global_indices(length, start, count):
    require(length > 0 and start >= 0 and count > 0, 'Invalid deterministic sampler extent')
    return ((start + i) % length for i in range(count))


def checkpoint_lineage(checkpoint, identity):
    """Only the explicit resume chain, never all matching runs in a directory."""
    checkpoint = resolve_path(checkpoint).resolve()
    receipt = validate_resume(checkpoint, identity, 2 ** 63)
    paths = [resolve_path(p) for p in receipt.get('retention_lineage', [str(checkpoint)])]
    require(paths and paths[-1] == checkpoint and len(paths) == len(set(paths)), 'Invalid explicit retention lineage')
    previous = -1
    verified = []
    for path in paths:
        require(path.is_relative_to(ROOT / 'experiments'), 'Checkpoint lineage escapes experiment outputs')
        match = re.fullmatch(r'lora_weights_step_(\d+)\.safetensors', path.name)
        require(match is not None and int(match.group(1)) > previous, 'Lineage steps must increase')
        previous = int(match.group(1))
        # An older retained checkpoint can refer to ancestors since retired by
        # a later committed child. Never reconstruct or scan for replacements.
        sidecar = path.with_suffix('.contract.json')
        if not path.exists() and not sidecar.exists():
            continue
        validate_resume(path, identity, 2 ** 63)
        verified.append(path)
    return verified


def remove_checkpoint_bundle(checkpoint, identity):
    checkpoint = resolve_path(checkpoint).resolve()
    require(checkpoint.is_relative_to(ROOT / 'experiments'), 'Cannot clean checkpoints outside experiment outputs')
    receipt = validate_resume(checkpoint, identity, 2 ** 63)
    # Revoke completion first; only named, SHA-verified files from this bundle.
    checkpoint.with_suffix('.contract.json').unlink()
    for name in receipt['files']:
        (checkpoint.parent / name).unlink()


def validate_media_protocol(runtime_protocol, cache_protocol):
    """V4 changes coordinates only; its immutable VAE/text inputs are V3 caches."""
    expected = {'reference_positions_v4_sequential_time': 'reference_positions_v3_shared_canvas',
                'reference_positions_v6_native_images_sequential_time': 'reference_positions_v5_native_images',
                'reference_positions_v7_refs_source_target': 'reference_positions_v5_native_images',
                'reference_positions_v8_shared_time_refs_source_target': 'reference_positions_v5_native_images'}.get(runtime_protocol, runtime_protocol)
    require(cache_protocol == expected, 'Runtime protocol and encoded media contract differ')
