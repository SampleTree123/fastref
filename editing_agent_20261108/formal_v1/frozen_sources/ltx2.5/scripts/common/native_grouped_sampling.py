"""Finite, fully identified mixed-source sampling; one real local batch per yield."""
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if (p / "common/project_layout.py").is_file()) / "common"))
from project_layout import ROOT as _PROJECT_ROOT, resolve_path, script_path, add_script_paths, resolve_cli_paths
add_script_paths()

from collections import Counter, defaultdict
import hashlib
import json
import math

import torch
from torch.utils.data import Sampler

SCHEMA = 'datamake_finite_grouped_mixture_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_sampling_plan(records, *, batch_size, world_size, accumulation, seed):
    """records are dataset-index ordered, with uid/source/repeat/group/tokens.

    Repeat counts are requested exposures, not extra loss weights. Padding is
    explicit in the resulting counts. A local batch never combines groups.
    """
    if not records or min(batch_size, world_size, accumulation) < 1:
        raise ValueError('Empty data or invalid batch/world/accumulation')
    if len({r['uid'] for r in records}) != len(records):
        raise ValueError('Duplicate source-qualified sample UID')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    groups = defaultdict(list)
    requested = Counter()
    for i, item in enumerate(records):
        if type(item['repeat']) is not int or item['repeat'] < 1 or item['tokens'] < 1:
            raise ValueError('Invalid repeat or token length')
        groups[digest(item['group'])].extend([i] * item['repeat'])
        requested[item['source']] += item['repeat']
    batches, group_padding = [], Counter()
    group_info = []
    for key, indices in sorted(groups.items()):
        permutation = torch.randperm(len(indices), generator=generator, device='cpu').tolist()
        ordered = [indices[p] for p in permutation]
        extra = (-len(ordered)) % batch_size
        padded = ordered + [ordered[k % len(ordered)] for k in range(extra)]
        group_padding.update(records[i]['source'] for i in padded[len(ordered):])
        group_info.append({'key': key, 'requested': len(indices), 'padded': extra,
                           'tokens': records[ordered[0]]['tokens']})
        batches.extend(padded[j:j + batch_size] for j in range(0, len(padded), batch_size))
    # Complete global optimizer windows: every rank gets exactly A full batches.
    multiple = world_size * accumulation
    needed = (-len(batches)) % multiple
    perm = torch.randperm(len(batches), generator=generator, device='cpu').tolist()
    extra_batches = [list(batches[perm[k % len(perm)]]) for k in range(needed)]
    window_padding = Counter(records[i]['source'] for batch in extra_batches for i in batch)
    batches.extend(extra_batches)
    # Similar token costs on the same DDP microstep minimize stragglers. Shuffle
    # complete global rows afterwards, so short/long sources interleave in time.
    tie_order = torch.randperm(len(batches), generator=generator, device='cpu').tolist()
    batches = [batches[i] for i in tie_order]
    batches.sort(key=lambda b: records[b[0]]['tokens'])
    rows = [batches[i:i + world_size] for i in range(0, len(batches), world_size)]
    for row in rows:
        order = torch.randperm(world_size, generator=generator, device='cpu').tolist()
        row[:] = [row[i] for i in order]
    order = torch.randperm(len(rows), generator=generator, device='cpu').tolist()
    batches = [batch for i in order for batch in rows[i]]
    appearances = Counter(i for batch in batches for i in batch)
    actual = Counter(records[i]['source'] for batch in batches for i in batch)
    if any(appearances[i] < item['repeat'] for i, item in enumerate(records)):
        raise AssertionError('Sampling silently dropped a requested occurrence')
    plan = {'schema': SCHEMA, 'batch_size': batch_size, 'world_size': world_size,
            'accumulation': accumulation, 'seed': seed, 'records_sha256': digest(records),
            'sample_uids': [r['uid'] for r in records], 'batches': batches,
            'global_microbatches': len(batches) // world_size,
            'optimizer_steps': len(batches) // multiple,
            'source_requested': dict(requested), 'source_actual': dict(actual),
            'group_padding': dict(group_padding), 'window_padding': dict(window_padding),
            'sample_extra_occurrences': {records[i]['uid']: appearances[i] - r['repeat']
                                         for i, r in enumerate(records) if appearances[i] != r['repeat']},
            'groups': group_info,
            'scheduling': 'seeded_group_shuffle_cost_sorted_global_rows_shuffled_v1'}
    plan['sha256'] = digest(plan)
    return plan


class PlannedBatchSampler(Sampler):
    """Accelerator shards these complete local batches exactly once by rank."""
    def __init__(self, plan, *, start_global_microbatch=0, stop_global_microbatch=None):
        unsigned = {k: v for k, v in plan.items() if k != 'sha256'}
        if digest(unsigned) != plan['sha256'] or plan['schema'] != SCHEMA:
            raise ValueError('Sampling plan signature mismatch')
        total = plan['global_microbatches']
        stop = total if stop_global_microbatch is None else stop_global_microbatch
        if not 0 <= start_global_microbatch < stop <= total:
            raise ValueError('Invalid or exhausted sampling cursor')
        self.plan = plan
        self.batch_size = plan['batch_size']
        self.drop_last = False
        self.start = start_global_microbatch * plan['world_size']
        self.stop = stop * plan['world_size']

    def __iter__(self):
        yield from self.plan['batches'][self.start:self.stop]

    def __len__(self):
        return self.stop - self.start


def sample_structure(sample):
    """Unbatched tensor structure; used by collation as an independent guard."""
    def latent(data):
        return (tuple(data['latents'].shape), data['fps'])
    def reference(ref):
        structure = (ref['type'], ref['view'], ref['index'], latent(ref['latent']))
        return (*structure, ref['reference_representation']) if ref.get('reference_representation') else structure
    return (latent(sample['video_latents']), latent(sample['source']),
            tuple(reference(r) for r in sample['references']),
            tuple(sample['conditions']['video_prompt_embeds'].shape),
            tuple(sample['conditions']['prompt_attention_mask'].shape))
