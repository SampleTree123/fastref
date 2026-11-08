"""Thin native trainer extension. Imported only for an actual GPU launch."""
from __future__ import annotations
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if (p / "common/project_layout.py").is_file()) / "common"))
from project_layout import ROOT as _PROJECT_ROOT, resolve_path, script_path, add_script_paths, resolve_cli_paths
add_script_paths()

import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from accelerate.utils import DistributedType, set_seed
from accelerate.scheduler import AcceleratedScheduler
from torch.utils.data import DataLoader, Sampler
from ltx_trainer.config import LtxTrainerConfig
import ltx_trainer.trainer as official_trainer
from ltx_trainer.trainer import LtxvTrainer
# Upstream uses LOCAL_RANK for process-global writes; enforce global rank here.
official_trainer.IS_MAIN_PROCESS = int(os.environ.get('RANK', '0')) == 0
from safetensors.torch import load_file
from peft import get_peft_model_state_dict

from native_cached_dataset import collate_single_sample, collate_reference_batch
from native_grouped_sampling import PlannedBatchSampler
from native_reference_strategy import NativeReferenceStrategy, install_grouped_video_forward
from native_training_contract import atomic_json, fingerprint, global_indices, require, sha_file, VIDEO_MODULES, checkpoint_lineage, remove_checkpoint_bundle


class ManifestTrainerConfig(LtxTrainerConfig):
    def _validate_data_dirs_exist(self):
        # Explicit external dataset validates every named sample/artifact. The
        # stock fixed-folder intersection would silently discard missing refs.
        pass


class GlobalCursorSampler(Sampler):
    """Global deterministic order; Accelerator shards batches across DDP ranks."""
    def __init__(self, length, start, count, *, order='sequential_repeat_v1', seed=0):
        self.length, self.start, self.count = length, start, count
        require(order in {'sequential_repeat_v1', 'deterministic_shuffle_v1'}, 'Unknown global sample order')
        self.order, self.seed = order, seed

    def __iter__(self):
        if self.order == 'sequential_repeat_v1':
            return global_indices(self.length, self.start, self.count)
        return self._shuffled()

    def _shuffled(self):
        require(self.length > 0 and self.start >= 0 and self.count > 0, 'Invalid shuffled sampler extent')
        cursor, remaining = self.start, self.count
        while remaining:
            epoch, offset = divmod(cursor, self.length)
            generator = torch.Generator(device='cpu').manual_seed((self.seed + epoch) % (2 ** 63))
            order = torch.randperm(self.length, generator=generator, device='cpu').tolist()
            take = min(remaining, self.length - offset)
            yield from order[offset:offset + take]
            cursor += take
            remaining -= take

    def __len__(self):
        return self.count


def same_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        return isinstance(actual, torch.Tensor) and torch.equal(actual.detach().cpu(), expected.detach().cpu())
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(same_tree(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(same_tree(a, b) for a, b in zip(actual, expected))
    return actual == expected


class NativeLoraTrainer(LtxvTrainer):
    def __init__(self, config, *, dataset, identity, receipt=None, sampling_plan=None, validation_steps=()):
        self.external_dataset = dataset
        self.launch_identity = identity
        self.resume_receipt = receipt
        self.sampling_plan = sampling_plan
        self._validation_steps = frozenset(validation_steps)
        self._microsteps_seen = 0
        self._retention_paths = checkpoint_lineage(config.model.load_checkpoint, identity) if receipt is not None else []
        super().__init__(config)

    def train(self, disable_progress_bars=False, step_callback=None):
        # BF16 autocast handles Linear input/weight conversion. PEFT's explicit
        # FP32 input copy is redundant; adapter weights and gradients stay FP32.
        # Cover the whole loop because checkpoint recomputation runs in backward,
        # after _training_step returns. The official helper restores flags on exit.
        from peft.helpers import disable_input_dtype_casting
        require(self._config.acceleration.mixed_precision_mode == 'bf16',
                'Skipping PEFT input copies requires the approved BF16 autocast path')
        callback = self._validation_save_callback(step_callback)
        with disable_input_dtype_casting(self._transformer):
            return super().train(disable_progress_bars=disable_progress_bars, step_callback=callback)

    def _validation_save_callback(self, next_callback=None):
        # Official callback runs on every rank after a complete optimizer update.
        # It adds non-periodic saves (e.g. 256) without exiting the training loop.
        def after_update(step, total_steps, sampled_videos_paths):
            if step in self._validation_steps and getattr(self, '_last_committed_step', None) != step:
                self._save_checkpoint()
            if next_callback is not None:
                next_callback(step, total_steps, sampled_videos_paths)
        return after_update

    def _load_models(self):
        super()._load_models()
        self._embeddings_processor.eval().requires_grad_(False)
        self._embeddings_processor.audio_connector = None
        strategy = NativeReferenceStrategy(
            self._config.training_strategy, protocol_version=self.launch_identity['config']['reference_protocol'],
            visual_reference_probability=self.launch_identity['config'].get('condition_dropout', {}).get('visual_reference_probability', 0.0))
        strategy.video_scale_factors = self._training_strategy.video_scale_factors
        require(strategy.get_checkpoint_metadata()['datamake_modality_contract'] == self.launch_identity['modality_contract'],
                'Runtime strategy contract changed')
        self._training_strategy = strategy
        if strategy.visual_reference_probability > 0:
            install_grouped_video_forward(self._transformer)

    def _setup_accelerator(self):
        super()._setup_accelerator()
        require(self._accelerator.distributed_type in (DistributedType.NO, DistributedType.MULTI_GPU),
                'This entry supports single CUDA/DDP only; FSDP cannot silently downgrade full state')
        require(self._accelerator.device.type == 'cuda', 'Actual training requires an allocated CUDA GPU')
        require(self._accelerator.num_processes == self.launch_identity['world_size'], 'Launcher world size mismatch')
        plugin = self._accelerator.state.dynamo_plugin
        require(str(plugin.backend).upper().split('.')[-1] == 'NO', 'Initial contract disables compile')

    def _collect_trainable_params(self):
        super()._collect_trainable_params()
        names = [n for n, p in self._transformer.named_parameters() if p.requires_grad]
        def allowed(name):
            return any(name.endswith(f'.{module}.lora_{part}.default.weight')
                       for module in VIDEO_MODULES for part in ('A', 'B')) and 'audio' not in name
        require(names and all(allowed(n) for n in names), 'Trainable parameters escape precise video LoRA allowlist')
        require(all(p.dtype == torch.float32 for p in self._trainable_params), 'Expected FP32 trainable PEFT adapters')
        require(all(not p.requires_grad for p in self._embeddings_processor.parameters()), 'Connector must stay frozen')
        self.trainable_parameter_count = sum(p.numel() for p in self._trainable_params)
        # Deliberately no parameter gradient hooks or per-tensor CUDA reductions.

    def _load_checkpoint(self):
        super()._load_checkpoint()
        step, state = self._resume_state
        if self.resume_receipt is None:
            require(step == 0 and state is None and self._config.model.load_checkpoint is None, 'Fresh run unexpectedly restored weights')
            return
        require(state is not None and step == self.resume_receipt['step'] and state.optimizer_state_dict is not None,
                'Explicit full resume could not resolve optimizer/step')
        self._check_resume_scheduler(state)
        # Check every adapter key, not just the keys that happened to be saved.
        saved = {k.removeprefix('diffusion_model.'): v for k, v in load_file(str(self._loaded_checkpoint_path)).items()}
        live = {k.removeprefix('base_model.model.'): v for k, v in get_peft_model_state_dict(self._transformer).items()}
        require(same_tree(live, saved), 'Full adapter key/tensor readback mismatch')

    def _check_resume_scheduler(self, state):
        # Official constant/None schedules deliberately have no scheduler object;
        # TrainingState.to_save_dict consequently omits their None state field.
        expected = self._config.optimization.scheduler_type not in (None, 'constant')
        require((state.lr_scheduler_state_dict is not None) == expected,
                'Full resume scheduler state does not match configured scheduler')
        if self._config.optimization.scheduler_type == 'linear':
            require(state.lr_scheduler_state_dict.get('total_iters') == self._scheduler_horizon()
                    and state.lr_scheduler_state_dict.get('last_epoch') == state.global_step,
                    'Linear scheduler horizon/update cursor differs from the complete sampling plan')
        return expected

    def _scheduler_horizon(self):
        plan = getattr(self, 'sampling_plan', None)
        horizon = plan['optimizer_steps'] if plan is not None else self._config.optimization.steps
        require(type(horizon) is int and 0 < self._config.optimization.steps <= horizon,
                'Scheduler horizon must cover this segment and the complete finite plan')
        return horizon

    def _create_scheduler(self, optimizer):
        # Official creation uses optimization.steps. Give it the complete plan,
        # while preserving the original segment stop for the training loop.
        original = self._config
        scheduler_config = original.model_copy(deep=True)
        scheduler_config.optimization.steps = self._scheduler_horizon()
        try:
            self._config = scheduler_config
            return super()._create_scheduler(optimizer)
        finally:
            self._config = original

    def _init_optimizer(self):
        super()._init_optimizer()
        if self._lr_scheduler is not None:
            require(isinstance(self._lr_scheduler, AcceleratedScheduler)
                    and self._lr_scheduler.step_with_optimizer,
                    'Expected optimizer-linked Accelerate scheduler')
            # Our horizon already counts global optimizer updates. Accelerate's
            # default split_batches=False would advance it world_size times.
            # Change ONLY scheduler counting; data batches remain unsplit.
            self._lr_scheduler.split_batches = True

    def _restore_training_state(self, state):
        require(state.optimizer_state_dict is not None, 'Full resume optimizer state incomplete')
        expected_scheduler = self._check_resume_scheduler(state)
        require((self._lr_scheduler is not None) == expected_scheduler,
                'Live scheduler does not match configured scheduler')
        require(super()._restore_training_state(state), 'Official optimizer/scheduler restore failed; refusing step0 fallback')
        require(same_tree(self._optimizer.state_dict(), state.optimizer_state_dict), 'Restored optimizer state differs')
        restored_scheduler = self._lr_scheduler.state_dict() if self._lr_scheduler is not None else None
        require(same_tree(restored_scheduler, state.lr_scheduler_state_dict), 'Restored scheduler state differs')
        rank = self._accelerator.process_index
        path = self._loaded_checkpoint_path.parent / f'rng_step_{state.global_step:05d}_rank_{rank:04d}.pt'
        # Receipt SHA was checked before any torch.load of this trusted local state.
        rng = torch.load(path, map_location='cpu', weights_only=False)
        require(rng['step'] == state.global_step and rng['rank'] == rank, 'Rank RNG identity mismatch')
        self._pending_rank_rng = rng
        self._restore_rank_rng(rng)
        self.full_restore_checked = True
        return True

    def _restore_rank_rng(self, rng):
        torch.random.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state(rng['cuda'], self._accelerator.device)
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])

    def _init_timestep_sampler(self):
        super()._init_timestep_sampler()
        # Official train() initializes W&B and the loader after state restore.
        # Restore again after those startup operations, immediately before the loop.
        if hasattr(self, '_pending_rank_rng'):
            self._restore_rank_rng(self._pending_rank_rng)
            del self._pending_rank_rng
        else:
            # Adapters were identically initialized and DDP-broadcast already;
            # draw independent sigma/noise streams for different data on each rank.
            set_seed(self._config.seed, device_specific=True)

    def _prepare_models_for_training(self):
        super()._prepare_models_for_training()
        model = self._accelerator.unwrap_model(self._transformer)
        base = model.get_base_model() if hasattr(model, 'get_base_model') else model
        require(getattr(base, '_enable_gradient_checkpointing', False) is True, 'Gradient checkpointing was not actually enabled')
        require(all(not p.requires_grad for p in self._embeddings_processor.parameters()), 'Connector unexpectedly trainable')

    def _init_dataloader(self):
        world = self._accelerator.num_processes
        initial_step = self._resume_state[0]
        accum = self._config.optimization.gradient_accumulation_steps
        generator = torch.Generator(device='cpu').manual_seed(self._config.seed)
        if self.sampling_plan is not None:
            require(self._accelerator.split_batches is False, 'Do not split a B4 local batch across ranks')
            require(self._accelerator.dispatch_batches in (None, False), 'Each rank must load its own full local batch')
            batch_sampler = PlannedBatchSampler(self.sampling_plan, start_global_microbatch=initial_step * accum,
                                               stop_global_microbatch=self._config.optimization.steps * accum)
            loader = DataLoader(self.external_dataset, batch_sampler=batch_sampler, num_workers=0,
                                collate_fn=collate_reference_batch, generator=generator)
        else:
            require(self._config.optimization.batch_size == 1, 'Legacy non-grouped loader is batch1 only')
            sampler = GlobalCursorSampler(len(self.external_dataset), initial_step * accum * world,
                                          (self._config.optimization.steps - initial_step) * accum * world,
                                          order=self.launch_identity['config']['data']['order'], seed=self._config.seed)
            loader = DataLoader(self.external_dataset, batch_size=1, sampler=sampler, num_workers=0,
                                collate_fn=collate_single_sample, generator=generator)
        self._dataloader = self._accelerator.prepare(loader)
        require(len(self._dataloader) == (self._config.optimization.steps - initial_step) * accum,
                'Unexpected Accelerator sharding; cursor contract invalid')

    def _training_step(self, batch):
        self._microsteps_seen += 1
        expected_sync = self._microsteps_seen % self._config.optimization.gradient_accumulation_steps == 0
        require(self._accelerator.sync_gradients == expected_sync,
                'Gradient accumulation synchronized outside its complete window')
        require(batch['video_latents']['latents'].shape[0] == self._config.optimization.batch_size,
                'Actual microbatch differs from configured batch size')
        if self.sampling_plan is not None:
            row = self._resume_state[0] * self._config.optimization.gradient_accumulation_steps + self._microsteps_seen - 1
            slot = row * self._accelerator.num_processes + self._accelerator.process_index
            expected = [self.sampling_plan['sample_uids'][i] for i in self.sampling_plan['batches'][slot]]
            require(list(batch['sample_uid']) == expected, 'Actual batch differs from identified sampling plan')
        output = super()._training_step(batch)
        require(output.loss.shape == (self._config.optimization.batch_size,), 'Loss must be one target average per sample')
        require(bool(torch.isfinite(output.loss).all()), 'Non-finite training loss')
        # Detached metrics only: optimize the unchanged per-sample mean above.
        values = output.loss.detach().to(torch.float64)
        track_reference_dropout = self._training_strategy.visual_reference_probability > 0
        totals = torch.zeros(8 if track_reference_dropout else 6, device=values.device, dtype=torch.float64)
        totals[0], totals[1] = values.sum(), values.numel()
        if track_reference_dropout:
            totals[6] = sum(self._training_strategy.last_reference_drop_mask)
            totals[7] = values.numel() if batch['references'] else 0
        names = batch.get('source_name', ['dataset'] * values.numel())
        for offset, source in ((2, 'ue'), (4, 'coinve')):
            mask = torch.tensor([name == source for name in names], device=values.device)
            totals[offset], totals[offset + 1] = values[mask].sum(), mask.sum()
        self._window_totals = getattr(self, '_window_totals', torch.zeros_like(totals)) + totals
        if expected_sync:
            global_totals = self._accelerator.reduce(self._window_totals, reduction='sum').cpu().tolist()
            self._window_totals.zero_()
            self._pending_window_metrics = {'train/global_window_loss': global_totals[0] / global_totals[1],
                                            'train/global_window_samples': global_totals[1]}
            if track_reference_dropout:
                self._pending_window_metrics['train/visual_reference_dropped_samples'] = global_totals[6]
                self._pending_window_metrics['train/visual_reference_eligible_samples'] = global_totals[7]
                if global_totals[7]:
                    self._pending_window_metrics['train/visual_reference_drop_fraction'] = global_totals[6] / global_totals[7]
            for offset, source in ((2, 'ue'), (4, 'coinve')):
                count = global_totals[offset + 1]
                self._pending_window_metrics[f'train/{source}_window_samples'] = count
                if count:
                    self._pending_window_metrics[f'train/{source}_window_loss'] = global_totals[offset] / count
        return output

    def _log_metrics(self, metrics):
        if 'train/global_step' in metrics and hasattr(self, '_pending_window_metrics'):
            metrics = {**metrics, **self._pending_window_metrics}
            del self._pending_window_metrics
        return super()._log_metrics(metrics)

    def _init_wandb(self, resume_run_id=None):
        if self._config.wandb.enabled:
            explicit = os.environ.get('WANDB_RUN_ID')
            if self.resume_receipt is not None:
                require(os.environ.get('WANDB_MODE', 'online') != 'offline', 'Strict W&B run resume requires online or disable W&B')
                require(resume_run_id and (explicit is None or explicit == resume_run_id), 'W&B resume ID missing/mismatched')
            else:
                require(os.environ.get('WANDB_RESUME', 'never') == 'never', 'Fresh training must not reuse a W&B run')
        super()._init_wandb(resume_run_id=resume_run_id)

    def _build_checkpoint_metadata(self):
        metadata = super()._build_checkpoint_metadata()
        metadata['datamake_training_identity_sha256'] = fingerprint(self.launch_identity)
        metadata['datamake_scheduler_horizon_steps'] = str(self._scheduler_horizon())
        metadata['datamake_scheduler_step_contract'] = 'one_step_per_global_optimizer_update_v1'
        return metadata

    def _cleanup_checkpoints(self):
        # Official save calls this before the state/receipt commit. Defer deletion.
        pass

    def _cleanup_training_states(self):
        pass

    def _save_checkpoint(self):
        directory = Path(self._config.output_dir) / 'checkpoints'
        directory.mkdir(parents=True, exist_ok=True)
        step = self._global_step
        checkpoint = directory / f'lora_weights_step_{step:05d}.safetensors'
        receipt_path = checkpoint.with_suffix('.contract.json')
        if getattr(self, '_last_committed_step', None) == step:
            # Final save may repeat the just-saved optimizer boundary; do not
            # overwrite the only complete bundle (especially keep_last_n=1).
            self._accelerator.wait_for_everyone()
            return checkpoint if self._accelerator.is_main_process else None
        if self._accelerator.is_main_process:
            receipt_path.unlink(missing_ok=True)  # No complete receipt during overwrite.
        # Avoid official duplicate-final-save retention deleting its current file.
        self._checkpoint_paths = [p for p in self._checkpoint_paths if p != checkpoint]
        state_path = directory / f'training_state_step_{step:05d}.pt'
        self._training_state_paths = [p for p in self._training_state_paths if p != state_path]
        rank = self._accelerator.process_index
        rng_path = directory / f'rng_step_{step:05d}_rank_{rank:04d}.pt'
        tmp = rng_path.with_suffix('.pt.tmp')
        torch.save({'step': step, 'rank': rank, 'torch': torch.random.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state(self._accelerator.device),
                    'python': random.getstate(), 'numpy': np.random.get_state()}, tmp)
        tmp.replace(rng_path)
        result = super()._save_checkpoint()
        self._accelerator.wait_for_everyone()
        if self._accelerator.is_main_process:
            # A receipt commits the complete bundle last. Partial saves are rejected.
            paths = [checkpoint, state_path] + [directory / f'rng_step_{step:05d}_rank_{r:04d}.pt'
                                              for r in range(self._accelerator.num_processes)]
            cursor = ({'next_global_microbatch': step * self._config.optimization.gradient_accumulation_steps,
                       'sampling_plan_sha256': self.sampling_plan['sha256']} if self.sampling_plan is not None else
                      {'next_global_microstep': step * self._config.optimization.gradient_accumulation_steps * self._accelerator.num_processes})
            keep = self._config.checkpoints.keep_last_n
            candidates = self._retention_paths + [checkpoint]
            require(len(candidates) == len(set(candidates)), 'Repeated checkpoint in resume lineage')
            from checkpoint_retention import commit_retention
            receipt = {'identity': self.launch_identity, 'step': step, **cursor,
                'resume_parent_checkpoint': str(self._config.model.load_checkpoint) if self._config.model.load_checkpoint else None,
                'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha_file(p)} for p in paths}}
            retained = commit_retention(checkpoint, candidates, keep, receipt, self.launch_identity,
                pin_purpose=('post_training_validation' if step in self._validation_steps else None))
            self._retention_paths = retained
            self._checkpoint_paths = [p for p in retained if p.parent == directory]
            retained_steps = {p.stem.split('step_')[-1] for p in self._checkpoint_paths}
            self._training_state_paths = [p for p in self._training_state_paths if p.stem.split('step_')[-1] in retained_steps]
        self._accelerator.wait_for_everyone()
        self._last_committed_step = step
        return result
