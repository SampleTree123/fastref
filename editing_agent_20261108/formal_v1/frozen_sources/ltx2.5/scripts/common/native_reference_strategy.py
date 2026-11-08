"""Experimental variable-reference packing; explicitly opts into candidate positions.

Reuses official patchification, target noise sampling, target-only loss and native
inference conditioning. No upstream edits, new embeddings, padding or reference loss.
"""
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if (p / "common/project_layout.py").is_file()) / "common"))
from project_layout import ROOT as _PROJECT_ROOT, resolve_path, script_path, add_script_paths, resolve_cli_paths
add_script_paths()

from dataclasses import dataclass, fields, replace
from collections import OrderedDict
import json
import math

import torch
from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core.types import DenoiseMask, LatentState
from ltx_core.model.transformer.modality import TokenTimesteps
from ltx_pipelines.utils.helpers import modality_from_latent_state
from ltx_trainer.training_strategies.flexible import FlexibleStrategy

from reference_positions import (apply_reference_position_protocol, validate_reference_slots,
                                PROTOCOL_VERSION, PROTOCOL_VERSION_V3, PROTOCOL_VERSION_V4, protocol_spec,
                                SHARED_CANVAS_PROTOCOLS, NATIVE_IMAGE_PROTOCOLS, ZERO_LOCAL_OFFSET_PROTOCOLS,
                                SEQUENTIAL_TIME_PROTOCOLS, NATIVE_IMAGE_CONTRACT, SEQUENTIAL_TIME_CONTRACT,
                                native_temporal_span, apply_sequential_time_layout,
                                REFS_FIRST_PROTOCOLS, REFS_FIRST_LAYOUT, sequential_time_contract,
                                rotate_state_prefix_to_end)
from contact_sheet_contract import validate_contact_sheet


SHARED_CANVAS_CONTRACT = 'source_canvas_contain_black_padding_before_vae_v1'
SHARED_TIME_CONTRACT = 'official_local_time_zero_origin_no_reference_offsets_v1'


def scalar(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Experimental strategy requires batch size one")
        return value[0]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Experimental strategy requires scalar metadata")
        return value.item()
    return value


def reference_segments(batch, *, protocol_version=PROTOCOL_VERSION, validate_metadata=True):
    segments = [("source", None, batch["source"])]
    for expected_index, item in enumerate(batch["references"]):
        if int(scalar(item["index"])) != expected_index:
            raise ValueError("Reference order changed")
        kind = scalar(item["type"])
        if kind not in {"motion", "identity", "hair", "garment", "scene", "outfit", "camera"}:
            raise ValueError("Unknown reference type")
        representation = scalar(item.get('reference_representation'))
        if protocol_version == PROTOCOL_VERSION_V4 and representation is not None:
            raise ValueError('V4 separate-images experiment does not accept contact sheets')
        if representation is not None:
            provenance = json.loads(scalar(item['contact_sheet_json']))
            if provenance.get('reference_representation') != representation:
                raise ValueError('Runtime contact-sheet provenance disagrees with representation')
            raw_shape = batch['source']['latents'].shape
            validate_contact_sheet({'type': kind, 'view': scalar(item['view']), **provenance},
                                   canvas_wh=[raw_shape[-1] * 32, raw_shape[-2] * 32],
                                   protocol=protocol_version, reference_count=len(batch['references']))
        else:
            validate_contact_sheet({'type': kind, 'view': scalar(item.get('view', ''))}, protocol=protocol_version)
        if kind in {"hair", "garment", "outfit"} and representation is None:
            expected_view = "front" if expected_index % 2 == 0 else "back"
            if scalar(item["view"]) != expected_view or len(batch["references"]) % 2:
                raise ValueError("Paired reference views must preserve front/back slots")
        role = "motion" if kind == "motion" else "image"
        slot = None if role == "motion" else expected_index
        segments.append((role, slot, item["latent"]))
    validate_reference_slots([{"role": role, "image_slot": slot} for role, slot, _ in segments], protocol_version=protocol_version)
    if protocol_version in SHARED_CANVAS_PROTOCOLS:
        validate_shared_canvas_request(batch, segments, validate_metadata=validate_metadata)
    elif protocol_version in NATIVE_IMAGE_PROTOCOLS:
        validate_shared_canvas_request(batch, segments, validate_metadata=validate_metadata, native_images=True)
    if protocol_version in REFS_FIRST_PROTOCOLS:
        segments = [*segments[1:], segments[0]]
    return segments


def validate_shared_canvas_latent(latent, source_shape, role, *, native_images=False):
    """Shape contract for both training records and direct native conditions."""
    if not isinstance(latent, torch.Tensor) or latent.ndim != 5:
        raise ValueError('Shared-canvas references require [B,C,F,H,W] latents')
    expected = tuple(source_shape)
    if role == 'image':
        if native_images:
            if tuple(latent.shape[:3]) != (*expected[:2], 1) or min(latent.shape[-2:]) < 1:
                raise ValueError('Native-size images need matching batch/channels and one nonempty frame')
            return
        expected = (*expected[:2], 1, *expected[3:])
    elif role not in {'source', 'motion', 'target'}:
        raise ValueError('Unknown shared-canvas role')
    if tuple(latent.shape) != expected:
        raise ValueError(f'Shared-canvas {role} shape {tuple(latent.shape)} does not match {expected}')


def validate_shared_canvas_request(batch, segments, *, validate_metadata=True, native_images=False):
    """Validate cached geometry/time metadata without reading target pixels.

    Actual contain/padding and temporal alignment belong to the versioned media
    cache. This checks its tensor/metadata contract, not semantic action alignment.
    """
    source = batch['source']['latents']
    if not isinstance(source, torch.Tensor) or source.ndim != 5:
        raise ValueError('Shared-canvas source requires [B,C,F,H,W]')
    records = list(segments)
    if 'video_latents' in batch:
        records.append(('target', None, batch['video_latents']))
    for role, _, raw in records:
        latent = raw['latents']
        validate_shared_canvas_latent(latent, source.shape, role, native_images=native_images)
        if latent.dtype != source.dtype or latent.device != source.device:
            raise ValueError('Shared-canvas latent dtype/device must match source')
        if validate_metadata:
            if scalar(raw['fps']) != 12:
                raise ValueError('Shared-canvas media must all use source FPS 12')
            declared = tuple(scalar(raw[k]) for k in ('num_frames', 'height', 'width'))
            if declared != tuple(latent.shape[2:]):
                raise ValueError('Shared-canvas latent geometry metadata disagrees with tensor')


def slice_collated_sample(batch, index, batch_size):
    def slice_value(value):
        if isinstance(value, torch.Tensor):
            if value.ndim == 0 or value.shape[0] != batch_size:
                raise ValueError("Batched field lacks its sample axis")
            return value[index:index + 1]
        if value is None:
            return None
        if isinstance(value, dict):
            return {k: slice_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)) and len(value) == batch_size:
            return value[index:index + 1]
        raise ValueError("Unexpected collated per-sample metadata")
    result = {k: slice_value(v) for k, v in batch.items() if k != 'references'}
    result['references'] = [{k: slice_value(v) for k, v in ref.items()} for ref in batch['references']]
    return result


def merge_model_inputs(items):
    """Only concatenate the B axis. No sample is appended to another's tokens."""
    def merge(values):
        first = values[0]
        if first is None:
            if any(v is not None for v in values):
                raise ValueError("Mixed enabled/disabled modality fields")
            return None
        if isinstance(first, torch.Tensor):
            if any(v.shape[1:] != first.shape[1:] or v.dtype != first.dtype for v in values):
                raise ValueError("Incompatible tensor fields in grouped microbatch")
            return torch.cat(values, dim=0)
        if isinstance(first, TokenTimesteps):
            if any(v.zero_index != first.zero_index for v in values):
                raise ValueError("Incompatible compact timestep tables")
            return TokenTimesteps(merge([v.values for v in values]), merge([v.indices for v in values]), first.zero_index)
        if hasattr(first, '__dataclass_fields__'):
            return type(first)(**{f.name: merge([getattr(v, f.name) for v in values]) for f in fields(first)})
        if any(v != first for v in values):
            raise ValueError("Incompatible modality flags")
        return first
    return merge(items)


VISUAL_DROPOUT_CONTRACT = 'all_optional_refs_per_sample_preserve_source_text_positions_v1'


@dataclass
class GroupedVideoInput:
    """At most two sequence lengths, handled inside one outer DDP forward.

    Groups retain full ModelInputs for unchanged target-only loss calculation.
    No padding or cross-sample attention is introduced.
    """
    groups: tuple
    sample_indices: tuple
    enabled: bool = True

    @property
    def sigma(self):
        values = torch.cat([group.video.sigma for group in self.groups])
        order = [i for indices in self.sample_indices for i in indices]
        inverse = torch.tensor(sorted(range(len(order)), key=order.__getitem__), device=values.device)
        return values.index_select(0, inverse)


def install_grouped_video_forward(model):
    """Install before DDP/PEFT; module names, parameters and state_dict stay intact."""
    if getattr(model, '_datamake_grouped_forward', False):
        return
    original_forward = model.forward

    def grouped_forward(*args, **kwargs):
        video = kwargs.get('video', args[0] if args else None)
        if not isinstance(video, GroupedVideoInput):
            return original_forward(*args, **kwargs)
        if args or kwargs.get('audio') is not None or kwargs.get('perturbations') is not None:
            raise ValueError('Grouped reference training supports video-only unperturbed calls')
        extra = {k: v for k, v in kwargs.items() if k not in {'video', 'audio', 'perturbations'}}
        predictions = []
        for group in video.groups:
            prediction, audio_prediction = original_forward(video=group.video, audio=None,
                                                              perturbations=None, **extra)
            if audio_prediction is not None:
                raise ValueError('Unexpected audio prediction')
            predictions.append(prediction)
        return predictions, None

    model.forward = grouped_forward
    model._datamake_grouped_forward = True


def keep_source_and_target(state, *, target_count, source_count, protocol):
    """Delete extra references after coordinates are assigned; never shift source/target."""
    total = state.latent.shape[1]
    if target_count + source_count > total:
        raise ValueError('Source/target token count exceeds state length')
    keep = (slice(total - target_count - source_count, None)
            if protocol in REFS_FIRST_PROTOCOLS else slice(0, target_count + source_count))
    attention = state.attention_mask
    if attention is not None:
        attention = attention[:, keep, keep]
    return replace(state, latent=state.latent[:, keep], positions=state.positions[:, :, keep],
                   clean_latent=state.clean_latent[:, keep],
                   denoise_mask=state.denoise_mask.with_same_values(state.denoise_mask.tensor[:, keep]),
                   keyframes_mask=None if state.keyframes_mask is None else state.keyframes_mask[:, keep],
                   attention_mask=attention)


class NativeReferenceStrategy(FlexibleStrategy):
    def __init__(self, config, *, protocol_version=PROTOCOL_VERSION, visual_reference_probability=0.0):
        protocol_spec(protocol_version)
        self.protocol_version = protocol_version
        if not math.isfinite(visual_reference_probability) or not 0 <= visual_reference_probability <= 1:
            raise ValueError('Visual reference dropout probability must be in [0,1]')
        self.visual_reference_probability = float(visual_reference_probability)
        if config.audio is not None or config.video is None or config.video.conditions:
            raise ValueError("Use generated video with no stock conditions; references come from the manifest")
        super().__init__(config)
        self._reference_position_cache = OrderedDict()

    def _patchify_latent_data(self, latent_data, modality_key):
        data = super()._patchify_latent_data(latent_data, modality_key)
        if self.protocol_version in ZERO_LOCAL_OFFSET_PROTOCOLS and modality_key == 'video':
            # Reuse the Python metadata that upstream already read from tensors.
            # Reading raw metadata again here would add four GPU synchronizations
            # per reference and per sample, particularly costly for 20-image batches.
            if data.fps != 12:
                raise ValueError('Shared-canvas media must all use source FPS 12')
            if (data.num_frames, data.height, data.width) != tuple(latent_data['latents'].shape[2:]):
                raise ValueError('Shared-canvas latent geometry metadata disagrees with tensor')
        return data

    def _reference_positions(self, data, role, slot):
        """Reuse validated formula output, never raw sample coordinates or latents.

        Callers still patchify/check each input and validate its reference slots.
        The only consumer concatenates these tensors into a new positions tensor;
        neither the cached tensor nor target positions are modified in place.
        """
        key = (self.protocol_version, role, slot, data.num_frames, data.height,
               data.width, data.fps, data.latents.device,
               tuple(data.latents.shape), data.latents.dtype, tuple(self.video_scale_factors))
        if key in self._reference_position_cache:
            self._reference_position_cache.move_to_end(key)
            return self._reference_position_cache[key]
        coords = self._get_video_positions(data.num_frames, data.height, data.width,
                                          1, data.fps, data.latents.device)
        coords = apply_reference_position_protocol(coords, role=role, image_slot=slot,
                                                   protocol_version=self.protocol_version)
        self._reference_position_cache[key] = coords
        # Bound retained device memory if future requests introduce more shapes.
        if len(self._reference_position_cache) > 128:
            self._reference_position_cache.popitem(last=False)
        return coords

    def prepare_training_inputs(self, batch, timestep_sampler, *, _drop_optional=None):
        batch_size = batch["video_latents"]["latents"].shape[0]
        if _drop_optional is None:
            # CPU RNG is rank-seeded and included in the existing resume RNG state.
            # With p=0 or no extra reference, consume no additional random draws.
            _drop_optional = ((torch.rand(batch_size) < self.visual_reference_probability).tolist()
                              if batch['references'] and self.visual_reference_probability > 0
                              else [False] * batch_size)
        if len(_drop_optional) != batch_size:
            raise ValueError('One reference-drop decision is required per sample')
        self.last_reference_drop_mask = tuple(bool(x) for x in _drop_optional)
        if batch_size > 1:
            items = [self.prepare_training_inputs(slice_collated_sample(batch, i, batch_size), timestep_sampler,
                                                  _drop_optional=[_drop_optional[i]]) for i in range(batch_size)]
            self.last_reference_drop_mask = tuple(bool(x) for x in _drop_optional)
            groups = OrderedDict()
            for i, item in enumerate(items):
                groups.setdefault(item.video.latent.shape[1], []).append(i)
            if len(groups) == 1:
                return merge_model_inputs(items)
            indices = tuple(tuple(value) for value in groups.values())
            grouped = GroupedVideoInput(tuple(merge_model_inputs([items[i] for i in ix]) for ix in indices), indices)
            return replace(items[0], video=grouped, video_loss_mask=None,
                           video_targets=torch.cat([item.video_targets for item in items]))
        if batch_size != 1:
            raise ValueError("Empty reference batch")
        if self.protocol_version not in ZERO_LOCAL_OFFSET_PROTOCOLS and scalar(batch["video_latents"]["fps"]) != 12:
            raise ValueError("Candidate reference protocol requires target FPS 12")
        # Shapes/dtypes/devices are checked without reading GPU tensor values;
        # _patchify_latent_data checks each target/source/reference's metadata once.
        segments = reference_segments(batch, protocol_version=self.protocol_version, validate_metadata=False)
        inputs = super().prepare_training_inputs(batch, timestep_sampler)
        target = inputs.video
        # Target/source share the local canvas; reject mismatched protocol bounds.
        apply_reference_position_protocol(target.positions, role="source", protocol_version=self.protocol_version)
        if target.context_mask is None or not bool((target.context_mask == 1).all()):
            raise ValueError("Native unmasked context requires all-valid post-connector context")
        tokens, positions, spans = [], [], []
        for role, slot, raw in segments:
            data = self._patchify_latent_data(raw, "video")
            if data.latents.requires_grad:
                raise ValueError("Reference cache must be detached")
            if data.fps != 12:
                raise ValueError("Candidate reference protocol requires 12 FPS")
            coords = self._reference_positions(data, role, slot)
            tokens.append(data.latents)
            positions.append(coords)
            if self.protocol_version in SEQUENTIAL_TIME_PROTOCOLS:
                spans.append(native_temporal_span(data.num_frames, data.fps, self.video_scale_factors.time))
        target_positions = target.positions
        if self.protocol_version in SEQUENTIAL_TIME_PROTOCOLS:
            shape = batch["video_latents"]["latents"].shape
            target_positions, positions = apply_sequential_time_layout(
                target_positions, positions, target_span=native_temporal_span(shape[2], 12.0, self.video_scale_factors.time),
                reference_spans=spans)
        reference_tokens = torch.cat(tokens, dim=1)
        length = reference_tokens.shape[1]
        target_length = target.latent.shape[1]
        data = batch["video_latents"]
        tokens_per_frame = (data['latents'].shape[-2] * data['latents'].shape[-1]
                            if self.protocol_version in ZERO_LOCAL_OFFSET_PROTOCOLS else
                            int(scalar(data["height"])) * int(scalar(data["width"])))
        marker = torch.zeros(1, target_length + length, 1, device=target.latent.device, dtype=torch.float32)
        marker[:, :tokens_per_frame] = 1
        # Match native inference's target-first layout, compact FP32 time table,
        # float marker and all-visible context. Algebraic equivalence alone was
        # insufficient: different BF16 kernels produced ~1.6% velocity differences.
        denoise = DenoiseMask(torch.cat([torch.ones(1, target_length, 1), torch.zeros(1, length, 1)], dim=1))
        state = LatentState(
            # Native GaussianNoiser/Euler retain the VAE latent dtype. The official
            # training interpolation promotes to FP32; explicitly cast at the model
            # boundary, as BF16 autocast's input projection would do, not the sigma.
            latent=torch.cat([target.latent, reference_tokens], dim=1).to(batch["video_latents"]["latents"].dtype),
            positions=torch.cat([target_positions, *positions], dim=2),
            denoise_mask=denoise.compact(target.latent.device),
            clean_latent=torch.cat([torch.zeros_like(target.latent), reference_tokens], dim=1),
            keyframes_mask=marker)
        loss_mask = torch.cat([inputs.video_loss_mask,
                               torch.zeros(1, length, dtype=torch.bool, device=target.latent.device)], dim=1)
        if self.protocol_version in REFS_FIRST_PROTOCOLS:
            state = rotate_state_prefix_to_end(state, target_length)
            loss_mask = torch.cat([loss_mask[:, target_length:], loss_mask[:, :target_length]], dim=1)
        if _drop_optional[0] and batch['references']:
            source_count = math.prod(batch['source']['latents'].shape[2:])
            state = keep_source_and_target(state, target_count=target_length, source_count=source_count,
                                           protocol=self.protocol_version)
            loss_mask = (loss_mask[:, -target_length-source_count:]
                         if self.protocol_version in REFS_FIRST_PROTOCOLS else loss_mask[:, :target_length+source_count])
        video = modality_from_latent_state(state, target.context, target.sigma)
        return replace(inputs, video=video, video_loss_mask=loss_mask)

    def compute_loss(self, video_pred, audio_pred, inputs):
        if isinstance(inputs.video, GroupedVideoInput):
            if audio_pred is not None or len(video_pred) != len(inputs.video.groups):
                raise ValueError('Grouped predictions do not match input groups')
            losses = torch.cat([self.compute_loss(pred, None, group)
                                for pred, group in zip(video_pred, inputs.video.groups)])
            order = [i for indices in inputs.video.sample_indices for i in indices]
            inverse = torch.tensor(sorted(range(len(order)), key=order.__getitem__), device=losses.device)
            return losses.index_select(0, inverse)
        if audio_pred is not None or video_pred is None or inputs.video_targets is None:
            raise ValueError("This strategy supervises generated video only")
        n = inputs.video_targets.shape[1]
        # Reuse official MSE and normalization after explicitly selecting the
        # target prefix; upstream's default selects the tail (its own layout).
        target_slice = slice(-n, None) if self.protocol_version in REFS_FIRST_PROTOCOLS else slice(None, n)
        return self._compute_modality_loss(video_pred[:, target_slice], inputs.video_targets, inputs.video_loss_mask[:, target_slice])

    def get_checkpoint_metadata(self):
        result = {"datamake_reference_position_protocol": self.protocol_version,
                "datamake_reference_layout": (REFS_FIRST_LAYOUT if self.protocol_version in REFS_FIRST_PROTOCOLS
                                              else "target_then_source_then_ordered_optional_refs"),
                "datamake_modality_contract": "native_compact_fp32_time_bf16_latent_all_visible_context_v1",
                "datamake_adapter_compute_contract": "unfused_peft_bf16_autocast",
                "datamake_target_first_frame_marker": "true_without_target_first_frame_conditioning"}
        if self.protocol_version in SHARED_CANVAS_PROTOCOLS:
            result.update(datamake_reference_canvas_contract=SHARED_CANVAS_CONTRACT,
                          datamake_reference_time_contract=(SEQUENTIAL_TIME_CONTRACT if self.protocol_version == PROTOCOL_VERSION_V4 else SHARED_TIME_CONTRACT))
        if self.protocol_version in NATIVE_IMAGE_PROTOCOLS:
            result.update(datamake_reference_canvas_contract=NATIVE_IMAGE_CONTRACT,
                          datamake_reference_time_contract=(sequential_time_contract(self.protocol_version) if self.protocol_version in SEQUENTIAL_TIME_PROTOCOLS else SHARED_TIME_CONTRACT))
        if self.visual_reference_probability > 0:
            result.update(datamake_visual_reference_dropout_probability=str(self.visual_reference_probability),
                          datamake_visual_reference_dropout_contract=VISUAL_DROPOUT_CONTRACT)
        return result


class NativeReferenceCondition(VideoConditionByReferenceLatent):
    """Native inference counterpart; references remain appended as upstream expects."""
    def __init__(self, latent, *, role, image_slot=None, protocol_version=PROTOCOL_VERSION):
        protocol_spec(protocol_version)
        self.protocol_version = protocol_version
        super().__init__(latent, strength=1.0)
        self.role = role
        self.image_slot = image_slot

    def apply_to(self, latent_state, latent_tools):
        if latent_tools.fps != 12:
            raise ValueError("Candidate reference protocol requires 12 FPS")
        if self.protocol_version in ZERO_LOCAL_OFFSET_PROTOCOLS:
            validate_shared_canvas_latent(self.latent, latent_tools.target_shape.to_torch_shape(), self.role,
                                          native_images=self.protocol_version in NATIVE_IMAGE_PROTOCOLS)
            if self.latent.dtype != latent_state.latent.dtype or self.latent.device != latent_state.latent.device:
                raise ValueError('Shared-canvas inference reference dtype/device differs from target')
        result = super().apply_to(latent_state, latent_tools)
        start = latent_state.positions.shape[2]
        coords = result.positions.clone()
        coords[:, :, start:] = apply_reference_position_protocol(
            coords[:, :, start:], role=self.role, image_slot=self.image_slot, protocol_version=self.protocol_version)
        return replace(result, positions=coords)
