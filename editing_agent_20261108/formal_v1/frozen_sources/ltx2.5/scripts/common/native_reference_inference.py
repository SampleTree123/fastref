"""Fixed-case native inference, using the same unfused PEFT path as training.

Owner: experiments/training/notes/mixed_training_vlm_plan_20261031.md. No target video values are
accepted by the generation function. Native scheduler/noiser/Euler/decoder are
reused; only DataMake reference positions and cached text assembly are adapted.
"""
from __future__ import annotations

import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if (p / "common/project_layout.py").is_file()) / "common"))
from project_layout import ROOT as _PROJECT_ROOT, resolve_path, script_path, add_script_paths, resolve_cli_paths
add_script_paths()

from pathlib import Path
from dataclasses import replace
import gc
import math

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load_file

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.batch_split import BatchSplitAdapter
from ltx_core.model.transformer import X0Model
from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape
from ltx_pipelines.utils.denoisers import SimpleDenoiser, GuidedDenoiser
from ltx_pipelines.utils.helpers import create_noised_state
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import VideoAudio
from ltx_trainer.model_loader import (
    embedding_weight_paths, load_embeddings_processor, load_transformer,
    read_video_scale_factors, load_text_encoder,
)

from native_reference_strategy import (NativeReferenceCondition, reference_segments,
                                       SHARED_CANVAS_CONTRACT, SHARED_TIME_CONTRACT,
                                       keep_source_and_target, VISUAL_DROPOUT_CONTRACT)
from reference_positions import (protocol_spec, PROTOCOL_VERSION_V3, PROTOCOL_VERSION_V4,
                                SHARED_CANVAS_PROTOCOLS, NATIVE_IMAGE_PROTOCOLS, NATIVE_IMAGE_CONTRACT,
                                SEQUENTIAL_TIME_PROTOCOLS, SEQUENTIAL_TIME_CONTRACT,
                                native_temporal_span, apply_sequential_time_layout,
                                REFS_FIRST_PROTOCOLS, REFS_FIRST_LAYOUT, sequential_time_contract,
                                rotate_state_prefix_to_end)


def validate_guidance(cfg_scale=1.0, stg_scale=0.0, guidance_rescale=0.0,
                      stg_blocks=(28,), guidance_batch_size=1):
    values = (cfg_scale, stg_scale, guidance_rescale)
    if not all(math.isfinite(v) for v in values) or cfg_scale < 1 or stg_scale < 0 or not 0 <= guidance_rescale <= 1:
        raise ValueError('Require finite CFG>=1, STG>=0 and rescale in [0,1]')
    if type(guidance_batch_size) is not int or guidance_batch_size < 1:
        raise ValueError('Guidance batch size must be positive')
    if (not isinstance(stg_blocks, (list, tuple)) or len(set(stg_blocks)) != len(stg_blocks)
            or any(type(i) is not int or i < 0 for i in stg_blocks) or (stg_scale != 0 and not stg_blocks)):
        raise ValueError('STG requires distinct nonnegative block indices')


@torch.inference_mode()
def encode_negative_context(weights, prompt, *, device):
    """Official Gemma + feature extractor + connector, once per worker.

    Run before loading the diffusion model. The full Gemma encoder is then
    released; only its small post-connector video context is retained on CPU.
    This is the same left-padded, non-enhanced text path as training caches.
    """
    if not isinstance(prompt, str):
        raise ValueError('Negative prompt must be a string (empty is allowed)')
    encoder = load_text_encoder(weights['text_encoders'], device=device,
                                dtype=torch.bfloat16).eval().requires_grad_(False)
    processor = load_embeddings_processor(
        checkpoint_path=embedding_weight_paths(weights['diffusion_models'], weights['text_encoders']),
        gemma_model_path=weights['text_encoders'], device=device,
        dtype=torch.bfloat16).eval().requires_grad_(False)
    processor.audio_connector = None
    hidden, mask = encoder.encode([prompt])[0]
    features, _ = processor.feature_extractor(hidden, mask, 'left')
    context, _, context_mask = processor.create_embeddings(
        features, None, convert_to_additive_mask(mask, features.dtype))
    if context_mask is None or not bool((context_mask == 1).all()) or not bool(torch.isfinite(context).all()):
        raise ValueError('Invalid negative post-connector context')
    result = context.cpu().contiguous()
    del encoder, processor, hidden, mask, features, context, context_mask
    gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    return result


def make_denoiser(model, context, *, negative_context=None, cfg_scale=1.0,
                  stg_scale=0.0, guidance_rescale=0.0, stg_blocks=(28,),
                  guidance_batch_size=1):
    """Use native guiders; the default branch retains the original exact call path."""
    validate_guidance(cfg_scale, stg_scale, guidance_rescale, stg_blocks, guidance_batch_size)
    x0_model = X0Model(model)
    if cfg_scale == 1.0 and stg_scale == 0.0 and guidance_rescale == 0.0:
        return x0_model, SimpleDenoiser(v_context=context, a_context=None)
    if stg_scale != 0 and max(stg_blocks) >= x0_model.num_blocks:
        raise ValueError('STG block index exceeds the transformer depth')
    if cfg_scale != 1.0:
        if negative_context is None or negative_context.shape != context.shape:
            raise ValueError('CFG requires negative context matching positive context shape')
        if negative_context.device != context.device or negative_context.dtype != context.dtype:
            raise ValueError('Negative context dtype/device differs from positive context')
    guider = MultiModalGuider(
        params=MultiModalGuiderParams(cfg_scale=cfg_scale, stg_scale=stg_scale,
            rescale_scale=guidance_rescale, stg_blocks=list(stg_blocks),
            modality_scale=1.0, skip_step=0), negative_context=negative_context)
    # Native helper splits cond/negative/STG into bounded-memory passes and
    # slices the official perturbation masks with each batch chunk.
    return (BatchSplitAdapter(x0_model, max_batch_size=guidance_batch_size),
            GuidedDenoiser(v_context=context, a_context=None, video_guider=guider,
                           audio_guider=None))


def validate_ref_cfg(ref_cfg):
    if not math.isfinite(ref_cfg) or ref_cfg < 1:
        raise ValueError('Reference CFG must be finite and >= 1')


class VisualReferenceDenoiser:
    """Add an independent optional-reference CFG term; source/text stay present.

    Reuses the official guided predictions and preserves their FP32 combination,
    full-packed rescale scope and final dtype. Only the new delta is target-only.
    """
    def __init__(self, base, context, *, source_count, target_count, protocol,
                 ref_cfg, cfg_scale, stg_scale, guidance_rescale):
        validate_ref_cfg(ref_cfg)
        self.base = base
        self.without_reference = SimpleDenoiser(v_context=context, a_context=None)
        self.source_count, self.target_count = source_count, target_count
        self.protocol, self.ref_cfg = protocol, ref_cfg
        self.cfg_scale, self.stg_scale, self.guidance_rescale = cfg_scale, stg_scale, guidance_rescale

    def __call__(self, transformer, video_state, audio_state, sigmas, step_index):
        if audio_state is not None:
            raise ValueError('Visual reference guidance is video-only')
        result = self.base(transformer, video_state, None, sigmas, step_index)
        if self.ref_cfg == 1 or video_state.latent.shape[1] == self.source_count + self.target_count:
            return result
        without = keep_source_and_target(video_state, target_count=self.target_count,
                                         source_count=self.source_count, protocol=self.protocol)
        negative = self.without_reference(transformer, without, None, sigmas, step_index).video.denoised
        video = result.video
        cond = video.cond if video.cond is not None else video.denoised
        dtype = cond.dtype
        cond = cond.float()
        def fp32(value):
            return value.float() if isinstance(value, torch.Tensor) else (0. if value is None else value)
        # Same operation order as official MultiModalGuider.calculate, including
        # its zero-weight video/audio modality term. Do not combine BF16 outputs.
        pred = (cond + (self.cfg_scale - 1) * (cond - fp32(video.uncond))
                + self.stg_scale * (cond - fp32(video.ptb))
                + 0. * (cond - fp32(video.mod)))
        target = slice(-self.target_count, None) if self.protocol in REFS_FIRST_PROTOCOLS else slice(0, self.target_count)
        pred[:, target] += (self.ref_cfg - 1) * (cond[:, target] - negative[:, target].float())
        if self.guidance_rescale != 0:
            factor = cond.std() / pred.std()
            pred = pred * (self.guidance_rescale * factor + (1 - self.guidance_rescale))
        return VideoAudio(replace(video, denoised=pred.to(dtype)), result.audio)


def validate_reference_checkpoint_metadata(metadata, protocol):
    protocol_spec(protocol)
    if metadata.get('datamake_reference_position_protocol') != protocol:
        raise ValueError('Checkpoint reference positions do not match inference')
    if metadata.get('datamake_adapter_compute_contract') != 'unfused_peft_bf16_autocast':
        raise ValueError('Checkpoint adapter compute contract does not match inference')
    if protocol in (*SHARED_CANVAS_PROTOCOLS, *NATIVE_IMAGE_PROTOCOLS):
        canvas_contract = NATIVE_IMAGE_CONTRACT if protocol in NATIVE_IMAGE_PROTOCOLS else SHARED_CANVAS_CONTRACT
        if (metadata.get('datamake_reference_canvas_contract') != canvas_contract
                or metadata.get('datamake_reference_time_contract') != (sequential_time_contract(protocol) if protocol in SEQUENTIAL_TIME_PROTOCOLS else SHARED_TIME_CONTRACT)):
            raise ValueError('Checkpoint shared-canvas/time contracts do not match inference')
    if protocol in REFS_FIRST_PROTOCOLS and metadata.get('datamake_reference_layout') != REFS_FIRST_LAYOUT:
        raise ValueError('Checkpoint reference packing does not match inference')


def load_inference_models(weights, lora_config, *, checkpoint=None, protocol, device, ref_cfg=1.0):
    """Fresh base also gets zero-effect PEFT, so base/trained kernels match."""
    # safetensors interprets bare 'cuda' as GPU 0 rather than the current rank.
    # Resolve it before any upstream checkpoint reader sees the device.
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    protocol_spec(protocol)
    validate_ref_cfg(ref_cfg)
    if checkpoint is None and ref_cfg > 1:
        raise ValueError('Reference CFG requires a checkpoint trained with visual reference dropout')
    if checkpoint is not None:
        checkpoint = resolve_path(checkpoint)
        with safe_open(str(checkpoint), framework='pt', device='cpu') as handle:
            metadata = handle.metadata() or {}
            validate_reference_checkpoint_metadata(metadata, protocol)
            if ref_cfg > 1 and (float(metadata.get('datamake_visual_reference_dropout_probability', '0')) <= 0
                                or metadata.get('datamake_visual_reference_dropout_contract') != VISUAL_DROPOUT_CONTRACT):
                raise ValueError('Checkpoint was not trained with the visual reference dropout contract')
    model = load_transformer(weights['diffusion_models'], device='cpu', dtype=torch.bfloat16)
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(
        r=lora_config['rank'], lora_alpha=lora_config['alpha'],
        lora_dropout=lora_config['dropout'], target_modules=lora_config['target_modules'],
        init_lora_weights=True))
    if any(p.dtype != torch.float32 for n, p in model.named_parameters() if 'lora_' in n):
        raise ValueError('Inference adapters must retain the training FP32 dtype')
    if checkpoint is not None:
        saved = {'base_model.model.' + k.removeprefix('diffusion_model.'): v
                 for k, v in load_file(str(checkpoint), device='cpu').items()}
        expected = get_peft_model_state_dict(model)
        if saved.keys() != expected.keys() or any(saved[k].shape != expected[k].shape for k in saved):
            raise ValueError('LoRA key set or shapes do not match configured modules/rank')
        set_peft_model_state_dict(model, saved)
        actual = get_peft_model_state_dict(model)
        if any(not torch.equal(actual[k], saved[k]) for k in saved):
            raise ValueError('Loaded LoRA tensors differ from saved checkpoint')
    # Preserve FP32 adapters; .to(dtype=BF16) here would silently change them.
    model = model.to(device=device).eval().requires_grad_(False)
    processor = load_embeddings_processor(
        checkpoint_path=embedding_weight_paths(weights['diffusion_models'], weights['text_encoders']),
        gemma_model_path=weights['text_encoders'], device=device, dtype=torch.bfloat16)
    processor.feature_extractor = None
    processor.audio_connector = None
    processor.eval().requires_grad_(False)
    return model, processor, read_video_scale_factors(weights['vae'])


def build_generation_state(source, references, *, scale_factors, protocol, seed):
    """B=1 only. The output canvas/duration come from source, never target pixels."""
    value = source['latents']
    if value.ndim != 5 or value.shape[0] != 1 or value.dtype != torch.bfloat16:
        raise ValueError('Expected one BF16 source latent [1,C,F,H,W]')
    fps = float(torch.as_tensor(source['fps']).item())
    if fps != 12:
        raise ValueError('Reference protocol currently requires 12 fps')
    tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1),
                             VideoLatentShape.from_torch_shape(value.shape),
                             fps=fps, scale_factors=scale_factors)
    conditions = []
    geometry = []
    for role, slot, raw in reference_segments({'source': source, 'references': references},
                                             protocol_version=protocol):
        latent = raw['latents']
        if latent.requires_grad:
            raise ValueError('Reference latent must be detached')
        geometry.append(tuple(latent.shape))
        conditions.append(NativeReferenceCondition(latent, role=role, image_slot=slot,
                                                   protocol_version=protocol))
    generator = torch.Generator(device=value.device).manual_seed(seed)
    state = create_noised_state(tools, conditions, GaussianNoiser(generator=generator),
                               dtype=value.dtype, device=value.device)
    if protocol in SEQUENTIAL_TIME_PROTOCOLS:
        target_count = tools.patchifier.get_token_count(tools.target_shape)
        cursor, coordinates, spans = target_count, [], []
        for shape in geometry:
            count = shape[2] * shape[3] * shape[4]
            coordinates.append(state.positions[:, :, cursor:cursor + count])
            spans.append(native_temporal_span(shape[2], fps, scale_factors.time))
            cursor += count
        if cursor != state.positions.shape[2]:
            raise ValueError('Sequential inference layout does not match reference tokens')
        target, refs = apply_sequential_time_layout(
            state.positions[:, :, :target_count], coordinates,
            target_span=native_temporal_span(value.shape[2], fps, scale_factors.time), reference_spans=spans)
        state = replace(state, positions=torch.cat([target, *refs], dim=2))
    if protocol in REFS_FIRST_PROTOCOLS:
        target_count = tools.patchifier.get_token_count(tools.target_shape)
        state = rotate_state_prefix_to_end(state, target_count)
    return tools, state


def extract_generated_latent(tools, state, *, protocol):
    """Give official output extraction its expected target-first layout."""
    if protocol in REFS_FIRST_PROTOCOLS:
        n = tools.patchifier.get_token_count(tools.target_shape)
        state = rotate_state_prefix_to_end(state, state.latent.shape[1] - n)
    return tools.unpatchify(tools.clear_conditioning(state)).latent


@torch.inference_mode()
def generate_latent(model, processor, *, source, references, conditions,
                    scale_factors, protocol, seed=42, steps=40,
                    negative_context=None, cfg_scale=1.0, stg_scale=0.0,
                    guidance_rescale=0.0, stg_blocks=(28,), guidance_batch_size=1, ref_cfg=1.0):
    validate_ref_cfg(ref_cfg)
    if steps < 2:
        raise ValueError('Native stretched LTX2Scheduler requires at least two inference steps')
    device = source['latents'].device
    if device.type != 'cuda':
        raise ValueError('Model inference requires an allocated CUDA device')
    tools, state = build_generation_state(source, references, scale_factors=scale_factors,
                                          protocol=protocol, seed=seed)
    # Training runs the frozen connector outside Accelerator's model autocast.
    # Preserve that boundary; only the transformer denoising pass uses AMP.
    features = conditions['video_prompt_embeds']
    mask = convert_to_additive_mask(conditions['prompt_attention_mask'], features.dtype)
    context, _, context_mask = processor.create_embeddings(features, None, mask)
    if context_mask is None or not bool((context_mask == 1).all()):
        raise ValueError('Expected all-valid post-connector text context')
    transformer, denoiser = make_denoiser(model, context, negative_context=negative_context,
        cfg_scale=cfg_scale, stg_scale=stg_scale, guidance_rescale=guidance_rescale,
        stg_blocks=stg_blocks, guidance_batch_size=guidance_batch_size)
    if ref_cfg > 1 and references:
        denoiser = VisualReferenceDenoiser(denoiser, context,
            source_count=math.prod(source['latents'].shape[2:]),
            target_count=tools.patchifier.get_token_count(tools.target_shape), protocol=protocol,
            ref_cfg=ref_cfg, cfg_scale=cfg_scale, stg_scale=stg_scale, guidance_rescale=guidance_rescale)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        sigmas = LTX2Scheduler().execute(steps=steps).to(device=device, dtype=torch.float32)
        result = euler_denoising_loop(sigmas, state, None, transformer, denoiser)
    n = tools.patchifier.get_token_count(tools.target_shape)
    refs = slice(None, -n) if protocol in REFS_FIRST_PROTOCOLS else slice(n, None)
    if not torch.equal(result.video.latent[:, refs], state.clean_latent[:, refs]):
        raise ValueError('Reference tokens changed during denoising')
    final = extract_generated_latent(tools, result.video, protocol=protocol)
    if not bool(torch.isfinite(final).all()):
        raise ValueError('Generated latent contains non-finite values')
    return final
