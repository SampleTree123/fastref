#!/usr/bin/env python3
"""Compact ONLY pre-connector video features; restore the full sequence on load.

This storage format drops finite padded feature rows, not connector registers or
sequence positions. It is valid only for the audited native connector protocol.
Real-weight GPU equivalence remains a separate required validation gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import os

import torch

PROTOCOL = "ltx25_preconnector_masked_rows_v1"
SOURCE_REVISION = "e98f6d927d83c92ef2be6aec76c8440b19306e39"
STAGE = "after_feature_extractor_before_create_embeddings"
SUPPORTED_DTYPES = {"torch.float32": torch.float32, "torch.float16": torch.float16,
                    "torch.bfloat16": torch.bfloat16, "torch.float64": torch.float64}
MASK_DTYPES = {torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def make_contract(*, sequence_length, feature_dim, num_learnable_registers,
                  encoder_identity, connector_identity, text_identity):
    """Identities must describe frozen weights/tokenizer/template/exact input.

    Use real revision/SHA identities, not a display label such as 'latest'.
    The caller must obtain dimensions/register count from the loaded model.
    """
    contract = {"protocol": PROTOCOL, "stage": STAGE, "source_revision": SOURCE_REVISION,
                "sequence_length": sequence_length, "feature_dim": feature_dim,
                "num_learnable_registers": num_learnable_registers,
                "encoder_identity": encoder_identity, "connector_identity": connector_identity,
                "text_identity": text_identity,
                "restore": "original_positions_and_full_sequence_zero_fill_padding",
                "modality": "video_only"}
    _validate_contract(contract)
    return contract


def _validate_contract(contract):
    _require(isinstance(contract, dict), "contract must be a dictionary")
    _require(contract.get("protocol") == PROTOCOL and contract.get("stage") == STAGE,
             "unsupported cache stage or protocol")
    _require(contract.get("source_revision") == SOURCE_REVISION, "native source revision needs new audit")
    _require(contract.get("modality") == "video_only", "only video features are supported")
    _require(contract.get("restore") == "original_positions_and_full_sequence_zero_fill_padding",
             "shortening the connector sequence is forbidden")
    for name in ("sequence_length", "feature_dim", "num_learnable_registers"):
        _require(type(contract.get(name)) is int and contract[name] > 0, f"invalid {name}")
    _require(contract["sequence_length"] % contract["num_learnable_registers"] == 0,
             "sequence length must be divisible by enabled register count")
    for name in ("encoder_identity", "connector_identity", "text_identity"):
        _require(isinstance(contract.get(name), str) and bool(contract[name].strip()), f"missing {name}")


def validate_connector(connector, contract):
    """Call against the loaded native connector before adopting this cache."""
    _validate_contract(contract)
    _require(getattr(connector, "num_learnable_registers", None) == contract["num_learnable_registers"],
             "connector register count mismatch or disabled registers")
    _require(getattr(connector, "inner_dim", None) == contract["feature_dim"], "connector feature dimension mismatch")
    _require(type(connector).__name__ == "Embeddings1DConnector", "unsupported connector implementation")


def _validate_mask(mask, length):
    _require(isinstance(mask, torch.Tensor) and mask.ndim == 1 and list(mask.shape) == [length],
             "mask must be binary [sequence_length]")
    _require(mask.dtype in MASK_DTYPES, "mask must be integer/bool, not additive/float")
    _require(bool(((mask == 0) | (mask == 1)).all()), "mask contains non-binary values")


def pack_text_features(video_prompt_embeds, prompt_attention_mask, *, contract):
    """Return CPU compact tensors. Does NOT mutate the original dense cache."""
    _validate_contract(contract)
    features = video_prompt_embeds
    _require(isinstance(features, torch.Tensor) and features.ndim == 2,
             "expected one sample of pre-connector features [S,D]")
    _require(list(features.shape) == [contract["sequence_length"], contract["feature_dim"]],
             "feature shape disagrees with contract")
    _require(str(features.dtype) in SUPPORTED_DTYPES, "unsupported floating feature dtype")
    _validate_mask(prompt_attention_mask, features.shape[0])
    # Native register replacement multiplies by 0 rather than torch.where.
    # NaN/Inf padding is NOT safely discarded: 0*NaN remains NaN.
    _require(bool(torch.isfinite(features).all()), "non-finite feature values, including padding, are not safe to discard")
    features = features.detach().cpu()
    mask = prompt_attention_mask.detach().cpu().clone().contiguous()
    positions = torch.nonzero(mask.bool(), as_tuple=False).flatten().to(torch.int64)
    metadata = {"contract": dict(contract), "original_shape": list(features.shape),
                "feature_dtype": str(features.dtype), "mask_dtype": str(mask.dtype),
                "valid_rows": len(positions)}
    return {"schema": PROTOCOL, "metadata": metadata, "protocol_signature": _signature(metadata),
            "prompt_attention_mask": mask, "valid_positions": positions,
            "video_valid_features": features.index_select(0, positions).contiguous()}


def unpack_text_features(compact, *, expected_contract):
    """Restore [S,D] plus original mask, keeping every connector position intact."""
    _validate_contract(expected_contract)
    _require(isinstance(compact, dict) and compact.get("schema") == PROTOCOL, "unsupported compact schema")
    metadata = compact.get("metadata")
    _require(isinstance(metadata, dict), "missing metadata")
    _require(compact.get("protocol_signature") == _signature(metadata), "protocol metadata signature mismatch")
    _require(metadata.get("contract") == expected_contract, "cache/model/input contract mismatch")
    shape = [expected_contract["sequence_length"], expected_contract["feature_dim"]]
    _require(metadata.get("original_shape") == shape, "original shape mismatch")
    mask = compact.get("prompt_attention_mask")
    _validate_mask(mask, shape[0])
    _require(str(mask.dtype) == metadata.get("mask_dtype"), "mask dtype mismatch")
    positions = compact.get("valid_positions")
    _require(isinstance(positions, torch.Tensor) and positions.dtype == torch.int64 and positions.ndim == 1,
             "positions must be int64 vector")
    expected_positions = torch.nonzero(mask.bool(), as_tuple=False).flatten().cpu()
    _require(torch.equal(positions.cpu(), expected_positions), "positions do not match original mask")
    _require(metadata.get("valid_rows") == len(expected_positions), "valid row count mismatch")
    values = compact.get("video_valid_features")
    _require(isinstance(values, torch.Tensor) and list(values.shape) == [len(positions), shape[1]],
             "packed feature shape mismatch")
    _require(str(values.dtype) == metadata.get("feature_dtype") and str(values.dtype) in SUPPORTED_DTYPES,
             "packed feature dtype mismatch")
    _require(bool(torch.isfinite(values).all()), "non-finite packed features")
    output = torch.zeros(shape, dtype=values.dtype, device="cpu")
    output.index_copy_(0, positions.cpu(), values.detach().cpu())
    return {"video_prompt_embeds": output, "prompt_attention_mask": mask.detach().cpu().clone()}


def _save_new(value, output):
    output = Path(output)
    _require(not output.exists(), "output already exists; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(filename)
    try:
        torch.save(value, temporary)
        # Hard-link publish refuses to replace an existing file, including races.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pack", "unpack"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args()
    _require(args.input.resolve() != args.output.resolve(), "original cache must not be overwritten")
    contract = json.loads(args.contract.read_text())
    cached = torch.load(args.input, map_location="cpu", weights_only=True)
    if args.mode == "pack":
        # Deliberately extract video fields; audio caches are untouched in input.
        result = pack_text_features(cached["video_prompt_embeds"], cached["prompt_attention_mask"], contract=contract)
    else:
        result = unpack_text_features(cached, expected_contract=contract)
    _save_new(result, args.output)
    print(json.dumps({"output": str(args.output), "mode": args.mode,
                      "scope": "video-only storage; full sequence restored before connector; GPU equivalence gate required"}))


if __name__ == "__main__":
    main()
