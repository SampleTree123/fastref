#!/usr/bin/env python3
"""GPU-only small-tensor equivalence for official downloaded FLA kernels."""
import argparse
import json
from pathlib import Path
import torch
from kernels import get_kernel
from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
from media_inputs import atomic_json

parser = argparse.ArgumentParser()
parser.add_argument("--revision", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(2)
torch.manual_seed(0)
kernel = get_kernel("kernels-community/fla", revision=args.revision)
results = []
with torch.inference_mode():
    for length in (1, 16):
        q, k, v = [torch.randn(1, length, 4, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        g = -torch.rand(1, length, 4, device="cuda")
        beta = torch.rand(1, length, 4, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(1, 4, 128, 128, device="cuda", dtype=torch.float32) if length == 1 else None
        kwargs = dict(g=g, beta=beta, initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
        reference = (modeling.torch_recurrent_gated_delta_rule if length == 1 else
                     modeling.torch_chunk_gated_delta_rule)(q, k, v, **kwargs)
        layer = (kernel.layers.recurrent_gated_delta_rule if length == 1 else
                 kernel.layers.chunk_gated_delta_rule)()
        actual = layer(q, k, v, **kwargs)
        comparisons = []
        for expected, observed in zip(reference, actual):
            difference = (expected.float() - observed.float()).abs()
            comparisons.append({"max_abs": difference.max().item(), "mean_abs": difference.mean().item(),
                                "allclose": torch.allclose(expected.float(), observed.float(), atol=.025, rtol=.025)})
        results.append({"sequence_length": length, "comparisons": comparisons})
report = {"revision": args.revision, "gpu": torch.cuda.get_device_name(0), "results": results,
          "passed": all(c["allclose"] for row in results for c in row["comparisons"])}
atomic_json(args.output, report)
print(json.dumps(report, indent=2))
if not report["passed"]:
    raise SystemExit(2)
