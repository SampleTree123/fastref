#!/usr/bin/env python3
"""Check evaluation examples against local official template, CPU only.

Owner: ../../datasets/history_selection_original/README.md
No media decoding, model loading, inference, or training.
"""
import hashlib
import json

from transformers import AutoProcessor

from evaluation_inputs import ROOT, parse_tool_output, template_kwargs


def main():
    directory = ROOT / "datasets/history_selection_original"
    inventory = json.loads((directory / "inventory.json").read_text())
    wanted = set(inventory["category_example_sample_ids"].values())
    rows = []
    with (directory / "eval.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["sample_id"] in wanted or (row["audit"]["chain_id"] == 8 and row["audit"]["edit_round"] in (3, 5)):
                rows.append(row)
    processor = AutoProcessor.from_pretrained(ROOT / "models/Qwen3.8-27B", local_files_only=True)
    checks = []
    for row in rows:
        model_input = row["model_input"]
        kwargs = template_kwargs(model_input)
        prefix = processor.apply_chat_template(model_input["messages"], **kwargs, tokenize=False, add_generation_prompt=True)
        full = processor.apply_chat_template(model_input["messages"] + [row["supervision"]["assistant_message"]],
                                              **kwargs, tokenize=False, add_generation_prompt=False)
        assert full.startswith(prefix)
        assert "<think>\n\n</think>" in prefix
        assert all(key not in prefix for key in ("/scratch/", "current_source_node_id", "current_target_node_id", "audit", "supervision"))
        answer = full[len(prefix):]
        parsed = parse_tool_output(answer, row["audit"]["edit_round"])
        assert parsed["valid"] and parsed["source_video_id"] == row["supervision"]["source_video_id"]
        blocks = model_input["messages"][1]["content"]
        assert prefix.count("<|image_pad|>") == sum(block["type"] == "image" for block in blocks)
        assert prefix.count("<|video_pad|>") == sum(block["type"] == "video" for block in blocks)
        checks.append({"sample_id": row["sample_id"], "selection_kind": row["audit"]["selection_kind"],
                       "prefix_preserved": True, "official_answer": answer,
                       "history_videos": sum(block.get("media_role") == "history" for block in blocks),
                       "image_count": prefix.count("<|image_pad|>"), "video_count": prefix.count("<|video_pad|>")})
    result = {"status": "passed", "scope": "official local processor chat template only; no media decode or full vision expansion, no model inference",
              "eval_jsonl_sha256": inventory["output"]["sha256"],
              "official_template_sha256": hashlib.sha256((ROOT / "models/Qwen3.8-27B/chat_template.jinja").read_bytes()).hexdigest(),
              "checks": checks}
    (directory / "template_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "sample_count": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
