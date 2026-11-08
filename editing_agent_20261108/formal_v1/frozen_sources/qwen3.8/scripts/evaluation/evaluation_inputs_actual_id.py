"""Actual non-negative history IDs; legacy -1 is invalid and earns no credit."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from evaluation_inputs import (assemble_model_input as legacy_assemble,
                               tools_for_history as legacy_tools, assistant_tool_call,
                               build_messages, template_kwargs,
                               parse_tool_output as legacy_parse)
ROOT = Path(__file__).resolve().parents[2]
TASK_PROMPT = ROOT / "configs/evaluation/actual_id/task_prompt.txt"
CONTRACT = "actual_id_v1"


@lru_cache(maxsize=1)
def task_prompt_text() -> str:
    return TASK_PROMPT.read_text().strip()


def tools_for_history(available_video_count: int) -> list[dict]:
    tools = legacy_tools(available_video_count)
    tools[0]["function"]["parameters"]["properties"]["source_video_id"]["enum"] = list(range(available_video_count))
    return tools


def assemble_model_input(chain: dict, round_number: int, dataset_root: Path,
                         task_prompt: str | None = None) -> dict:
    result = legacy_assemble(chain, round_number, dataset_root,
                             task_prompt or task_prompt_text())
    result["tools"] = tools_for_history(round_number)
    return result


def parse_tool_output(raw: str, available_video_count: int | None = None) -> dict:
    parsed = legacy_parse(raw, available_video_count)
    if parsed["source_video_id"] == -1:
        return {"valid": False, "source_video_id": -1, "error": "negative_id_forbidden"}
    return parsed


def validate_row(row: dict) -> None:
    count = row["audit"]["edit_round"]
    if type(count) is not int or count < 1:
        raise ValueError("edit_round must be a positive integer")
    label = row["supervision"]["source_video_id"]
    parent = row["supervision"]["resolved_parent_video_id"]
    if type(label) is not int or not 0 <= label < count or type(parent) is not int or parent != label:
        raise ValueError("Actual-ID labels must equal the actual parent in available history")
    expected = list(range(count))
    tools = row["model_input"]["tools"]
    if len(tools) != 1 or tools[0]["function"]["parameters"]["properties"]["source_video_id"]["enum"] != expected:
        raise ValueError("Actual-ID tool enum must contain only available non-negative IDs")
    system = row["model_input"]["messages"][0]
    if system["role"] != "system" or system["content"].strip() != task_prompt_text():
        raise ValueError("Actual-ID system prompt differs from the frozen contract")


def score_prediction(raw: str, row: dict) -> dict:
    count = row["audit"]["edit_round"]
    label = row["supervision"]["source_video_id"]
    if type(label) is not int or not 0 <= label < count:
        raise ValueError("Actual-ID scoring refuses a negative or unavailable label")
    parsed = parse_tool_output(raw, count)
    value = parsed["source_video_id"]
    correct = bool(parsed["valid"] and value == label)
    return {**parsed, "strict_correct": correct, "resolved_correct": correct,
            "resolved_parent_video_id": value if parsed["valid"] else None,
            "runtime_fallback_source_video_id": None,
            "selection_contract": CONTRACT}
