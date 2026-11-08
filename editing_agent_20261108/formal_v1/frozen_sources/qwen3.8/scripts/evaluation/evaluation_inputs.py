"""Shared input-only assembly and strict official Qwen tool-output scoring.

No model imports or media decoding. Paths are loader metadata, never rendered text.
Owner: datasets/history_selection_original/README.md
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
TASK_PROMPT = ROOT / "configs/evaluation/task_prompt.txt"
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "select_source_video",
        "description": "Choose the existing baseline video for the current edit.",
        "parameters": {
            "type": "object",
            "properties": {"source_video_id": {"type": "integer"}},
            "required": ["source_video_id"],
            "additionalProperties": False,
        },
    },
}


def tools_for_history(available_video_count: int) -> list[dict]:
    if available_video_count < 1:
        raise ValueError("At least the original video must exist")
    tool = deepcopy(TOOL_SCHEMA)
    tool["function"]["parameters"]["properties"]["source_video_id"]["enum"] = [
        -1, *range(available_video_count)
    ]
    return [tool]


def assemble_model_input(chain: dict, round_number: int, dataset_root: Path,
                         task_prompt: str | None = None) -> dict:
    """Use only legal history plus current prompt/references, never current labels.

    This function intentionally does not read current source/target fields or
    source_selection. A target file reused in legal history remains visible there.
    """
    if not 1 <= round_number <= len(chain["turns"]):
        raise ValueError("Round outside chain")
    turns = chain["turns"]
    current = turns[round_number - 1]
    history = chain["history_node_ids"][:round_number]
    content = []

    def text(value):
        content.append({"type": "text", "text": value})

    def video(node):
        path = dataset_root / "dataset_v1/nodes/nodes" / f"node_{node}.mp4"
        content.append({"type": "video", "path": str(path.resolve()), "media_role": "history"})

    def references(refs):
        if not refs:
            text("Editing references: none.")
        for number, ref in enumerate(refs, 1):
            path = Path(ref.get("absolute_path") or dataset_root / "dataset_v1" / ref["path"])
            view = ref.get("view")
            text(f"Editing reference {number}: {ref['type']}" + (f", {view}." if view else "."))
            content.append({"type": "video" if ref["type"] == "motion" else "image",
                            "path": str(path.resolve()),
                            "media_role": "motion_reference" if ref["type"] == "motion" else "image_reference"})

    text("Available history. Video 0 is the original video.")
    video(history[0])
    for index, past in enumerate(turns[:round_number - 1], 1):
        text(f"Completed edit {index}: started from video {past['source_history_index']}. "
             f"The user instruction was: {past['prompt']['en']}")
        references(past["references"])
        text(f"Result of completed edit {index}: video {index}.")
        video(history[index])
    text(f"Current request: {current['prompt']['en']}")
    references(current["references"])
    return {"messages": [{"role": "system", "content": task_prompt or TASK_PROMPT.read_text().strip()},
                         {"role": "user", "content": content}],
            "tools": tools_for_history(len(history)), "enable_thinking": False,
            "add_vision_id": False}


def build_messages(model_input: dict) -> list[dict]:
    """Return messages only; caller handles path-to-official-media conversion."""
    return deepcopy(model_input["messages"])


def template_kwargs(model_input: dict) -> dict:
    """Pass with official processor.apply_chat_template, add_generation_prompt=True."""
    return {key: deepcopy(model_input[key]) for key in
            ("tools", "enable_thinking", "add_vision_id")}


def assistant_tool_call(source_video_id: int) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
        "name": "select_source_video", "arguments": {"source_video_id": source_video_id}}}]}


# Full match: exactly one official XML call; no prose, extra arguments, duplicate
# calls, JSON substitutions, or non-integer values. The optional end token is the
# official assistant terminator when decoding with skip_special_tokens=False.
_TOOL_OUTPUT = re.compile(
    r"\s*<tool_call>\s*<function=select_source_video>\s*"
    r"<parameter=source_video_id>\s*(-1|0|[1-9][0-9]*)\s*</parameter>\s*"
    r"</function>\s*</tool_call>\s*(?:<\|im_end\|>\s*)?"
)


def parse_tool_output(raw: str, available_video_count: int | None = None) -> dict:
    """Parse only generated continuation, not the prompt; invalid never gets credit."""
    if not isinstance(raw, str):
        return {"valid": False, "source_video_id": None, "error": "non_text_output"}
    match = _TOOL_OUTPUT.fullmatch(raw)
    if match is None:
        return {"valid": False, "source_video_id": None, "error": "invalid_tool_format"}
    value = int(match.group(1))
    if available_video_count is not None and not -1 <= value < available_video_count:
        return {"valid": False, "source_video_id": value, "error": "out_of_range"}
    return {"valid": True, "source_video_id": value, "error": None}


def score_prediction(raw: str, row: dict) -> dict:
    count = row["audit"]["edit_round"]
    parsed = parse_tool_output(raw, count)
    label = row["supervision"]["source_video_id"]
    parent = row["supervision"]["resolved_parent_video_id"]
    value = parsed["source_video_id"]
    resolved = (count - 1 if value == -1 else value) if parsed["valid"] else None
    return {**parsed, "strict_correct": bool(parsed["valid"] and value == label),
            "resolved_correct": bool(parsed["valid"] and resolved == parent),
            "resolved_parent_video_id": resolved,
            "runtime_fallback_source_video_id": value if parsed["valid"] else -1}
