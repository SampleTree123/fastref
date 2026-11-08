"""Bounded, file-backed Qwen -> LTX editing loop.

Borrowed organization: official Qwen-Agent FnCallAgent's tool schemas, bounded
model/tool loop and tool-result feedback. Model formatting stays with the local
official Transformers template. This is a task-specific harness, not a claim of
running Qwen-Agent itself. No ground-truth labels are accepted by its entry point.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
from media_inputs import atomic_json, sha_file
from contracts import (CONTRACT_VERSION, canonical, digest, parse_tool_calls,
                       assistant_call, public_history, validate_public_chain,
                       card_tools, selection_tools, edit_tools, review_tools)

PROMPTS = ROOT / "configs/editing_agent/prompts"
DEFAULTS = {"seed": 42, "max_inspect_calls": 1, "max_inspect_videos": 3,
            "max_corrections": 1, "observe_edit_references": False,
            "observe_review_references": True}


class StageFailure(RuntimeError):
    pass


def settings(config=None):
    result = {**DEFAULTS, **(config or {})}
    if result["max_corrections"] != 1 or result["max_inspect_calls"] not in (0, 1):
        raise ValueError("This experiment permits one correction and at most one inspection")
    if not 1 <= result["max_inspect_videos"] <= 3:
        raise ValueError("Inspect one to three history videos")
    return result


def prompt(name, config=None):
    directory = Path((config or {}).get("prompts_dir", PROMPTS))
    return (directory / f"{name}.txt").read_text().strip()


def video_block(path):
    return {"type": "video", "path": str(Path(path).resolve()), "media_role": "history"}


def reference_blocks(references):
    blocks = []
    for ref in references:
        blocks.append({"type": "text", "text": f"Editing reference {ref['reference_id']}: {ref['type']}, {ref.get('view', '')}."})
        motion = ref["type"] in ("motion", "action", "video") or Path(ref["path"]).suffix.lower() in (".mp4", ".mov", ".mkv")
        blocks.append({"type": "video" if motion else "image", "path": str(Path(ref["path"]).resolve()),
                       "media_role": "motion_reference" if motion else "image_reference"})
    return blocks


def _result_call(result, tools):
    # Always reparse raw output, even if a backend also supplied parsed arguments.
    return parse_tool_calls(result["raw_output"], tools)[0]


def _attempt(messages, tools, backend, output, stage, counters, validate=None):
    """One invalid model output may be corrected; never infer a source fallback."""
    output = Path(output)
    while True:
        number = counters.get("calls", 0)
        counters["calls"] = number + 1
        result = backend.call(messages, tools)
        record = {"stage": stage, "messages": deepcopy(messages), "tools": deepcopy(tools), **result}
        try:
            call = _result_call(result, tools)
            if validate:
                validate(call)
            record.update(status="valid", parsed_call=call)
            atomic_json(output / f"{stage}_call_{number:02d}.json", record)
            return call
        except (ValueError, TypeError, KeyError) as exc:
            record.update(status="invalid", validation_error=str(exc))
            atomic_json(output / f"{stage}_call_{number:02d}.json", record)
            if counters.get("corrections", 0) >= 1:
                raise StageFailure(f"{stage}: invalid output after one correction: {exc}") from exc
            counters["corrections"] = counters.get("corrections", 0) + 1
            # Keep the exact model answer for correction, without executing it.
            messages.append({"role": "assistant", "content": result["raw_output"].replace("<|im_end|>", "")})
            messages.append({"role": "user", "content":
                "The previous tool call was not executed. Error: " + str(exc) +
                ". Correct it once using exactly one of the available tools and all required parameters. Do not invent a default ID."})


def observe_video(video_path, video_id, backend, output_dir, config=None):
    """Describe one actual video; optional content-addressed cache holds no IDs."""
    config = settings(config)
    path = Path(video_path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Missing or empty observed video: " + str(path))
    if type(video_id) is not int or video_id < 0:
        raise ValueError("Video ID must be nonnegative")
    media_cache = getattr(backend, "cache", None)
    video_sha = media_cache.content_digest(path) if media_cache is not None else sha_file(path)
    identity = {"video_sha256": video_sha, "backend": getattr(backend, "identity", {"kind": type(backend).__name__}),
                "prompt": prompt("observe", config), "contract": CONTRACT_VERSION}
    cache_root = config.get("observed_card_cache")
    cache_path = Path(cache_root) / (digest(identity) + ".json") if cache_root else None
    content = None
    if cache_path and cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if cached.get("identity") != identity:
            raise ValueError("Observed card cache identity differs")
        from contracts import validate_schema, CARD
        validate_schema(cached["card"], CARD)
        content = cached["card"]
    if content is None:
        messages = [{"role": "system", "content": prompt("observe", config)},
                    {"role": "user", "content": [{"type": "text", "text": "Describe this video's actual visible contents."}, video_block(path)]}]
        call = _attempt(messages, card_tools(), backend, output_dir, "observe", {})
        content = call["arguments"]
        if cache_path:
            atomic_json(cache_path, {"identity": identity, "card": content})
    card = {"video_id": video_id, "video_path": str(path), **content,
            "observation_identity": digest(identity)}
    atomic_json(Path(output_dir) / "card.json", card)
    return card


def build_observed_history(public_items, backend, output_dir, config=None):
    """Prepare understanding-only history from actual prior videos, not labels."""
    history = {"videos": [], "edits": []}
    for i, item in enumerate(public_items):
        if item["video_id"] != i or (i and not 0 <= item["parent_id"] < i):
            raise ValueError("History IDs/parents must refer to earlier available videos")
        card = observe_video(item["video_path"], i, backend, Path(output_dir) / f"video_{i:03d}", config)
        card["parent_id"] = item.get("parent_id")
        history["videos"].append(card)
        if i:
            history["edits"].append({"source_video_id": item["parent_id"], "output_video_id": i,
                "user_prompt": item["user_prompt"], "editing_prompt": item.get("editing_prompt", item["user_prompt"]),
                "review": {"note": "Existing recorded edit; no generated-vs-source review was performed by this harness."}})
    return history


def choose_and_edit_request(history, turn, backend, output_dir, config=None):
    config = settings(config)
    output = Path(output_dir)
    # The understanding worker naturally holds a list of observed prior videos.
    # Convert only its public fields, never metadata or expected answers.
    if isinstance(history, list):
        items = history
        history = {"videos": items, "edits": [
            {"source_video_id": v["parent_id"], "output_video_id": v["video_id"],
             "user_prompt": v.get("user_prompt", ""),
             "editing_prompt": v.get("editing_prompt", v.get("user_prompt", "")),
             "review": v.get("review", {"note": "Existing recorded history; no comparison review available."})}
            for v in items if v["video_id"] > 0]}
    ids = [v["video_id"] for v in history["videos"]]
    if ids != list(range(len(ids))):
        raise ValueError("History IDs must be consecutive from zero")
    by_id = {v["video_id"]: v for v in history["videos"]}
    messages = [{"role": "system", "content": prompt("select", config)},
                {"role": "user", "content": "History:\n" + canonical(public_history(history)) +
                 "\nCurrent user request:\n" + turn["user_prompt"]}]
    # Do not pass this turn's visual references or their asset descriptions here.
    counters = {}
    tools = selection_tools(ids, allow_inspect=bool(config["max_inspect_calls"]), max_inspect_videos=config["max_inspect_videos"])
    call = _attempt(messages, tools, backend, output, "select", counters)
    inspected = []
    if call["name"] == "inspect_history":
        inspected = call["arguments"]["video_ids"]
        messages.append(assistant_call(call))
        messages.append({"role": "tool", "name": "inspect_history", "content": canonical({"viewed_video_ids": inspected})})
        content = []
        for video_id in inspected:
            content.extend([{"type": "text", "text": f"Existing history video {video_id}:"}, video_block(by_id[video_id]["video_path"])])
        content.append({"type": "text", "text": "Select the source video now. No more inspection calls are available."})
        messages.append({"role": "user", "content": content})
        call = _attempt(messages, selection_tools(ids, allow_inspect=False), backend, output, "select", counters)
    source_id = call["arguments"]["source_video_id"]
    refs = turn["references"]
    ref_ids = [r["reference_id"] for r in refs]
    descriptions = [{k: r[k] for k in ("reference_id", "type", "view") if k in r} for r in refs]
    editing_content = [{"type": "text", "text": canonical({
        "source_video_id": source_id, "user_prompt": turn["user_prompt"], "editing_references": descriptions})}]
    if config["observe_edit_references"]:
        editing_content.extend(reference_blocks(refs))
    edit_messages = [{"role": "system", "content": prompt("edit", config)},
                     {"role": "user", "content": editing_content}]
    def validate_edit(call):
        if call["arguments"]["reference_ids"] != ref_ids:
            raise ValueError("All current editing references must be passed in their original order")
    edit_counters = {}
    edit = _attempt(edit_messages, edit_tools(source_id, refs), backend, output, "edit", edit_counters, validate_edit)
    result = {**edit["arguments"], "inspected_video_ids": inspected,
              "selection_model_calls": counters["calls"], "selection_corrections": counters.get("corrections", 0),
              "editing_model_calls": edit_counters["calls"], "editing_corrections": edit_counters.get("corrections", 0)}
    atomic_json(output / "edit_request.json", result)
    return result


def review_output(source_video, output_video, turn, editing_prompt, backend, output_dir, config=None):
    config = settings(config)
    content = [{"type": "text", "text": "SOURCE BEFORE THIS EDIT:"}, video_block(source_video),
               {"type": "text", "text": "NEW OUTPUT AFTER THIS EDIT:"}, video_block(output_video),
               {"type": "text", "text": canonical({"user_request": turn["user_prompt"], "executed_editing_prompt": editing_prompt})}]
    if config["observe_review_references"]:
        content.extend(reference_blocks(turn["references"]))
    messages = [{"role": "system", "content": prompt("review", config)}, {"role": "user", "content": content}]
    call = _attempt(messages, review_tools(), backend, output_dir, "review", {})
    atomic_json(Path(output_dir) / "review.json", call["arguments"])
    return call["arguments"]


def _identity(chain, backend, config):
    paths = [chain["root_video"]] + [r["path"] for t in chain["turns"] for r in t["references"]]
    media = []
    for item in sorted(set(paths)):
        p = Path(item).resolve(); st = p.stat()
        media.append({"path": str(p), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns})
    return {"contract": CONTRACT_VERSION, "chain": chain, "config": config,
            "backend": getattr(backend, "identity", {"kind": type(backend).__name__}),
            "code": {p.name: sha_file(p) for p in (Path(__file__), Path(__file__).with_name("contracts.py"))},
            "prompts": {name: prompt(name, config) for name in ("observe", "select", "edit", "review")},
            "input_media": media}


def run_chain(public_chain, output_dir, backend, editor, config=None):
    """Run or resume one real chain. Caller continues other chains on failure.

    editor(request) must return status='ok' and video_path. GPU scheduling and
    file-queue transport are owned by the parent launcher, not this module.
    """
    validate_public_chain(public_chain)
    config = settings(config)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    identity = _identity(public_chain, backend, config)
    state_path = output / "state.json"
    with (output / ".chain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["identity"] != identity:
                raise ValueError("Cannot reuse a chain directory after inputs, prompts, code or model settings changed")
            if state["status"] in ("completed", "failed"):
                return state
        else:
            state = {"identity": identity, "chain_id": public_chain["chain_id"], "status": "running",
                     "started_unix": time.time(), "history": {"videos": [], "edits": []}, "completed_turns": 0}
            atomic_json(state_path, state)
        try:
            if not state["history"]["videos"]:
                card = observe_video(public_chain["root_video"], 0, backend, output / "initial", config)
                card["parent_id"] = None
                state["history"]["videos"].append(card)
                atomic_json(state_path, state)
            for turn in public_chain["turns"][state["completed_turns"]:]:
                turn_id = turn["turn_id"]
                directory = output / f"turn_{turn_id:02d}"
                progress_path = directory / "progress.json"
                progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"turn_id": turn_id}
                if "request" not in progress:
                    progress["request"] = choose_and_edit_request(state["history"], turn, backend, directory, config)
                    atomic_json(progress_path, progress)
                request = progress["request"]
                source_id = request["source_video_id"]
                source = state["history"]["videos"][source_id]["video_path"]
                if "generation" not in progress:
                    generated = editor({"request_id": public_chain["chain_id"] + f"_turn_{turn_id:02d}",
                        "source_video": source, "prompt": request["prompt"],
                        "references": deepcopy(turn["references"]), "output_dir": str(directory / "generation"),
                        "seed": config["seed"]})
                    if generated.get("status") != "ok":
                        raise StageFailure("LTX tool failed: " + str(generated.get("error", generated)))
                    if not Path(generated["video_path"]).is_file():
                        raise StageFailure("LTX tool returned no usable video")
                    progress["generation"] = generated
                    atomic_json(progress_path, progress)
                video_path = str(Path(progress["generation"]["video_path"]).resolve())
                if not Path(video_path).is_file() or Path(video_path).stat().st_size == 0:
                    raise StageFailure("Saved generated video is missing or empty")
                if "review" not in progress:
                    progress["review"] = review_output(source, video_path, turn, request["prompt"], backend, directory, config)
                    atomic_json(progress_path, progress)
                record = progress["review"]
                card = {"video_id": turn_id, "parent_id": source_id, "video_path": video_path, **record["card"]}
                edit = {"source_video_id": source_id, "output_video_id": turn_id,
                        "user_prompt": turn["user_prompt"], "editing_prompt": request["prompt"],
                        "reference_ids": request["reference_ids"], "review": record["review"]}
                atomic_json(output / "videos" / f"video_{turn_id:03d}.json", card)
                atomic_json(output / "edits" / f"edit_{turn_id:03d}.json", edit)
                state["history"]["videos"].append(card)
                state["history"]["edits"].append(edit)
                state["completed_turns"] = turn_id
                atomic_json(state_path, state)
                print(canonical({"event": "editing_turn_complete", "chain_id": public_chain["chain_id"],
                                 "turn_id": turn_id, "source_video_id": source_id, "video_path": video_path}), flush=True)
            state["status"] = "completed"
        except Exception as exc:
            state.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        finally:
            state["finished_unix"] = time.time()
            atomic_json(state_path, state)
        return state
