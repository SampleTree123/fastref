"""Input-only schemas and validation for the editing agent; no model imports.

Owner: scripts/editing_agent/README.md. Tool envelope uses the local official
Qwen chat template. Qwen-Agent inspires the bounded dispatch loop; this module
does not claim to run that framework or silently repair malformed arguments.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import html
import json
import re

ATTRIBUTES = ("identity", "hairstyle", "hair_color", "clothing", "action", "camera", "scene")
CONTRACT_VERSION = "editing_agent_memory_tools_v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def obj(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}


def tool(name, description, parameters):
    return {"type": "function", "function": {
        "name": name, "description": description, "parameters": parameters}}


STRING = {"type": "string", "minLength": 1, "maxLength": 2000}
STRING_LIST = {"type": "array", "items": STRING, "maxItems": 20}
CARD = obj({"observed": obj({key: STRING for key in ATTRIBUTES}),
            "uncertain": STRING_LIST})
REVIEW = obj({"requested_change": STRING, "observed_change": STRING,
              "fulfillment": {"type": "string", "enum": ["fulfilled", "partial", "not_fulfilled", "uncertain"]},
              "unexpected_changes": STRING_LIST, "uncertain": STRING_LIST})


def card_tools():
    return [tool("record_video_card", "Record only the visible contents of this video, not inferred editing intentions.", CARD)]


def selection_tools(video_ids, *, allow_inspect=True, max_inspect_videos=3):
    ids = sorted(video_ids)
    if not ids or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("History requires nonnegative integer video IDs")
    result = [tool("select_source_video", "Select one existing video as the source of this edit. Normal continuation means the latest ID.",
                   obj({"source_video_id": {"type": "integer", "enum": ids}}))]
    if allow_inspect:
        result.append(tool("inspect_history", "View up to three existing videos if the written history does not distinguish the requested source. Do not inspect editing reference assets to choose history.",
                           obj({"video_ids": {"type": "array", "items": {"type": "integer", "enum": ids},
                                               "minItems": 1, "maxItems": max_inspect_videos, "uniqueItems": True}})))
    return result


def edit_tools(source_id, references):
    ids = [r["reference_id"] for r in references]
    items = {"type": "string"}
    if ids:
        items["enum"] = ids
    return [tool("edit_video", "Edit the selected source using an English editing-only instruction and all current editing references in their original order.",
                 obj({"source_video_id": {"type": "integer", "enum": [source_id]},
                      "prompt": {"type": "string", "minLength": 1, "maxLength": 6000},
                      "reference_ids": {"type": "array", "items": items,
                                        "minItems": len(ids), "maxItems": len(ids), "uniqueItems": True}}))]


def review_tools():
    return [tool("record_edit_result", "Record the output video's own visible properties separately from source/output comparison and instruction fulfillment.",
                 obj({"card": CARD, "review": REVIEW}))]


def validate_schema(value, schema, path="arguments"):
    """Small supported JSON-schema subset, strict booleans vs integers."""
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        properties = schema["properties"]
        if set(schema.get("required", [])) - set(value):
            raise ValueError(f"{path} is missing required fields")
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError(f"{path} contains unknown fields")
        for key, item in value.items():
            if key in properties:
                validate_schema(item, properties[key], path + "." + key)
    elif kind == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array")
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 1000000):
            raise ValueError(f"{path} has an invalid number of items")
        if schema.get("uniqueItems") and len({canonical(x) for x in value}) != len(value):
            raise ValueError(f"{path} contains duplicate items")
        for item in value:
            validate_schema(item, schema["items"], path + "[]")
    elif kind == "integer":
        if type(value) is not int:
            raise ValueError(f"{path} must be an integer")
    elif kind == "string":
        if not isinstance(value, str) or not schema.get("minLength", 0) <= len(value.strip()) <= schema.get("maxLength", 1000000):
            raise ValueError(f"{path} must be a nonempty bounded string")
    else:
        raise ValueError("Unsupported schema kind: " + str(kind))
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is not one of the available choices")


def _json_object(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError("Duplicate JSON field: " + k)
        result[k] = v
    return result


def parse_tool_calls(raw, tools):
    """Parse official XML or standard JSON envelopes, never a naked answer.

    Text outside the one envelope is retained in raw logs but is not executed.
    Known completed thinking blocks are removed before parsing. No bracket,
    spelling, type or unavailable-ID repair is performed here.
    """
    if not isinstance(raw, str):
        raise ValueError("Model output is not text")
    body = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    body = body.replace("<|im_end|>", "").strip()
    matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", body, re.S))
    if len(matches) != 1:
        raise ValueError("Require exactly one complete tool_call envelope")
    outside = body[:matches[0].start()] + body[matches[0].end():]
    if any(tag in outside for tag in ("<tool_call", "</tool_call", "<function=", "<parameter=", "<think>")):
        raise ValueError("Unclosed or extra tool markup")
    payload = matches[0].group(1)
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    if payload.lstrip().startswith("{"):
        parsed = json.loads(payload, object_pairs_hook=_json_object)
        if set(parsed) != {"name", "arguments"}:
            raise ValueError("JSON tool call requires only name and arguments")
        name, arguments = parsed["name"], parsed["arguments"]
    else:
        function = re.fullmatch(r"<function=([a-zA-Z_][a-zA-Z0-9_]*)>\s*(.*?)\s*</function>", payload, re.S)
        if function is None:
            raise ValueError("Invalid function envelope")
        name = function.group(1)
        if name not in schemas:
            raise ValueError("Unavailable tool: " + name)
        parameters = function.group(2)
        fields = list(re.finditer(r"<parameter=([a-zA-Z_][a-zA-Z0-9_]*)>\s*(.*?)\s*</parameter>", parameters, re.S))
        if re.sub(r"<parameter=[a-zA-Z_][a-zA-Z0-9_]*>.*?</parameter>", "", parameters, flags=re.S).strip():
            raise ValueError("Unexpected parameter text")
        arguments = {}
        for field in fields:
            key, value = field.group(1), html.unescape(field.group(2).strip())
            if key in arguments or key not in schemas[name]["properties"]:
                raise ValueError("Duplicate or unknown parameter: " + key)
            kind = schemas[name]["properties"][key]["type"]
            if kind == "string":
                if value.startswith('"') and value.endswith('"'):
                    value = json.loads(value)
            else:
                value = json.loads(value, object_pairs_hook=_json_object)
            arguments[key] = value
    if name not in schemas:
        raise ValueError("Unavailable tool: " + str(name))
    validate_schema(arguments, schemas[name])
    return [{"name": name, "arguments": arguments}]


def assistant_call(call):
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": deepcopy(call)}]}


def public_history(history):
    """Allowlist model-visible text; no paths, ground truth attributes or labels."""
    return {"videos": [{k: deepcopy(v[k]) for k in
                        ("video_id", "parent_id", "observed", "uncertain") if k in v}
                       for v in history["videos"]],
            "edits": [{k: deepcopy(e[k]) for k in
                       ("source_video_id", "output_video_id", "user_prompt", "editing_prompt", "review") if k in e}
                      for e in history.get("edits", [])]}


def validate_public_chain(chain):
    if not isinstance(chain, dict) or set(chain) - {"schema", "chain_id", "split", "root_video", "turns"}:
        raise ValueError("Public chain contains unknown fields; never pass private labels")
    if "schema" in chain and chain["schema"] != "editing_agent_chain_v1":
        raise ValueError("Unknown public chain schema")
    if not isinstance(chain.get("chain_id"), str) or not isinstance(chain.get("root_video"), str):
        raise ValueError("Chain ID and root video must be strings")
    turns = chain.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("Chain requires turns")
    for i, turn in enumerate(turns, 1):
        if set(turn) - {"turn_id", "user_prompt", "references"}:
            raise ValueError("Public turn contains unknown fields")
        if type(turn.get("turn_id")) is not int or turn["turn_id"] != i or not isinstance(turn.get("user_prompt"), str):
            raise ValueError("Turn IDs must be consecutive and prompts must be strings")
        refs = turn.get("references")
        if not isinstance(refs, list) or len({r["reference_id"] for r in refs}) != len(refs):
            raise ValueError("References require unique IDs")
        for ref in refs:
            if set(ref) - {"reference_id", "type", "path", "view"} or not all(isinstance(ref.get(k), str) for k in ("reference_id", "type", "path")):
                raise ValueError("Reference fields must only identify current editing assets")
