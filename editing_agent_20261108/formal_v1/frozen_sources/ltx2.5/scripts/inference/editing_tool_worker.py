#!/usr/bin/env python3
"""Persistent file-queue adapter for native LTX editing. See editing_tool_README.md."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from project_layout import ROOT, add_script_paths
add_script_paths()
from editing_tool_runtime import NativeVideoEditor, atomic_json, digest, normalize_request


def process_request(path, queue, editor):
    """One bounded call. The caller decides whether to issue a corrected request."""
    raw = json.loads(path.read_text())
    response_path = queue / "responses" / path.name
    request_sha = digest(raw)
    if response_path.exists():
        old = json.loads(response_path.read_text())
        if old.get("request_sha256") != request_sha or old.get("worker_identity_sha256") != editor.identity_sha:
            raise ValueError("Request ID reused with changed input or worker configuration")
        if old.get("status") == "error":
            return old
        # Re-check input file contents and the committed output before reuse.
        # generate() returns immediately when the receipt still matches.
    try:
        normalized = normalize_request(raw)
        if path.stem != normalized["request_id"]:
            raise ValueError("Request filename must equal request_id")
        response = editor.generate(raw)
    except Exception as error:
        response = {"request_id": path.stem, "status": "error", "error_type": type(error).__name__,
                    "error": str(error), "traceback": traceback.format_exc(), "finished_unix": time.time()}
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    response.update(request_sha256=request_sha, worker_identity_sha256=editor.identity_sha)
    atomic_json(response_path, response)
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--ref-cfg", type=float, default=1.0)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    args = parser.parse_args()
    queue = args.queue.resolve()
    queue.mkdir(parents=True, exist_ok=True)
    for name in ("requests", "responses"):
        (queue / name).mkdir(exist_ok=True)
    lock = (queue / ".worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (queue / "STOP").exists():
        raise SystemExit("Queue STOP exists; use a new queue or explicitly remove it before restart")
    atomic_json(queue / "ready.json", {"status": "loading", "pid": os.getpid(), "started_unix": time.time()})
    try:
        editor = NativeVideoEditor(args.config, args.checkpoint, args.cache or queue / "encoding_cache",
                                   steps=args.steps, ref_cfg=args.ref_cfg)
        old = queue / "worker_identity.json"
        if old.exists() and json.loads(old.read_text()) != editor.identity:
            raise ValueError("Queue contains a different frozen worker identity; use a new queue")
        atomic_json(old, editor.identity)
        atomic_json(queue / "ready.json", {"status": "ready", "pid": os.getpid(),
                    "worker_identity_sha256": editor.identity_sha, "ready_unix": time.time()})
        print(json.dumps({"event": "ready", "queue": str(queue), "identity": editor.identity_sha}), flush=True)
        observed = {}
        while not (queue / "STOP").exists():
            for request in sorted((queue / "requests").glob("*.json")):
                stat = request.stat()
                signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                if observed.get(request.name) == signature:
                    continue
                if (queue / "STOP").exists():
                    break
                result = process_request(request, queue, editor)
                observed[request.name] = signature
                print(json.dumps({"event": "response", "request_id": request.stem,
                                  "status": result["status"], "video_path": result.get("video_path"),
                                  "timings": result.get("timings"), "error": result.get("error")}), flush=True)
            time.sleep(args.poll_seconds)
        atomic_json(queue / "ready.json", {"status": "stopped", "pid": os.getpid(), "finished_unix": time.time()})
    except BaseException as error:
        atomic_json(queue / "ready.json", {"status": "error", "error": str(error),
                    "traceback": traceback.format_exc(), "finished_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
