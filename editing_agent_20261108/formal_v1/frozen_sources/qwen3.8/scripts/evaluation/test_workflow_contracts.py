"""CPU-only regression checks for immutable aggregation and bounded orchestration.

Run: python -m unittest discover -s scripts/evaluation -p test_workflow_contracts.py -v
No real subprocess launch, Slurm command, model loading, or GPU use.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from evaluation_inputs import score_prediction
from media_inputs import sha_file
import run_allocation
import run_workflow
from run_model_eval import digest
from summarize_results import summarize


def tool_call(value):
    return ("<tool_call>\n<function=select_source_video>\n<parameter=source_video_id>\n"
            f"{value}\n</parameter>\n</function>\n</tool_call><|im_end|>")


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="qwen_workflow_test_", dir="/tmp")
        self.root = Path(self.temporary.name)
        self.rows = [{"sample_id": f"sample_{index}", "audit": {
            "original_split": "test", "selection_kind": "latest", "edit_round": 2},
            "supervision": {"source_video_id": -1, "resolved_parent_video_id": 1}}
            for index in range(3)]
        self.dataset = self.root / "dataset.jsonl"
        self.dataset.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        self.manifests = {}
        for profile in ("low", "high"):
            directory = self.root / profile
            (directory / "records").mkdir(parents=True)
            identity = {"dataset_sha256": sha_file(self.dataset), "config": {"name": profile},
                        "prepare_only": False, "code": {"example": "fixed"}}
            manifest = {"identity": identity, "run_fingerprint": digest(identity)}
            (directory / "expected_run.json").write_text(json.dumps(manifest))
            self.manifests[profile] = manifest

    def tearDown(self):
        self.temporary.cleanup()

    def record(self, profile_name, index, raw=None, **updates):
        row = self.rows[index]
        raw = tool_call(-1) if raw is None else raw
        record = {"sample_id": row["sample_id"], "profile": profile_name,
                  "run_fingerprint": self.manifests[profile_name]["run_fingerprint"],
                  "row_sha256": digest(row), "status": "ok", "raw_output": raw,
                  "scores": score_prediction(raw, row), "total_seconds": 2,
                  "generation_seconds": 1, "prepare_seconds": 1, "gpu_peak_allocated_bytes": 1024}
        record.update(updates)
        path = self.root / profile_name / "records" / (row["sample_id"] + ".json")
        path.write_text(json.dumps(record))
        return path

    def test_partial_invalid_and_error_all_remain_visible(self):
        self.record("low", 0)
        self.record("low", 1, raw="malformed")
        self.record("low", 2, status="error", raw_output=None, scores=None)
        self.record("high", 0)
        result = summarize(self.dataset, self.root)
        low = result["profiles"]["low"]["groups"]["all"]
        high = result["profiles"]["high"]["groups"]["all"]
        self.assertEqual((low["answered"], low["runtime_error"], low["strict_correct"]), (2, 1, 1))
        self.assertEqual(low["invalid_format_or_range"], 1)
        self.assertEqual(low["strict_accuracy_observed_including_errors"], 1 / 3)
        self.assertEqual(high["missing"], 2)
        self.assertFalse(result["complete"])
        self.assertEqual(result["paired"]["answered_in_both"], 1)

    def test_single_stale_fingerprint_is_incompatible_without_mixing(self):
        for profile in ("low", "high"):
            for index in range(3):
                self.record(profile, index, run_fingerprint="all_records_are_stale")
        result = summarize(self.dataset, self.root)
        self.assertFalse(result["complete"])
        self.assertFalse(result["mixed_fingerprints"])
        for profile in ("low", "high"):
            count = result["profiles"][profile]["groups"]["all"]
            self.assertEqual((count["answered"], count["incompatible"], count["strict_correct"]), (0, 3, 0))

    def test_mixed_profile_row_hash_and_wrong_scores_are_not_credited(self):
        self.record("low", 0)
        self.record("low", 1, run_fingerprint="old")
        self.record("low", 2, profile="high")
        self.record("high", 0, row_sha256="wrong")
        self.record("high", 1, raw="invalid", scores={"valid": True, "strict_correct": True, "resolved_correct": True})
        self.record("high", 2)
        result = summarize(self.dataset, self.root)
        self.assertTrue(result["mixed_fingerprints"])
        for profile in ("low", "high"):
            count = result["profiles"][profile]["groups"]["all"]
            self.assertEqual((count["answered"], count["incompatible"]), (1, 2))
        self.assertFalse(result["complete"])

    def test_missing_or_changed_dataset_manifest_never_accepts_records(self):
        self.record("low", 0)
        self.record("high", 0)
        (self.root / "low/expected_run.json").unlink()
        manifest = self.manifests["high"]
        manifest["identity"]["dataset_sha256"] = "wrong_dataset"
        manifest["run_fingerprint"] = digest(manifest["identity"])
        (self.root / "high/expected_run.json").write_text(json.dumps(manifest))
        result = summarize(self.dataset, self.root)
        for profile in ("low", "high"):
            count = result["profiles"][profile]["groups"]["all"]
            self.assertEqual((count["answered"], count["incompatible"], count["missing"]), (0, 1, 2))
        self.assertFalse(result["complete"])

    def test_duplicate_expected_ids_rejected(self):
        self.dataset.write_text(json.dumps(self.rows[0]) + "\n" + json.dumps(self.rows[0]) + "\n")
        with self.assertRaises(ValueError):
            summarize(self.dataset, self.root)

    def test_all_compatible_terminal_records_finish_with_errors_counted(self):
        for profile in ("low", "high"):
            self.record(profile, 0)
            self.record(profile, 1, raw=tool_call(1))
            self.record(profile, 2, status="error", raw_output=None, scores=None)
        result = summarize(self.dataset, self.root)
        self.assertTrue(result["complete"])
        self.assertFalse(result["all_inference_succeeded"])
        count = result["profiles"]["low"]["groups"]["all"]
        self.assertEqual(count["strict_accuracy_observed_including_errors"], 1 / 3)
        self.assertEqual(count["resolved_accuracy_observed_including_errors"], 2 / 3)

    def test_invalid_manifest_config_is_reported_instead_of_crashing(self):
        manifest = self.manifests["low"]
        manifest["identity"]["config"] = []
        manifest["run_fingerprint"] = digest(manifest["identity"])
        (self.root / "low/expected_run.json").write_text(json.dumps(manifest))
        self.record("low", 0)
        result = summarize(self.dataset, self.root)
        self.assertEqual(result["profiles"]["low"]["manifest_error"], "manifest_profile_mismatch")
        self.assertEqual(result["profiles"]["low"]["groups"]["all"]["incompatible"], 1)


class Child:
    def __init__(self, stubborn=False):
        self.returncode = None
        self.stubborn = stubborn
        self.terminated = self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.stubborn:
            self.returncode = -signal.SIGTERM

    def kill(self):
        self.killed = True
        self.returncode = -signal.SIGKILL

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake child", timeout)
        return self.returncode


class AllocationTests(unittest.TestCase):
    def test_signal_during_startup_prevents_later_replicas(self):
        handlers, children, streams = {}, [], []

        def register(sig, handler):
            old = handlers.get(sig, signal.SIG_DFL)
            handlers[sig] = handler
            return old

        def spawn(*args, **kwargs):
            children.append(Child())
            streams.append(kwargs["stdout"])
            return children[-1]

        def sleep(_):
            handlers[signal.SIGTERM](signal.SIGTERM, None)

        with tempfile.TemporaryDirectory(dir="/tmp") as directory, \
             patch.object(sys, "argv", ["run_allocation", "--profile", "low", "--output", directory]), \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}), \
             patch.object(run_allocation.signal, "signal", side_effect=register), \
             patch.object(run_allocation.subprocess, "Popen", side_effect=spawn), \
             patch.object(run_allocation.time, "sleep", side_effect=sleep):
            code = run_allocation.main()
        self.assertEqual(code, 128 + signal.SIGTERM)
        self.assertEqual(len(children), 1)
        self.assertTrue(children[0].terminated)
        self.assertIsNotNone(children[0].returncode)
        self.assertTrue(all(stream.closed for stream in streams))

    def test_stubborn_child_is_killed_and_reaped(self):
        child = Child(stubborn=True)
        run_allocation.terminate_children([child], grace_seconds=0)
        self.assertTrue(child.terminated)
        self.assertTrue(child.killed)
        self.assertEqual(child.returncode, -signal.SIGKILL)


class WorkflowTests(unittest.TestCase):
    def test_expiry_continues_only_to_second_authorized_allocation_then_stops_errors(self):
        calls, commands = {}, []

        def state(job):
            calls[job] = calls.get(job, 0) + 1
            if job == 193061:
                return "RUNNING" if calls[job] == 1 else "FINISHED"
            return "PENDING" if calls[job] == 1 else "RUNNING"

        def spawn(command, **kwargs):
            commands.append(command)
            child = Child()
            child.returncode = 143 if len(commands) == 1 else 2
            return child

        with tempfile.TemporaryDirectory(dir="/tmp") as directory, \
             patch.object(sys, "argv", ["run_workflow", "--output", directory]), \
             patch.object(run_workflow, "job_state", side_effect=state), \
             patch.object(run_workflow.subprocess, "Popen", side_effect=spawn), \
             patch.object(run_workflow.time, "sleep"):
            code = run_workflow.main()
            saved = json.loads((Path(directory) / "state.json").read_text())
        self.assertEqual(code, 2)
        self.assertEqual(len(commands), 2)
        self.assertIn("--jobid=193061", commands[0])
        self.assertIn("--jobid=193287", commands[1])
        self.assertEqual(saved["status"], "needs_error_review")
        self.assertEqual(len(saved["attempts"]), 2)
        self.assertTrue(all(command[0] == "srun" for command in commands))


if __name__ == "__main__":
    unittest.main()
