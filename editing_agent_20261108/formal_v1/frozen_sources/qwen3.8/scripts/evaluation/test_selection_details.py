"""Small CPU fixture for supplementary reporting; no media/model/GPU reads."""
import json
from pathlib import Path
import tempfile
import unittest

from evaluation_inputs import score_prediction
from media_inputs import sha_file
from run_model_eval import digest
from summarize_selection_details import summarize_details, render_markdown


def tool(value):
    return f"<tool_call><function=select_source_video><parameter=source_video_id>{value}</parameter></function></tool_call><|im_end|>"


class DetailTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="qwen_details_", dir="/tmp")
        self.root = Path(self.temporary.name)
        self.rows = [{"sample_id": f"s{i}", "audit": {"original_split": "train" if i%2 else "test",
                     "selection_kind": "latest", "edit_round": 2},
                     "supervision": {"source_video_id": -1, "resolved_parent_video_id": 1}} for i in range(5)]
        self.dataset = self.root/"dataset.jsonl"
        self.dataset.write_text("".join(json.dumps(row)+"\n" for row in self.rows))
        self.manifests = {}
        for profile in ("low", "high"):
            (self.root/profile/"records").mkdir(parents=True)
            identity = {"dataset_sha256": sha_file(self.dataset), "config": {"name": profile}, "prepare_only": False}
            self.manifests[profile] = {"identity": identity, "run_fingerprint": digest(identity)}
            (self.root/profile/"expected_run.json").write_text(json.dumps(self.manifests[profile]))

    def tearDown(self):
        self.temporary.cleanup()

    def record(self, profile, index, raw, **changes):
        row = self.rows[index]
        record = {"sample_id": row["sample_id"], "profile": profile, "row_sha256": digest(row),
                  "run_fingerprint": self.manifests[profile]["run_fingerprint"], "status": "ok",
                  "raw_output": raw, "scores": score_prediction(raw,row), "total_seconds": 2,
                  "prepare_seconds": 1, "generation_seconds": 1, "gpu_peak_allocated_bytes": 1024**3,
                  "created_at": f"2026-11-06T00:00:0{index}+00:00"}
        record.update(changes)
        (self.root/profile/"records"/(row["sample_id"]+".json")).write_text(json.dumps(record))

    def test_formats_bounds_equivalence_missing_and_error_pairs(self):
        self.record("low",0,tool(1)); self.record("high",0,tool(-1))
        self.record("low",1,"1"); self.record("high",1,tool(99))
        self.record("low",2,tool(-2))
        self.record("low",3,"",status="error",raw_output=None,scores=None)
        self.record("high",3,tool(-1))
        report = summarize_details(self.dataset,self.root)
        low,high = [report["profiles"][p]["groups"]["all"] for p in ("low","high")]
        self.assertEqual((low["answered"],low["runtime_error"],low["missing"]),(3,1,1))
        self.assertEqual((low["format_error"],low["out_of_range"]),(1,1))
        self.assertEqual((high["out_of_range"],high["missing"]),(1,2))
        self.assertEqual(low["resolved_accuracy_observed_including_runtime_errors"],.25)
        self.assertEqual(low["resolved_accuracy_expected_lower_bound"],.2)
        pairs = report["paired"]["terminal_in_both_including_errors"]
        self.assertEqual((pairs["n"],pairs["not_in_pair"]),(3,2))
        self.assertEqual(pairs["groups"]["all"]["strict"]["high_only_correct"],2)
        self.assertEqual(pairs["groups"]["all"]["resolved"]["both_correct"],1)
        self.assertEqual(pairs["groups"]["all"]["resolved"]["high_only_correct"],1)
        self.assertFalse(report["complete"])
        self.assertIn("部分结果",render_markdown(report))

    def test_incompatible_identity_never_gets_credit(self):
        self.record("low",0,tool(-1),run_fingerprint="old")
        self.record("high",0,tool(-1),scores={"valid":True,"strict_correct":False,"resolved_correct":True})
        report = summarize_details(self.dataset,self.root)
        for profile in ("low","high"):
            counts = report["profiles"][profile]["groups"]["all"]
            self.assertEqual((counts["incompatible"],counts["missing"],counts["strict_correct"]),(1,4,0))
        self.assertEqual(report["paired"]["terminal_in_both_including_errors"]["n"],0)


if __name__ == "__main__":
    unittest.main()
