"""CPU contract checks; run with python scripts/inference/test_editing_tool.py."""
import json
from pathlib import Path
import tempfile
import unittest

from editing_tool_worker import process_request
from editing_tool_runtime import atomic_json, dynamic_video_profile, normalize_request


class FakeEditor:
    identity_sha = "test_identity"

    def generate(self, request):
        return {"request_id": request["request_id"], "status": "ok",
                "video_path": str(Path(request["output_dir"]) / "edited.mp4")}


class EditingToolContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "generated_previous_round.mp4"
        self.source.write_bytes(b"fixture")
        self.raw = {"request_id": "chain1_turn2", "source_video": str(self.source),
                    "prompt": "Change the hair color to blue.", "references": [],
                    "output_dir": str(self.root / "output"), "seed": 42}

    def tearDown(self):
        self.temp.cleanup()

    def test_arbitrary_source_and_current_prompt(self):
        result = normalize_request(self.raw)
        self.assertEqual(result["source_video"], str(self.source))
        self.assertEqual(result["prompt"], self.raw["prompt"])
        self.assertEqual(result["references"], [])

    def test_no_target_or_dataset_index_channel(self):
        for key in ("target_video", "target_latent", "sample_id", "gt_parent_id"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                normalize_request({**self.raw, key: "unused"})

    def test_invalid_reference_is_rejected(self):
        ref = {"reference_id": "cloth", "type": "outfit", "view": "front", "path": str(self.source)}
        normalize_request({**self.raw, "references": [ref]})
        with self.assertRaises(ValueError):
            normalize_request({**self.raw, "references": [ref, ref]})
        with self.assertRaises(ValueError):
            normalize_request({**self.raw, "references": [{**ref, "type": "target"}]})

    def test_generated_video_keeps_training_sampling(self):
        base = {"source_fps": 30, "source_frames": 300, "source_dimensions": [[1920, 1080]],
                "fps": 12, "num_frames": 121, "video_canvas": {"landscape": [768, 448]}}
        result = dynamic_video_profile(base, {"fps": "12", "frames": 121, "width": 768, "height": 448})
        self.assertEqual(result["source_frames"], 121)
        self.assertEqual(result["source_fps"], 12)
        self.assertEqual(result["num_frames"], 121)
        self.assertEqual(base["source_fps"], 30)
        self.assertEqual(result["video_canvas"], base["video_canvas"])

    def test_queue_response_and_repeat_id(self):
        queue = self.root / "queue"
        request = queue / "requests" / "chain1_turn2.json"
        atomic_json(request, self.raw)
        first = process_request(request, queue, FakeEditor())
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first, process_request(request, queue, FakeEditor()))
        self.assertEqual(first, json.loads((queue / "responses" / request.name).read_text()))
        atomic_json(request, {**self.raw, "prompt": "Change the hair color to red."})
        with self.assertRaisesRegex(ValueError, "Request ID reused"):
            process_request(request, queue, FakeEditor())

    def test_bad_request_is_a_response_not_worker_failure(self):
        queue = self.root / "queue"
        request = queue / "requests" / "wrong_name.json"
        atomic_json(request, self.raw)
        response = process_request(request, queue, FakeEditor())
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["error_type"], "ValueError")
        self.assertIn("filename", response["error"])


if __name__ == "__main__":
    unittest.main()
