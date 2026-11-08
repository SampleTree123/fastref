"""CPU checks for the Qwen to native LTX file protocol and launch settings."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from launch_allocation import ltx_command
from queue_client import LTXQueueClient, atomic_json


class QueueProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.request = {'request_id': 'original', 'source_video': str(self.root / 'source.mp4'),
                        'prompt': 'Change hair color to blue.', 'references': [],
                        'output_dir': str(self.root / 'result'), 'seed': 42}

    def tearDown(self):
        self.temp.cleanup()

    def test_worker_startup_failure_is_immediate(self):
        atomic_json(self.root / 'ready.json', {'status': 'error', 'error': 'weights failed'})
        with self.assertRaisesRegex(RuntimeError, 'weights failed'):
            LTXQueueClient(self.root, timeout=0)(self.request)

    def test_real_worker_response_contract(self):
        video = self.root / 'edited.mp4'
        video.write_bytes(b'test video')
        def responder():
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                pending = list((self.root / 'requests').glob('*.json'))
                if pending:
                    raw = json.loads(pending[0].read_text())
                    sha = hashlib.sha256(json.dumps(raw, ensure_ascii=False, sort_keys=True,
                                                      separators=(',', ':')).encode()).hexdigest()
                    atomic_json(self.root / 'responses' / pending[0].name,
                                {'request_id': raw['request_id'], 'request_sha256': sha,
                                 'status': 'ok', 'video_path': str(video)})
                    return
                time.sleep(.01)
        thread = threading.Thread(target=responder)
        thread.start()
        result = LTXQueueClient(self.root, timeout=3)(self.request)
        thread.join()
        self.assertEqual(result['video_path'], str(video))
        self.assertEqual(result, LTXQueueClient(self.root, timeout=0)(self.request))
        video.unlink()
        with self.assertRaisesRegex(ValueError, 'missing or empty'):
            LTXQueueClient(self.root, timeout=0)(self.request)

    def test_launch_settings_are_explicit(self):
        config = {'config': 'train.yaml', 'checkpoint': 'checkpoint.pt', 'steps': 40,
                  'ref_cfg': 1, 'cfg_scale': 3, 'stg_scale': 1,
                  'guidance_rescale': .7, 'stg_blocks': [28]}
        (self.root / 'train.yaml').touch()
        (self.root / 'checkpoint.pt').touch()
        with patch('launch_allocation.LTX', self.root):
            command = ltx_command(config, self.root / 'queue')
            self.assertEqual(command[command.index('--steps') + 1], '40')
            self.assertEqual(command[command.index('--config') + 1], str(self.root / 'train.yaml'))
            self.assertEqual(command[command.index('--ref-cfg') + 1], '1')
            with self.assertRaisesRegex(ValueError, 'cfg_scale'):
                ltx_command({**config, 'cfg_scale': 5}, self.root / 'queue')


if __name__ == '__main__':
    unittest.main()
