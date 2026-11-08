#!/usr/bin/env python3
"""CPU fake-media checks; no Qwen/LTX weights, CUDA or Slurm access."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from queue_client import atomic_json
from reuse_editor import StrictReuseEditor, digest, file_sha


class Fallback:
    def __init__(self, queue):
        self.queue, self.calls = queue, []

    def __call__(self, request):
        self.calls.append(deepcopy(request))
        return {'status': 'ok', 'test_real_generation_requested': True}


class ReuseChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.previous = self.root / 'v2'
        self.directory = self.previous / 'cases/old_case/turn_01/generation'
        self.directory.mkdir(parents=True)
        self.source = self.root / 'source.mp4'; self.source.write_bytes(b'source RGB')
        self.ref1 = self.root / 'front.png'; self.ref1.write_bytes(b'front RGB')
        self.ref2 = self.root / 'back.png'; self.ref2.write_bytes(b'back RGB')
        self.video = self.directory / 'edited.mp4'; self.video.write_bytes(b'previous actual LTX output')
        self.queue = self.root / 'queue'; self.queue.mkdir()
        self.identity = {'schema': 'datamake_editing_tool_v1', 'checkpoint_sha256': 'a'*64,
                         'config_sha256': 'b'*64, 'reference_protocol': {'layout': 'sequential'},
                         'weights_sha256': 'c'*64, 'runtime_sha256': 'd'*64,
                         'steps': 40, 'cfg_scale': 3, 'ref_cfg': 1, 'profile': {'frames': 121}}
        self.write_worker()
        self.old_request = {'request_id': 'edit_old', 'source_video': str(self.source), 'prompt': 'Match this outfit.',
                            'references': [{'reference_id': 'ref_1', 'path': str(self.ref1), 'type': 'outfit', 'view': 'front'},
                                           {'reference_id': 'ref_2', 'path': str(self.ref2), 'type': 'outfit', 'view': 'back'}],
                            'seed': 42, 'output_dir': str(self.directory)}
        self.media_sha = [file_sha(p) for p in (self.source, self.ref1, self.ref2)]
        self.old = {'status': 'ok', 'request': self.old_request, 'input_media_sha256': self.media_sha,
                    'worker_identity_sha256': digest(self.identity), 'video_path': str(self.video),
                    'video_sha256': file_sha(self.video), 'bytes': self.video.stat().st_size,
                    'execution_sha256': digest({'request': self.old_request, 'media_sha256': self.media_sha,
                                                'worker': digest(self.identity)})}
        self.receipt = self.directory / 'editing_generation.json'; atomic_json(self.receipt, self.old)
        self.current = {**deepcopy(self.old_request), 'request_id': 'entirely_new_case_turn',
                        'output_dir': str(self.root / 'v3/cases/new_case/turn_01/generation')}
        self.fallback = Fallback(self.queue)

    def tearDown(self):
        self.temp.cleanup()

    def write_worker(self):
        atomic_json(self.queue / 'worker_identity.json', self.identity)
        atomic_json(self.queue / 'ready.json', {'status': 'ready', 'worker_identity_sha256': digest(self.identity)})

    def editor(self):
        return StrictReuseEditor(self.fallback, self.previous)

    def test_same_content_at_other_paths_reuses_real_output_and_writes_provenance(self):
        for key, source in [('source_video', self.source)]:
            duplicate = self.root / 'renamed_source.mp4'; duplicate.write_bytes(source.read_bytes())
            self.current[key] = str(duplicate)
        duplicate = self.root / 'renamed_front.png'; duplicate.write_bytes(self.ref1.read_bytes())
        self.current['references'][0]['path'] = str(duplicate)
        result = self.editor()(self.current)
        self.assertTrue(result['reused'])
        self.assertEqual(result['video_path'], str(self.video))
        self.assertFalse(self.fallback.calls)
        proof = json.loads((Path(self.current['output_dir']) / 'reuse_provenance.json').read_text())
        self.assertTrue(proof['not_ground_truth'])
        self.assertEqual(proof['previous_receipt'], str(self.receipt))
        self.assertFalse((Path(self.current['output_dir']) / 'editing_generation.json').exists())

    def test_prompt_source_reference_seed_and_order_changes_do_not_reuse(self):
        different = self.root / 'different.bin'; different.write_bytes(b'different bytes')
        cases = []
        for key, value in [('prompt', 'Different clothing.'), ('source_video', str(different)), ('seed', 43)]:
            request = deepcopy(self.current); request[key] = value; cases.append(request)
        request = deepcopy(self.current); request['references'][0]['path'] = str(different); cases.append(request)
        request = deepcopy(self.current); request['references'].reverse(); cases.append(request)
        request = deepcopy(self.current); request['references'][0]['view'] = 'back'; cases.append(request)
        request = deepcopy(self.current); request['references'][0]['type'] = 'garment'; cases.append(request)
        editor = self.editor()
        for request in cases:
            self.assertTrue(editor(request)['test_real_generation_requested'])
        self.assertEqual(len(self.fallback.calls), len(cases))

    def test_checkpoint_protocol_steps_cfg_changes_do_not_reuse(self):
        original = deepcopy(self.identity)
        for key, value in [('checkpoint_sha256', 'f'*64), ('reference_protocol', {'layout': 'original'}),
                           ('steps', 30), ('cfg_scale', 5), ('runtime_sha256', 'f'*64)]:
            self.identity = {**original, key: value}; self.write_worker()
            self.assertTrue(self.editor()(self.current)['test_real_generation_requested'])

    def test_corrupt_output_and_changed_receipt_are_rejected(self):
        editor = self.editor()
        self.video.write_bytes(b'bad actual output')
        self.assertTrue(editor(self.current)['test_real_generation_requested'])
        self.video.write_bytes(b'previous actual LTX output')
        altered = {**self.old, 'execution_sha256': 'e'*64}; atomic_json(self.receipt, altered)
        self.assertTrue(editor(self.current)['test_real_generation_requested'])

    def test_old_input_mutation_and_false_ready_identity_are_rejected(self):
        duplicate = self.root / 'same_source.mp4'; duplicate.write_bytes(self.source.read_bytes())
        self.current['source_video'] = str(duplicate)
        editor = self.editor()
        self.source.write_bytes(b'old source changed')
        self.assertTrue(editor(self.current)['test_real_generation_requested'])
        atomic_json(self.queue / 'ready.json', {'status': 'ready', 'worker_identity_sha256': '0'*64})
        self.assertTrue(editor(self.current)['test_real_generation_requested'])

    def test_no_current_ready_worker_never_uses_previous_model_identity(self):
        (self.queue / 'ready.json').unlink()
        self.assertTrue(self.editor()(self.current)['test_real_generation_requested'])

    def test_optional_wait_for_missing_or_loading_worker(self):
        for loading in (False, True):
            if loading:
                atomic_json(self.queue / 'ready.json', {'status': 'loading'})
            else:
                (self.queue / 'ready.json').unlink(missing_ok=True)
            editor = StrictReuseEditor(self.fallback, self.previous, wait_ready_seconds=120)
            with patch('reuse_editor.time.sleep', side_effect=lambda _: self.write_worker()) as slept:
                self.assertTrue(editor(self.current)['reused'])
                self.assertEqual(slept.call_count, 1)

    def test_ready_identity_mismatch_does_not_wait(self):
        atomic_json(self.queue / 'ready.json', {'status': 'ready', 'worker_identity_sha256': '0'*64})
        editor = StrictReuseEditor(self.fallback, self.previous, wait_ready_seconds=120)
        with patch('reuse_editor.time.sleep', side_effect=AssertionError('Must not wait on identity mismatch')):
            self.assertTrue(editor(self.current)['test_real_generation_requested'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
