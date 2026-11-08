"""Fake provenance checks only: no model/GPU work and no image-quality claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from audit_runtime import audit, digest
from queue_client import atomic_json
from reuse_editor import StrictReuseEditor


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RuntimeAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.run = self.root / 'run'; self.case = self.run / 'cases/c1'; self.turn = self.case / 'turn_01'
        self.source = self.root / 'source.mp4'; self.source.write_bytes(b'actual initial source')
        self.output = self.turn / 'generation/edited.mp4'; self.output.parent.mkdir(parents=True)
        self.output.write_bytes(b'actual generated output')
        self.row = {'chain_id': 'c1', 'root_video': str(self.source), 'turns': [
            {'turn_id': 1, 'user_prompt': 'Change hair to blue.', 'references': []}]}
        self.identity = {'schema': 'datamake_editing_tool_v1', 'checkpoint_sha256': 'a'*64,
                         'config_sha256': 'b'*64, 'reference_protocol': 'v7', 'weights_sha256': 'c'*64,
                         'runtime_sha256': 'd'*64, 'steps': 40}
        self.queue = self.run / 'queues/q0'
        atomic_json(self.queue / 'worker_identity.json', self.identity)
        atomic_json(self.queue / 'ready.json', {'status': 'ready', 'worker_identity_sha256': digest(self.identity)})
        self.request = {'request_id': 'test_1', 'source_video': str(self.source), 'prompt': 'Change hair to blue.',
                        'references': [], 'output_dir': str(self.output.parent), 'seed': 42}
        self.generation = self.native(self.request, self.output)
        content = {'observed': {'hair_color': 'blue'}, 'uncertain': []}
        self.card = {'video_id': 1, 'video_path': str(self.output), 'parent_id': 0, **deepcopy(content)}
        root_card = {'video_id': 0, 'video_path': str(self.source), 'parent_id': None,
                     'observed': {'hair_color': 'black'}, 'uncertain': []}
        self.progress = {'request': {'source_video_id': 0, 'prompt': self.request['prompt'], 'reference_ids': []},
                         'generation': self.generation, 'review': {'card': deepcopy(content), 'review': {'fulfillment': 'uncertain'}}}
        edit = {'source_video_id': 0, 'output_video_id': 1, 'user_prompt': self.row['turns'][0]['user_prompt'],
                'editing_prompt': self.request['prompt'], 'reference_ids': [], 'review': self.progress['review']['review']}
        self.state = {'status': 'completed', 'completed_turns': 1,
                      'identity': {'chain': self.row, 'prompts': {'observe': 'Observe actual video.'}, 'config': {'seed': 42}},
                      'history': {'videos': [root_card, self.card], 'edits': [edit]}}
        atomic_json(self.turn / 'output_observation/card.json', {k: v for k, v in self.card.items() if k != 'parent_id'})
        self.observation_call = {'status': 'valid', 'messages': [
            {'role': 'system', 'content': 'Observe actual video.'},
            {'role': 'user', 'content': [{'type': 'text', 'text': "Describe this video's actual visible contents."},
                {'type': 'video', 'path': str(self.output), 'media_role': 'history'}]}],
            'parsed_call': {'arguments': deepcopy(content)}}
        atomic_json(self.turn / 'output_observation/observe_call_00.json', self.observation_call)
        self.persist()

    def tearDown(self):
        self.temp.cleanup()

    def native(self, request, output):
        hashes = [sha(request['source_video'])]
        record = {'status': 'ok', 'request': deepcopy(request), 'input_media_sha256': hashes,
                  'worker_identity_sha256': digest(self.identity), 'video_path': str(output),
                  'video_sha256': sha(output), 'bytes': output.stat().st_size,
                  'execution_sha256': digest({'request': request, 'media_sha256': hashes, 'worker': digest(self.identity)})}
        atomic_json(output.parent / 'editing_generation.json', record)
        return record

    def persist(self):
        atomic_json(self.turn / 'progress.json', self.progress); atomic_json(self.case / 'state.json', self.state)
        for video in self.state['history']['videos']:
            atomic_json(self.case / f"videos/video_{video['video_id']:03d}.json", video)
        atomic_json(self.case / 'edits/edit_001.json', self.state['history']['edits'][0])

    def result(self, **kwargs):
        return audit(self.run, [self.row], **kwargs)

    def test_valid_evidence_and_explicit_legacy_skip(self):
        self.assertTrue(self.result()['success'])
        (self.turn / 'output_observation/card.json').unlink()
        self.assertFalse(self.result()['success'])
        legacy = self.result(legacy_v2=True)
        self.assertTrue(legacy['success'])
        self.assertIn('legacy_v2', legacy['scope'])
        self.assertIn('not_checked', legacy['cases'][0]['turns'][0]['observation'])

    def test_replacing_actual_source_with_gt_fails_even_with_consistent_receipt(self):
        gt = self.root / 'gt.mp4'; gt.write_bytes(b'ground truth target')
        changed = {**self.request, 'source_video': str(gt)}
        self.progress['generation'] = self.native(changed, self.output); self.persist()
        result = self.result()
        self.assertFalse(result['success']); self.assertIn('source was replaced', str(result['cases'][0]['failures']))

    def test_comparison_overwriting_card_fails(self):
        self.card['observed']['hair_color'] = 'red'
        self.progress['review']['card']['observed']['hair_color'] = 'red'; self.persist()
        result = self.result()
        self.assertFalse(result['success']); self.assertIn('changed after independent', str(result['cases'][0]['failures']))

    def test_hidden_source_or_instruction_in_observation_fails(self):
        self.observation_call['messages'][1]['content'].append({'type': 'text', 'text': 'Requested hair color is blue.'})
        atomic_json(self.turn / 'output_observation/observe_call_00.json', self.observation_call)
        self.assertFalse(self.result()['success'])

    def test_broken_state_is_reported_with_case_path(self):
        (self.case / 'state.json').write_text('{broken')
        result = self.result()
        self.assertFalse(result['success'])
        self.assertEqual(result['cases'][0]['failures'][0]['path'], str(self.case))

    def test_verified_actual_generation_reuse_is_accepted_without_new_native_receipt(self):
        previous = self.root / 'previous'; old_output = previous / 'cases/old/turn_01/generation/edited.mp4'
        old_output.parent.mkdir(parents=True); old_output.write_bytes(self.output.read_bytes())
        self.native({**self.request, 'request_id': 'old', 'output_dir': str(old_output.parent)}, old_output)
        class Fallback:
            def __init__(self, queue): self.queue = queue
            def __call__(self, request): raise AssertionError('Unexpected generation')
        reused = StrictReuseEditor(Fallback(self.queue), previous)(self.request)
        self.assertTrue(reused['reused'])
        (self.output.parent / 'editing_generation.json').unlink(); self.output.unlink()
        self.progress['generation'] = reused; self.card['video_path'] = str(old_output)
        observed = json.loads((self.turn / 'output_observation/card.json').read_text()); observed['video_path'] = str(old_output)
        atomic_json(self.turn / 'output_observation/card.json', observed)
        self.observation_call['messages'][1]['content'][1]['path'] = str(old_output)
        atomic_json(self.turn / 'output_observation/observe_call_00.json', self.observation_call); self.persist()
        result = self.result(); self.assertTrue(result['success'], result)
        old_output.write_bytes(b'changed output bytes')
        self.assertFalse(self.result()['success'])


if __name__ == '__main__':
    unittest.main()
