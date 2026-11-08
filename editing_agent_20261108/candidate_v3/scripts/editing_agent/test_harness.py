"""CPU behavior checks for memory, source routing, bounded correction and resume."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from contracts import (ATTRIBUTES, parse_tool_calls, selection_tools, edit_tools,
                       card_tools, review_tools, validate_public_chain)
from harness import observe_video, choose_and_edit_request, run_chain, review_output, StageFailure


def card(color="black"):
    return {"observed": {k: color if k == "hair_color" else "Visible " + k for k in ATTRIBUTES}, "uncertain": []}


def answer(name, arguments):
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call><|im_end|>"


class FakeBackend:
    identity = {"test_backend": 1}
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []
    def call(self, messages, tools, output_path=None):
        self.calls.append(deepcopy({"messages": messages, "tools": tools}))
        return {"raw_output": next(self.answers)}


class HarnessTests(unittest.TestCase):
    def test_official_xml_and_strict_arguments(self):
        tools = selection_tools([0, 1])
        text = "<tool_call><function=select_source_video><parameter=source_video_id>1</parameter></function></tool_call><|im_end|>"
        self.assertEqual(parse_tool_calls(text, tools)[0]["arguments"]["source_video_id"], 1)
        for value in (-1, 2, True, "1"):
            with self.assertRaises(ValueError):
                parse_tool_calls(answer("select_source_video", {"source_video_id": value}), tools)
        with self.assertRaises(ValueError):
            parse_tool_calls(text + text, tools)
        with self.assertRaises(ValueError):
            parse_tool_calls('<tool_call>{"name":"select_source_video","arguments":{"source_video_id":0,"source_video_id":1}}</tool_call>', tools)

    def test_structured_observation_xml(self):
        c = card()
        raw = '<tool_call><function=record_video_card><parameter=observed>' + json.dumps(c['observed']) + '</parameter><parameter=uncertain>[]</parameter></function></tool_call>'
        self.assertEqual(parse_tool_calls(raw, card_tools())[0]['arguments'], c)

    def test_private_labels_rejected(self):
        base = {"chain_id": "x", "split": "debug", "root_video": "/tmp/a.mp4", "turns": [{"turn_id": 1, "user_prompt": "red hair", "references": []}]}
        validate_public_chain(base)
        validate_public_chain({**base, "schema": "editing_agent_chain_v1"})
        with self.assertRaises(ValueError):
            validate_public_chain({**base, "schema": "private_or_unknown"})
        base["turns"][0]["expected_source_video_id"] = 0
        with self.assertRaises(ValueError):
            validate_public_chain(base)

    def test_selection_has_no_current_reference_pixels_or_descriptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = {"videos": [{"video_id": 0, "parent_id": None, "video_path": "/secret/a.mp4", **card()}], "edits": []}
            turn = {"turn_id": 1, "user_prompt": "Change clothing to the supplied images.", "references": [{"reference_id": "ref_1", "type": "garment", "path": "/secret/newdress.png", "view": "front"}]}
            backend = FakeBackend([answer("select_source_video", {"source_video_id": 0}), answer("edit_video", {"source_video_id": 0, "prompt": "Match clothing to the reference image.", "reference_ids": ["ref_1"]})])
            result = choose_and_edit_request(history, turn, backend, tmp)
            selection = json.dumps(backend.calls[0])
            self.assertNotIn("newdress", selection)
            self.assertNotIn("garment", selection)
            self.assertNotIn("/secret", selection)
            self.assertEqual(result["reference_ids"], ["ref_1"])

    def test_one_correction_never_latest_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = [{"video_id": i, "parent_id": None if i == 0 else 0, "video_path": "unused", **card()} for i in range(2)]
            backend = FakeBackend([answer("select_source_video", {"source_video_id": -1})] * 3)
            with self.assertRaises(StageFailure):
                choose_and_edit_request(history, {"turn_id": 2, "user_prompt": "Continue editing.", "references": []}, backend, tmp)
            self.assertEqual(len(backend.calls), 2)

    def test_inspect_then_select_keeps_references_out_of_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = [{"video_id": i, "parent_id": None if i == 0 else 0, "video_path": f"/video/{i}.mp4", **card()} for i in range(2)]
            backend = FakeBackend([answer("inspect_history", {"video_ids": [0, 1]}), answer("select_source_video", {"source_video_id": 0}), answer("edit_video", {"source_video_id": 0, "prompt": "Make hair red.", "reference_ids": []})])
            result = choose_and_edit_request(history, {"turn_id": 2, "user_prompt": "Use the darker hair version; make hair red.", "references": []}, backend, tmp)
            self.assertEqual(result['inspected_video_ids'], [0, 1])
            self.assertEqual(len(backend.calls[1]['tools']), 1)
            self.assertEqual(backend.calls[1]['messages'][-1]['content'][1]['path'], '/video/0.mp4')

    def test_dropping_current_reference_is_invalid_and_can_be_corrected_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = [{"video_id": 0, "parent_id": None, "video_path": "unused", **card()}]
            turn = {"turn_id": 1, "user_prompt": "Match the shirt.", "references": [{"reference_id": "r", "type": "garment", "path": "unused"}]}
            backend = FakeBackend([answer("select_source_video", {"source_video_id": 0}), answer("edit_video", {"source_video_id": 0, "prompt": "Match shirt.", "reference_ids": []}), answer("edit_video", {"source_video_id": 0, "prompt": "Match shirt.", "reference_ids": ["r"]})])
            result = choose_and_edit_request(history, turn, backend, tmp)
            self.assertEqual(result['editing_corrections'], 1)

    def test_real_generated_parent_and_no_regeneration_after_completed_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root.mp4"; root.write_bytes(b"source-video")
            chain = {"chain_id": "c", "split": "debug", "root_video": str(root), "turns": [
                {"turn_id": 1, "user_prompt": "Make hair red.", "references": []},
                {"turn_id": 2, "user_prompt": "Continue with blue hair.", "references": []}]}
            review = lambda color: {"card": card(color), "review": {"requested_change": "hair color", "observed_change": color,
                     "fulfillment": "fulfilled", "unexpected_changes": [], "uncertain": []}}
            backend = FakeBackend([answer("record_video_card", card()),
                answer("select_source_video", {"source_video_id": 0}), answer("edit_video", {"source_video_id": 0, "prompt": "Red hair.", "reference_ids": []}), answer("record_video_card", card("red")), answer("record_edit_result", review("red")["review"]),
                answer("select_source_video", {"source_video_id": 1}), answer("edit_video", {"source_video_id": 1, "prompt": "Blue hair.", "reference_ids": []}), answer("record_video_card", card("blue")), answer("record_edit_result", review("blue")["review"])])
            requests = []
            def editor(request):
                requests.append(request)
                path = Path(request["output_dir"]) / "generated.mp4"; path.parent.mkdir(parents=True); path.write_bytes(b"actual-generated")
                return {"status": "ok", "video_path": str(path)}
            out = Path(tmp) / "out"
            state = run_chain(chain, out, backend, editor)
            self.assertEqual(state['status'], 'completed', state.get('error'))
            self.assertEqual(requests[1]['source_video'], str(out/'turn_01/generation/generated.mp4'))
            self.assertEqual(state['history']['videos'][2]['parent_id'], 1)
            self.assertEqual(run_chain(chain, out, backend, editor)['status'], 'completed')
            self.assertEqual(len(requests), 2)
            chain['turns'][0]['user_prompt'] = 'Changed input'
            with self.assertRaises(ValueError):
                run_chain(chain, out, backend, editor)

    def test_observed_card_cache_not_user_intent_and_renumbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'a.mp4'; source.write_bytes(b'actual-video')
            backend = FakeBackend([answer('record_video_card', card('brown'))])
            cfg = {'observed_card_cache': str(Path(tmp)/'cache')}
            first = observe_video(source, 0, backend, Path(tmp)/'one', cfg)
            second = observe_video(source, 5, backend, Path(tmp)/'two', cfg)
            self.assertEqual(len(backend.calls), 1)
            self.assertEqual(second['video_id'], 5)
            self.assertEqual(first['observed'], second['observed'])

    def test_missing_card_attribute_requires_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'video.mp4'; source.write_bytes(b'video')
            incomplete = card(); del incomplete['observed']['scene']
            backend = FakeBackend([answer('record_video_card', incomplete), answer('record_video_card', card())])
            result = observe_video(source, 0, backend, Path(tmp) / 'observation')
            self.assertEqual(set(result['observed']), set(ATTRIBUTES))
            self.assertEqual(len(backend.calls), 2)

    def test_interrupted_posthook_resumes_without_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'root.mp4'; source.write_bytes(b'video')
            chain = {'chain_id': 'resume', 'split': 'debug', 'root_video': str(source), 'turns': [
                {'turn_id': 1, 'user_prompt': 'Make hair red.', 'references': []}]}
            def answers_then_interrupt():
                yield answer('record_video_card', card())
                yield answer('select_source_video', {'source_video_id': 0})
                yield answer('edit_video', {'source_video_id': 0, 'prompt': 'Make hair red.', 'reference_ids': []})
                raise KeyboardInterrupt('Simulated worker interruption after generation')
            requests = []
            def editor(request):
                requests.append(request)
                path = Path(request['output_dir']) / 'video.mp4'
                path.parent.mkdir(parents=True); path.write_bytes(b'generated-video')
                return {'status': 'ok', 'video_path': str(path)}
            directory = Path(tmp) / 'out'
            with self.assertRaises(KeyboardInterrupt):
                run_chain(chain, directory, FakeBackend(answers_then_interrupt()), editor)
            self.assertEqual(json.loads((directory/'state.json').read_text())['status'], 'running')
            review = {'card': card('red'), 'review': {'requested_change': 'hair color', 'observed_change': 'red hair',
                      'fulfillment': 'fulfilled', 'unexpected_changes': [], 'uncertain': []}}
            backend = FakeBackend([answer('record_video_card', card('red')), answer('record_edit_result', review['review'])])
            result = run_chain(chain, directory, backend, editor)
            self.assertEqual(result['status'], 'completed', result.get('error'))
            self.assertEqual(len(requests), 1)
            self.assertEqual(len(backend.calls), 2)

    def test_invalid_model_answer_is_terminal_not_automatic_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'root.mp4'; source.write_bytes(b'video')
            chain = {'chain_id': 'bad', 'root_video': str(source), 'turns': [
                {'turn_id': 1, 'user_prompt': 'Change hair.', 'references': []}]}
            backend = FakeBackend(['wrong', 'wrong'])
            editor = lambda request: self.fail('Must not generate after invalid observation')
            directory = Path(tmp)/'out'
            state = run_chain(chain, directory, backend, editor)
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(run_chain(chain, directory, backend, editor)['status'], 'failed')
            self.assertEqual(len(backend.calls), 2)

    def test_independent_output_card_and_original_plus_executed_review_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source.mp4'; source.write_bytes(b'source')
            output = Path(tmp)/'output.mp4'; output.write_bytes(b'output')
            reference = Path(tmp)/'target.png'; reference.write_bytes(b'reference')
            turn = {'user_prompt': 'RETURN_TO_SECRET_HISTORY_VIDEO_7. Change hair blue.',
                    'references': [{'reference_id': 'ref_1', 'type': 'hair', 'path': str(reference), 'view': 'front'}]}
            comparison = {'requested_change': 'blue hair', 'observed_change': 'hair remains brown',
                          'fulfillment': 'not_fulfilled', 'unexpected_changes': [], 'uncertain': []}
            backend = FakeBackend([answer('record_video_card', card('brown')), answer('record_edit_result', comparison)])
            result = review_output(source, output, turn, 'Change hair blue.', backend, Path(tmp)/'review')
            observation = json.dumps(backend.calls[0]['messages'])
            self.assertIn(str(output), observation)
            self.assertNotIn(str(source), observation)
            self.assertNotIn(str(reference), observation)
            self.assertNotIn('blue', observation)
            self.assertNotIn('SECRET_HISTORY', observation)
            comparison_input = json.dumps(backend.calls[1]['messages'])
            self.assertIn(str(source), comparison_input)
            self.assertIn(str(output), comparison_input)
            self.assertIn(str(reference), comparison_input)
            self.assertIn('Change hair blue.', comparison_input)
            self.assertIn('SECRET_HISTORY', comparison_input)
            self.assertIn('Already executed', comparison_input)
            self.assertIn('historical appearance words are not observed facts', comparison_input)
            self.assertEqual(result['card'], card('brown'))
            self.assertEqual(result['review'], comparison)
            self.assertNotIn('video_id', result['card'])
            with self.assertRaises(ValueError):
                parse_tool_calls(answer('record_edit_result', {'card': card(), 'review': comparison}), review_tools())

    def test_interrupted_comparison_reuses_independent_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source.mp4'; source.write_bytes(b'source')
            output = Path(tmp)/'output.mp4'; output.write_bytes(b'output')
            turn = {'user_prompt': 'Change hair blue.', 'references': []}
            def answers_then_interrupt():
                yield answer('record_video_card', card('brown'))
                raise KeyboardInterrupt('Interruption during comparison, after card saved')
            directory = Path(tmp)/'review'
            with self.assertRaises(KeyboardInterrupt):
                review_output(source, output, turn, 'Change hair blue.', FakeBackend(answers_then_interrupt()), directory)
            comparison = {'requested_change': 'blue hair', 'observed_change': 'brown hair remains',
                          'fulfillment': 'not_fulfilled', 'unexpected_changes': [], 'uncertain': []}
            backend = FakeBackend([answer('record_edit_result', comparison)])
            result = review_output(source, output, turn, 'Change hair blue.', backend, directory)
            self.assertEqual(len(backend.calls), 1)
            self.assertEqual(backend.calls[0]['tools'][0]['function']['name'], 'record_edit_result')
            self.assertEqual(result['card'], card('brown'))


if __name__ == '__main__':
    unittest.main()
