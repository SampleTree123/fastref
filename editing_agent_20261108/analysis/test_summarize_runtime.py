"""Small CPU proofs for counts, retries vs inspections, and card reuse wording."""
import json
from pathlib import Path
import tempfile
import unittest
from summarize_runtime import summarize


class SummaryTests(unittest.TestCase):
    def test_inspection_followup_is_not_a_format_retry(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name); case = run / 'cases/q1'; case.mkdir(parents=True)
            def save(relative, data):
                path = run / relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            valid = {'stage': 'select', 'status': 'valid', 'messages': [{'role': 'user', 'content': 'Choose'}],
                     'prepare_seconds': 1., 'generation_seconds': 2., 'total_seconds': 3.,
                     'tools': [{'function': {'name': 'select_source_video', 'parameters': {'properties': {'source_video_id': {'enum': [0, 1, 2]}}}}}]}
            save('cases/q1/decision/select_call_00.json', {**valid, 'parsed_call': {'name': 'inspect_history', 'arguments': {'video_ids': [0, 2]}}})
            save('cases/q1/decision/select_call_01.json', {**valid, 'messages': [{'role': 'user', 'content': [{'type': 'video', 'path': '/fake'}]}],
                'parsed_call': {'name': 'select_source_video', 'arguments': {'source_video_id': 0}}})
            save('cases/q1/decision/edit_call_00.json', {'stage': 'edit', 'status': 'invalid', 'messages': [], 'validation_error': 'missing field'})
            save('cases/q1/decision/edit_call_01.json', {'stage': 'edit', 'status': 'valid',
                'messages': [{'role': 'user', 'content': 'The previous tool call was not executed. Error: missing field'}],
                'parsed_call': {'name': 'edit_video', 'arguments': {}}})
            for qid in ['q1', 'q2']:
                save(f'cases/{qid}/worker_result.json', {'status': 'ok', 'seconds': 8.})
                save(f'cases/{qid}/observed_history.json', [{'observation_identity': 'shared'}])
            save('observed_cards/shared/cached_card.json', {'observation_identity': 'shared'})
            public = [{'sample_id': qid, 'history': []} for qid in ['q1', 'q2', 'q3']]
            result = summarize(run, public, [{'sample_id': 'q1', 'selection_kind': 'explicit_id'}])
            self.assertEqual(result['history_inspection']['accepted_tool_calls'], 1)
            self.assertEqual(result['history_inspection']['viewed_video_slots'], 2)
            self.assertEqual(result['stages']['select']['correction_calls'], 0)
            self.assertEqual(result['stages']['edit']['correction_calls_valid'], 1)
            self.assertEqual(result['source_selection']['location_counts'], {'root': 1})
            self.assertEqual(result['observed_card_references']['repeated_identity_assignments'], 1)
            self.assertEqual(result['case_status']['pending'], 1)
            self.assertEqual(result['durations_by_stage']['select']['total_seconds']['sum_seconds'], 6.)

    def test_invalid_call_is_not_automatically_terminal(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name); decision = run / 'cases/q/decision'; decision.mkdir(parents=True)
            (decision / 'select_call_00.json').write_text(json.dumps({'stage': 'select', 'status': 'invalid', 'messages': []}))
            public = [{'sample_id': 'q', 'history': []}]
            result = summarize(run, public)
            self.assertEqual(result['stages']['select']['groups_ending_invalid_without_confirmed_terminal_failure'], 1)
            self.assertEqual(result['stages']['select']['groups_with_confirmed_terminal_tool_failure'], 0)
            (decision.parent / 'worker_result.json').write_text(json.dumps({'status': 'error', 'error': 'select: invalid output after one correction: nope'}))
            result = summarize(run, public)
            self.assertEqual(result['stages']['select']['groups_with_confirmed_terminal_tool_failure'], 1)
            self.assertEqual(result['terminal_cases'][0]['confirmed_terminal_tool_stage'], 'select')


if __name__ == '__main__':
    unittest.main()
