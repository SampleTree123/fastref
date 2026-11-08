#!/usr/bin/env python3
"""Guard the semantic boundary: no -1 credit, exact XML, and available IDs only."""
import json
from pathlib import Path
import tempfile
import unittest
from evaluation_inputs_actual_id import parse_tool_output, score_prediction, tools_for_history
from evaluation_inputs import score_prediction as old_score
from run_model_eval_actual_id import require_expected_run


def output(value):
    return f'<tool_call><function=select_source_video><parameter=source_video_id>{value}</parameter></function></tool_call>'


class ActualIDContracts(unittest.TestCase):
    def row(self):
        return {'audit': {'edit_round': 4}, 'supervision': {'source_video_id': 3, 'resolved_parent_video_id': 3}}

    def test_negative_one_never_resolves_or_receives_credit(self):
        actual = score_prediction(output(-1), self.row())
        self.assertFalse(actual['valid'])
        self.assertFalse(actual['strict_correct'])
        self.assertFalse(actual['resolved_correct'])
        self.assertIsNone(actual['resolved_parent_video_id'])
        self.assertIsNone(actual['runtime_fallback_source_video_id'])
        self.assertTrue(old_score(output(-1), self.row())['resolved_correct'])

    def test_actual_latest_and_earlier_are_distinct(self):
        self.assertTrue(score_prediction(output(3), self.row())['strict_correct'])
        self.assertFalse(score_prediction(output(2), self.row())['strict_correct'])
        self.assertFalse(parse_tool_output(output(4), 4)['valid'])

    def test_only_one_exact_official_call(self):
        for raw in ('3', '{"source_video_id":3}', output('3.0'), 'Here: '+output(3), output(3)*2):
            self.assertFalse(parse_tool_output(raw, 4)['valid'], raw)
        self.assertTrue(parse_tool_output(output(3)+'<|im_end|>', 4)['valid'])

    def test_schema_and_labels_exclude_negative_id(self):
        self.assertEqual(tools_for_history(4)[0]['function']['parameters']['properties']['source_video_id']['enum'], [0,1,2,3])
        row = self.row()
        row['supervision']['source_video_id'] = -1
        with self.assertRaises(ValueError):
            score_prediction(output(3), row)

    def test_resume_refuses_changed_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            require_expected_run(root, 'first', {'contract': 'actual_id_v1'})
            require_expected_run(root, 'first', {'contract': 'actual_id_v1'})
            with self.assertRaises(ValueError):
                require_expected_run(root, 'second', {'contract': 'legacy'})


if __name__ == '__main__':
    unittest.main()
