#!/usr/bin/env python3
"""CPU integration checks using real public/labels and temporary fake results."""
import json
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'datasets/editing_agent_v1'
REPORT = Path(__file__).with_name('report.py')


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class MediaParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []
        self.cells = []
        self.expanded_details = 0

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag in ('video', 'img'):
            self.sources.extend(v for k, v in attrs if k == 'src')
        if tag == 'article' and 'chain-cell' in values.get('class', ''):
            self.cells.append(values)
        if tag == 'details' and 'open' in values:
            self.expanded_details += 1


class ReportChecks(unittest.TestCase):
    def setUp(self):
        # Same filesystem permits hard links to original media; no video copies
        # and no GPU calls. Every temporary output is removed after the test.
        self.temp = tempfile.TemporaryDirectory(prefix='.report_test_', dir=ROOT)
        self.run = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, public, labels, limit, gallery=False):
        args = [sys.executable, str(REPORT), '--run', str(self.run), '--dataset', str(DATA / public),
                '--labels', str(DATA / labels), '--limit', str(limit)]
        if gallery:
            args.append('--gallery')
        subprocess.run(args, check=True, capture_output=True, text=True)
        return json.loads((self.run / 'report/summary.json').read_text())

    def test_understanding_correct_wrong_error_pending_and_shared_calls(self):
        public = read(DATA / 'understanding_test.jsonl')[:4]
        labels = read(DATA / 'private/understanding_test_labels.jsonl')[:4]
        for index in (0, 1):
            case = self.run / 'cases' / public[index]['sample_id']
            expected = labels[index]['expected_source_video_id']
            predicted = expected if index == 0 else (expected + 1) % len(public[index]['history'])
            save(case / 'worker_result.json', {'status': 'ok', 'result': {'source_video_id': predicted}})
            save(case / 'decision/select_call_00.json', {'status': 'valid'})
        save(self.run / 'cases' / public[2]['sample_id'] / 'worker_result.json',
             {'status': 'error', 'error': 'fixture error'})
        save(self.run / 'cases' / public[2]['sample_id'] / 'decision/select_call_00.json', {'status': 'invalid'})
        save(self.run / 'observed_cards/shared_video/observe_call_00.json', {'status': 'invalid'})
        save(self.run / 'observed_cards/shared_video/observe_call_01.json', {'status': 'valid'})
        summary = self.execute('understanding_test.jsonl', 'private/understanding_test_labels.jsonl', 4)
        self.assertEqual((summary['cases_ok'], summary['cases_error'], summary['cases_pending']), (2, 1, 1))
        self.assertEqual((summary['route_correct'], summary['route_attempted'], summary['route_total_expected']), (1, 2, 4))
        self.assertEqual(summary['accuracy_over_all_expected'], 0.25)
        self.assertEqual(summary['agreement_over_attempted'], 0.5)
        self.assertEqual((summary['format_calls'], summary['format_invalid_calls']), (5, 2))
        self.assertEqual(summary['route_metric'], 'source_id_accuracy')
        self.assertFalse(summary['all_cases_finished'])
        self.assertFalse(summary['all_cases_succeeded'])

    def test_partial_chain_routes_and_portable_gallery(self):
        public = read(DATA / 'chains_debug.jsonl')[:3]
        labels = read(DATA / 'private/chains_debug_labels.jsonl')[:3]
        generated = self.run / 'fake_generated.mp4'
        generated.write_bytes(b'CPU fixture: not a playable generated result')
        first = self.run / 'cases' / public[0]['chain_id']
        save(first / 'worker_result.json', {'status': 'ok'})
        for turn in labels[0]['turns']:
            save(first / f"turn_{turn['turn_id']:02d}" / 'progress.json',
                 {'request': {'source_video_id': turn['expected_source_video_id'], 'prompt': '<edit & preserve>'},
                  'generation': {'video_path': str(generated)}, 'review': {'card': {}}})
        second = self.run / 'cases' / public[1]['chain_id']
        save(second / 'worker_result.json', {'status': 'error', 'error': 'fixture LTX failure'})
        save(second / 'turn_01/progress.json',
             {'request': {'source_video_id': labels[1]['turns'][0]['expected_source_video_id'], 'prompt': 'edit'}})
        # The third chain is still running: its valid first decision counts even
        # before a terminal worker result exists, but later turns remain pending.
        third = self.run / 'cases' / public[2]['chain_id']
        save(third / 'turn_01/progress.json',
             {'request': {'source_video_id': labels[2]['turns'][0]['expected_source_video_id'], 'prompt': 'edit'}})
        # Generation can finish while the post-generation review is pending.
        # Its media must remain visible even before progress records generation.
        finished_before_review = self.run / 'finished_before_review.mp4'
        finished_before_review.write_bytes(b'CPU fixture: generation committed before review')
        save(third / 'turn_01/generation/editing_generation.json',
             {'status': 'ok', 'video_path': str(finished_before_review)})
        summary = self.execute('chains_debug.jsonl', 'private/chains_debug_labels.jsonl', 3, gallery=True)
        expected_attempts = len(labels[0]['turns']) + 2
        self.assertEqual((summary['cases_ok'], summary['cases_error'], summary['cases_pending']), (1, 1, 1))
        self.assertEqual(summary['route_total_expected'], sum(len(x['turns']) for x in labels))
        self.assertEqual(summary['route_attempted'], expected_attempts)
        self.assertEqual(summary['route_correct'], expected_attempts)
        self.assertEqual(summary['route_metric'], 'planned_route_agreement')
        page = (self.run / 'report/index.html').read_text()
        parser = MediaParser(); parser.feed(page)
        self.assertGreater(len(parser.sources), 6)
        for src in parser.sources:
            self.assertTrue(src.startswith('media/'), src)
            self.assertTrue((self.run / 'report' / src).is_file(), src)
            self.assertFalse((self.run / 'report' / src).is_symlink(), src)
        self.assertIn('&lt;edit &amp; preserve&gt;', page)
        self.assertNotIn('/scratch/', page)
        self.assertNotIn('target_attributes', page)
        # Two aligned rows per case, including the source video in both rows.
        self.assertEqual(len(parser.cells), sum(2 * (len(x['turns']) + 1) for x in public))
        self.assertEqual(len([c for c in parser.cells if c['data-video-id'] == '0']), 6)
        actual_cells = [c for c in parser.cells if 'data-parent-id' in c]
        self.assertEqual(len(actual_cells), sum(len(x['turns']) for x in public))
        for cell, turn in zip(actual_cells, labels[0]['turns']):
            self.assertEqual(cell['data-parent-id'], str(turn['expected_source_video_id']))
        self.assertIn('回溯编辑', page)
        self.assertIn('chain-scroll', page)
        self.assertIn('grid-template-columns:repeat(var(--columns),280px)', page)
        self.assertEqual(parser.expanded_details, 0)
        self.assertTrue(any((self.run / 'report' / src).read_bytes() == finished_before_review.read_bytes()
                            for src in parser.sources if src.endswith('.mp4')))

    def test_finished_error_is_not_success_and_empty_denominator(self):
        public = read(DATA / 'understanding_test.jsonl')[0]
        save(self.run / 'cases' / public['sample_id'] / 'worker_result.json', {'status': 'error', 'error': 'failed'})
        summary = self.execute('understanding_test.jsonl', 'private/understanding_test_labels.jsonl', 1)
        self.assertTrue(summary['all_cases_finished'])
        self.assertFalse(summary['all_cases_succeeded'])
        self.assertIsNone(summary['agreement_over_attempted'])
        self.assertEqual(summary['route_correct'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
