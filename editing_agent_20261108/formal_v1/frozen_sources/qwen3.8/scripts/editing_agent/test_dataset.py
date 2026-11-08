#!/usr/bin/env python3
"""CPU checks for public/private separation and real graph-backed examples."""
import copy
import json
from pathlib import Path
import unittest

import build_dataset as build


DATA = build.ROOT / 'datasets/editing_agent_v1'


class DatasetChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.debug = list(build.rows(DATA / 'chains_debug.jsonl'))
        cls.chains = list(build.rows(DATA / 'chains_test.jsonl'))
        cls.questions = list(build.rows(DATA / 'understanding_test.jsonl'))
        cls.dlabels = list(build.rows(DATA / 'private/chains_debug_labels.jsonl'))
        cls.clabels = list(build.rows(DATA / 'private/chains_test_labels.jsonl'))
        cls.qlabels = list(build.rows(DATA / 'private/understanding_test_labels.jsonl'))
        cls.audit = json.loads((DATA / 'audit.json').read_text())

    def test_counts_and_prefix_coverage(self):
        self.assertEqual((len(self.debug), len(self.chains), len(self.questions)), (6, 128, 1000))
        for n in (6, 30, 60, 128):
            labels = self.dlabels if n == 6 else self.clabels[:n]
            self.assertEqual({t['axis'] for c in labels for t in c['turns']}, set(build.AXES))
            self.assertEqual({c['length'] for c in labels}, {3, 4, 5})
            self.assertEqual({c['topology'] for c in labels}, {'linear', 'fork'})

    def test_inputs_and_labels_match_without_leaking(self):
        for public, private, key, chain in (
            (self.debug, self.dlabels, 'chain_id', True),
            (self.chains, self.clabels, 'chain_id', True),
            (self.questions, self.qlabels, 'sample_id', False),
        ):
            self.assertEqual([p[key] for p in public], [p[key] for p in private])
            self.assertEqual(len({p[key] for p in public}), len(public))
            for p, label in zip(public, private):
                build.check_public(p, is_chain=chain)
                self.assertEqual(build.digest(p), label['public_sha256'])
                self.assertEqual(label['original_split'], 'train' if public is self.debug else 'test')

    def test_debug_group_and_nodes_do_not_overlap_test(self):
        train_components = {x['component_id'] for x in self.dlabels}
        test_components = {x['component_id'] for x in self.clabels + self.qlabels}
        self.assertFalse(train_components & test_components)
        train_nodes = {n for x in self.dlabels for n in x['history_node_ids']}
        test_nodes = {n for x in self.clabels + self.qlabels for n in x['history_node_ids']}
        test_nodes.update(x['target_node_id'] for x in self.qlabels)
        self.assertFalse(train_nodes & test_nodes)

    def test_media_and_hashes(self):
        for file, info in self.audit['files'].items():
            self.assertEqual(build.sha(DATA / file), info['sha256'])
        for row in self.debug + self.chains:
            self.assertTrue(Path(row['root_video']).is_file())
            for turn in row['turns']:
                for ref in turn['references']:
                    self.assertTrue(Path(ref['path']).is_file())
        for row in self.questions:
            for h in row['history']:
                self.assertTrue(Path(h['video_path']).is_file())
        self.assertEqual(self.audit['understanding']['selection_evidence_axes'], {'camera': 250, 'hair_color': 250})

    def test_graph_edges_and_actual_sources(self):
        # Read the source manifests independently of the builder's graph loader.
        edges = {}
        for split in ('train', 'val', 'test', 'reserved'):
            for edge in build.rows(build.VARIANT / split / 'samples.jsonl'):
                edges[edge['sample_id']] = edge
        for label in self.clabels + self.dlabels:
            for t in label['turns']:
                edge = edges[t['edge_id']]
                self.assertEqual(edge['source_node_id'], label['history_node_ids'][t['expected_source_video_id']])
                self.assertEqual(edge['target_node_id'], label['history_node_ids'][t['turn_id']])
        for row, label in zip(self.questions, self.qlabels):
            edge = edges[label['current_edge_id']]
            self.assertEqual(edge['source_node_id'], label['history_node_ids'][label['expected_source_video_id']])
            self.assertEqual(edge['target_node_id'], label['target_node_id'])
            for h, edge_id in zip(row['history'][1:], label['completed_edge_ids']):
                edge = edges[edge_id]
                self.assertEqual(edge['source_node_id'], label['history_node_ids'][h['parent_id']])
                self.assertEqual(edge['target_node_id'], label['history_node_ids'][h['video_id']])

    def test_rejects_gt_fields_and_future_parent(self):
        question = copy.deepcopy(self.questions[0])
        question['turn']['expected_source_video_id'] = 0
        with self.assertRaisesRegex(ValueError, 'Private fields'):
            build.check_public(question, is_chain=False)
        question = copy.deepcopy(self.questions[0])
        question['history'][1]['parent_id'] = 1
        with self.assertRaisesRegex(ValueError, 'earlier available'):
            build.check_public(question, is_chain=False)

    def test_event_wording_is_not_visual_content(self):
        turn = {'source_history_index': 1, 'edit_round': 4, 'prompt': {'en': 'Please use the version produced by the first clothing change.'}}
        self.assertEqual(build.selection_kind(turn), 'event_after')
        turn['prompt']['en'] = 'Please use the version where the person has black hair.'
        self.assertEqual(build.selection_kind(turn), 'content_description_match')


if __name__ == '__main__':
    unittest.main(verbosity=2)
