#!/usr/bin/env python3
"""CPU-only campaign/freezing simulations; never starts model workers or Slurm."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import freeze
import run_campaign as campaign
from queue_client import atomic_json


class FreezeChecks(unittest.TestCase):
    def test_snapshot_verify_and_reject_changed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'qwen'; root.mkdir()
            source = root / 'input.json'; source.write_text('{"version":1}')
            output = root / 'run'
            with patch.object(freeze, 'ROOT', root), patch.object(freeze, 'files', return_value=[source]):
                with patch.object(sys, 'argv', ['freeze', '--output', str(output)]):
                    freeze.main()
                self.assertEqual((output / 'frozen_sources/qwen/input.json').read_text(), source.read_text())
                with patch.object(sys, 'argv', ['freeze', '--output', str(output), '--verify']):
                    freeze.main()
                    source.write_text('{"version":2}')
                    with self.assertRaisesRegex(ValueError, 'Frozen input/code changed'):
                        freeze.main()


class CampaignChecks(unittest.TestCase):
    def test_cached_measurement_does_not_replace_real_throughput(self):
        stages = [{'mode': 'closed_loop', 'seconds_per_new_case': 80},
                  {'mode': 'closed_loop', 'seconds_per_new_case': None},
                  {'mode': 'understanding', 'seconds_per_new_case': 1000}]
        self.assertEqual(campaign.measured_seconds_per_chain(stages), 80)

    def simulate(self, root, *, hours=17, clock=None, failures=()):
        clock = clock or [1000.0]
        data = root / 'datasets/editing_agent_v1'; data.mkdir(parents=True, exist_ok=True)
        for name, n, field in [('understanding_test.jsonl', 1000, 'sample_id'), ('chains_test.jsonl', 128, 'chain_id')]:
            (data / name).write_text(''.join(json.dumps({field: f'case_{i}'})+'\n' for i in range(n)))
        output = root / 'run'; calls = []

        def fake_run(args, timeout):
            self.assertGreater(timeout, 0)
            script = Path(args[0]).name
            args = [str(x) for x in args]
            def val(flag): return args[args.index(flag)+1]
            if script == 'freeze.py':
                return
            if script == 'launch_allocation.py':
                mode, limit = val('--mode'), int(val('--limit'))
                calls.append((mode, limit))
                path = Path(val('--output'))
                dispatch = [json.loads(line) for line in Path(val('--dataset')).read_text().splitlines()]
                self.assertEqual(len(dispatch), limit)
                for row in dispatch:
                    cid = row.get('chain_id', row.get('sample_id'))
                    result = path / 'cases' / cid / 'worker_result.json'
                    if not result.exists():
                        atomic_json(result, {'status': 'error' if mode=='closed_loop' and cid in failures else 'ok'})
                        clock[0] += .1 if mode == 'understanding' else 10
                    else:
                        self.fail('Committed successes/errors must not be dispatched again')
            elif script == 'report.py':
                path = Path(val('--run')); limit = int(val('--limit'))
                results = [json.loads(p.read_text()) for i in range(limit)
                           if (p := path / 'cases' / f'case_{i}' / 'worker_result.json').exists()]
                ok = sum(r['status']=='ok' for r in results)
                errors = sum(r['status']=='error' for r in results)
                atomic_json(path / 'report/summary.json', {'cases_ok': ok, 'cases_error': errors, 'cases_pending': limit-ok-errors})
            else:
                self.fail(script)

        with patch.object(campaign, 'ROOT', root), patch.object(campaign, 'run_command', side_effect=fake_run), \
             patch.object(campaign.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(campaign.signal, 'signal'), patch.dict(os.environ, {'SLURM_JOB_END_TIME': ''}), \
             patch.object(sys, 'argv', ['campaign', '--output', str(output), '--gpus', '8', '--hours', str(hours)]):
            campaign.main()
        return calls, json.loads((output / 'campaign.json').read_text()), clock

    def test_order_expansion_and_resume_without_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls, state, clock = self.simulate(root)
            self.assertEqual(calls, [('understanding', 1000), ('closed_loop', 30), ('closed_loop', 30),
                                     ('closed_loop', 40), ('closed_loop', 28)])
            self.assertEqual(state['closed_loop_requested'], 128)
            self.assertEqual(campaign.measured_seconds_per_chain(state['stages']), 10)
            calls, resumed, _ = self.simulate(root, clock=clock)
            self.assertEqual(calls, [])
            self.assertEqual(resumed['closed_loop_requested'], 128)
            self.assertEqual(campaign.measured_seconds_per_chain(resumed['stages']), 10)

    def test_remaining_time_stops_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, state, _ = self.simulate(Path(tmp), hours=.9)
            self.assertEqual(calls, [('understanding', 1000), ('closed_loop', 30)])
            self.assertEqual(state['closed_loop_requested'], 30)
            self.assertIn('Insufficient', state['expansion_stopped_reason'])

    def test_two_failed_chains_are_retained_and_two_new_chains_fill_minimum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls, state, clock = self.simulate(root, failures={'case_0', 'case_1'})
            self.assertEqual(calls[:3], [('understanding', 1000), ('closed_loop', 30), ('closed_loop', 2)])
            supplement = next(s for s in state['stages'] if s['mode']=='closed_loop' and s['limit']==32)
            self.assertEqual(supplement['summary'], {'cases_ok':30, 'cases_error':2, 'cases_pending':0})
            self.assertEqual(supplement['dispatched_cases'], 2)
            self.assertFalse(state['minimum_pending'])
            self.assertEqual(state['minimum_completed'], 126)
            calls, resumed, _ = self.simulate(root, clock=clock, failures={'case_0', 'case_1'})
            self.assertEqual(calls, [])
            self.assertFalse(resumed['minimum_pending'])

    def test_two_failures_with_insufficient_time_do_not_claim_thirty_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, state, _ = self.simulate(Path(tmp), hours=.85, failures={'case_0', 'case_1'})
            self.assertEqual(calls, [('understanding', 1000), ('closed_loop', 30)])
            self.assertEqual(state['status'], 'paused_time_limit')
            self.assertTrue(state['minimum_pending'])
            self.assertEqual(state['minimum_completed'], 28)

    def test_command_timeout_reaps_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / 'sleep.py'
            marker = Path(tmp) / 'pid'
            script.write_text('import os,time\nfrom pathlib import Path\nPath('+repr(str(marker))+').write_text(str(os.getpid()))\ntime.sleep(30)\n')
            original_popen = subprocess.Popen
            processes = []
            def capture(*args, **kwargs):
                proc = original_popen(*args, **kwargs); processes.append(proc); return proc
            with patch.object(campaign.subprocess, 'Popen', side_effect=capture):
                with self.assertRaises(subprocess.TimeoutExpired):
                    campaign.run_command([script], timeout=.2)
            pid = processes[0].pid
            self.assertIsNotNone(processes[0].poll())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
