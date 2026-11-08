#!/usr/bin/env python3
"""CPU mocks only: no scheduler command is ever executed by these tests."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import handoff


class HandoffChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name).resolve()
        self.manifest = self.output / 'freeze_manifest.json'
        self.manifest.write_text('{"files":{}}')
        self.owner = {'output': str(self.output), 'job_id': '194849', 'step_id': '12', 'gpus': 4,
                      'pid': 123, 'started_unix': 10,
                      'freeze_manifest_sha256': hashlib.sha256(self.manifest.read_bytes()).hexdigest()}
        self.save_owner()
        self.commands = []

    def tearDown(self):
        self.tmp.cleanup()

    def save_owner(self):
        (self.output / 'campaign_owner.json').write_text(json.dumps(self.owner))

    def runner(self, command, **kwargs):
        self.commands.append(command)
        return SimpleNamespace(stdout=f'StepId=194849.12 UserId=test({os.getuid()}) State=RUNNING')

    def test_only_authorized_step_is_signalled_then_lock_waited(self):
        checks = iter([False, False, False, True])
        clock = [0]
        def sleep(_): clock[0] += 1
        result = handoff.handoff(self.output, '195999', timeout=5, run=self.runner,
                                 available=lambda _: next(checks), clock=lambda: clock[0], sleep=sleep)
        self.assertEqual(result['status'], 'handoff_ready')
        self.assertFalse(result['source_allocation_cancelled'])
        self.assertEqual([c for c in self.commands if c[0] == 'scancel'],
                         [['scancel', '--signal=TERM', '194849.12']])

    def test_timeout_never_cancels_allocation(self):
        clock = [0]
        def sleep(_): clock[0] += 1
        with self.assertRaises(TimeoutError):
            handoff.handoff(self.output, '195999', timeout=3, run=self.runner,
                            available=lambda _: False, clock=lambda: clock[0], sleep=sleep)
        self.assertEqual([c for c in self.commands if c[0] == 'scancel'],
                         [['scancel', '--signal=TERM', '194849.12']])

    def test_wrong_output_job_or_non_numeric_step_refused(self):
        original = dict(self.owner)
        for change in ({'output': '/different/output'}, {'job_id': '194133'},
                       {'step_id': 'batch'}, {'step_id': 'extern'}, {'gpus': 8}):
            self.owner = {**original, **change}; self.save_owner(); self.commands.clear()
            with self.assertRaises(ValueError):
                handoff.handoff(self.output, '195999', run=self.runner, available=lambda _: False)
            self.assertFalse(any(c[0] == 'scancel' for c in self.commands))

    def test_wrong_slurm_user_or_step_refused(self):
        with self.assertRaises(ValueError):
            handoff.verify_step(f'StepId=194849.13 UserId=test({os.getuid()})', '194849.12', os.getuid())
        with self.assertRaises(ValueError):
            handoff.verify_step(f'StepId=194849.12 UserId=test({os.getuid()+1})', '194849.12', os.getuid())

    def test_free_output_does_not_signal_anything(self):
        result = handoff.handoff(self.output, '195999', run=self.runner, available=lambda _: True)
        self.assertEqual(result['status'], 'no_active_campaign')
        self.assertFalse(any(c[0] in ('scancel', 'scontrol') for c in self.commands))

    def test_real_shared_lock_detection(self):
        self.assertTrue(handoff.lock_available(self.output))
        with (self.output / '.campaign.lock').open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertFalse(handoff.lock_available(self.output))
        self.assertTrue(handoff.lock_available(self.output))


if __name__ == '__main__':
    unittest.main(verbosity=2)
