"""Keep validation checkpoints; atomically commit lineage before retiring bundles."""
from contextlib import contextmanager, ExitStack
import fcntl
import json
from pathlib import Path
import time


@contextmanager
def retention_locks(checkpoints):
    """A resume chain may span directories; acquire every directory in fixed order."""
    directories = sorted({Path(p).resolve().parent for p in checkpoints}, key=str)
    with ExitStack() as stack:
        for directory in directories:
            handle = stack.enter_context((directory / '.retention.lock').open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX)
        yield  # Closing handles also releases all locks, including on exceptions.


@contextmanager
def retention_lock(checkpoint):
    with retention_locks([checkpoint]):
        yield


def marker(checkpoint):
    return Path(checkpoint).with_suffix('.validation.json')


def is_pinned(checkpoint):
    return marker(checkpoint).is_file()


def pin_for_validation(checkpoint, purpose):
    from native_training_contract import validate_resume, atomic_json, fingerprint, require
    checkpoint = Path(checkpoint).resolve()
    with retention_lock(checkpoint):
        receipt = json.loads(checkpoint.with_suffix('.contract.json').read_text())
        path = marker(checkpoint)
        identity_sha = fingerprint(receipt['identity'])
        files_sha = fingerprint(receipt['files'])
        if path.exists():
            selected = json.loads(path.read_text())
            if selected.get('identity_sha256') == identity_sha and selected.get('files_sha256') == files_sha:
                # The controller already verified the complete bundle before pinning.
                # Workers check completeness without serially hashing it again.
                require(selected['checkpoint'] == checkpoint.name and selected['step'] == receipt['step'],
                        'Validation marker refers to another checkpoint')
                require(all((checkpoint.parent / name).is_file()
                            and (checkpoint.parent / name).stat().st_size == evidence['bytes']
                            for name, evidence in receipt['files'].items()), 'Pinned checkpoint bundle is incomplete')
                return
        validate_resume(checkpoint, receipt['identity'], 2 ** 63)
        atomic_json(path, {'checkpoint': checkpoint.name, 'step': receipt['step'],
                           'selected_unix': time.time(), 'purpose': str(purpose),
                           'identity_sha256': identity_sha, 'files_sha256': files_sha})


def select_retained(candidates, keep):
    if type(keep) is not int or keep < 1:
        raise ValueError('Retain at least the newest checkpoint')
    recent = set(candidates[-keep:])
    return [p for p in candidates if p in recent or is_pinned(p)]


def remove_if_unpinned(checkpoint, identity):
    from native_training_contract import remove_checkpoint_bundle
    with retention_lock(checkpoint):
        if is_pinned(checkpoint):
            return False
        remove_checkpoint_bundle(checkpoint, identity)
        return True


def commit_retention(checkpoint, candidates, keep, receipt, identity, *, pin_purpose=None):
    """Select, publish the new lineage and delete under the same directory locks.

    A concurrent validation either pins before selection and is retained in the
    new lineage, or sees that deletion already completed. No late pin is omitted.
    Publish the complete new checkpoint before deleting any predecessor.
    """
    from native_training_contract import atomic_json, remove_checkpoint_bundle, require, fingerprint
    checkpoint = Path(checkpoint).resolve()
    candidates = [Path(p).resolve() for p in candidates]
    require(candidates and candidates[-1] == checkpoint and len(set(candidates)) == len(candidates),
            'Retention must end with the newly committed checkpoint exactly once')
    with retention_locks(candidates):
        retained = select_retained(candidates, keep)
        committed = {**receipt, 'retention_lineage': [str(p) for p in retained]}
        if pin_purpose is not None:
            # The saver just hashed the complete bundle. Publish its selection
            # before the completion receipt: after a crash, a resumable selected
            # checkpoint must already be pinned. Readers hold this same lock.
            atomic_json(marker(checkpoint), {
                'checkpoint': checkpoint.name, 'step': receipt['step'],
                'selected_unix': time.time(), 'purpose': str(pin_purpose),
                'identity_sha256': fingerprint(receipt['identity']),
                'files_sha256': fingerprint(receipt['files'])})
        atomic_json(checkpoint.with_suffix('.contract.json'), committed)
        for old in candidates:
            if old not in retained:
                remove_checkpoint_bundle(old, identity)
        return retained


def finish_retention(trainer, final_checkpoint):
    """Call only after successful completion of the complete training plan."""
    final_checkpoint = Path(final_checkpoint).resolve()
    receipt = json.loads(final_checkpoint.with_suffix('.contract.json').read_text())
    trainer._retention_paths = commit_retention(final_checkpoint, trainer._retention_paths, 1,
                                                receipt, trainer.launch_identity)
