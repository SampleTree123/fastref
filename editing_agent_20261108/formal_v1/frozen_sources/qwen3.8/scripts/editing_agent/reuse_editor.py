"""Optional development-only reuse of identical, verified actual LTX generations.

Usage: editor = StrictReuseEditor(LTXQueueClient(queue), previous_run=v2_run)
Then call editor(request) through the unchanged harness. Qwen decisions and
post-generation reviews still run normally. No labels or target videos are read.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

from queue_client import atomic_json


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def semantic_key(request, media_sha, worker_sha):
    """Only computation-relevant inputs; file names and conversation IDs aren't inputs."""
    refs = request['references']
    if len(media_sha) != 1 + len(refs):
        raise ValueError('Receipt media count differs from request')
    return digest({'source_sha256': media_sha[0], 'prompt': request['prompt'].strip(),
                   'references': [{'sha256': sha, 'type': ref['type'], 'view': ref.get('view', '')}
                                  for ref, sha in zip(refs, media_sha[1:])],
                   'seed': request.get('seed', 42), 'worker_identity_sha256': worker_sha})


class StrictReuseEditor:
    """Wrap a queue client without changing model inference or current worker code.

    A previous receipt is considered only when its execution digest, input
    media bytes, output bytes and current complete worker identity all match.
    No current worker ready identity means no reuse: the normal queue handles it.
    """
    def __init__(self, fallback, previous_run, ready_path=None, wait_ready_seconds=0):
        self.fallback = fallback
        self.previous_run = Path(previous_run).resolve()
        self.ready_path = Path(ready_path) if ready_path is not None else Path(fallback.queue) / 'ready.json'
        if wait_ready_seconds < 0:
            raise ValueError('wait_ready_seconds must be nonnegative')
        self.wait_ready_seconds = wait_ready_seconds
        self.stats = Counter()
        self.index = defaultdict(list)
        # Only actual generation receipts under cases; never public/private GT,
        # gallery targets, or an entire model workspace searched indiscriminately.
        for path in sorted((self.previous_run / 'cases').rglob('editing_generation.json')):
            try:
                old = json.loads(path.read_text())
                if old.get('status') != 'ok':
                    continue
                self._validate_execution_receipt(old)
                key = semantic_key(old['request'], old['input_media_sha256'], old['worker_identity_sha256'])
                self.index[key].append((path, file_sha(path)))
            except (OSError, ValueError, TypeError, KeyError):
                self.stats['invalid_receipt'] += 1

    @staticmethod
    def _validate_execution_receipt(record):
        expected = digest({'request': record['request'], 'media_sha256': record['input_media_sha256'],
                           'worker': record['worker_identity_sha256']})
        if record['execution_sha256'] != expected:
            raise ValueError('Previous execution digest differs')
        hashes = [record['worker_identity_sha256'], record['video_sha256'], *record['input_media_sha256']]
        if any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value) for value in hashes):
            raise ValueError('Receipt lacks valid content hashes')

    def _current_worker(self):
        try:
            ready = json.loads(self.ready_path.read_text())
            identity = json.loads(self.ready_path.with_name('worker_identity.json').read_text())
            required = {'schema', 'checkpoint_sha256', 'config_sha256', 'reference_protocol',
                        'weights_sha256', 'steps', 'runtime_sha256'}
            if not required <= identity.keys() or ready.get('status') != 'ready':
                return None
            if identity['schema'] != 'datamake_editing_tool_v1' or type(identity['steps']) is not int or identity['steps'] <= 0:
                return None
            actual = digest(identity)
            return actual if ready.get('worker_identity_sha256') == actual else None
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def _wait_current_worker(self):
        deadline = time.monotonic() + self.wait_ready_seconds
        while True:
            try:
                ready = json.loads(self.ready_path.read_text())
                # A ready-but-different identity must never wait until another
                # identity happens to match. Loading/missing is the only wait.
                if ready.get('status') != 'loading':
                    return self._current_worker()
            except FileNotFoundError:
                pass
            except (OSError, ValueError, TypeError):
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(.25, remaining))

    @staticmethod
    def _inputs(request):
        # Reject extra inference controls rather than silently omit them from the
        # match key. The native editor performs full validation on fallback.
        allowed = {'request_id', 'source_video', 'prompt', 'references', 'output_dir', 'seed'}
        if set(request) - allowed or not {'request_id', 'source_video', 'prompt', 'references', 'output_dir'} <= request.keys():
            raise ValueError('Unsupported request fields for reuse')
        if not isinstance(request['prompt'], str) or not request['prompt'].strip() or len(request['prompt']) > 16000:
            raise ValueError('Empty prompt')
        seed = request.get('seed', 42)
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('Seed is not an integer')
        if not isinstance(request['request_id'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}', request['request_id']):
            raise ValueError('Invalid request ID')
        if not Path(request['output_dir']).is_absolute() or not isinstance(request['references'], list):
            raise ValueError('Invalid output directory or reference list')
        paths = [request['source_video']]
        ids = set()
        for ref in request['references']:
            if set(ref) - {'reference_id', 'path', 'type', 'view'}:
                raise ValueError('Unsupported reference fields')
            if not isinstance(ref['reference_id'], (str, int)) or isinstance(ref['reference_id'], bool) or str(ref['reference_id']) in ids:
                raise ValueError('Duplicate reference ID')
            if ref['type'] not in {'motion', 'identity', 'hair', 'garment', 'scene', 'outfit', 'camera'} or not isinstance(ref.get('view', ''), str):
                raise ValueError('Unsupported reference type/view')
            ids.add(str(ref['reference_id']))
            paths.append(ref['path'])
        if any(not Path(p).is_absolute() or not Path(p).is_file() for p in paths):
            raise ValueError('Inputs must be existing absolute files')
        return paths, [file_sha(p) for p in paths]

    def __call__(self, request):
        began = time.monotonic()
        worker = self._wait_current_worker()
        if worker is not None:
            try:
                current_paths, current_sha = self._inputs(request)
                key = semantic_key(request, current_sha, worker)
            except (OSError, ValueError, TypeError, KeyError):
                key = None
            for path, receipt_sha in self.index.get(key, []):
                try:
                    # Revalidate on every hit, not just constructor time.
                    if file_sha(path) != receipt_sha:
                        raise ValueError('Previous receipt changed')
                    old = json.loads(path.read_text())
                    self._validate_execution_receipt(old)
                    _, old_sha = self._inputs(old['request'])
                    if old_sha != old['input_media_sha256'] or old_sha != current_sha:
                        raise ValueError('Previous input media changed')
                    video = Path(old['video_path']).resolve()
                    if video != (path.parent / 'edited.mp4').resolve():
                        raise ValueError('Receipt video is not its committed actual generation')
                    if not video.is_file() or video.stat().st_size != old['bytes'] or old['bytes'] <= 0 or file_sha(video) != old['video_sha256']:
                        raise ValueError('Previously generated output changed or is missing')
                    if self._current_worker() != worker:
                        raise ValueError('Worker identity changed during lookup')
                    proof = {'kind': 'previous_actual_ltx_generation', 'not_ground_truth': True,
                             'previous_run': str(self.previous_run), 'previous_receipt': str(path),
                             'previous_receipt_sha256': receipt_sha, 'previous_execution_sha256': old['execution_sha256'],
                             'semantic_key_sha256': key, 'worker_identity_sha256': worker,
                             'current_request': request, 'current_input_paths': current_paths,
                             'input_media_sha256': current_sha, 'video_path': str(video),
                             'video_sha256': old['video_sha256'], 'verified_unix': time.time()}
                    atomic_json(Path(request['output_dir']) / 'reuse_provenance.json', proof)
                    self.stats['reused'] += 1
                    return {'status': 'ok', 'request_id': request['request_id'], 'video_path': str(video),
                            'video_sha256': old['video_sha256'], 'bytes': old['bytes'],
                            'worker_identity_sha256': worker, 'reused': True, 'reuse_provenance': proof,
                            'timings': {'reuse_verification_seconds': time.monotonic() - began}}
                except (OSError, ValueError, TypeError, KeyError):
                    self.stats['rejected_candidate'] += 1
        self.stats['generated_fallback'] += 1
        return self.fallback(request)
