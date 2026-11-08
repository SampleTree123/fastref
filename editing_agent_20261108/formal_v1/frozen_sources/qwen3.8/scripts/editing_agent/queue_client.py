"""Local file-backed LTX tool; arguments are supplied by the checked harness."""
import hashlib
import json
import time
import os
import tempfile
from pathlib import Path


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.tmp-', suffix='.json')
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class LTXQueueClient:
    def __init__(self, queue, timeout=1800):
        self.queue, self.timeout = Path(queue), timeout

    def __call__(self, request):
        request = dict(request)
        # Give identical logical names in different chains distinct queue keys.
        material = json.dumps(request, ensure_ascii=False, sort_keys=True)
        request['request_id'] = 'edit_' + hashlib.sha256(material.encode()).hexdigest()[:24]
        request_id = request['request_id']
        self.queue.mkdir(parents=True, exist_ok=True)
        request_path = self.queue / 'requests' / (request_id + '.json')
        response_path = self.queue / 'responses' / (request_id + '.json')
        if request_path.exists():
            if json.loads(request_path.read_text()) != request:
                raise ValueError('Tool request identity changed')
        else:
            atomic_json(request_path, request)
        start = time.monotonic()
        while not response_path.exists():
            if (self.queue / 'fatal.json').exists():
                raise RuntimeError('LTX worker failed: ' + (self.queue / 'fatal.json').read_text())
            ready = self.queue / 'ready.json'
            if ready.exists():
                state = json.loads(ready.read_text())
                if state.get('status') in ('error', 'stopped'):
                    raise RuntimeError('LTX worker unavailable: ' + json.dumps(state, ensure_ascii=False))
            if time.monotonic() - start > self.timeout:
                raise TimeoutError('LTX tool did not return within limit: ' + request_id)
            time.sleep(0.5)
        result = json.loads(response_path.read_text())
        if result.get('request_id') != request_id:
            raise ValueError('LTX response ID mismatch')
        request_sha = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True,
                                                 separators=(',', ':')).encode()).hexdigest()
        if result.get('request_sha256') != request_sha:
            raise ValueError('LTX response does not match the submitted request')
        if result.get('status') not in ('ok', 'error'):
            raise ValueError('LTX response has no valid completion status')
        if result['status'] == 'ok':
            video = Path(result.get('video_path', ''))
            if not video.is_absolute() or not video.is_file() or video.stat().st_size == 0:
                raise ValueError('LTX response video is missing or empty')
        return result
