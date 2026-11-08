"""Reuse a successful payload checksum while the file's identity is unchanged."""
import hashlib


def file_signature(path):
    stat = path.stat()
    return {'device': stat.st_dev, 'inode': stat.st_ino, 'bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}


def verify_payload(path, artifact):
    before = file_signature(path)
    if before['bytes'] != artifact['bytes']:
        raise ValueError(f'Cache size changed: {path}')
    if artifact.get('verified_stat') == before and artifact.get('verified_sha256') == artifact['sha256']:
        return before
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    after = file_signature(path)
    if before != after or digest.hexdigest() != artifact['sha256']:
        raise ValueError(f'Cache changed or checksum mismatch: {path}')
    return after
