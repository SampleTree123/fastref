#!/usr/bin/env python3
"""Copy a completed gallery to a self-contained folder, ready for local viewing or GitHub.

Existing explicit .zip callers remain supported. Normal output is a folder with
index.html and regular media files; source symlinks are materialized on export.
"""
from __future__ import annotations
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if (p / "common/project_layout.py").is_file()) / "common"))
from project_layout import ROOT as _PROJECT_ROOT, resolve_path, script_path, add_script_paths, resolve_cli_paths
add_script_paths()

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import shutil
from urllib.parse import unquote, urlsplit
import uuid
from zipfile import ZIP_STORED, ZipFile, ZipInfo


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_file(path):
    digest=hashlib.sha256()
    with resolve_path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


class MediaReferences(HTMLParser):
    """Collect static embedded media; reject unsupported external dependencies."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.paths=set();self.in_style=False

    def reference(self, url, *, anchor=False):
        if not url or url.startswith('#'):
            return
        parsed=urlsplit(url)
        if parsed.scheme == 'data':
            return  # Already contained in HTML.
        if anchor and (parsed.scheme or parsed.netloc):
            return  # Informational outgoing links are not embedded resources.
        require(not parsed.scheme and not parsed.netloc, f'External embedded resource is not offline: {url}')
        relative=unquote(parsed.path)
        require(relative and '\\' not in relative and '\x00' not in relative, f'Unsafe media URL: {url}')
        pure=PurePosixPath(relative)
        require(not pure.is_absolute() and '..' not in pure.parts, f'Unsafe archive-relative media path: {url}')
        if anchor and str(pure) in ('index.html','comparison.json'):
            return
        require(len(pure.parts)>=2 and pure.parts[0] in ('media', 'cases'), f'Unsupported non-media embedded resource: {url}')
        self.paths.add(pure.as_posix())

    def css(self, text):
        require('@import' not in text.lower(), 'CSS imports are not supported by this offline exporter')
        for match in re.finditer(r'url\(\s*([\'"]?)(.*?)\1\s*\)',text,re.I):
            self.reference(match[2])

    def handle_starttag(self, tag, attrs):
        values=dict(attrs)
        require('srcset' not in values, 'srcset needs explicit exporter support; refusing to omit media')
        if tag=='style':self.in_style=True
        for attribute in ('src','poster'):
            if values.get(attribute):self.reference(values[attribute])
        if values.get('href'):
            self.reference(values['href'],anchor=tag=='a')
        if values.get('style'):self.css(values['style'])

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag,attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag=='style':self.in_style=False

    def handle_data(self, data):
        if self.in_style:self.css(data)


def referenced_media(html_text):
    parser=MediaReferences();parser.feed(html_text);parser.close()
    require(parser.paths,'Gallery has no media references')
    return sorted(parser.paths)


def regular_info(name):
    info=ZipInfo(name)
    info.compress_type=ZIP_STORED
    info.create_system=3
    info.external_attr=(stat.S_IFREG|0o644)<<16
    return info


def export_folder(gallery, output):
    gallery = Path(gallery).resolve()
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), 'Export destination already exists')
    html_path = gallery/'index.html'
    html_bytes = html_path.read_bytes()
    references = referenced_media(html_bytes.decode('utf-8'))
    targets = {name: (gallery/name).resolve(strict=True) for name in references}
    require(all(path.is_file() for path in targets.values()), 'Missing media file')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name('.'+output.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.mkdir()
        (temporary/'index.html').write_bytes(html_bytes)
        (temporary/'.nojekyll').write_text('')
        for name, source in targets.items():
            dest = temporary/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            stem = PurePosixPath(name).stem
            if re.fullmatch('[0-9a-f]{64}', stem):
                require(sha_file(dest) == stem, f'Media differs from gallery content hash: {name}')
        require(html_path.read_bytes() == html_bytes, 'Gallery changed during export')
        temporary.rename(output)
        return {'status': 'completed', 'directory': str(output), 'media_count': len(references),
                'html_sha256': hashlib.sha256(html_bytes).hexdigest()}
    finally:
        if temporary.exists(): shutil.rmtree(temporary)


def export_gallery(gallery, output):
    gallery=Path(gallery).resolve();output=Path(output).absolute()
    if output.suffix.lower() != '.zip':
        return export_folder(gallery, output)
    receipt_path=output.with_suffix(output.suffix+'.receipt.json')
    require(not output.exists() and not output.is_symlink() and not receipt_path.exists(), 'Export destination already exists; choose a new filename')
    html_path=gallery/'index.html';comparison_path=gallery/'comparison.json'
    html_bytes=html_path.read_bytes();comparison_bytes=comparison_path.read_bytes()
    comparison=json.loads(comparison_bytes)
    html_sha=hashlib.sha256(html_bytes).hexdigest()
    require(comparison.get('status')=='completed' and comparison.get('html_sha256')==html_sha,
            'Gallery is incomplete or index.html differs from its receipt')
    media=referenced_media(html_bytes.decode('utf-8'))
    # Resolve only for reading: archive names retain their safe relative paths.
    targets={name:(gallery/name).resolve(strict=True) for name in media}
    require(all(path.is_file() for path in targets.values()),'A referenced media file is missing or not a regular file')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name('.'+output.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary_receipt=temporary.with_suffix('.receipt.tmp')
    try:
        with ZipFile(temporary,'w',compression=ZIP_STORED,allowZip64=True) as archive:
            archive.writestr(regular_info('index.html'),html_bytes)
            archive.writestr(regular_info('comparison.json'),comparison_bytes)
            for name,path in targets.items():
                before=path.stat();digest=hashlib.sha256();count=0
                require(stat.S_ISREG(before.st_mode),f'Not a regular media file: {name}')
                with path.open('rb') as source, archive.open(regular_info(name),'w',force_zip64=True) as dest:
                    for chunk in iter(lambda:source.read(1024*1024),b''):
                        dest.write(chunk);digest.update(chunk);count+=len(chunk)
                after=path.stat()
                require(count==before.st_size and (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
                        ==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),f'Media changed during export: {name}')
                stem=PurePosixPath(name).stem
                if re.fullmatch('[0-9a-f]{64}',stem):
                    require(digest.hexdigest()==stem,f'Media differs from gallery content hash: {name}')
        require(html_path.read_bytes()==html_bytes and comparison_path.read_bytes()==comparison_bytes,
                'Gallery changed during export; wait for its update to finish')
        receipt={'schema':'datamake_offline_gallery_export_v1','status':'completed',
                 'created_at':datetime.now(timezone.utc).isoformat(),'gallery':str(gallery),
                 'archive':str(output),'archive_bytes':temporary.stat().st_size,'archive_sha256':sha_file(temporary),
                 'media_count':len(media),'entry_count':len(media)+2,'compression':'ZIP_STORED',
                 'html_sha256':html_sha,'comparison_sha256':hashlib.sha256(comparison_bytes).hexdigest()}
        temporary_receipt.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
        # Publishing an already closed ZIP by hard link is atomic and never
        # clobbers a concurrently created destination. Temporary files are local
        # to output.parent so this does not require a cross-filesystem rename.
        os.link(temporary,output)
        os.link(temporary_receipt,receipt_path)
        return receipt
    finally:
        temporary.unlink(missing_ok=True);temporary_receipt.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); resolve_cli_paths(args)
    print(json.dumps(export_gallery(args.gallery,args.output),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
