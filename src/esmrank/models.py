"""Download the pretrained model files from Zenodo."""
import gzip
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .paths import (MODEL_DIR_ENV, REQUIRED_MODEL_FILES, default_model_dir,
                    find_model_dir, missing_model_files)

ZENODO_RECORD = '18773439'
ARCHIVE_NAME = 'ESMRank.tar.gz'
ARCHIVE_URL = f'https://zenodo.org/records/{ZENODO_RECORD}/files/{ARCHIVE_NAME}?download=1'
# md5 as shown on the Zenodo record page; skip the check with --no-verify.
ARCHIVE_MD5 = '5e073fe91a50523ff03baf0f4307773c'


def _md5_of(path):
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            md5.update(chunk)
    return md5.hexdigest()


def _download(url, dest, expected_md5=None):
    """Download ``url`` to ``dest`` (via ``dest.part``; only kept if the checksum matches)."""
    dest = Path(dest)
    part = dest.with_name(dest.name + '.part')
    md5 = hashlib.md5()
    with urllib.request.urlopen(url) as resp, open(part, 'wb') as out:
        total = int(resp.headers.get('Content-Length') or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            md5.update(chunk)
            done += len(chunk)
            if total:
                print(f'\rdownloading {done / 1e6:,.0f} / {total / 1e6:,.0f} MB',
                      end='', file=sys.stderr, flush=True)
    print(file=sys.stderr)
    if expected_md5 and md5.hexdigest() != expected_md5:
        part.unlink()
        raise RuntimeError(
            f'checksum mismatch for {ARCHIVE_NAME}: got {md5.hexdigest()}, '
            f'expected {expected_md5} (re-run with --no-verify to skip the check)')
    os.replace(part, dest)


def _safe_extract(archive, dest):
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, 'data_filter'):
            tar.extractall(dest, filter='data')
            return
        dest_root = Path(dest).resolve()
        for member in tar.getmembers():
            target = (dest_root / member.name).resolve()
            if dest_root != target and dest_root not in target.parents:
                raise RuntimeError(f'unsafe path in archive: {member.name}')
        tar.extractall(dest)


def _locate(extracted, name):
    """Find ``name`` anywhere in the extracted archive -> (path, is_gzipped) or (None, False)."""
    hits = sorted(Path(extracted).rglob(name))
    if hits:
        return hits[0], False
    hits = sorted(Path(extracted).rglob(name + '.gz'))
    if hits:
        return hits[0], True
    return None, False


def _place(src, target, gzipped=False):
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    if gzipped:
        with gzip.open(src, 'rb') as f_in, open(target, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        shutil.move(str(src), str(target))


def _describe(extracted, limit=25):
    files = [p for p in Path(extracted).rglob('*') if p.is_file()]
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    lines = [f'  {p.relative_to(extracted)}  ({p.stat().st_size / 1e6:,.1f} MB)'
             for p in files[:limit]]
    more = f'\n  ... and {len(files) - limit} more files' if len(files) > limit else ''
    return '\n'.join(lines) + more


def download_models(dest=None, force=False, verify=True, archive=None):
    """Download and unpack the model files into ``dest`` (default: user cache).

    Each required file is looked up anywhere inside the archive. The downloaded
    archive is kept next to ``dest`` if installation fails, so a retry does not
    download it again. Pass ``archive=`` to use an already downloaded
    ``ESMRank.tar.gz`` instead.
    """
    dest = Path(dest) if dest else default_model_dir()
    if not force and not missing_model_files(dest):
        print(f'model files already present in {dest} (use --force to re-download)')
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    downloaded = None
    if archive is not None:
        archive = Path(archive)
        if verify and _md5_of(archive) != ARCHIVE_MD5:
            raise RuntimeError(
                f'{archive} does not match the expected md5 (use --no-verify to skip the check)')
    else:
        dl_dir = dest.parent / '.esmrank-download'
        dl_dir.mkdir(exist_ok=True)
        archive = downloaded = dl_dir / ARCHIVE_NAME
        if archive.exists() and (not verify or _md5_of(archive) == ARCHIVE_MD5):
            print(f'reusing previously downloaded {archive}')
        else:
            print(f'fetching {ARCHIVE_URL}')
            _download(ARCHIVE_URL, archive, ARCHIVE_MD5 if verify else None)

    with tempfile.TemporaryDirectory(dir=dest.parent, prefix='.esmrank-dl-') as tmp:
        print('extracting archive')
        extracted = Path(tmp) / 'extracted'
        extracted.mkdir()
        _safe_extract(archive, extracted)

        dest.mkdir(parents=True, exist_ok=True)
        found = {name: _locate(extracted, name) for name in REQUIRED_MODEL_FILES}
        absent = [name for name, (src, _) in found.items() if src is None]
        listing = _describe(extracted) if absent else ''  # before files are moved out
        for name, (src, gzipped) in found.items():
            if src is not None:
                _place(src, dest / name, gzipped)

        missing = missing_model_files(dest)
        if missing:
            raise RuntimeError(
                f'{ARCHIVE_NAME} does not contain: {", ".join(missing)}.\n'
                f'Largest files in the archive:\n{listing}\n'
                f'The archive is kept at {archive}; if you obtain the missing file(s) '
                f'elsewhere, copy them into {dest}.')

    if downloaded is not None:
        downloaded.unlink(missing_ok=True)
        try:
            downloaded.parent.rmdir()
        except OSError:
            pass
    print(f'models installed in {dest}')
    return dest


def ensure_models(model_dir=None, *, verify=True):
    """Return a directory with the model files, downloading them only if missing.

    With ``model_dir=None`` the usual lookup applies (``$ESMRANK_MODEL_DIR``,
    ``./model``, the user cache); if nothing complete is found, the files are
    downloaded into ``$ESMRANK_MODEL_DIR`` when set, else the user cache.
    With an explicit ``model_dir`` that is incomplete, they are downloaded there.
    """
    if model_dir is None:
        try:
            return find_model_dir()
        except FileNotFoundError:
            model_dir = os.environ.get(MODEL_DIR_ENV) or default_model_dir()
    elif not missing_model_files(model_dir):
        return Path(model_dir)
    return download_models(dest=model_dir, verify=verify)
