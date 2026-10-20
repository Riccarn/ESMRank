"""Download the pretrained model files from Zenodo."""
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


def _download(url, dest, expected_md5=None):
    md5 = hashlib.md5()
    with urllib.request.urlopen(url) as resp, open(dest, 'wb') as out:
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
        raise RuntimeError(
            f'checksum mismatch for {ARCHIVE_NAME}: got {md5.hexdigest()}, '
            f'expected {expected_md5} (re-run with --no-verify to skip the check)')


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


def _find_model_root(extracted):
    """Directory (inside the extracted archive) that holds the model files."""
    for marker in REQUIRED_MODEL_FILES:
        hits = sorted(Path(extracted).rglob(marker))
        if hits:
            return hits[0].parent
    return None


def download_models(dest=None, force=False, verify=True):
    """Download and unpack the model files into ``dest`` (default: user cache)."""
    dest = Path(dest) if dest else default_model_dir()
    if not force and not missing_model_files(dest):
        print(f'model files already present in {dest} (use --force to re-download)')
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix='.esmrank-dl-') as tmp:
        archive = Path(tmp) / ARCHIVE_NAME
        print(f'fetching {ARCHIVE_URL}')
        _download(ARCHIVE_URL, archive, ARCHIVE_MD5 if verify else None)

        print('extracting archive')
        extracted = Path(tmp) / 'extracted'
        extracted.mkdir()
        _safe_extract(archive, extracted)

        root = _find_model_root(extracted)
        if root is None:
            raise RuntimeError(
                f'none of {", ".join(REQUIRED_MODEL_FILES)} found in {ARCHIVE_NAME}')

        dest.mkdir(parents=True, exist_ok=True)
        for item in root.iterdir():
            target = dest / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))

    missing = missing_model_files(dest)
    if missing:
        raise RuntimeError(f'download finished but {dest} is missing: {", ".join(missing)}')
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
