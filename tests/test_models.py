import sys
from pathlib import Path

import pytest

import esmrank
from esmrank import models, paths


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """No model anywhere: empty cwd, empty cache, no env override."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.delenv(paths.MODEL_DIR_ENV, raising=False)
    return tmp_path


def _fake_download(calls):
    def fake(dest=None, force=False, verify=True):
        dest = Path(dest)  # like the real download_models
        calls.append(dest)
        for name in paths.REQUIRED_MODEL_FILES:
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            (dest / name).write_text('x')
        return dest
    return fake


def _populate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in paths.REQUIRED_MODEL_FILES:
        (directory / name).write_text('x')


def test_ensure_models_downloads_to_cache_when_missing(isolated, monkeypatch):
    calls = []
    monkeypatch.setattr(models, 'download_models', _fake_download(calls))
    out = models.ensure_models()
    assert calls == [paths.default_model_dir()]
    assert out == paths.default_model_dir()


def test_ensure_models_uses_env_dir_as_download_target(isolated, monkeypatch):
    target = isolated / 'mymodels'
    monkeypatch.setenv(paths.MODEL_DIR_ENV, str(target))
    calls = []
    monkeypatch.setattr(models, 'download_models', _fake_download(calls))
    assert models.ensure_models() == target
    assert calls == [target]


def test_ensure_models_does_not_download_when_present(isolated, monkeypatch):
    _populate(isolated / 'model')  # legacy ./model layout
    monkeypatch.setattr(models, 'download_models',
                        lambda *a, **k: pytest.fail('should not download'))
    assert models.ensure_models() == Path('model')


def test_ensure_models_explicit_dir(isolated, monkeypatch):
    explicit = isolated / 'here'
    calls = []
    monkeypatch.setattr(models, 'download_models', _fake_download(calls))
    assert models.ensure_models(explicit) == explicit
    assert calls == [explicit]

    calls.clear()  # now complete: no second download
    assert models.ensure_models(explicit) == explicit
    assert calls == []


def test_find_model_dir_raises_without_models(isolated):
    with pytest.raises(FileNotFoundError, match='download-models'):
        paths.find_model_dir()


def test_public_api_is_lazy_and_resolves():
    assert esmrank.ensure_models is models.ensure_models
    assert esmrank.find_model_dir is paths.find_model_dir
    assert 'rank_variants' in dir(esmrank)
    assert 'torch' not in sys.modules
    with pytest.raises(AttributeError):
        esmrank.does_not_exist


# --- archive handling (no network: _download is replaced by a copy) ----------

import gzip
import tarfile


def _make_archive(path, files):
    """files: {path_in_archive: bytes}"""
    with tarfile.open(path, 'w:gz') as tar:
        for name, data in files.items():
            src = path.parent / ('_tmp_' + Path(name).name)
            src.write_bytes(data)
            tar.add(src, arcname=name)
            src.unlink()


@pytest.fixture
def fake_zenodo(tmp_path, monkeypatch):
    """Make download_models 'download' a locally built archive."""
    state = {'archive': None, 'calls': 0}

    def fake_download(url, dest, expected_md5=None):
        state['calls'] += 1
        Path(dest).write_bytes(state['archive'].read_bytes())

    monkeypatch.setattr(models, '_download', fake_download)
    return state


def test_download_finds_files_in_different_directories(tmp_path, fake_zenodo):
    fake_zenodo['archive'] = tmp_path / 'a.tar.gz'
    _make_archive(fake_zenodo['archive'], {
        'ESMRank/model/esm_xgb_rank_rrf.json': b'{}',
        'ESMRank/pfam/Pfam-A.hmm': b'HMMER3',
    })
    dest = tmp_path / 'out' / 'model'
    assert models.download_models(dest=dest, verify=False) == dest
    assert (dest / 'esm_xgb_rank_rrf.json').read_bytes() == b'{}'
    assert (dest / 'Pfam-A.hmm').read_bytes() == b'HMMER3'
    assert not (dest.parent / '.esmrank-download').exists()  # cleaned up


def test_download_decompresses_gzipped_pfam(tmp_path, fake_zenodo):
    fake_zenodo['archive'] = tmp_path / 'a.tar.gz'
    _make_archive(fake_zenodo['archive'], {
        'esm_xgb_rank_rrf.json': b'{}',
        'Pfam-A.hmm.gz': gzip.compress(b'HMMER3'),
    })
    dest = tmp_path / 'model'
    models.download_models(dest=dest, verify=False)
    assert (dest / 'Pfam-A.hmm').read_bytes() == b'HMMER3'


def test_missing_file_reports_archive_content_and_keeps_archive(tmp_path, fake_zenodo):
    fake_zenodo['archive'] = tmp_path / 'a.tar.gz'
    _make_archive(fake_zenodo['archive'], {'ESMRank/model/esm_xgb_rank_rrf.json': b'{}'})
    dest = tmp_path / 'out' / 'model'
    with pytest.raises(RuntimeError, match=r'does not contain: Pfam-A\.hmm') as exc:
        models.download_models(dest=dest, verify=False)
    assert 'esm_xgb_rank_rrf.json' in str(exc.value)  # listing of the archive
    kept = dest.parent / '.esmrank-download' / models.ARCHIVE_NAME
    assert kept.exists()

    # a retry reuses the kept archive instead of downloading again
    calls_before = fake_zenodo['calls']
    with pytest.raises(RuntimeError):
        models.download_models(dest=dest, verify=False)
    assert fake_zenodo['calls'] == calls_before


def test_download_from_local_archive_skips_network(tmp_path, fake_zenodo, monkeypatch):
    archive = tmp_path / 'ESMRank.tar.gz'
    _make_archive(archive, {'m/esm_xgb_rank_rrf.json': b'{}', 'm/Pfam-A.hmm': b'H'})
    monkeypatch.setattr(models, '_download',
                        lambda *a, **k: pytest.fail('should not download'))
    dest = tmp_path / 'model'
    models.download_models(dest=dest, verify=False, archive=archive)
    assert not models.missing_model_files(dest)
    assert archive.exists()  # a user-provided archive is never deleted


def test_local_archive_checksum_is_verified(tmp_path):
    archive = tmp_path / 'ESMRank.tar.gz'
    _make_archive(archive, {'x': b'y'})
    with pytest.raises(RuntimeError, match='md5'):
        models.download_models(dest=tmp_path / 'model', archive=archive)
