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
