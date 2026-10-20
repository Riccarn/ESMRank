"""Locating the pretrained model files (XGBoost ranker + Pfam-A HMMs)."""
import os
from pathlib import Path

# Files the pipeline needs inside the model directory.
REQUIRED_MODEL_FILES = ('esm_xgb_rank_rrf.json', 'Pfam-A.hmm')

MODEL_DIR_ENV = 'ESMRANK_MODEL_DIR'


def default_model_dir():
    """Per-user location used by ``esmrank download-models``."""
    cache_home = os.environ.get('XDG_CACHE_HOME') or str(Path.home() / '.cache')
    return Path(cache_home) / 'esmrank' / 'model'


def missing_model_files(model_dir):
    model_dir = Path(model_dir)
    return [f for f in REQUIRED_MODEL_FILES if not (model_dir / f).is_file()]


def find_model_dir(explicit=None):
    """Return the model directory to use.

    Order: ``explicit`` argument, ``$ESMRANK_MODEL_DIR``, ``./model`` (layout of the
    original repository checkout), then the per-user cache directory.
    """
    if explicit is not None:
        missing = missing_model_files(explicit)
        if missing:
            raise FileNotFoundError(
                f'model directory {explicit} is missing: {", ".join(missing)}')
        return Path(explicit)

    candidates = []
    if os.environ.get(MODEL_DIR_ENV):
        candidates.append(Path(os.environ[MODEL_DIR_ENV]))
    candidates.append(Path('model'))
    candidates.append(default_model_dir())

    for cand in candidates:
        if not missing_model_files(cand):
            return cand

    searched = '\n  '.join(str(c) for c in candidates)
    raise FileNotFoundError(
        'ESMRank model files not found. Run `esmrank download-models`, or point '
        f'--model-dir / ${MODEL_DIR_ENV} at a directory containing '
        f'{" and ".join(REQUIRED_MODEL_FILES)}.\nSearched:\n  {searched}')
