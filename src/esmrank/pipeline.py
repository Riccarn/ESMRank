"""End-to-end ESMRank pipeline (variant generation -> features -> ranking -> profile)."""
from pathlib import Path

from .models import ensure_models
from .paths import find_model_dir
from .variants import generate_variants, validate_mode


def run_pipeline(input_path, *, outdir='output', csv=False, sub=False,
                 indel=False, alanines=False, model_dir=None, esm_model=None,
                 batch_size=None, gpu_ids=None, auto_download=False):
    """Run the whole pipeline and return the per-sample output directory.

    Results are written to ``<outdir>/<input file stem>/`` (``predictions.csv``,
    and ``profile.tsv`` / ``profile.png`` in -sub / -alanines mode).

    With ``auto_download=True`` missing model files are fetched first (see
    :func:`esmrank.models.ensure_models`); otherwise a missing model raises
    ``FileNotFoundError``.
    """
    # fail fast, before any expensive step
    validate_mode(csv=csv, sub=sub, indel=indel, alanines=alanines)
    model_dir = ensure_models(model_dir) if auto_download else find_model_dir(model_dir)

    sample_name = Path(input_path).stem
    sample_dir = Path(outdir) / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    # heavy imports (torch, xgboost, pyhmmer, ...) only once we actually run
    from .esm_features import extract_esm_features
    from .manual_features import extract_manual_features
    from .merge import merge_features
    from .predict import predict
    from .profile_builder import build_profile

    print('generating variants')
    generate_variants(input_path, sample_dir, csv=csv, sub=sub,
                      indel=indel, alanines=alanines)

    print('shallow features extraction')
    extract_manual_features(sample_dir, model_dir)

    print('deep features extraction')
    extract_esm_features(sample_dir, model_name=esm_model,
                         worker_batch_size=batch_size, gpu_ids=gpu_ids)

    print('merge features')
    merge_features(sample_dir)

    print('make predictions')
    predict(sample_dir, model_dir)

    print('building profile')
    build_profile(sample_dir, sub=sub, alanines=alanines)

    return sample_dir


def rank_variants(input_path, **kwargs):
    """Like :func:`run_pipeline`, but return the predictions as a DataFrame.

    Columns: ``ids`` (variant identifier) and ``prediction`` (ranking score).
    Accepts the same keyword arguments as :func:`run_pipeline`.
    """
    import pandas as pd

    sample_dir = run_pipeline(input_path, **kwargs)
    return pd.read_csv(Path(sample_dir) / 'predictions.csv')
