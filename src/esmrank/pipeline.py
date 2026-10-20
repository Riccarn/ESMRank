"""End-to-end ESMRank pipeline (variant generation -> features -> ranking -> profile)."""
from pathlib import Path

from .inputs import resolve_input
from .models import ensure_models
from .paths import find_model_dir
from .variants import (generate_variants, generate_variants_from_sequence,
                       validate_mode, write_variants_table)


def run_pipeline(data=None, *, sequence=None, name=None, outdir='output',
                 csv=False, sub=False, indel=False, alanines=False,
                 model_dir=None, esm_model=None, batch_size=None, gpu_ids=None,
                 auto_download=False):
    """Run the whole pipeline and return the per-sample output directory.

    The input can be a FASTA file, a protein sequence string (or the text of a
    single-record FASTA), or - for a table of variants - a CSV file (``csv=True``)
    or a pandas DataFrame with columns ``hgvsp``, ``seq_wt`` and ``seq_mu``.
    See :func:`esmrank.inputs.resolve_input` for how strings are interpreted;
    ``sequence=`` passes a sequence explicitly. A sequence needs at least one of
    ``sub``, ``indel``, ``alanines``; a DataFrame is a variants table and cannot
    be combined with them.

    Results are written to ``<outdir>/<name>/`` (``predictions.csv``, and
    ``profile.tsv`` / ``profile.png`` in sub / alanines mode). ``name`` defaults to
    the input file name, the FASTA header, or a short hash of the input.

    With ``auto_download=True`` missing model files are fetched first (see
    :func:`esmrank.models.ensure_models`); otherwise a missing model raises
    ``FileNotFoundError``.
    """
    # fail fast, before any expensive step
    inp = resolve_input(data, sequence=sequence, csv=csv, name=name)
    if inp.kind == 'dataframe' and (sub or indel or alanines):
        raise ValueError('a DataFrame is a table of variants: sub / indel / alanines '
                         'only apply when the input is a sequence')
    if inp.kind == 'sequence' and csv:
        raise ValueError('csv=True needs a CSV file or a DataFrame, not a sequence')
    csv_mode = inp.kind in ('csv_file', 'dataframe')
    validate_mode(csv=csv_mode, sub=sub, indel=indel, alanines=alanines)
    model_dir = ensure_models(model_dir) if auto_download else find_model_dir(model_dir)

    sample_dir = Path(outdir) / inp.name
    sample_dir.mkdir(parents=True, exist_ok=True)

    # heavy imports (torch, xgboost, pyhmmer, ...) only once we actually run
    from .esm_features import extract_esm_features
    from .manual_features import extract_manual_features
    from .merge import merge_features
    from .predict import predict
    from .profile_builder import build_profile

    print('generating variants')
    if inp.kind == 'sequence':
        generate_variants_from_sequence(inp.value, sample_dir, sub=sub,
                                        indel=indel, alanines=alanines)
    elif inp.kind == 'dataframe':
        write_variants_table(inp.value, sample_dir)
    else:
        generate_variants(inp.value, sample_dir, csv=csv_mode, sub=sub,
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


def rank_variants(data=None, **kwargs):
    """Like :func:`run_pipeline`, but return the predictions as a DataFrame.

    Columns: ``ids`` (variant identifier) and ``prediction`` (ranking score).
    Accepts the same arguments as :func:`run_pipeline`.
    """
    import pandas as pd

    sample_dir = run_pipeline(data, **kwargs)
    return pd.read_csv(Path(sample_dir) / 'predictions.csv')
