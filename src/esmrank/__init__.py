"""ESMRank: a learn-to-rank approach for protein variant effect prediction."""

__version__ = '0.1.0'

__all__ = ['__version__', 'run_pipeline']


def run_pipeline(*args, **kwargs):
    """Run the full ESMRank pipeline (see :func:`esmrank.pipeline.run_pipeline`)."""
    from .pipeline import run_pipeline as _run_pipeline
    return _run_pipeline(*args, **kwargs)
