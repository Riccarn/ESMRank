"""Command line interface: ``esmrank run`` and ``esmrank download-models``."""
import argparse
import sys
from pathlib import Path

from . import __version__


def _gpu_list(value):
    return [int(x) for x in value.split(',') if x.strip()]


def build_parser():
    parser = argparse.ArgumentParser(
        prog='esmrank',
        description='ESMRank: a learn-to-rank approach for protein variant effect prediction.')
    parser.add_argument('--version', action='version', version=f'esmrank {__version__}')
    sub_parsers = parser.add_subparsers(dest='command', required=True)

    run = sub_parsers.add_parser(
        'run', help='run the pipeline on a FASTA file or a variants CSV',
        description=(
            'Input is either a single-sequence FASTA (variants are generated with '
            '-sub / -indel / -alanines) or, with -csv, a CSV with columns '
            'hgvsp, seq_wt and seq_mu.'))
    run.add_argument('--input', '-i', required=True, help='input FASTA or CSV file')
    run.add_argument('-csv', '--csv', action='store_true',
                     help='read variants directly from a CSV (default: FASTA mode)')
    run.add_argument('-sub', '--sub', action='store_true',
                     help='generate all possible missense substitutions')
    run.add_argument('-indel', '--indel', action='store_true',
                     help='generate all possible single insertions and deletions')
    run.add_argument('-alanines', '--alanines', action='store_true',
                     help='alanine scanning only (exclusive with -sub / -indel)')
    run.add_argument('--outdir', default='output',
                     help='output root; results go in <outdir>/<input name>/ (default: output)')
    run.add_argument('--model-dir', default=None,
                     help='directory with the pretrained model files '
                          '(default: $ESMRANK_MODEL_DIR, ./model, then the user cache)')
    run.add_argument('--download', action='store_true',
                     help='download the model files first if they are not already present')
    run.add_argument('--gpus', type=_gpu_list, default=None, metavar='0,1',
                     help='comma-separated CUDA device ids (default: all, or $ESM_GPU_IDS)')
    run.add_argument('--batch-size', type=int, default=None,
                     help='worker batch size for the ESM step (default: 128, or $ESM_WORKER_BATCH_SIZE)')
    run.add_argument('--esm-model', default=None,
                     help='ESM-2 checkpoint name (default: esm2_t33_650M_UR50D, or $ESM_MODEL_NAME)')

    dl = sub_parsers.add_parser(
        'download-models', help='download the pretrained model files from Zenodo (~830 MB)')
    dl.add_argument('--dest', default=None,
                    help='where to put the files (default: ~/.cache/esmrank/model)')
    dl.add_argument('--force', action='store_true', help='download again even if present')
    dl.add_argument('--no-verify', action='store_true', help='skip the md5 check')
    dl.add_argument('--archive', default=None,
                    help='use an already downloaded ESMRank.tar.gz instead of fetching it')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.command == 'download-models':
        from .models import download_models
        download_models(dest=args.dest, force=args.force, verify=not args.no_verify,
                        archive=args.archive)
        return 0

    from .pipeline import run_pipeline
    try:
        run_pipeline(
            Path(args.input), outdir=args.outdir, csv=args.csv, sub=args.sub,
            indel=args.indel, alanines=args.alanines, model_dir=args.model_dir,
            esm_model=args.esm_model, batch_size=args.batch_size, gpu_ids=args.gpus,
            auto_download=args.download)
    except (ValueError, FileNotFoundError) as exc:
        print(f'esmrank: error: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
