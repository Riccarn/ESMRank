"""Step 1: generate the variants to score (or pass a user CSV through)."""
from pathlib import Path

import pandas as pd

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY*'


def validate_mode(csv=False, sub=False, indel=False, alanines=False):
    """CSV and FASTA modes are mutually exclusive; -alanines excludes -sub/-indel."""
    if csv == bool(sub or indel or alanines):
        raise ValueError(
            'choose either csv mode (-csv) or FASTA mode with at least one of '
            '-sub, -indel, -alanines')
    if alanines and (sub or indel):
        raise ValueError('-alanines is mutually exclusive with -sub and -indel')


def read_single_fasta_sequence(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    return ''.join(line.strip() for line in lines[1:])


def build_variants(seq, *, sub=False, indel=False, alanines=False):
    """Return a DataFrame indexed by hgvsp with seq_wt / seq_mu columns."""
    aa = AMINO_ACIDS
    res = {}

    if alanines:
        res.update({
            f'{seq[i]}{i+1}A' if seq[i] != 'A' else f'A{i+1}G': {
                'seq_wt': seq,
                'seq_mu': seq[:i] + ('A' if seq[i] != 'A' else 'G') + seq[i+1:]
            }
            for i in range(len(seq))
        })

    if sub:
        res.update({  # substitutions
            f'{seq[i]}{i+1}{j}': {
                'seq_wt': seq,
                'seq_mu': seq[:i] + j + seq[i+1:]
            }
            for i in range(len(seq))
            for j in aa
            if not (j == '*' and i == 0)
        })

    if indel:
        res.update({  # deletions
            f'{seq[i]}{i+1}DEL': {
                'seq_wt': seq,
                'seq_mu': seq[:i] + seq[i+1:]
            }
            for i in range(len(seq))
        })
        res.update({  # insertions
            f'{i+1}INS{j}': {
                'seq_wt': seq,
                'seq_mu': seq[:i] + j + seq[i:]
            }
            for i in range(len(seq))
            for j in aa[:-1]
        })

    res = pd.DataFrame.from_dict(res, orient='index')
    res.index.name = 'hgvsp'
    return res


def generate_variants(input_path, sample_dir, *, csv=False, sub=False,
                      indel=False, alanines=False):
    """Write ``<sample_dir>/temp.csv`` with the variants to score."""
    validate_mode(csv=csv, sub=sub, indel=indel, alanines=alanines)
    out_file = Path(sample_dir) / 'temp.csv'

    if csv:
        pd.read_csv(input_path).to_csv(out_file, index=None)
        return out_file

    seq = read_single_fasta_sequence(input_path)
    build_variants(seq, sub=sub, indel=indel, alanines=alanines).to_csv(out_file)
    return out_file


def generate_variants_from_sequence(seq, sample_dir, *, sub=False, indel=False,
                                    alanines=False):
    """Like :func:`generate_variants` for a sequence string instead of a FASTA file."""
    validate_mode(csv=False, sub=sub, indel=indel, alanines=alanines)
    out_file = Path(sample_dir) / 'temp.csv'
    build_variants(seq, sub=sub, indel=indel, alanines=alanines).to_csv(out_file)
    return out_file


def write_variants_table(df, sample_dir):
    """Like csv mode of :func:`generate_variants`, with a DataFrame instead of a CSV file."""
    out_file = Path(sample_dir) / 'temp.csv'
    df.to_csv(out_file, index=None)
    return out_file
