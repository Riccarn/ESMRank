"""Work out what the user passed as input: a file, a sequence string or a DataFrame."""
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

# The 20 standard amino acids: the only letters the pipeline can score.
AMINO_ACIDS_20 = frozenset('ACDEFGHIKLMNPQRSTVWY')
# A string that is not an existing file is read as a sequence only if it is all
# uppercase letters and at least this long (shorter ones must be passed with sequence=).
MIN_AUTODETECT_LENGTH = 10
VARIANT_COLUMNS = ('hgvsp', 'seq_wt', 'seq_mu')

_NAME_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')


@dataclass
class ResolvedInput:
    kind: str       # 'fasta_file' | 'csv_file' | 'sequence' | 'dataframe'
    value: object   # path, cleaned sequence, or DataFrame
    name: str       # name of the output sub-directory


def clean_sequence(seq):
    """Normalise a protein sequence (case, whitespace, final stop) and validate it."""
    seq = ''.join(seq.split()).upper().rstrip('*')
    if not seq:
        raise ValueError('empty sequence')
    bad = sorted(set(seq) - AMINO_ACIDS_20)
    if bad:
        raise ValueError(
            f'invalid characters in sequence: {"".join(bad)} '
            f'(only the 20 standard amino acids are supported)')
    return seq


def parse_fasta_text(text):
    """Return (header, sequence) from the text of a single-record FASTA."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines or not lines[0].startswith('>'):
        raise ValueError('FASTA text must start with a ">" header line')
    if sum(line.startswith('>') for line in lines) != 1:
        raise ValueError('FASTA text must contain exactly one sequence')
    return lines[0][1:].strip(), ''.join(lines[1:])


def check_name(name):
    if not _NAME_RE.fullmatch(name):
        raise ValueError(
            f'invalid name {name!r}: use letters, digits, ".", "_" or "-" '
            f'(it becomes a directory name)')
    return name


def _short_hash(data):
    return hashlib.sha1(data).hexdigest()[:8]


def _looks_like_sequence(text):
    text = text.strip()
    return len(text) >= MIN_AUTODETECT_LENGTH and re.fullmatch(r'[A-Z]+\*?', text) is not None


def _from_sequence(seq, name):
    seq = clean_sequence(seq)
    return ResolvedInput('sequence', seq, name or f'seq_{_short_hash(seq.encode())}')


def _from_fasta_text(text, name):
    header, seq = parse_fasta_text(text)
    if name is None and header:
        token = re.sub(r'[^A-Za-z0-9._-]', '_', header.split()[0]).lstrip('._-')
        name = token or None
    return _from_sequence(seq, name)


def _from_dataframe(df, name):
    import pandas as pd

    df = df.copy()
    if 'hgvsp' not in df.columns and df.index.name == 'hgvsp':
        df = df.reset_index()
    missing = [c for c in VARIANT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f'the DataFrame needs the columns {", ".join(VARIANT_COLUMNS)}; '
            f'missing: {", ".join(missing)}')
    if df.empty:
        raise ValueError('the DataFrame has no rows')
    if name is None:
        hashed = pd.util.hash_pandas_object(df[list(VARIANT_COLUMNS)], index=False)
        name = f'df_{_short_hash(hashed.values.tobytes())}'
    return ResolvedInput('dataframe', df, name)


def resolve_input(data=None, *, sequence=None, csv=False, name=None):
    """Turn the user's input into a :class:`ResolvedInput`.

    ``data`` can be:

    - a pandas DataFrame with columns ``hgvsp``, ``seq_wt``, ``seq_mu`` (variants table);
    - a path (str or PathLike) to a FASTA file, or to a CSV file when ``csv=True``;
    - a protein sequence string, or the text of a single-record FASTA. A string is
      treated as a path if such a file exists, otherwise as a sequence when it is
      FASTA text (starts with ">") or all uppercase letters and at least
      ``MIN_AUTODETECT_LENGTH`` long.

    ``sequence=`` passes a sequence explicitly (any length, any case).
    ``name`` sets the output sub-directory; by default it comes from the file name,
    the FASTA header, or a short hash of the content.
    """
    import pandas as pd

    if (data is None) == (sequence is None):
        raise ValueError('pass exactly one of: the input (file path, sequence or '
                         'DataFrame) or sequence=...')
    if name is not None:
        name = check_name(name)

    if sequence is not None:
        if not isinstance(sequence, str):
            raise TypeError('sequence must be a string')
        return _from_fasta_text(sequence, name) if sequence.lstrip().startswith('>') \
            else _from_sequence(sequence, name)

    if isinstance(data, pd.DataFrame):
        return _from_dataframe(data, name)

    if isinstance(data, str) and not os.path.exists(data):
        if data.lstrip().startswith('>'):
            return _from_fasta_text(data, name)
        if _looks_like_sequence(data):
            return _from_sequence(data, name)
        raise FileNotFoundError(
            f'input file not found: {data!r} (to pass a short sequence directly, '
            f'use sequence=...)')

    if isinstance(data, (str, os.PathLike)):
        path = Path(data)
        if not path.is_file():
            raise FileNotFoundError(f'input file not found: {str(path)!r}')
        return ResolvedInput('csv_file' if csv else 'fasta_file', path, name or path.stem)

    raise TypeError(
        f'unsupported input of type {type(data).__name__}: expected a file path, '
        f'a sequence string or a pandas DataFrame')
