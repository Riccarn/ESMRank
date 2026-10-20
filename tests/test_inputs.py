import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from esmrank import pipeline
from esmrank.inputs import resolve_input

SEQ = 'MKTAYIAKQR'  # 10 aa: the shortest string recognised as a sequence


# --- resolve_input ----------------------------------------------------------

def test_existing_fasta_file(tmp_path):
    f = tmp_path / 'prot.fasta'
    f.write_text('>p\nMKT\n')
    for given in (f, str(f)):
        r = resolve_input(given)
        assert (r.kind, r.name, r.value) == ('fasta_file', 'prot', f)


def test_existing_csv_file_needs_csv_flag(tmp_path):
    f = tmp_path / 'vars.csv'
    f.write_text('hgvsp,seq_wt,seq_mu\n')
    assert resolve_input(f, csv=True).kind == 'csv_file'
    assert resolve_input(f).kind == 'fasta_file'


def test_sequence_string_is_detected_and_name_is_stable():
    a, b = resolve_input(SEQ), resolve_input(SEQ)
    assert (a.kind, a.value) == ('sequence', SEQ)
    assert a.name == b.name and a.name.startswith('seq_')
    assert resolve_input(SEQ + 'A').name != a.name


def test_explicit_sequence_accepts_short_and_lowercase():
    r = resolve_input(sequence='mkt a\n')
    assert (r.kind, r.value) == ('sequence', 'MKTA')
    assert resolve_input(sequence=SEQ + '*').value == SEQ


def test_missing_file_is_not_mistaken_for_a_sequence():
    for given in ('variants.csv', 'mkt', 'variants', 'MKT'):
        with pytest.raises(FileNotFoundError, match='sequence='):
            resolve_input(given)


def test_invalid_sequence_characters():
    with pytest.raises(ValueError, match='BX'):
        resolve_input('MKTAYBXAKQR')


def test_fasta_text_uses_header_for_the_name():
    r = resolve_input('>sp|P12345 my protein\nMKTAY\nIAKQR\n')
    assert (r.kind, r.value) == ('sequence', SEQ)
    assert r.name == 'sp_P12345'
    assert resolve_input('>x\nMKT', name='given').name == 'given'
    with pytest.raises(ValueError, match='exactly one'):
        resolve_input('>a\nMKT\n>b\nMKT\n')


def test_dataframe_input():
    df = pd.DataFrame({'hgvsp': ['M1A'], 'seq_wt': ['MK'], 'seq_mu': ['AK'], 'extra': [1.5]})
    r = resolve_input(df)
    assert r.kind == 'dataframe' and r.name.startswith('df_')
    assert list(r.value.columns) == ['hgvsp', 'seq_wt', 'seq_mu', 'extra']
    assert r.name == resolve_input(df.copy()).name
    assert r.value is not df  # the caller's DataFrame is not modified


def test_dataframe_with_hgvsp_as_index():
    df = pd.DataFrame({'seq_wt': ['MK'], 'seq_mu': ['AK']}, index=pd.Index(['M1A'], name='hgvsp'))
    assert list(resolve_input(df).value['hgvsp']) == ['M1A']


@pytest.mark.parametrize('df, msg', [
    (pd.DataFrame({'hgvsp': ['a'], 'seq_wt': ['M']}), 'seq_mu'),
    (pd.DataFrame(columns=['hgvsp', 'seq_wt', 'seq_mu']), 'no rows'),
])
def test_invalid_dataframe(df, msg):
    with pytest.raises(ValueError, match=msg):
        resolve_input(df)


def test_argument_errors():
    with pytest.raises(ValueError, match='exactly one'):
        resolve_input()
    with pytest.raises(ValueError, match='exactly one'):
        resolve_input(SEQ, sequence=SEQ)
    with pytest.raises(ValueError, match='invalid name'):
        resolve_input(SEQ, name='../x')
    with pytest.raises(TypeError):
        resolve_input(42)


# --- whole pipeline with the heavy steps replaced by stubs --------------------

@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'find_model_dir', lambda explicit=None: tmp_path)

    def stub(name, **attrs):
        module = types.ModuleType(f'esmrank.{name}')
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, f'esmrank.{name}', module)

    def predict(sample_dir, model_dir):
        ids = pd.read_csv(Path(sample_dir) / 'temp.csv')['hgvsp']
        pd.DataFrame({'ids': ids, 'prediction': range(len(ids))}).to_csv(
            Path(sample_dir) / 'predictions.csv', index=False)

    nothing = lambda *a, **k: None  # noqa: E731
    stub('esm_features', extract_esm_features=nothing)
    stub('manual_features', extract_manual_features=nothing)
    stub('merge', merge_features=nothing)
    stub('predict', predict=predict)
    stub('profile_builder', build_profile=nothing)
    return tmp_path / 'out'


def test_pipeline_with_a_sequence_string(stubbed):
    df = pipeline.rank_variants(SEQ, alanines=True, outdir=stubbed)
    assert list(df['ids']) == ['M1A', 'K2A', 'T3A', 'A4G', 'Y5A', 'I6A', 'A7G', 'K8A', 'Q9A', 'R10A']
    sample_dir, = list(stubbed.iterdir())
    assert sample_dir.name.startswith('seq_')
    assert (sample_dir / 'temp.csv').is_file()


def test_sequence_keyword_and_string_give_the_same_run(stubbed):
    pipeline.run_pipeline(SEQ, sub=True, outdir=stubbed)
    pipeline.run_pipeline(sequence=SEQ, sub=True, outdir=stubbed)
    assert len(list(stubbed.iterdir())) == 1


def test_pipeline_with_fasta_text_and_name(stubbed):
    out = pipeline.run_pipeline('>my_protein\nMKTAY\nIAKQR\n', alanines=True, outdir=stubbed)
    assert out.name == 'my_protein'
    out = pipeline.run_pipeline(SEQ, alanines=True, outdir=stubbed, name='custom')
    assert out.name == 'custom'


def test_pipeline_with_a_dataframe_keeps_extra_columns(stubbed):
    df = pd.DataFrame({'hgvsp': ['M1A', 'K2A'], 'seq_wt': [SEQ, SEQ],
                       'seq_mu': ['A' + SEQ[1:], SEQ[0] + 'A' + SEQ[2:]], 'assay': [1, 2]})
    result = pipeline.rank_variants(df, outdir=stubbed)
    assert list(result['ids']) == ['M1A', 'K2A']
    temp = pd.read_csv(next(stubbed.iterdir()) / 'temp.csv')
    assert list(temp.columns) == ['hgvsp', 'seq_wt', 'seq_mu', 'assay']
    assert 'hgvsp' in df.columns and df.index.name is None  # caller's df untouched


def test_pipeline_with_files_still_works(stubbed, tmp_path):
    fasta = tmp_path / 'protein.fasta'
    fasta.write_text('>p\nMKTAYIAKQR\n')
    assert pipeline.run_pipeline(fasta, alanines=True, outdir=stubbed).name == 'protein'

    table = tmp_path / 'variants.csv'
    pd.DataFrame({'hgvsp': ['M1A'], 'seq_wt': [SEQ], 'seq_mu': ['A' + SEQ[1:]]}).to_csv(table, index=False)
    assert pipeline.run_pipeline(table, csv=True, outdir=stubbed).name == 'variants'


def test_pipeline_mode_errors_happen_before_any_work(stubbed):
    df = pd.DataFrame({'hgvsp': ['M1A'], 'seq_wt': [SEQ], 'seq_mu': [SEQ]})
    with pytest.raises(ValueError, match='table of variants'):
        pipeline.run_pipeline(df, sub=True, outdir=stubbed)
    with pytest.raises(ValueError, match='csv=True'):
        pipeline.run_pipeline(SEQ, csv=True, outdir=stubbed)
    with pytest.raises(ValueError, match='choose either'):
        pipeline.run_pipeline(SEQ, outdir=stubbed)  # sequence without sub/indel/alanines
    assert not stubbed.exists()  # nothing was created


def test_cli_input_is_always_a_file_never_a_sequence(tmp_path, monkeypatch, capsys):
    from esmrank import cli
    monkeypatch.chdir(tmp_path)
    assert cli.main(['run', '--input', SEQ, '-sub']) == 2
    assert 'input file not found' in capsys.readouterr().err
    assert not (tmp_path / 'output').exists()
