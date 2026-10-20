import pytest

from esmrank.variants import build_variants, generate_variants, validate_mode


def test_substitutions_skip_start_stop_and_keep_ids():
    df = build_variants('MA', sub=True)
    # 2 positions x 20 aa, plus '*' only for position 2
    assert len(df) == 2 * 20 + 1
    assert 'M1*' not in df.index and 'A2*' in df.index
    assert df.loc['A2V', 'seq_mu'] == 'MV'
    assert (df['seq_wt'] == 'MA').all()


def test_alanine_scan_replaces_alanine_with_glycine():
    df = build_variants('MAK', alanines=True)
    assert list(df.index) == ['M1A', 'A2G', 'K3A']
    assert list(df['seq_mu']) == ['AAK', 'MGK', 'MAA']


def test_indels():
    df = build_variants('MA', indel=True)
    assert df.loc['M1DEL', 'seq_mu'] == 'A'
    assert df.loc['2INSG', 'seq_mu'] == 'MGA'
    assert len(df) == 2 + 2 * 20


@pytest.mark.parametrize('kwargs', [
    dict(csv=True, sub=True),
    dict(csv=False),
    dict(alanines=True, sub=True),
    dict(alanines=True, indel=True),
])
def test_invalid_mode_combinations(kwargs):
    with pytest.raises(ValueError):
        validate_mode(**kwargs)


def test_generate_variants_from_fasta_with_crlf(tmp_path):
    fasta = tmp_path / 'p.fasta'
    fasta.write_bytes(b'>p\r\nMA\r\nK\r\n')
    out = generate_variants(fasta, tmp_path, alanines=True)
    assert out.name == 'temp.csv'
    assert out.read_text().splitlines()[0] == 'hgvsp,seq_wt,seq_mu'
    assert 'K3A,MAK,MAA' in out.read_text()
