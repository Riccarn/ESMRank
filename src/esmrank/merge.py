"""Step 4: merge ESM features and manual features into the model dataset."""
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    'embedding_distance', 'structural_distance', 'Residue Weight', 'pKa1',
    'pl4', 'P1', 'NCISC', 'wt_len', 'rel_pos', 'distance', 'aggregation',
    'dist_to_domain', 'entropy', 'instability', 'flexibility', 'gravy',
    'isoelectric_point', 'tm_idx', 'esm_score',
]


def merge_features(sample_dir):
    sample_dir = Path(sample_dir)
    esm_features = sample_dir / 'esm_features.csv'
    manual_features = sample_dir / 'manual_features.csv'
    out_file = sample_dir / 'dataset.csv'

    X = pd.read_csv(esm_features, index_col='ids')
    Y = pd.read_csv(manual_features, index_col=0)

    if 'Unnamed: 0' in X.columns:
        del X['Unnamed: 0']

    Z = X.join(Y)

    Z['esm_score'] = Z.wt_score - Z.mu_score
    del Z['wt_score'], Z['mu_score']

    Z.dropna(subset=REQUIRED_COLUMNS).to_csv(out_file)
    return out_file
