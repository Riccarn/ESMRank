"""Step 2: hand-crafted (shallow) features + masked sequences for the ESM step."""
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tqdm import tqdm

from .feature_lib import load_pfam, extract_profiles, distance_to_family
from .feature_lib import flexibility_around_mutation, compute_tm_index, asymmetric_window
from .feature_lib import entropy_around_mutation, compute_aggregation_profile
from .mask_sequences import align_and_mask

ID_KEY = 'hgvsp'
WT_SEQ_KEY = 'seq_wt'
MU_SEQ_KEY = 'seq_mu'


def _load_aminoacid_table():
    with resources.files('esmrank').joinpath('data/aminoacids.csv').open('r') as fh:
        return pd.read_csv(fh).set_index('Letter')


def extract_manual_features(sample_dir, model_dir):
    """Reads ``temp.csv``; writes ``manual_features.csv`` and ``masked_sequences.csv``."""
    sample_dir = Path(sample_dir)
    in_file = sample_dir / 'temp.csv'
    features_file = sample_dir / 'manual_features.csv'
    sequences_file = sample_dir / 'masked_sequences.csv'
    extra_features = sample_dir / 'temp.csv'  # extra features (same as in_file if none)
    id_key, wt_seq_key, mu_seq_key = ID_KEY, WT_SEQ_KEY, MU_SEQ_KEY

    print('loading input data')
    X = pd.read_csv(in_file).dropna(subset=[id_key, wt_seq_key, mu_seq_key])

    alphabet = set('ACDEFGHIKLMNPQRSTVWY*')
    X = X.loc[[all([c in alphabet for c in p])
               for p in tqdm(X[wt_seq_key], desc='parsing wt sequences')]]
    X = X.loc[[all([c in alphabet for c in p])
               for p in tqdm(X[mu_seq_key], desc='parsing mu sequences')]]

    X.loc[:, wt_seq_key] = [c.split('*')[0] for c in X[wt_seq_key]]
    X.loc[:, mu_seq_key] = [c.split('*')[0] for c in X[mu_seq_key]]

    # LOADING FEATURES
    print('loading aminoacids features')
    aa_features = _load_aminoacid_table()
    aa_map = {aa: aa_features.loc[aa].to_numpy() for aa in aa_features.index}

    print('running pfam hmmscan')
    hmms = load_pfam(Path(model_dir) / 'Pfam-A.hmm')
    families = extract_profiles(list(X[wt_seq_key].unique()), hmms)

    analysis = {}
    for seq in tqdm(list(X[wt_seq_key].unique()), desc='performing wt analysis'):
        curr = ProteinAnalysis(seq)
        analysis[seq] = {
            'instability': curr.instability_index(),
            'flexibility': curr.flexibility(),
            'gravy': curr.gravy(),
            'isop': curr.isoelectric_point(),
            'aggreg': compute_aggregation_profile(seq),
            'tm_idx': compute_tm_index(seq),
            'families': np.array([[dom['start_idx']-1, dom['stop_idx']-1] for dom in families[seq]]),
        }

    features = {}
    masked_sequences = {'id': [], 'wt': [], 'mu': [], 'masked_wt': [], 'masked_mu': []}

    for _, (uv, wt, mu) in tqdm(X[[id_key, wt_seq_key, mu_seq_key]].iterrows(),
                                total=X.shape[0], desc='extracting features'):

        assert len(wt) > 0

        mu = mu.split('*')[0]

        # wt-mu alignment
        masked_wt, masked_mu, distance = align_and_mask(wt, mu)

        if np.isnan(distance):
            print(f'error on {uv}, skipping')
            continue

        # initialize feature set
        feat = np.zeros(5)
        k = 1

        wt_m = masked_wt.replace('<mask>', '*')
        mu_m = masked_mu.replace('<mask>', '*')

        # accumulate features for wt masked aminoacids
        for i in range(len(wt)):
            if wt_m[i] == '*':
                k += 1
                feat += aa_map[wt[i]]

        pos = 0

        # accumulate features for mu masked aminoacids
        for i in range(len(mu)):
            if mu_m[i] == '*':
                if pos == 0:
                    pos = i
                feat -= aa_map[mu[i]]

        if pos >= len(wt):
            pos = len(wt) - 1

        analysis_mu = ProteinAnalysis(mu)

        features[uv] = np.concatenate([feat, [
            len(wt),
            pos / len(mu),
            distance / len(wt),
            analysis[wt]['aggreg'][pos] - compute_aggregation_profile(mu)[pos],
            distance_to_family(analysis[wt]['families'], pos) if len(analysis[wt]['families']) > 0 else -1,
            entropy_around_mutation(mu, pos),
            analysis[wt]['instability'] - analysis_mu.instability_index(),
            flexibility_around_mutation(analysis[wt]['flexibility'], analysis_mu.flexibility(), pos),
            analysis[wt]['gravy'] - analysis_mu.gravy(),
            analysis[wt]['isop'] - analysis_mu.isoelectric_point(),
            analysis[wt]['tm_idx'] - compute_tm_index(mu)
        ]])

        masked_sequences['id'].append(uv)

        start_idx, stop_idx = asymmetric_window(pos, len(wt), 400)
        masked_sequences['wt'].append(wt[start_idx:stop_idx])
        masked_sequences['masked_wt'].append(wt_m[start_idx:stop_idx].replace('*', '<mask>'))

        start_idx, stop_idx = asymmetric_window(pos, len(mu), 400)
        masked_sequences['mu'].append(mu[start_idx:stop_idx])
        masked_sequences['masked_mu'].append(mu_m[start_idx:stop_idx].replace('*', '<mask>'))

    print('writing masked sequences file')
    pd.DataFrame.from_dict(masked_sequences).to_csv(sequences_file, index=None)

    cnames = [list(aa_features.columns) + ['wt_len', 'rel_pos', 'distance',
                                           'aggregation', 'dist_to_domain',
                                           'entropy', 'instability', 'flexibility',
                                           'gravy', 'isoelectric_point', 'tm_idx']]

    print('building feature dataframe')
    feat = pd.DataFrame.from_dict(features, orient='index')
    feat.columns = cnames

    # join feat to X
    feat = feat.join(X.set_index(id_key))
    del feat[wt_seq_key], feat[mu_seq_key]

    print('joining computed features and extra features')
    X = feat.join(pd.read_csv(extra_features).set_index(id_key), how='left', lsuffix='_1')

    X.columns = [c[0] if isinstance(c, tuple) else c for c in X.columns]

    # remove common columns between data and extra features
    if wt_seq_key in X.columns:
        del X[wt_seq_key]
    if mu_seq_key in X.columns:
        del X[mu_seq_key]
    for c in X.columns:
        if c.endswith('_1'):
            del X[c]

    print('writing manual features file')
    X.to_csv(features_file)
    print('finished')
    return features_file
