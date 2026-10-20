"""Step 6: aggregate per-variant scores into a smoothed per-position profile."""
import re
from pathlib import Path

WINDOW_SIZE = 5


def build_profile(sample_dir, *, sub=False, alanines=False):
    """Write ``profile.tsv`` / ``profile.png``; returns None if the mode has no profile."""
    import matplotlib.pyplot as plt
    import pandas as pd

    mode = 'alanines' if alanines else 'sub' if sub else 'not supported'
    if mode not in {'sub', 'alanines'}:
        print('To build a profile run in fasta mode with -sub or -alanines flags')
        return None

    sample_dir = Path(sample_dir)
    in_file = sample_dir / 'predictions.csv'
    out_file = sample_dir / 'profile.tsv'

    data = pd.read_csv(in_file)

    data['pos'] = data['ids'].map(lambda x: int(re.findall(r'\d+', x)[0]))
    if mode == 'sub':
        position_scores = data.groupby('pos').prediction.mean()
        position_scores = pd.DataFrame(position_scores).reset_index()
    else:
        position_scores = data[['pos', 'prediction']].copy()

    position_scores['score_smooth'] = position_scores['prediction'].rolling(
        window=WINDOW_SIZE, center=True, min_periods=1, win_type='gaussian').mean(std=1)
    result = position_scores[['pos', 'score_smooth']].copy()
    result.columns = ['pos', 'score']
    result.to_csv(out_file, index=False, sep='\t')
    result.plot(x='pos', y='score')
    plt.savefig(out_file.with_suffix('.png'), dpi=300)
    plt.close()
    return out_file
