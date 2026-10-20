"""Step 5: score the variants with the pretrained XGBoost ranker."""
from pathlib import Path

MODEL_FILE = 'esm_xgb_rank_rrf.json'


def predict(sample_dir, model_dir):
    import pandas as pd
    import xgboost as xgb

    sample_dir = Path(sample_dir)

    print('loading model')
    model = xgb.XGBRanker()
    model.load_model(str(Path(model_dir) / MODEL_FILE))

    print('loading features')
    data = pd.read_csv(sample_dir / 'dataset.csv')

    print('running predictions')
    data['prediction'] = model.predict(data[model.feature_names_in_])

    print('writing results')
    out_file = sample_dir / 'predictions.csv'
    data[['ids', 'prediction']].to_csv(out_file, index=None)
    return out_file
