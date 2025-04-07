# code/train.py
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    args = parser.parse_args()

    df = pd.read_csv(f'{args.train}/train.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))

