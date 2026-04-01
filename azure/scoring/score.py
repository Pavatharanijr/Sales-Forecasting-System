"""
Azure ML scoring script for the ensemble forecast model.
"""
import os
import json
import pickle
import numpy as np


def init():
    global artifacts
    model_dir = os.environ["AZUREML_MODEL_DIR"]
    model_file = os.listdir(model_dir)[0]
    with open(os.path.join(model_dir, model_file), "rb") as f:
        artifacts = pickle.load(f)


def run(raw_data: str) -> str:
    data = json.loads(raw_data)
    features = data["features"]  # list of dicts with FEATURE_COLS keys
    feat_cols = artifacts["feature_cols"]
    xgb, lgb, meta = artifacts["xgb"], artifacts["lgb"], artifacts["meta"]

    X = np.array([[row[c] for c in feat_cols] for row in features])
    preds = meta.predict(np.column_stack([xgb.predict(X), lgb.predict(X)]))
    return json.dumps({"predictions": preds.tolist()})
