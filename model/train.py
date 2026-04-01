"""
Ensemble forecasting: XGBoost + LightGBM stacked, with Prophet trend baseline.
Explainability via SHAP. Saves model artifacts to model/artifacts/.
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import shap
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = "model/artifacts"
FEATURE_COLS = ["lag_1", "lag_2", "lag_4", "rolling_mean_4", "rolling_std_4",
                "month", "quarter", "week", "year", "trend"]


def load_data(freq: str) -> pd.DataFrame:
    path = f"data/processed/sales_{freq}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run data/prepare_data.py first. Missing: {path}")
    return pd.read_csv(path, parse_dates=["ds"])


def train_prophet(df: pd.DataFrame) -> Prophet:
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(df[["ds", "y"]])
    return m


def train_ensemble(df: pd.DataFrame, freq: str):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    X = df[FEATURE_COLS].values
    y = df["y"].values

    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                       subsample=0.8, colsample_bytree=0.8, random_state=42)
    lgb = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)

    tscv = TimeSeriesSplit(n_splits=5)
    xgb_oof = np.zeros(len(y))
    lgb_oof = np.zeros(len(y))

    for train_idx, val_idx in tscv.split(X):
        xgb.fit(X[train_idx], y[train_idx])
        lgb.fit(X[train_idx], y[train_idx])
        xgb_oof[val_idx] = xgb.predict(X[val_idx])
        lgb_oof[val_idx] = lgb.predict(X[val_idx])

    # Meta-learner stacking
    meta_X = np.column_stack([xgb_oof, lgb_oof])
    meta = Ridge(alpha=1.0)
    meta.fit(meta_X, y)

    # Final fit on full data
    xgb.fit(X, y)
    lgb.fit(X, y)

    # SHAP explainer on XGBoost
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X)

    # Metrics
    preds = meta.predict(np.column_stack([xgb.predict(X), lgb.predict(X)]))
    mape = mean_absolute_percentage_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))

    artifacts = {
        "xgb": xgb, "lgb": lgb, "meta": meta,
        "explainer": explainer, "shap_values": shap_values,
        "feature_cols": FEATURE_COLS, "df": df,
        "metrics": {"mape": mape, "rmse": rmse}
    }

    path = f"{ARTIFACTS_DIR}/ensemble_{freq}.pkl"
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)

    with mlflow.start_run(run_name=f"ensemble_{freq}"):
        mlflow.log_params({"freq": freq, "n_estimators": 300, "meta": "Ridge"})
        mlflow.log_metrics({"mape": mape, "rmse": rmse})
        mlflow.sklearn.log_model(meta, "meta_model")

    print(f"[{freq}] MAPE={mape:.4f}  RMSE={rmse:.2f}  -> {path}")
    return artifacts


def forecast_future(artifacts: dict, df: pd.DataFrame, periods: int) -> pd.DataFrame:
    xgb, lgb, meta = artifacts["xgb"], artifacts["lgb"], artifacts["meta"]
    last = df.iloc[-1].copy()
    history = list(df["y"].values)
    rows = []

    for i in range(periods):
        lag1 = history[-1]
        lag2 = history[-2] if len(history) >= 2 else lag1
        lag4 = history[-4] if len(history) >= 4 else lag1
        roll4 = np.mean(history[-4:]) if len(history) >= 4 else lag1
        std4 = np.std(history[-4:]) if len(history) >= 4 else 0
        next_ds = last["ds"] + pd.tseries.frequencies.to_offset(df.attrs.get("freq", "W"))
        row = {
            "ds": next_ds, "lag_1": lag1, "lag_2": lag2, "lag_4": lag4,
            "rolling_mean_4": roll4, "rolling_std_4": std4,
            "month": next_ds.month, "quarter": next_ds.quarter,
            "week": next_ds.isocalendar()[1], "year": next_ds.year,
            "trend": last["trend"] + i + 1
        }
        X_row = np.array([[row[c] for c in FEATURE_COLS]])
        pred = meta.predict(np.column_stack([xgb.predict(X_row), lgb.predict(X_row)]))[0]
        row["y_pred"] = max(pred, 0)
        rows.append(row)
        history.append(pred)
        last = pd.Series(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    for freq in ["W", "MS", "QS"]:
        df = load_data(freq)
        train_ensemble(df, freq)
