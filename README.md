# Sales Forecasting System

Ensemble ML forecasting (XGBoost + LightGBM + Ridge stacking) with SHAP explainability and a Deep Ocean Streamlit UI.

---

## Project Structure

```
Sales Forecasting System/
├── app.py                    # Streamlit UI (Deep Ocean theme)
├── requirements.txt
└── model/
    ├── train.py              # Ensemble training + SHAP + MLflow logging
    └── __init__.py
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Upload your dataset
- Upload any sales CSV using the file uploader on the main page
- The file must contain a date column and a sales/revenue column
- Recommended: [Kaggle Superstore Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

### 4. Train and forecast
- Select forecast frequency (Weekly / Monthly / Quarterly) from the sidebar
- Set the number of forecast periods using the slider
- Click **Train / Retrain Model** to train the ensemble
- Explore results across the Forecast, Explainability, and Data Analysis tabs

---

## Features

| Feature | Details |
|---|---|
| Dataset | Any sales CSV (Kaggle Superstore recommended) |
| Frequencies | Weekly / Monthly / Quarterly |
| Models | XGBoost + LightGBM stacked with Ridge meta-learner |
| Explainability | SHAP feature importance + waterfall chart |
| Tracking | MLflow experiment logging |
| UI Theme | Deep Ocean (Streamlit) |

---

## Metrics Tracked

| Metric | Description |
|---|---|
| MAPE | Mean Absolute Percentage Error |
| RMSE | Root Mean Squared Error |
| Forecast Total | Sum of all predicted periods |
| Avg per Period | Mean predicted value per period |

---

## Model Architecture

Raw sales data is aggregated and resampled to the chosen frequency. Time-based features (lags, rolling statistics, calendar features) are engineered and fed into two gradient boosting models — XGBoost and LightGBM. Their predictions are stacked using a Ridge meta-learner trained via TimeSeriesSplit cross-validation to prevent data leakage. SHAP values are computed on the XGBoost model to explain each prediction.

---

## Live Demo

Deployed on Streamlit Cloud:
[https://pavatharanijr-sales-forecasting-system.streamlit.app](https://pavatharanijr-sales-forecasting-system.streamlit.app)
