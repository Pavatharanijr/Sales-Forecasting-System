# 🌊 Sales Forecasting System

Ensemble ML forecasting (XGBoost + LightGBM + Ridge stacking) with SHAP explainability,
a Deep Ocean Streamlit UI, and Azure ML deployment.

---

## Project Structure

```
Sales Forecasting System/
├── app.py                    # Streamlit UI (Deep Ocean theme)
├── requirements.txt
├── data/
│   ├── prepare_data.py       # Kaggle download + feature engineering
│   ├── raw/                  # Raw CSV (auto-downloaded)
│   └── processed/            # Resampled + feature-engineered CSVs
├── model/
│   ├── train.py              # Ensemble training + SHAP + MLflow logging
│   ├── artifacts/            # Saved .pkl model files
│   └── __init__.py
└── azure/
    ├── deploy_azure.py       # Azure ML managed endpoint deployment
    ├── conda_env.yml         # Azure ML environment spec
    ├── config.json           # Azure workspace credentials (fill in)
    └── scoring/
        └── score.py          # Azure ML scoring script
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Kaggle API
Place your `kaggle.json` at `~/.kaggle/kaggle.json`:
```json
{"username": "<kaggle-username>", "key": "<kaggle-api-key>"}
```
Get it from: https://www.kaggle.com/settings → API → Create New Token

### 3. Download & prepare data
```bash
python data/prepare_data.py
```
This downloads the **Superstore Sales** dataset and generates:
- `data/processed/sales_W.csv`   (Weekly)
- `data/processed/sales_MS.csv`  (Monthly)
- `data/processed/sales_QS.csv`  (Quarterly)

### 4. Train the ensemble model
```bash
python model/train.py
```
Or click **Train / Retrain Model** in the Streamlit sidebar.

### 5. Run the app
```bash
streamlit run app.py
```

---

## Azure Deployment

### 1. Fill in Azure config
Edit `azure/config.json`:
```json
{
  "subscription_id": "<your-azure-subscription-id>",
  "resource_group": "<your-resource-group>",
  "workspace_name": "<your-aml-workspace-name>"
}
```

### 2. Login to Azure
```bash
az login
```

### 3. Deploy
```bash
# Deploy weekly model (W / MS / QS)
python azure/deploy_azure.py W
```

---

## Features

| Feature | Details |
|---|---|
| Dataset | Kaggle Superstore Sales |
| Frequencies | Weekly / Monthly / Quarterly |
| Models | XGBoost + LightGBM → Ridge meta-learner |
| Explainability | SHAP feature importance + waterfall |
| Tracking | MLflow experiment logging |
| UI Theme | Deep Ocean (Streamlit) |
| Cloud | Azure ML Managed Online Endpoint |

---

## Metrics Tracked
- **MAPE** — Mean Absolute Percentage Error
- **RMSE** — Root Mean Squared Error
- **Forecast Total** — Sum of predicted periods
- **Avg per Period** — Mean predicted value
