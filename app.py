"""
Sales Forecasting System — Deep Ocean Theme
Run: streamlit run app.py
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap
import streamlit as st
from model.train import train_ensemble, forecast_future

warnings.filterwarnings("ignore")

OCEAN = {
    "bg":      "#020B18",
    "surface": "#041C2C",
    "card":    "#062840",
    "border":  "#0A4A6E",
    "accent1": "#00C2FF",
    "accent2": "#00FFD1",
    "accent3": "#7B61FF",
    "text":    "#D6F0FF",
    "subtext": "#7ABFDF",
    "danger":  "#FF4D6D",
}

FEATURE_COLS = ["lag_1", "lag_2", "lag_4", "rolling_mean_4", "rolling_std_4",
                "month", "quarter", "week", "year", "trend"]

st.set_page_config(page_title="Sales Forecasting", layout="wide")

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; background-color: {OCEAN['bg']}; color: {OCEAN['text']}; }}
  .stApp {{ background-color: {OCEAN['bg']}; }}
  section[data-testid="stSidebar"] {{ background: {OCEAN['surface']}; border-right: 1px solid {OCEAN['border']}; }}
  .ocean-card {{ background: {OCEAN['card']}; border: 1px solid {OCEAN['border']}; border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; }}
  .metric-tile {{ background: linear-gradient(135deg, {OCEAN['card']}, {OCEAN['surface']}); border: 1px solid {OCEAN['border']}; border-radius: 10px; padding: 18px; text-align: center; }}
  .metric-value {{ font-size: 2rem; font-weight: 700; color: {OCEAN['accent1']}; }}
  .metric-label {{ font-size: 0.8rem; color: {OCEAN['subtext']}; text-transform: uppercase; letter-spacing: 1px; }}
  .stButton > button {{ background: linear-gradient(90deg, {OCEAN['accent1']}, {OCEAN['accent3']}); color: {OCEAN['bg']}; border: none; border-radius: 8px; font-weight: 600; padding: 10px 28px; }}
  .stButton > button:hover {{ opacity: 0.85; }}
  .stTabs [data-baseweb="tab-list"] {{ background: {OCEAN['surface']}; border-radius: 8px; gap: 12px; padding: 6px; }}
  .stTabs [data-baseweb="tab"] {{ color: {OCEAN['subtext']}; border-radius: 6px; flex: 1; justify-content: center; font-weight: 600; font-size: 0.95rem; padding: 10px 0; border: 1px solid {OCEAN['border']}; }}
  .stTabs [aria-selected="true"] {{ background: {OCEAN['card']}; color: {OCEAN['accent1']}; border-color: {OCEAN['accent1']}; }}
  .stTabs [data-baseweb="tab-panel"] {{ background: {OCEAN['card']}; border: 1px solid {OCEAN['border']}; border-radius: 0 0 12px 12px; padding: 24px; margin-top: -1px; }}
  .stSelectbox label, .stSlider label {{ color: #FFFFFF !important; font-weight: 600 !important; font-size: 0.9rem !important; }}
  p[data-testid="stWidgetLabel"] {{ color: #FFFFFF !important; }}
  hr {{ border-color: {OCEAN['border']}; }}
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: {OCEAN['bg']}; }}
  ::-webkit-scrollbar-thumb {{ background: {OCEAN['border']}; border-radius: 3px; }}
  h1, h2, h3 {{ color: {OCEAN['accent1']}; }}
  [data-testid="stFileUploader"] span {{ color: #000000 !important; font-weight: 600 !important; }}
  [data-testid="stFileUploader"] small {{ color: #000000 !important; }}
  [data-testid="stFileUploaderFileName"] {{ color: #000000 !important; font-weight: 600 !important; }}
  [data-testid="stFileUploaderFileSize"] {{ display: none !important; }}
  [data-testid="stFileUploaderFileName"] {{ display: none !important; }}
  [data-testid="stFileUploaderDeleteBtn"] {{ display: none !important; }}
  [data-testid="stFileUploaderFile"] {{ display: none !important; }}
  [data-testid="stFileUploaderDropzoneInstructions"] {{ display: none !important; }}
  [data-testid="stFileUploaderDropzone"] svg {{ display: none !important; }}
</style>
""", unsafe_allow_html=True)

FREQ_MAP = {"Weekly": "W", "Monthly": "MS", "Quarterly": "QS"}
PERIODS_DEFAULT = {"Weekly": 12, "Monthly": 6, "Quarterly": 4}
ARTIFACTS_DIR = "model/artifacts"


def plotly_layout(title=""):
    return dict(
        title=dict(text=title, font=dict(color=OCEAN["accent1"], size=16)),
        paper_bgcolor=OCEAN["card"], plot_bgcolor=OCEAN["surface"],
        font=dict(color=OCEAN["text"], family="Inter"),
        xaxis=dict(gridcolor=OCEAN["border"], zerolinecolor=OCEAN["border"]),
        yaxis=dict(gridcolor=OCEAN["border"], zerolinecolor=OCEAN["border"]),
        legend=dict(bgcolor=OCEAN["card"], bordercolor=OCEAN["border"]),
        margin=dict(l=40, r=20, t=50, b=40),
    )


def engineer_features(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df = df.groupby("Order Date")["Sales"].sum().reset_index()
    df = df.set_index("Order Date").resample(freq).sum().reset_index()
    df.columns = ["ds", "y"]
    df = df[df["y"] > 0].copy()
    df["lag_1"] = df["y"].shift(1)
    df["lag_2"] = df["y"].shift(2)
    df["lag_4"] = df["y"].shift(4)
    df["rolling_mean_4"] = df["y"].shift(1).rolling(4).mean()
    df["rolling_std_4"] = df["y"].shift(1).rolling(4).std()
    df["month"] = df["ds"].dt.month
    df["quarter"] = df["ds"].dt.quarter
    df["week"] = df["ds"].dt.isocalendar().week.astype(int)
    df["year"] = df["ds"].dt.year
    df["trend"] = np.arange(len(df))
    return df.dropna()


@st.cache_resource(show_spinner=False)
def get_artifacts(freq: str):
    path = f"{ARTIFACTS_DIR}/ensemble_{freq}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def render_metrics(artifacts, future):
    m = artifacts["metrics"]
    tiles = [
        ("MAPE", f"{m['mape']*100:.2f}%"),
        ("RMSE", f"${m['rmse']:,.0f}"),
        ("Forecast Total", f"${future['y_pred'].sum():,.0f}"),
        ("Avg per Period", f"${future['y_pred'].mean():,.0f}"),
    ]
    for col, (label, value) in zip(st.columns(4), tiles):
        col.markdown(f"""
        <div class="metric-tile">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)


def render_forecast_chart(df, future, freq_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], name="Historical",
                             line=dict(color=OCEAN["accent2"], width=2),
                             mode="lines+markers", marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=future["ds"], y=future["y_pred"], name="Forecast",
                             line=dict(color=OCEAN["accent1"], width=2.5, dash="dot"),
                             mode="lines+markers", marker=dict(size=6, symbol="diamond")))
    upper, lower = future["y_pred"] * 1.15, future["y_pred"] * 0.85
    fig.add_trace(go.Scatter(
        x=pd.concat([future["ds"], future["ds"][::-1]]),
        y=pd.concat([upper, lower[::-1]]),
        fill="toself", fillcolor="rgba(0,194,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence Band"
    ))
    fig.update_layout(**plotly_layout(f"{freq_label} Sales Forecast"))
    st.plotly_chart(fig, use_container_width=True)


def render_shap(artifacts):
    shap_vals = artifacts["shap_values"]
    feat_cols = artifacts["feature_cols"]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({"Feature": feat_cols, "Mean |SHAP|": mean_abs}).sort_values("Mean |SHAP|")
    fig = go.Figure(go.Bar(
        x=shap_df["Mean |SHAP|"], y=shap_df["Feature"], orientation="h",
        marker=dict(color=shap_df["Mean |SHAP|"],
                    colorscale=[[0, OCEAN["accent3"]], [1, OCEAN["accent1"]]], showscale=False)
    ))
    fig.update_layout(**plotly_layout("Feature Importance (SHAP)"))
    st.plotly_chart(fig, use_container_width=True)

    last_shap = shap_vals[-1]
    wf_df = pd.DataFrame({"Feature": feat_cols, "SHAP Value": last_shap}).sort_values("SHAP Value")
    colors = [OCEAN["danger"] if v < 0 else OCEAN["accent2"] for v in wf_df["SHAP Value"]]
    fig2 = go.Figure(go.Bar(x=wf_df["SHAP Value"], y=wf_df["Feature"],
                            orientation="h", marker_color=colors))
    fig2.update_layout(**plotly_layout("SHAP Waterfall (Last Period)"))
    st.plotly_chart(fig2, use_container_width=True)


def render_decomposition(df):
    fig = px.line(df, x="ds", y="y")
    fig.update_traces(line_color=OCEAN["accent2"])
    fig.update_layout(**plotly_layout("Historical Sales Trend"))
    st.plotly_chart(fig, use_container_width=True)

    df2 = df.copy()
    df2["rolling_mean"] = df2["y"].rolling(4).mean()
    df2["rolling_std"] = df2["y"].rolling(4).std()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df2["ds"], y=df2["y"], name="Sales",
                              line=dict(color=OCEAN["accent2"], width=1.5)))
    fig2.add_trace(go.Scatter(x=df2["ds"], y=df2["rolling_mean"], name="4-Period MA",
                              line=dict(color=OCEAN["accent1"], width=2)))
    fig2.add_trace(go.Scatter(x=df2["ds"], y=df2["rolling_std"], name="Rolling Std",
                              line=dict(color=OCEAN["accent3"], width=1.5, dash="dash")))
    fig2.update_layout(**plotly_layout("Rolling Statistics"))
    st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        m = df.copy()
        m["month_name"] = m["ds"].dt.strftime("%b")
        m_avg = m.groupby("month_name")["y"].mean().reset_index()
        fig3 = px.bar(m_avg, x="month_name", y="y", color="y",
                      color_continuous_scale=[[0, OCEAN["accent3"]], [1, OCEAN["accent1"]]])
        fig3.update_layout(**plotly_layout("Avg Sales by Month"))
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        df["quarter_label"] = "Q" + df["ds"].dt.quarter.astype(str)
        q_avg = df.groupby("quarter_label")["y"].mean().reset_index()
        fig4 = px.bar(q_avg, x="quarter_label", y="y", color="y",
                      color_continuous_scale=[[0, OCEAN["accent3"]], [1, OCEAN["accent2"]]])
        fig4.update_layout(**plotly_layout("Avg Sales by Quarter"))
        st.plotly_chart(fig4, use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:20px 0 10px">
        <h2 style="margin:4px 0;color:{OCEAN['accent1']}">Sales Forecast</h2>
        <p style="color:{OCEAN['subtext']};font-size:0.8rem">Ensemble + Explainability</p>
    </div><hr>
    """, unsafe_allow_html=True)

    freq_label = st.selectbox("Forecast Frequency", list(FREQ_MAP.keys()))
    freq = FREQ_MAP[freq_label]
    periods = st.slider("Forecast Periods", min_value=2, max_value=24,
                        value=PERIODS_DEFAULT[freq_label])

    st.markdown("<hr>", unsafe_allow_html=True)
    train_btn = st.button("Train / Retrain Model")



# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:10px 0 20px">
    <h1 style="margin:0;font-size:2rem;color:#FFFFFF;font-weight:700">Sales Forecasting System</h1>
    <p style="color:{OCEAN['subtext']};margin:4px 0 0">
        Ensemble ML · SHAP Explainability · {freq_label} Granularity
    </p>
</div>
""", unsafe_allow_html=True)

# ── Upload on main page ───────────────────────────────────────────────────────
st.markdown(f'<p style="color:{OCEAN["accent2"]};font-weight:600;margin-bottom:4px">Upload Dataset (CSV)</p>',
            unsafe_allow_html=True)
uploaded = st.file_uploader("CSV with Order Date and Sales columns",
                            type=["csv"], label_visibility="collapsed")

if uploaded is not None:
    size_kb = uploaded.size / 1024
    size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
    st.markdown(f"""
    <div style="background:{OCEAN['card']};border:1px solid {OCEAN['accent2']};
         border-radius:8px;padding:12px 16px;margin-top:8px;margin-bottom:8px">
        <span style="color:{OCEAN['accent2']};font-weight:600">{uploaded.name}</span>
        <span style="color:{OCEAN['subtext']};font-size:0.85rem;margin-left:12px">{size_str}</span>
    </div>
    """, unsafe_allow_html=True)

if uploaded is None:
    st.markdown(f"""
    <div class="ocean-card" style="border-color:{OCEAN['accent1']};text-align:center;padding:30px;margin-top:16px">
        <h3 style="color:{OCEAN['accent1']}">Upload Your Dataset to Get Started</h3>
        <p style="color:{OCEAN['subtext']}">
            Upload any CSV with <b>Order Date</b> and <b>Sales</b> columns.<br>
            Recommended: <a href="https://www.kaggle.com/datasets/vivek468/superstore-dataset-final"
            style="color:{OCEAN['accent2']}">Kaggle Superstore Dataset</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Process uploaded file ─────────────────────────────────────────────────────
try:
    raw = pd.read_csv(uploaded, encoding="latin-1")
    # Auto-detect columns if not named exactly
    if "Order Date" not in raw.columns:
        date_col = next((c for c in raw.columns if "date" in c.lower()), None)
        if date_col:
            raw = raw.rename(columns={date_col: "Order Date"})
    if "Sales" not in raw.columns:
        sales_col = next((c for c in raw.columns
                          if any(k in c.lower() for k in ["sales", "revenue", "amount"])), None)
        if sales_col:
            raw = raw.rename(columns={sales_col: "Sales"})

    if "Order Date" not in raw.columns or "Sales" not in raw.columns:
        st.error(f"Could not find date/sales columns. Found: {list(raw.columns)}")
        st.stop()

    df = engineer_features(raw, freq)
    df.attrs["freq"] = freq

except Exception as e:
    st.error(f"Failed to process file: {e}")
    st.stop()

# ── Train ─────────────────────────────────────────────────────────────────────
if train_btn:
    with st.spinner("Training ensemble model..."):
        artifacts = train_ensemble(df, freq)
    st.cache_resource.clear()
    st.success("Model trained successfully!")

artifacts = get_artifacts(freq)

if artifacts is None:
    st.markdown(f"""
    <div class="ocean-card" style="border-color:{OCEAN['accent3']}">
        <h3 style="color:{OCEAN['accent3']}">Model Not Trained Yet</h3>
        <p>Click <b>Train / Retrain Model</b> in the sidebar to train the ensemble.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

future = forecast_future(artifacts, df, periods)

render_metrics(artifacts, future)
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["  Forecast  ", "  Explainability  ", "  Data Analysis  "])

with tab1:
    render_forecast_chart(df, future, freq_label)
    display_df = future[["ds", "y_pred"]].copy()
    display_df.columns = ["Period", "Forecasted Sales ($)"]
    display_df["Forecasted Sales ($)"] = display_df["Forecasted Sales ($)"].map("${:,.2f}".format)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab2:
    render_shap(artifacts)
    st.markdown(f"""
    <div class="ocean-card">
        <h3 style="color:#FFFFFF;font-weight:700">How to Read SHAP Values</h3>
        <p style="color:#FFFFFF">
        • <span style="color:{OCEAN['accent1']}">Mean |SHAP|</span> — overall feature importance.<br>
        • <span style="color:{OCEAN['accent2']}">Positive values</span> push the forecast higher.<br>
        • <span style="color:{OCEAN['danger']}">Negative values</span> pull the forecast lower.<br>
        • <b>lag_1</b> (last period sales) is typically the strongest driver.
        </p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    render_decomposition(df)
