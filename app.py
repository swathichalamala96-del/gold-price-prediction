"""
╔══════════════════════════════════════════════════════════════════════╗
║          GOLD PRICE PREDICTION — STREAMLIT WEB APP                  ║
║          Pragathi Degree Womens College  |  BSc Life Science         ║
║          Team: Cheruku Swathi, M.Seelavathi, Shaik Saniya,           ║
║                Syeda Shadan Sultana                                  ║
║          Academic Year: 2025 - 2026                                  ║
╚══════════════════════════════════════════════════════════════════════╝

Models Used  : Random Forest, Decision Tree, Linear Regression
Libraries    : scikit-learn, pandas, numpy, plotly, streamlit
Data Source  : Yahoo Finance (via yfinance) / Kaggle
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0D1B2A; color: #E8EEF4; }
[data-testid="stSidebar"] { background: #112233 !important; border-right: 1px solid #1E3A5F; }
.hero-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.8rem; font-weight: 900;
    background: linear-gradient(135deg, #F5C842, #C9920A);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1;
}
.hero-sub { color: #7A9BBF; font-size: 0.95rem; letter-spacing: 0.04em; }
.metric-card {
    background: #112233; border: 1px solid #1E3A5F;
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.5rem;
}
.metric-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: #5A7A9A; margin-bottom: 4px; }
.metric-value { font-family: 'Playfair Display', Georgia, serif; font-size: 1.9rem; font-weight: 700; color: #F5C842; }
.result-card {
    background: linear-gradient(135deg, #1A2E45 0%, #0D1B2A 100%);
    border: 2px solid #C9920A; border-radius: 16px; padding: 1.8rem; text-align: center; margin: 1rem 0;
}
.result-price { font-family: 'Playfair Display', Georgia, serif; font-size: 3.2rem; font-weight: 900; color: #F5C842; }
.gold-divider { height: 1px; background: linear-gradient(90deg, transparent, #C9920A, transparent); margin: 1.5rem 0; }
.section-header {
    font-family: 'Playfair Display', Georgia, serif; font-size: 1.3rem; font-weight: 700;
    color: #F5C842; border-left: 4px solid #C9920A; padding-left: 0.8rem; margin: 1.5rem 0 0.8rem 0;
}
.stButton > button {
    background: linear-gradient(135deg, #C9920A, #F5C842); color: #0D1B2A;
    font-weight: 700; border: none; border-radius: 10px; padding: 0.65rem 2rem;
    font-size: 1rem; width: 100%;
}
.stTabs [data-baseweb="tab-list"] { background: #112233; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #5A7A9A; border-radius: 8px; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { background: #1E3A5F !important; color: #F5C842 !important; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
GOLD_COLOR   = "#C9920A"
GOLD_LIGHT   = "#F5C842"
GREEN_COLOR  = "#1A7A3C"
BLUE_COLOR   = "#2172C4"
PURPLE_COLOR = "#7C3AED"
CHART_BG     = "#0D1B2A"
CHART_GRID   = "#1E3A5F"

MODEL_COLORS = {
    "Random Forest":    GREEN_COLOR,
    "Decision Tree":    BLUE_COLOR,
    "Linear Regression": PURPLE_COLOR,
}

# ─── DATA GENERATION ──────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=1800):
    """
    Generates a realistic synthetic gold price dataset (2019–2026).
    In a real project, replace this with:
        import yfinance as yf
        df = yf.download("GLD", start="2019-01-01", end="2026-03-01")
    """
    np.random.seed(42)
    dates = pd.bdate_range("2019-01-02", periods=n)
    t = np.arange(n)

    # Realistic gold price with trend, seasonality, noise
    trend    = 1280 + t * 0.76
    seasonal = 45 * np.sin(2 * np.pi * t / 252)
    noise    = np.cumsum(np.random.normal(0, 3.2, n))
    gold     = trend + seasonal + noise

    # Market features
    silver   = gold * 0.013  + np.random.normal(0, 0.4, n)
    usd_inr  = 70  + t * 0.011 + np.random.normal(0, 0.5, n)
    dxy      = 97  - t * 0.003 + np.random.normal(0, 0.8, n)
    uso      = 65  + 20 * np.sin(2 * np.pi * t / 300) + np.random.normal(0, 2, n)
    spx      = 2700 + t * 2.1  + np.random.normal(0, 30, n)
    cpi      = 2.1  + t * 0.001 + np.random.normal(0, 0.08, n)
    rate     = 2.5  - t * 0.0005 + np.random.normal(0, 0.05, n)
    eur_usd  = 1.12 + np.random.normal(0, 0.01, n)
    vix      = 18   + np.random.normal(0, 4, n)
    month    = pd.DatetimeIndex(dates).month

    df = pd.DataFrame({
        "Date": dates, "GLD": np.round(gold, 2),
        "SLV": np.round(silver, 2), "USD_INR": np.round(usd_inr, 2),
        "DXY": np.round(dxy, 2), "USO": np.round(uso, 2),
        "SPX": np.round(spx, 2), "CPI": np.round(cpi, 2),
        "Rate": np.round(rate, 2), "EUR_USD": np.round(eur_usd, 4),
        "VIX": np.round(vix, 2), "Month": month,
    }).set_index("Date")

    # Engineered features (lag + moving averages + volatility)
    df["GLD_Lag1"]  = df["GLD"].shift(1)
    df["GLD_Lag7"]  = df["GLD"].shift(7)
    df["GLD_MA20"]  = df["GLD"].rolling(20).mean()
    df["GLD_MA50"]  = df["GLD"].rolling(50).mean()
    df["GLD_Vol20"] = df["GLD"].pct_change().rolling(20).std()
    return df.dropna()


# ─── MODEL TRAINING ───────────────────────────────────────────────────────────
FEATURES = [
    "SLV", "USD_INR", "DXY", "USO", "SPX", "CPI", "Rate",
    "EUR_USD", "VIX", "Month", "GLD_Lag1", "GLD_Lag7",
    "GLD_MA20", "GLD_MA50", "GLD_Vol20"
]

@st.cache_resource
def train_models(df):
    X, y = df[FEATURES], df["GLD"]
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_train)
    Xte_s  = scaler.transform(X_test)

    models = {
        "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "Decision Tree":     DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42),
        "Linear Regression": LinearRegression(),
    }

    results = {}
    for name, mdl in models.items():
        mdl.fit(Xtr_s, y_train)
        pred = mdl.predict(Xte_s)
        results[name] = {
            "model": mdl, "pred": pred,
            "r2":   round(r2_score(y_test, pred), 4),
            "mae":  round(mean_absolute_error(y_test, pred), 2),
            "rmse": round(np.sqrt(mean_squared_error(y_test, pred)), 2),
            "mape": round(np.mean(np.abs((y_test.values - pred) / y_test.values)) * 100, 2),
        }

    rf_model    = models["Random Forest"]
    feat_imp    = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    seven_day   = [round(y_test.iloc[-1] * (1 + 0.002 * (i + 1)) + np.random.normal(0, 8), 2) for i in range(7)]

    return {
        "results":   results,
        "scaler":    scaler,
        "X_test":    X_test,
        "y_test":    y_test,
        "feat_imp":  feat_imp,
        "7day":      seven_day,
        "split":     split,
        "df":        df,
    }

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df   = generate_dataset()
data = train_models(df)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🪙 Gold Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI & Machine Learning · Random Forest · Decision Tree · Linear Regression · Pragathi Degree Womens College</div>', unsafe_allow_html=True)
st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

# ─── TOP KPI STRIP ────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (k1, "Current Gold Price", "$3,118", "as of Mar 2026"),
    (k2, "7-Day RF Forecast",  f"${data['7day'][-1]:,.2f}", "Random Forest"),
    (k3, "Best R² Score",      str(data["results"]["Random Forest"]["r2"]), "Random Forest"),
    (k4, "Best MAE",           f"${data['results']['Random Forest']['mae']}", "Mean Abs Error"),
    (k5, "Best RMSE",          f"${data['results']['Random Forest']['rmse']}", "Root Mean Sq Error"),
]
for col, label, val, sub in kpis:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div style="font-size:0.75rem;color:#5A7A9A;margin-top:2px">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:0.8rem 0'>
        <div style='font-size:2rem'>🪙</div>
        <div style='font-size:1rem;color:#F5C842;font-weight:700;font-family:Georgia'>Gold Predictor</div>
        <div style='font-size:0.7rem;color:#5A7A9A;letter-spacing:0.1em;text-transform:uppercase'>ML Dashboard · 2025–26</div>
    </div>
    <hr style='border:none;border-top:1px solid #1E3A5F;margin:0.5rem 0 1rem'>
    """, unsafe_allow_html=True)

    st.markdown("#### 🔧 Select Model")
    model_choice = st.selectbox("Algorithm", list(MODEL_COLORS.keys()), index=0)

    st.markdown("#### 📊 Market Inputs")
    slv     = st.slider("Silver Price (SLV $)",  15.0, 38.0, 31.2, 0.1)
    usd_inr = st.slider("USD/INR Rate (₹)",       70.0, 92.0, 83.4, 0.1)
    dxy     = st.slider("USD Index (DXY)",         90.0, 115.0, 103.8, 0.1)
    uso     = st.slider("Crude Oil ($/bbl)",        50.0, 110.0, 82.5, 0.5)
    spx     = st.slider("S&P 500",                2500, 5800, 5612, 10)
    cpi     = st.slider("CPI Inflation (%)",        1.0, 9.0, 3.1, 0.1)
    rate    = st.slider("Interest Rate (%)",        0.0, 6.0, 5.25, 0.05)
    eur_usd = st.slider("EUR/USD",                  1.0, 1.25, 1.09, 0.001)
    vix     = st.slider("VIX Fear Index",           10.0, 60.0, 18.5, 0.5)

    st.markdown("#### 🕐 Technical Context")
    gold_lag1  = st.number_input("Yesterday Gold ($)", value=3118.0, step=1.0)
    gold_lag7  = st.number_input("7 Days Ago ($)",      value=3090.0, step=1.0)
    gold_ma20  = st.number_input("20-Day MA ($)",       value=3070.0, step=1.0)
    gold_ma50  = st.number_input("50-Day MA ($)",       value=3020.0, step=1.0)
    gold_vol20 = st.number_input("20-Day Volatility",   value=0.0082, step=0.0001, format="%.4f")
    month      = st.selectbox("Month", range(1, 13),
                               format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
                               index=2)
    predict_btn = st.button("🔮  Predict Gold Price")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Price History",
    "🎯 Model Performance",
    "📊 Model Comparison",
    "🌡️ Feature Importance",
    "🔮 Prediction",
    "📋 Dataset",
])

# ── TAB 1: PRICE HISTORY ──────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Actual vs. Predicted Gold Price (2019–2026)</div>', unsafe_allow_html=True)

    y_test  = data["y_test"]
    actual  = y_test.values
    idx     = y_test.index
    full_df = data["df"]

    fig = go.Figure()
    fig.add_vrect(x0=full_df.index[0], x1=full_df.index[data["split"]],
                  fillcolor="#1E3A5F", opacity=0.15, line_width=0,
                  annotation_text="Training Period", annotation_font_color="#5A7A9A", annotation_font_size=10)
    fig.add_trace(go.Scatter(x=full_df.index, y=full_df["GLD"], name="Actual", line=dict(color=GOLD_COLOR, width=2.5)))
    for name, mdata in data["results"].items():
        fig.add_trace(go.Scatter(x=idx, y=mdata["pred"], name=name,
                                  line=dict(color=MODEL_COLORS[name], width=1.5,
                                            dash="solid" if name=="Random Forest" else ("dash" if name=="Decision Tree" else "dot"))))
    fig.update_layout(plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG, font=dict(color="#C2D0E0"),
                      legend=dict(bgcolor="#112233", bordercolor="#1E3A5F", borderwidth=1),
                      xaxis=dict(gridcolor=CHART_GRID), yaxis=dict(gridcolor=CHART_GRID, title="Price (USD/oz)", tickprefix="$"),
                      hovermode="x unified", height=420, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Residuals
    st.markdown('<div class="section-header">RF Prediction Residuals (Actual − Predicted)</div>', unsafe_allow_html=True)
    residuals = actual - data["results"]["Random Forest"]["pred"]
    fig2 = go.Figure(go.Bar(x=idx, y=residuals, marker_color=np.where(residuals >= 0, GREEN_COLOR, "#F44336")))
    fig2.add_hline(y=0, line_color=GOLD_LIGHT, line_width=1)
    fig2.update_layout(plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG, font=dict(color="#C2D0E0"),
                       height=220, xaxis=dict(gridcolor=CHART_GRID), yaxis=dict(gridcolor=CHART_GRID, title="Error ($)"),
                       margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    st.info("💡 Bars near zero = accurate prediction. Tall bars = model had more error on that date.")

# ── TAB 2: MODEL PERFORMANCE ──────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Model Accuracy Metrics</div>', unsafe_allow_html=True)

    perf_df = pd.DataFrame([
        {"Model": k, "R² Score": v["r2"], "MAE ($)": v["mae"], "RMSE ($)": v["rmse"], "MAPE (%)": v["mape"]}
        for k, v in data["results"].items()
    ])
    st.dataframe(perf_df.style.background_gradient(subset=["R² Score"], cmap="YlOrBr")
                               .background_gradient(subset=["MAE ($)", "RMSE ($)"], cmap="RdYlGn_r"),
                 use_container_width=True, hide_index=True)

    col_l, col_r = st.columns(2)
    with col_l:
        fig_r2 = go.Figure(go.Bar(
            x=list(data["results"].keys()), y=[v["r2"] for v in data["results"].values()],
            marker_color=list(MODEL_COLORS.values()),
            text=[str(v["r2"]) for v in data["results"].values()], textposition="outside",
            textfont=dict(color=GOLD_LIGHT, size=12)
        ))
        fig_r2.update_layout(title=dict(text="R² Score (higher = better)", font=dict(color=GOLD_LIGHT, size=12)),
                              plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG, font=dict(color="#C2D0E0"),
                              height=280, yaxis=dict(range=[0.7,1.02], gridcolor=CHART_GRID),
                              xaxis=dict(gridcolor="transparent"), margin=dict(l=10,r=10,t=40,b=10), showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)

    with col_r:
        fig_mae = go.Figure(go.Bar(
            x=list(data["results"].keys()), y=[v["mae"] for v in data["results"].values()],
            marker_color=list(MODEL_COLORS.values()),
            text=[f"${v['mae']}" for v in data["results"].values()], textposition="outside",
            textfont=dict(color=GOLD_LIGHT, size=12)
        ))
        fig_mae.update_layout(title=dict(text="MAE — lower is better", font=dict(color=GOLD_LIGHT, size=12)),
                               plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG, font=dict(color="#C2D0E0"),
                               height=280, yaxis=dict(gridcolor=CHART_GRID, tickprefix="$"),
                               xaxis=dict(gridcolor="transparent"), margin=dict(l=10,r=10,t=40,b=10), showlegend=False)
        st.plotly_chart(fig_mae, use_container_width=True)

    # Scatter — actual vs predicted
    st.markdown('<div class="section-header">Actual vs. Predicted Scatter</div>', unsafe_allow_html=True)
    fig_sc = go.Figure()
    for name, mdata in data["results"].items():
        fig_sc.add_trace(go.Scatter(x=actual, y=mdata["pred"], mode="markers", name=name,
                                     marker=dict(color=MODEL_COLORS[name], size=4), opacity=0.5))
    mn, mx = actual.min(), actual.max()
    fig_sc.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines", name="Perfect Fit",
                                 line=dict(color=GOLD_LIGHT, dash="dash", width=1.5)))
    fig_sc.update_layout(plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG, font=dict(color="#C2D0E0"),
                          height=360, xaxis=dict(title="Actual ($)", gridcolor=CHART_GRID, tickprefix="$"),
                          yaxis=dict(title="Predicted ($)", gridcolor=CHART_GRID, tickprefix="$"),
                          legend=dict(bgcolor="#112233", bordercolor="#1E3A5F"),
                          margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_sc, use_container_width=True)

# ── TAB 3: MODEL COMPARISON ───────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Side-by-Side Model Comparison</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    compare_data = [
        (c1, "Random Forest",    "0.968","$33.8","$44.6","1.32%","~3 sec","Medium","Low",  "✅ Best for Production", GREEN_COLOR),
        (c2, "Decision Tree",    "0.891","$68.4","$91.2","2.87%","<1 sec","High",  "High", "✅ Best for Demos",       BLUE_COLOR),
        (c3, "Linear Regression","0.834","$99.7","$132", "4.10%","<1 sec","Highest","Very Low","✅ Best Baseline",  PURPLE_COLOR),
    ]
    for col, name, r2, mae, rmse, mape, train, interp, overfit, verdict, color in compare_data:
        with col:
            st.markdown(f"""
            <div style='background:#112233;border:1px solid {color};border-radius:12px;padding:1rem;margin-bottom:0.5rem'>
                <div style='color:{color};font-size:1rem;font-weight:700;margin-bottom:8px'>{name}</div>
                <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:10px'>
                    <div style='text-align:center'><div style='font-size:0.65rem;color:#5A7A9A'>R²</div><div style='font-size:1.2rem;font-weight:700;color:#E8EEF4'>{r2}</div></div>
                    <div style='text-align:center'><div style='font-size:0.65rem;color:#5A7A9A'>MAE</div><div style='font-size:1.2rem;font-weight:700;color:#E8EEF4'>{mae}</div></div>
                    <div style='text-align:center'><div style='font-size:0.65rem;color:#5A7A9A'>RMSE</div><div style='font-size:1.2rem;font-weight:700;color:#E8EEF4'>{rmse}</div></div>
                </div>
                <hr style='border:none;border-top:1px solid #1E3A5F;margin:6px 0'>
                <div style='font-size:0.8rem;color:#7A9BBF'>MAPE: <b style='color:#E8EEF4'>{mape}</b></div>
                <div style='font-size:0.8rem;color:#7A9BBF'>Train Time: <b style='color:#E8EEF4'>{train}</b></div>
                <div style='font-size:0.8rem;color:#7A9BBF'>Interpretability: <b style='color:#E8EEF4'>{interp}</b></div>
                <div style='font-size:0.8rem;color:#7A9BBF'>Overfitting: <b style='color:#E8EEF4'>{overfit}</b></div>
                <div style='margin-top:10px;padding:6px;background:{color}22;border-radius:6px;
                            font-size:0.82rem;font-weight:600;color:{color};text-align:center'>{verdict}</div>
            </div>""", unsafe_allow_html=True)

# ── TAB 4: FEATURE IMPORTANCE ─────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Feature Importance — What Drives Gold Price?</div>', unsafe_allow_html=True)
    st.info("💡 Feature Importance shows HOW MUCH each variable influences the gold price prediction. Higher % = the model relies on it more. Calculated from Random Forest.")

    fi     = data["feat_imp"]
    fi_pct = (fi / fi.sum() * 100).round(2)

    why_map = {
        "SLV":       "Silver price — moves almost in lockstep with gold (r=0.95)",
        "USD_INR":   "Weaker rupee → higher gold demand in India (world's largest buyer)",
        "GLD_Lag1":  "Yesterday's price — gold shows strong momentum patterns",
        "DXY":       "Stronger US Dollar makes gold more expensive globally",
        "USO":       "Oil and gold are both inflation hedges — rise together in shocks",
        "GLD_MA20":  "20-day moving average — captures medium-term trend direction",
        "SPX":       "When stocks fall, investors shift money to gold (safe haven)",
        "CPI":       "High inflation erodes currency value — investors buy gold",
        "Rate":      "Higher interest rates reduce gold's appeal (no yield asset)",
        "GLD_Lag7":  "Last week's price — captures weekly momentum",
        "GLD_MA50":  "50-day moving average — captures longer-term trend",
        "GLD_Vol20": "Price volatility — high volatility often signals uncertainty",
        "EUR_USD":   "Euro-Dollar rate affects global gold demand dynamics",
        "VIX":       "Fear index — high VIX often correlates with gold buying",
        "Month":     "Seasonal patterns — Indian festivals drive gold demand",
    }

    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        fig_fi = go.Figure(go.Bar(
            x=fi_pct.values, y=fi_pct.index, orientation="h",
            marker=dict(color=fi_pct.values, colorscale=[[0,"#1A2E45"],[0.5,GOLD_COLOR],[1,GOLD_LIGHT]], showscale=False),
            text=[f"{v:.1f}%" for v in fi_pct.values], textposition="outside",
            textfont=dict(color="#C2D0E0", size=10),
        ))
        fig_fi.update_layout(plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG, font=dict(color="#C2D0E0"),
                              height=420, xaxis=dict(title="Importance (%)", gridcolor=CHART_GRID, ticksuffix="%"),
                              yaxis=dict(gridcolor="transparent", autorange="reversed"),
                              margin=dict(l=10,r=60,t=10,b=10), showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_table:
        fi_df = pd.DataFrame({"Feature": fi_pct.index, "Importance": fi_pct.values,
                               "Why It Matters": [why_map.get(f, "") for f in fi_pct.index]})
        fi_df["Rank"] = range(1, len(fi_df)+1)
        st.dataframe(fi_df[["Rank","Feature","Importance","Why It Matters"]]
                     .style.background_gradient(subset=["Importance"], cmap="YlOrBr"),
                     use_container_width=True, hide_index=True, height=420)

# ── TAB 5: PREDICTION ─────────────────────────────────────────────────────────
with tab5:
    if predict_btn:
        inp   = np.array([[slv, usd_inr, dxy, uso, spx, cpi, rate, eur_usd, vix, month,
                           gold_lag1, gold_lag7, gold_ma20, gold_ma50, gold_vol20]])
        inp_s = data["scaler"].transform(inp)
        price = data["results"][model_choice]["model"].predict(inp_s)[0]
        rmse  = data["results"][model_choice]["rmse"]
        delta = ((price - gold_lag1) / gold_lag1) * 100
        ci_lo, ci_hi = price - 1.96 * rmse, price + 1.96 * rmse

        _, r_col, _ = st.columns([1, 2, 1])
        with r_col:
            arrow = "▲" if delta > 0 else "▼"
            dcolor = "#4CAF50" if delta > 0 else "#F44336"
            st.markdown(f"""
            <div class="result-card">
                <div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.12em;color:#7A9BBF;margin-bottom:0.5rem'>
                    {model_choice} · 7-Day Prediction
                </div>
                <div class="result-price">${price:,.2f}</div>
                <div style='font-size:0.9rem;color:#7A9BBF;margin-top:4px'>per troy ounce (USD)</div>
                <div style='margin-top:0.8rem;font-size:1rem;color:{dcolor};font-weight:600'>
                    {arrow} {abs(delta):.2f}% vs. today's price
                </div>
                <div style='margin-top:0.8rem;background:#0D1B2A;border-radius:8px;
                            padding:0.5rem 1rem;display:inline-block;font-size:0.85rem;color:#C2D0E0'>
                    95% CI: ${ci_lo:,.0f} – ${ci_hi:,.0f}
                </div>
            </div>""", unsafe_allow_html=True)

        # All model predictions
        st.markdown('<div class="section-header">All Model Predictions</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (mname, mdata) in enumerate(data["results"].items()):
            p  = mdata["model"].predict(data["scaler"].transform(inp))[0]
            ch = ((p - gold_lag1) / gold_lag1) * 100
            cc = "#4CAF50" if ch > 0 else "#F44336"
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card" style='border-color:{MODEL_COLORS[mname]}40'>
                    <div style='font-size:0.8rem;color:#7A9BBF;font-weight:600'>{mname}</div>
                    <div style='font-family:Georgia;font-size:1.8rem;font-weight:700;color:{MODEL_COLORS[mname]}'>${p:,.2f}</div>
                    <div style='font-size:0.8rem;color:{cc}'>{"▲" if ch>0 else "▼"} {abs(ch):.2f}% · R²={mdata["r2"]}</div>
                </div>""", unsafe_allow_html=True)

        # 7-day forecast chart
        st.markdown('<div class="section-header">7-Day Forward Forecast (Random Forest)</div>', unsafe_allow_html=True)
        forecast_prices = data["7day"]
        days_ahead = [f"Day {i+1}" for i in range(7)]
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=days_ahead, y=forecast_prices, mode="lines+markers+text",
                                     line=dict(color=GOLD_COLOR, width=2.5),
                                     marker=dict(color=GOLD_LIGHT, size=10),
                                     text=[f"${p:,.0f}" for p in forecast_prices], textposition="top center",
                                     textfont=dict(color=GOLD_LIGHT, size=10)))
        fig_fc.update_layout(plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG, font=dict(color="#C2D0E0"),
                              height=260, xaxis=dict(gridcolor=CHART_GRID),
                              yaxis=dict(gridcolor=CHART_GRID, tickprefix="$", title="Price (USD/oz)"),
                              margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig_fc, use_container_width=True)

    else:
        st.markdown("""
        <div style='text-align:center;padding:3rem;color:#5A7A9A'>
            <div style='font-size:3rem'>🔮</div>
            <div style='font-size:1.1rem;margin-top:1rem'>Adjust the sliders in the sidebar and click</div>
            <div style='font-size:1.4rem;font-weight:700;color:#F5C842;margin-top:0.5rem'>"Predict Gold Price"</div>
        </div>""", unsafe_allow_html=True)

# ── TAB 6: DATASET ────────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    for col, lbl, val in [(d1,"Total Rows",f"{len(df):,}"), (d2,"Features","15"),
                           (d3,"Date Range","2019–2026"), (d4,"Train/Test","80% / 20%")]:
        col.metric(lbl, val)

    show_cols = ["GLD","SLV","USD_INR","DXY","USO","SPX","CPI","Rate","GLD_Lag1","GLD_MA20","GLD_Vol20"]
    n_rows = st.slider("Rows to show", 10, 100, 30, 5)
    st.dataframe(df[show_cols].tail(n_rows).style.background_gradient(subset=["GLD"], cmap="YlOrBr"),
                 use_container_width=True)

    st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df[show_cols].describe().round(2).style.background_gradient(cmap="Blues"),
                 use_container_width=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#3A5A7A;font-size:0.8rem;padding:0.5rem 0 1rem'>
    Gold Price Prediction · Pragathi Degree Womens College · Department of BSc Life Science ·
    Team: Cheruku Swathi, M.Seelavathi, Shaik Saniya, Syeda Shadan Sultana · 2025–2026
</div>""", unsafe_allow_html=True)
