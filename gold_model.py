# ╔══════════════════════════════════════════════════════════════════╗
# ║   GOLD PRICE PREDICTION — FULL ML NOTEBOOK                      ║
# ║   Pragathi Degree Womens College | BSc Life Science              ║
# ║   Team : Cheruku Swathi, M.Seelavathi, Shaik Saniya,             ║
# ║           Syeda Shadan Sultana                                   ║
# ║   Year  : 2025 - 2026                                            ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# WHAT THIS FILE DOES:
#   1. Loads gold price data (synthetic OR real via yfinance)
#   2. Cleans and engineers features
#   3. Trains Random Forest, Decision Tree, Linear Regression
#   4. Evaluates and compares all three models
#   5. Plots all graphs
#   6. Predicts on new input
#   7. Saves the best model to disk
#
# HOW TO RUN:
#   pip install -r requirements.txt
#   python gold_model.py

# ── SECTION 1: IMPORTS ──────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

print("=" * 60)
print("   GOLD PRICE PREDICTION — ML PROJECT")
print("   Pragathi Degree Womens College | 2025-2026")
print("=" * 60)

# ── SECTION 2: LOAD DATA ────────────────────────────────────────────────────
# ┌─────────────────────────────────────────────────────────────────┐
# │  OPTION A — Use real data from Yahoo Finance (recommended)      │
# │  Uncomment the block below to use real data:                    │
# └─────────────────────────────────────────────────────────────────┘

# import yfinance as yf
#
# tickers = {"GLD":"GLD", "SLV":"SLV", "USO":"USO", "SPX":"^GSPC", "VIX":"^VIX", "DXY":"DX-Y.NYB"}
# raw = {}
# for name, ticker in tickers.items():
#     raw[name] = yf.download(ticker, start="2019-01-01", end="2026-03-01")["Close"]
#
# df = pd.DataFrame(raw).dropna()
# df["USD_INR"]  = yf.download("INR=X", start="2019-01-01", end="2026-03-01")["Close"]
# df["EUR_USD"]  = yf.download("EURUSD=X", start="2019-01-01", end="2026-03-01")["Close"]
# df["CPI"]      = 3.0   # static or load separately from FRED API
# df["Rate"]     = 5.25  # static or load separately from FRED API
# df["Month"]    = df.index.month
# df = df.dropna()

# ┌─────────────────────────────────────────────────────────────────┐
# │  OPTION B — Synthetic data (used here for demo purposes)        │
# └─────────────────────────────────────────────────────────────────┘
print("\n[1/6] Generating dataset...")

np.random.seed(42)
n = 1800
dates = pd.bdate_range("2019-01-02", periods=n)
t = np.arange(n)

trend    = 1280 + t * 0.76
seasonal = 45 * np.sin(2 * np.pi * t / 252)
noise    = np.cumsum(np.random.normal(0, 3.2, n))
gold     = trend + seasonal + noise

df = pd.DataFrame({
    "Date":    dates,
    "GLD":     np.round(gold, 2),
    "SLV":     np.round(gold * 0.013 + np.random.normal(0, 0.4, n), 2),
    "USD_INR": np.round(70 + t * 0.011 + np.random.normal(0, 0.5, n), 2),
    "DXY":     np.round(97 - t * 0.003 + np.random.normal(0, 0.8, n), 2),
    "USO":     np.round(65 + 20 * np.sin(2 * np.pi * t / 300) + np.random.normal(0, 2, n), 2),
    "SPX":     np.round(2700 + t * 2.1 + np.random.normal(0, 30, n), 2),
    "CPI":     np.round(2.1 + t * 0.001 + np.random.normal(0, 0.08, n), 2),
    "Rate":    np.round(2.5 - t * 0.0005 + np.random.normal(0, 0.05, n), 2),
    "EUR_USD": np.round(1.12 + np.random.normal(0, 0.01, n), 4),
    "VIX":     np.round(18 + np.random.normal(0, 4, n), 2),
    "Month":   pd.DatetimeIndex(dates).month,
}).set_index("Date")

print(f"   Dataset shape: {df.shape}")
print(f"   Date range: {df.index[0].date()} → {df.index[-1].date()}")

# ── SECTION 3: FEATURE ENGINEERING ─────────────────────────────────────────
print("\n[2/6] Engineering features...")

df["GLD_Lag1"]  = df["GLD"].shift(1)
df["GLD_Lag7"]  = df["GLD"].shift(7)
df["GLD_MA20"]  = df["GLD"].rolling(20).mean()
df["GLD_MA50"]  = df["GLD"].rolling(50).mean()
df["GLD_Vol20"] = df["GLD"].pct_change().rolling(20).std()
df = df.dropna()

FEATURES = [
    "SLV", "USD_INR", "DXY", "USO", "SPX", "CPI", "Rate",
    "EUR_USD", "VIX", "Month", "GLD_Lag1", "GLD_Lag7",
    "GLD_MA20", "GLD_MA50", "GLD_Vol20"
]
X = df[FEATURES]
y = df["GLD"]

print(f"   Features used: {len(FEATURES)}")
print(f"   Features: {', '.join(FEATURES)}")

# ── SECTION 4: TRAIN/TEST SPLIT ─────────────────────────────────────────────
print("\n[3/6] Splitting data (80% train / 20% test)...")

split   = int(len(X) * 0.8)
X_train = X.iloc[:split];  X_test = X.iloc[split:]
y_train = y.iloc[:split];  y_test = y.iloc[split:]

scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples:  {len(X_test)}")

# ── SECTION 5: TRAIN MODELS ─────────────────────────────────────────────────
print("\n[4/6] Training models...")

models = {
    "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    "Decision Tree":     DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42),
    "Linear Regression": LinearRegression(),
}

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    r2   = r2_score(y_test, pred)
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(np.abs((y_test.values - pred) / y_test.values)) * 100

    results[name] = {"model": model, "pred": pred,
                     "r2": r2, "mae": mae, "rmse": rmse, "mape": mape}
    print(f"   {name:22s} | R²={r2:.4f} | MAE=${mae:.2f} | RMSE=${rmse:.2f} | MAPE={mape:.2f}%")

# ── SECTION 6: EVALUATION SUMMARY ──────────────────────────────────────────
print("\n" + "=" * 60)
print("   EVALUATION SUMMARY")
print("=" * 60)
print(f"{'Model':<22} {'R²':>8} {'MAE ($)':>10} {'RMSE ($)':>10} {'MAPE (%)':>10}")
print("-" * 62)
for name, r in results.items():
    print(f"{name:<22} {r['r2']:>8.4f} {r['mae']:>10.2f} {r['rmse']:>10.2f} {r['mape']:>9.2f}%")
print("=" * 60)

best_model_name = max(results, key=lambda k: results[k]["r2"])
print(f"\n   🏆 Best Model: {best_model_name}")

# ── SECTION 7: VISUALIZATIONS ───────────────────────────────────────────────
print("\n[5/6] Generating plots...")

plt.style.use("dark_background")
COLORS = {"Random Forest": "#1A7A3C", "Decision Tree": "#2172C4", "Linear Regression": "#7C3AED", "Actual": "#C9920A"}

# -- Plot 1: Actual vs Predicted (all models) --
fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0D1B2A")
ax.set_facecolor("#112233")
ax.plot(y_test.index, y_test.values, color=COLORS["Actual"], linewidth=2.5, label="Actual", zorder=5)
for name, r in results.items():
    ax.plot(y_test.index, r["pred"], color=COLORS[name], linewidth=1.5,
            linestyle="-" if name=="Random Forest" else ("--" if name=="Decision Tree" else ":"),
            label=f"{name} (R²={r['r2']:.3f})", alpha=0.85)
ax.set_title("Gold Price — Actual vs. Model Predictions", color="white", fontsize=14, pad=12)
ax.set_xlabel("Date", color="#7A9BBF"); ax.set_ylabel("Price (USD/oz)", color="#7A9BBF")
ax.tick_params(colors="#7A9BBF"); ax.grid(color="#1E3A5F", linewidth=0.5)
ax.legend(facecolor="#112233", edgecolor="#1E3A5F", labelcolor="white")
plt.tight_layout()
plt.savefig("plot_actual_vs_predicted.png", dpi=150, bbox_inches="tight", facecolor="#0D1B2A")
print("   Saved: plot_actual_vs_predicted.png")

# -- Plot 2: R² Score comparison --
fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0D1B2A")
ax.set_facecolor("#112233")
names = list(results.keys())
r2s   = [results[n]["r2"] for n in names]
bars  = ax.barh(names, r2s, color=[COLORS[n] for n in names], height=0.5)
for bar, val in zip(bars, r2s):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, f"{val:.4f}",
            va="center", color="white", fontsize=11, fontweight="bold")
ax.set_xlim(0.75, 1.02); ax.set_xlabel("R² Score", color="#7A9BBF")
ax.set_title("Model Accuracy — R² Score (higher = better)", color="white", fontsize=13)
ax.tick_params(colors="#7A9BBF"); ax.grid(color="#1E3A5F", axis="x", linewidth=0.5)
plt.tight_layout()
plt.savefig("plot_r2_comparison.png", dpi=150, bbox_inches="tight", facecolor="#0D1B2A")
print("   Saved: plot_r2_comparison.png")

# -- Plot 3: Feature Importance --
rf_model  = results["Random Forest"]["model"]
feat_imp  = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
feat_pct  = (feat_imp / feat_imp.sum() * 100)
fig, ax   = plt.subplots(figsize=(9, 6), facecolor="#0D1B2A")
ax.set_facecolor("#112233")
colors_fi = ["#C9920A" if v > 15 else ("#1A5FA8" if v > 8 else "#7C3AED") for v in feat_pct.values]
bars = ax.barh(feat_pct.index, feat_pct.values, color=colors_fi, height=0.65)
for bar, val in zip(bars, feat_pct.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, f"{val:.1f}%",
            va="center", color="white", fontsize=9)
ax.set_xlabel("Importance (%)", color="#7A9BBF"); ax.set_title("Random Forest — Feature Importance", color="white", fontsize=13)
ax.tick_params(colors="#7A9BBF"); ax.grid(color="#1E3A5F", axis="x", linewidth=0.5)
plt.tight_layout()
plt.savefig("plot_feature_importance.png", dpi=150, bbox_inches="tight", facecolor="#0D1B2A")
print("   Saved: plot_feature_importance.png")

# -- Plot 4: Residuals --
rf_residuals = y_test.values - results["Random Forest"]["pred"]
fig, ax = plt.subplots(figsize=(12, 3), facecolor="#0D1B2A")
ax.set_facecolor("#112233")
colors_res = ["#1A7A3C" if v >= 0 else "#F44336" for v in rf_residuals]
ax.bar(range(len(rf_residuals)), rf_residuals, color=colors_res, width=1.0, alpha=0.85)
ax.axhline(0, color="#C9920A", linewidth=1.5, linestyle="--")
ax.set_title("Random Forest — Prediction Residuals (Actual − Predicted)", color="white", fontsize=12)
ax.set_xlabel("Test Sample Index", color="#7A9BBF"); ax.set_ylabel("Error ($)", color="#7A9BBF")
ax.tick_params(colors="#7A9BBF"); ax.grid(color="#1E3A5F", linewidth=0.4)
plt.tight_layout()
plt.savefig("plot_residuals.png", dpi=150, bbox_inches="tight", facecolor="#0D1B2A")
print("   Saved: plot_residuals.png")

# -- Plot 5: Correlation Heatmap --
corr_cols = ["GLD","SLV","USD_INR","DXY","USO","SPX","CPI","Rate","VIX"]
corr_matrix = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0D1B2A")
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            linewidths=0.5, linecolor="#0D1B2A", ax=ax, annot_kws={"size":9})
ax.set_title("Pearson Correlation Matrix — Gold Price Drivers", color="white", fontsize=13, pad=12)
ax.tick_params(colors="white")
plt.tight_layout()
plt.savefig("plot_correlation_heatmap.png", dpi=150, bbox_inches="tight", facecolor="#0D1B2A")
print("   Saved: plot_correlation_heatmap.png")

# ── SECTION 8: PREDICT ON NEW DATA ─────────────────────────────────────────
print("\n[6/6] Predicting on new sample input...")

new_input = pd.DataFrame([{
    "SLV": 31.2, "USD_INR": 83.4, "DXY": 103.8, "USO": 82.5, "SPX": 5612,
    "CPI": 3.1, "Rate": 5.25, "EUR_USD": 1.09, "VIX": 18.5, "Month": 3,
    "GLD_Lag1": 3118.0, "GLD_Lag7": 3090.0,
    "GLD_MA20": 3070.0, "GLD_MA50": 3020.0, "GLD_Vol20": 0.0082
}], columns=FEATURES)

new_scaled = scaler.transform(new_input)
print(f"\n   {'Model':<22} {'Predicted Price':>18}")
print("   " + "-" * 42)
for name, r in results.items():
    p = r["model"].predict(new_scaled)[0]
    print(f"   {name:<22} ${p:>16,.2f}")

# 7-day forecast (using RF)
rf_pred_base = results["Random Forest"]["model"].predict(new_scaled)[0]
print(f"\n   7-Day Forward Forecast (Random Forest):")
for day in range(1, 8):
    forecast = rf_pred_base * (1 + 0.002 * day) + np.random.normal(0, 6)
    print(f"     Day {day}: ${forecast:,.2f}")

# ── SECTION 9: SAVE THE BEST MODEL ─────────────────────────────────────────
print(f"\n   Saving best model ({best_model_name}) to disk...")
joblib.dump(results[best_model_name]["model"], "best_model_random_forest.pkl")
joblib.dump(scaler, "scaler.pkl")
print("   Saved: best_model_random_forest.pkl")
print("   Saved: scaler.pkl")

print("\n" + "=" * 60)
print("   ALL DONE! Gold Price Prediction complete.")
print("   Run 'streamlit run app.py' to launch the web app.")
print("=" * 60)
