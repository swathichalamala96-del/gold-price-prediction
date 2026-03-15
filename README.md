# 🪙 Gold Price Prediction — AI & Machine Learning

> **Pragathi Degree Womens College | Department of BSc Life Science**
> Academic Year: 2025 – 2026

---

## 👩‍💻 Team Members
| Name | Role |
|------|------|
| Cheruku Swathi | Model Training & Streamlit App |
| M. Seelavathi | Data Collection & EDA |
| Shaik Saniya | Feature Engineering & Visualization |
| Syeda Shadan Sultana | Model Evaluation & Presentation |

---

## 📌 Project Overview
This project predicts the **daily closing price of gold (USD per troy ounce)** using three Machine Learning regression models trained on 7 years of real market data (2019–2026).

Gold prices are influenced by macroeconomic factors like exchange rates, inflation, crude oil, and stock markets. Our ML pipeline captures these patterns to generate accurate price predictions.

---

## 🤖 Models Used
| Model | R² Score | MAE | RMSE | MAPE |
|-------|----------|-----|------|------|
| **Random Forest** ✅ | **0.968** | **$33.8** | **$44.6** | **1.32%** |
| Decision Tree | 0.891 | $68.4 | $91.2 | 2.87% |
| Linear Regression | 0.834 | $99.7 | $132.0 | 4.10% |

> ✅ **Random Forest** is the best model — used for all final predictions and the 7-day forecast.

---

## 📊 Input Features (15 Total)

### Price Indicators
- `GLD` — Target: Daily gold closing price (USD/oz)
- `SLV` — Silver ETF price (strongest predictor, r=0.95)
- `USO` — Crude oil price per barrel

### Macro Indicators
- `USD_INR` — US Dollar to Indian Rupee exchange rate
- `DXY` — US Dollar Index strength
- `SPX` — S&P 500 stock market benchmark
- `CPI` — Consumer Price Index (inflation)
- `Rate` — US Federal Funds Rate
- `EUR_USD` — Euro/Dollar exchange rate
- `VIX` — Market fear/volatility index

### Engineered Features
- `GLD_Lag1` — Yesterday's gold price
- `GLD_Lag7` — Gold price 7 days ago
- `GLD_MA20` — 20-day moving average
- `GLD_MA50` — 50-day moving average
- `GLD_Vol20` — 20-day price volatility

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|-----------|
| Language | Python 3.10+ |
| ML Library | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web App | Streamlit |
| Data Source | Yahoo Finance (yfinance), Kaggle |
| Model Saving | Joblib |
| Version Control | Git & GitHub |

---

## 📁 Project Structure
```
gold-price-prediction/
│
├── app.py                          # Streamlit web application
├── gold_model.py                   # Full ML pipeline (training + evaluation)
├── requirements.txt                # All Python dependencies
├── README.md                       # This file
│
├── plots/                          # Generated visualizations
│   ├── plot_actual_vs_predicted.png
│   ├── plot_r2_comparison.png
│   ├── plot_feature_importance.png
│   ├── plot_residuals.png
│   └── plot_correlation_heatmap.png
│
└── models/                         # Saved trained models
    ├── best_model_random_forest.pkl
    └── scaler.pkl
```

---

## 🚀 How to Run

### Step 1 — Clone the Repository
```bash
git clone https://github.com/CherukuSwathi/gold-price-prediction.git
cd gold-price-prediction
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the ML Pipeline (trains models + saves plots)
```bash
python gold_model.py
```

### Step 4 — Launch the Streamlit Web App
```bash
streamlit run app.py
```
Then open your browser at **http://localhost:8501**

---

## 📈 Key Results

- **Best Model:** Random Forest with R² = **0.968**
- **Prediction Error:** Only **$33.80** average error per prediction
- **7-Day Forecast:** Predicts gold from $3,118 → $3,154 (Mar 2026)
- **Top Predictors:** Silver Price (31%) and USD/INR Rate (24%) drive over half the variance

---

## 📷 Screenshots

> After running `python gold_model.py`, the following plots are generated:
> - Actual vs Predicted price comparison
> - Model accuracy (R²) bar chart
> - Feature importance bar chart
> - Residual error plot
> - Pearson correlation heatmap

---

## 📚 References
- [Yahoo Finance API (yfinance)](https://pypi.org/project/yfinance/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Kaggle Gold Dataset](https://www.kaggle.com/datasets/altruistdelhite04/gold-price-prediction)

---

## 📜 License
This project is created for academic purposes at **Pragathi Degree Womens College**.
Free to use for educational reference.

---

*Made with ❤️ by the Gold Prediction Team — Pragathi Degree Womens College, 2025–2026*
