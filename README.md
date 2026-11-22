# Ensemble Learning Dashboard: Next-Day Temperature in Sri Lanka 🇱🇰🌡️

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sri-lanka-weather-ensemble.streamlit.app/)

An interactive machine learning dashboard demonstrating **Ensemble Learning** techniques to forecast next-day mean temperature across six major Sri Lankan districts (**Colombo, Kandy, Galle, Jaffna, Anuradhapura, and Nuwara Eliya**).

🔗 **Live Dashboard:** [sri-lanka-weather-ensemble.streamlit.app](https://sri-lanka-weather-ensemble.streamlit.app/)

---

## 📌 Project Overview

Weather patterns across Sri Lanka exhibit significant microclimatic variation—from the tropical coastal climate of Colombo and Galle to the dry northern plains of Jaffna and the cold central highlands of Nuwara Eliya. 

This project explores whether combining multiple diverse learning algorithms via **Ensemble Learning** outperforms individual models and baseline persistence forecasts.

---

## 🔬 Ensemble Families & Models Evaluated

| Ensemble Paradigm | Algorithms Used | Key Concept |
|---|---|---|
| **Bagging** | Random Forest, Extra Trees | Reduces variance by aggregating decorrelated trees trained on bootstrap samples |
| **Boosting** | XGBoost, LightGBM, CatBoost, Gradient Boosting, AdaBoost | Reduces bias by sequentially fitting learners to the residuals of previous trees |
| **Stacking** | Random Forest + XGBoost + ANN (MLP) &rarr; Ridge Meta-Learner | Combines diverse base model predictions using a meta-regressor |
| **Hybrid (Tabular)** | Residual Hybrid: RF + ANN | ANN captures global non-linear representations; Random Forest learns residual errors |
| **Hybrid (Time Series)** | Residual Hybrid: RF + ARMA(2,2) | ARMA models temporal autocorrelation; Random Forest learns remaining weather features |
| **Baselines** | Persistence baseline ("tomorrow = today"), Ridge Regression | Reference benchmarks |

---

## 📊 Key Results

* **Best Model Overall:** **Hybrid RF + ARMA** (Test RMSE: **0.541 °C**, $R^2 = 0.984$)
* **Best Tabular Stacking:** **RF + XGB + ANN &rarr; Ridge** (Test RMSE: **0.543 °C**, $R^2 = 0.984$)
* **Gain over Persistence Baseline:** **7.7% reduction in RMSE** compared to standard naive forecasting (0.586 °C).

---

## 📁 Repository Structure

```text
├── app.py                             # Streamlit dashboard application
├── ensemble_utils.py                  # Shared feature engineering & ResidualHybrid class
├── Ensemble_Learning_Dashboard.ipynb  # End-to-end Python notebook (EDA, training, evaluation)
├── requirements.txt                   # Deployment dependencies
├── sri_lanka_weather.csv              # Full dataset (2015–2025)
├── raw_data/                          # District-wise raw CSVs
└── models/                            # Saved trained models, meta data & results
```

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nirojiha26/ensemble-weather-dashboard.git
   cd ensemble-weather-dashboard
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the dashboard:**
   ```bash
   streamlit run app.py
   ```
