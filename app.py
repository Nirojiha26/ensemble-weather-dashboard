"""
Ensemble Learning Dashboard: next-day temperature for Sri Lankan districts.
Run with:  streamlit run app.py
Needs the files created by the notebook in ./models and ensemble_utils.py next to this file.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ensemble_utils import DISTRICTS, make_input_row  # also needed so joblib can unpickle ResidualHybrid

MODELS_DIR = "models"
FAMILY_COLORS = {"Bagging": "#2a9d8f", "Boosting": "#e76f51", "Stacking": "#6a4c93",
                 "Hybrid": "#f4a261", "Baseline": "#9a9a9a", "Time series": "#577590"}

st.set_page_config(page_title="Sri Lanka Temperature: Ensemble Dashboard", page_icon="🌡️", layout="wide")

if not os.path.exists(os.path.join(MODELS_DIR, "meta.json")):
    st.error("The `models/` folder was not found. Run the notebook first (Run All), then start the dashboard "
             "from the same folder with `streamlit run app.py`.")
    st.stop()


@st.cache_data
def load_tables():
    results = pd.read_csv(f"{MODELS_DIR}/results.csv")
    latest = pd.read_csv(f"{MODELS_DIR}/latest_by_district.csv", parse_dates=["date"])
    test_pred = pd.read_csv(f"{MODELS_DIR}/test_predictions.csv", parse_dates=["date"])
    importances = pd.read_csv(f"{MODELS_DIR}/feature_importances.csv", index_col=0)
    meta = json.load(open(f"{MODELS_DIR}/meta.json"))
    return results, latest, test_pred, importances, meta


@st.cache_resource
def load_model(filename):
    return joblib.load(os.path.join(MODELS_DIR, filename))


results, latest, test_pred, importances, meta = load_tables()
name_map = meta["name_map"]
family_of = dict(zip(results["Model"], results["Family"]))
rmse_of = dict(zip(results["Model"], results["RMSE"]))
persistence_rmse = float(results.loc[results["Model"] == "Baseline: Persistence", "RMSE"].iloc[0])
live_models = [m for m in results["Model"] if m in name_map]          # models that can predict live
best_live = min(live_models, key=lambda m: rmse_of[m])
best_overall = results.sort_values("RMSE").iloc[0]

# ------------------------------------------------------------------ sidebar
INPUTS = [
    ("temperature_2m_max", "Today's max temperature (°C)"),
    ("temperature_2m_min", "Today's min temperature (°C)"),
    ("temperature_2m_mean", "Today's mean temperature (°C)"),
    ("precipitation_sum", "Today's rainfall (mm)"),
    ("wind_speed_10m_max", "Max wind speed (km/h)"),
    ("wind_gusts_10m_max", "Max wind gusts (km/h)"),
    ("shortwave_radiation_sum", "Solar radiation (MJ/m²)"),
    ("et0_fao_evapotranspiration", "Evapotranspiration (mm)"),
]
HISTORY = [
    ("temp_lag1", "Mean temp 1 day ago (°C)"),
    ("temp_lag2", "Mean temp 2 days ago (°C)"),
    ("temp_lag3", "Mean temp 3 days ago (°C)"),
    ("temp_roll7", "7-day average temp (°C)"),
    ("rain_roll7", "7-day total rainfall (mm)"),
]

st.sidebar.title("🌡️ Inputs")
district = st.sidebar.selectbox("District", DISTRICTS, index=0)
row = latest[latest["district"] == district].iloc[0]
st.sidebar.caption(f"Pre-filled with the last observed day in the data ({row['date'].date()}). Change any value.")

if st.sidebar.button("Reset to latest observed values"):
    for col, _ in INPUTS + HISTORY:
        st.session_state.pop(f"{district}_{col}", None)
    st.session_state.pop(f"date_{district}", None)
    st.rerun()

base_date = st.sidebar.date_input("Today's date (sets the season)", value=row["date"].date(), key=f"date_{district}")

values = {}
st.sidebar.subheader("Today's weather")
for col, label in INPUTS:
    values[col] = st.sidebar.number_input(label, value=float(round(row[col], 2)), step=0.1, format="%.2f",
                                          key=f"{district}_{col}")
with st.sidebar.expander("Recent history", expanded=True):
    for col, label in HISTORY:
        values[col] = st.number_input(label, value=float(round(row[col], 2)), step=0.1, format="%.2f",
                                      key=f"{district}_{col}")

st.sidebar.subheader("Model")
default_idx = live_models.index(best_live)
chosen = st.sidebar.selectbox("Primary model", live_models, index=default_idx,
                              help="Every saved model is also shown side by side in the Prediction tab.")

# ------------------------------------------------------------------ header
st.title("Ensemble Learning Dashboard: next-day temperature in Sri Lanka")
st.caption("Bagging, boosting, stacking and hybrid models compared on daily weather for six Sri Lankan districts "
           "(2015-2025, Open-Meteo/ERA5 reanalysis). Test period: 2024-2025.")

c1, c2, c3 = st.columns(3)
c1.metric("Best model overall (test RMSE)", best_overall["Model"], f"{best_overall['RMSE']:.3f} °C", delta_color="off")
c2.metric("Persistence baseline RMSE", f"{persistence_rmse:.3f} °C")
gain = 100 * (persistence_rmse - best_overall["RMSE"]) / persistence_rmse
c3.metric("Best model's gain over persistence", f"{gain:.1f} %")

tab_pred, tab_cmp, tab_imp, tab_fit, tab_about = st.tabs(
    ["🔮 Prediction", "📊 Model comparison", "🧭 Feature importance", "📈 Actual vs predicted", "ℹ️ About"])

# ------------------------------------------------------------------ prediction
with tab_pred:
    X = make_input_row(values, district, base_date)
    problems = []
    if values["temperature_2m_min"] > values["temperature_2m_max"]:
        problems.append("Min temperature is higher than max temperature.")
    if not (values["temperature_2m_min"] - 0.01 <= values["temperature_2m_mean"] <= values["temperature_2m_max"] + 0.01):
        problems.append("Mean temperature should lie between min and max.")
    for p in problems:
        st.warning(p)

    preds = {}
    for m in live_models:
        preds[m] = float(load_model(name_map[m]).predict(X)[0])

    today_mean = values["temperature_2m_mean"]
    chosen_pred = preds[chosen]
    a, b, c = st.columns(3)
    a.metric(f"{chosen}: tomorrow's mean temperature in {district}", f"{chosen_pred:.2f} °C",
             f"{chosen_pred - today_mean:+.2f} °C vs today")
    all_vals = np.array(list(preds.values()))
    b.metric("All-model average", f"{all_vals.mean():.2f} °C")
    spread = float(all_vals.max() - all_vals.min())
    level = "High" if spread < 0.3 else ("Medium" if spread < 0.8 else "Low")
    c.metric("Model agreement", level, f"spread {spread:.2f} °C", delta_color="off")
    st.caption(f"Typical test error of the chosen model: about ±{rmse_of[chosen]:.2f} °C (RMSE). "
               "Agreement = how far apart the highest and lowest model predictions are for this input.")

    pdf = pd.DataFrame({"Model": list(preds), "Prediction": list(preds.values())})
    pdf["Family"] = pdf["Model"].map(family_of)
    pdf = pdf.sort_values("Prediction")
    fig = px.bar(pdf, x="Prediction", y="Model", color="Family", orientation="h",
                 color_discrete_map=FAMILY_COLORS, text=pdf["Prediction"].round(2))
    lo = float(pdf["Prediction"].min()) - 0.5
    hi = float(pdf["Prediction"].max()) + 0.5
    fig.update_xaxes(range=[lo, hi], title="Predicted next-day mean temperature (°C)")
    fig.add_vline(x=today_mean, line_dash="dash", line_color="grey", annotation_text="today")
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0), yaxis_title=None)
    st.plotly_chart(fig)
    st.info("The Hybrid RF + ARMA model is evaluated in the notebook but cannot be run here: it needs the district's "
            "full daily history to compute the ARMA forecast.")

# ------------------------------------------------------------------ comparison
with tab_cmp:
    metric = st.radio("Metric", ["RMSE", "MAE", "R2"], horizontal=True)
    cmp = results.copy()
    cmp["RMSE gain vs persistence (%)"] = (100 * (persistence_rmse - cmp["RMSE"]) / persistence_rmse).round(2)
    asc = metric != "R2"
    cmp_sorted = cmp.sort_values(metric, ascending=asc)
    fig2 = px.bar(cmp_sorted.iloc[::-1], x=metric, y="Model", color="Family", orientation="h",
                  color_discrete_map=FAMILY_COLORS, text=cmp_sorted.iloc[::-1][metric].round(4))
    vals = cmp[metric]
    pad = (vals.max() - vals.min()) * 0.15 + 1e-4
    fig2.update_xaxes(range=[vals.min() - pad, vals.max() + pad])
    fig2.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0), yaxis_title=None)
    st.plotly_chart(fig2)
    st.caption("The axis is zoomed on purpose: the models differ by only a few hundredths of a °C, "
               "so look at the numbers as well as the bar lengths.")
    st.subheader("Best result per family")
    fam = cmp.sort_values("RMSE").groupby("Family").first()[["Model", "RMSE", "MAE", "R2"]].sort_values("RMSE")
    st.dataframe(fam)
    st.subheader("Full results table")
    st.dataframe(cmp.sort_values("RMSE").reset_index(drop=True))

# ------------------------------------------------------------------ importance
with tab_imp:
    imp_model = st.selectbox("Model", list(importances.columns),
                             index=list(importances.columns).index("Random Forest") if "Random Forest" in importances.columns else 0)
    topn = st.slider("Number of features", 5, len(importances), 12)
    top = importances[imp_model].sort_values(ascending=False).head(topn).iloc[::-1]
    fig3 = px.bar(x=top.values, y=top.index, orientation="h", labels={"x": "Relative importance", "y": ""},
                  color_discrete_sequence=["#2a9d8f"])
    fig3.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig3)
    st.caption("Impurity-based importance of tree models. Highly correlated features (today's mean/max/min and the "
               "lags) share importance between them, so read this as a rough guide.")

# ------------------------------------------------------------------ actual vs predicted
with tab_fit:
    d2 = st.selectbox("District", DISTRICTS, index=DISTRICTS.index(district), key="fit_district")
    pred_cols = [c for c in test_pred.columns if c not in ("date", "district", "actual")]
    default_models = [m for m in [best_overall["Model"], "Random Forest", "XGBoost"] if m in pred_cols]
    show = st.multiselect("Models", pred_cols, default=default_models)
    days = st.slider("Show the last N days of the test period", 30, 730, 180)
    sub = test_pred[test_pred["district"] == d2].sort_values("date").tail(days)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=sub["date"], y=sub["actual"], name="Actual", line=dict(color="black", width=2)))
    for m in show:
        fig4.add_trace(go.Scatter(x=sub["date"], y=sub[m], name=m, line=dict(width=1.4), opacity=0.85))
    fig4.update_layout(height=450, yaxis_title="Mean temperature (°C)", margin=dict(l=0, r=0, t=30, b=0),
                       legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig4)
    if show:
        err = pd.DataFrame({"Model": show,
                            "RMSE in this window (°C)": [float(np.sqrt(np.mean((sub[m] - sub["actual"]) ** 2))) for m in show]})
        st.dataframe(err.round(3))

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown("""
**Task:** predict the next day's mean temperature (°C) for six Sri Lankan districts
(Colombo, Kandy, Galle, Jaffna, Anuradhapura, Nuwara Eliya).

**Data:** daily weather 2015-2025 from the Open-Meteo historical API (ERA5 reanalysis) at each district's coordinates.
These are reanalysis values, not raw weather-station readings.

**Models**
- **Bagging:** Random Forest, Extra Trees (many deep trees on bootstrap samples, averaged, which lowers variance).
- **Boosting:** AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost (trees added one after another to fix earlier errors, which lowers bias).
- **Stacking:** Random Forest + XGBoost + ANN combined by a Ridge meta-learner.
- **Hybrids:** RF + ANN (RF learns the ANN's residuals) and RF + ARMA (RF learns the ARMA residuals).
- **Baselines:** persistence ("tomorrow = today") and Ridge regression.

**Evaluation:** chronological split (train 2015-2023, test 2024-2025); metrics RMSE, MAE, R².

**Limitations:** one-day-ahead only; reanalysis data; hyper-parameters set by hand.
""")
