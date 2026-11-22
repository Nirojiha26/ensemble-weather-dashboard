"""
Shared helpers for the Sri Lankan weather ensemble project.
Used by BOTH the notebook and the Streamlit dashboard so that features
are built identically in training and in real-time inference.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

DISTRICTS = ["Colombo", "Kandy", "Galle", "Jaffna", "Anuradhapura", "Nuwara Eliya"]

RAW_NUM = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
]
ENGINEERED = ["temp_lag1", "temp_lag2", "temp_lag3", "temp_roll7", "rain_roll7",
              "doy_sin", "doy_cos"]
DISTRICT_COLS = [f"district_{d}" for d in DISTRICTS]
FEATURE_COLS = RAW_NUM + ENGINEERED + DISTRICT_COLS
TARGET = "target_temp_next_day"


def add_features(df):
    """df: cleaned daily data with columns date, district + RAW_NUM.
    Returns a frame with engineered features and the next-day target
    (target is NaN on the last day of every district)."""
    df = df.sort_values(["district", "date"]).reset_index(drop=True).copy()
    g = df.groupby("district", group_keys=False)
    t = df["temperature_2m_mean"]
    df["temp_lag1"] = g["temperature_2m_mean"].shift(1)
    df["temp_lag2"] = g["temperature_2m_mean"].shift(2)
    df["temp_lag3"] = g["temperature_2m_mean"].shift(3)
    df["temp_roll7"] = g["temperature_2m_mean"].transform(lambda s: s.rolling(7).mean())
    df["rain_roll7"] = g["precipitation_sum"].transform(lambda s: s.rolling(7).sum())
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for d in DISTRICTS:
        df[f"district_{d}"] = (df["district"] == d).astype(int)
    df[TARGET] = g["temperature_2m_mean"].shift(-1)
    return df


def make_input_row(values: dict, district: str, date) -> pd.DataFrame:
    """Build ONE model-ready row from user-entered values (used by the dashboard)."""
    row = {c: float(values[c]) for c in RAW_NUM + ["temp_lag1", "temp_lag2", "temp_lag3",
                                                     "temp_roll7", "rain_roll7"]}
    doy = pd.Timestamp(date).dayofyear
    row["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    row["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for d in DISTRICTS:
        row[f"district_{d}"] = int(d == district)
    return pd.DataFrame([row])[FEATURE_COLS]


class ResidualHybrid(BaseEstimator, RegressorMixin):
    """Hybrid: a first model (e.g. ANN) captures the main pattern, then a second
    model (e.g. Random Forest) is trained on the first model's residuals.
    Prediction = base prediction + residual prediction."""

    def __init__(self, base_model=None, residual_model=None):
        self.base_model = base_model
        self.residual_model = residual_model

    def fit(self, X, y):
        self.base_ = clone(self.base_model).fit(X, y)
        resid = np.asarray(y) - self.base_.predict(X)
        self.resid_ = clone(self.residual_model).fit(X, resid)
        return self

    def predict(self, X):
        return self.base_.predict(X) + self.resid_.predict(X)
