"""Faithful re-run of the modelling pipeline used in the 2023 dissertation
(original notebook: final thesis code, Research Question 4).

This is kept *only* to show the dissertation's reported numbers are
reproducible. It contains a methodological flaw that is documented, not
hidden (see README, "Corrections to the original analysis"):

* the target ``Job_Loss_Probability`` is ``standardised(TOT_EMP) * Probability``
  (a z-score times a probability, so it is not a probability and can be
  negative), and
* both of its ingredients (``TOT_EMP`` and ``Probability``) are also used as
  input features, so the models largely re-learn a known formula (target
  leakage). The R-squared of 0.82 therefore does not measure the ability to
  predict automation risk.

The corrected modelling lives in :mod:`automation_risk.models`.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import RANDOM_STATE
from .data import merge_raw

CATEGORICAL = ["OCC_CODE", "Occupation", "OCC_TITLE", "OCC_GROUP", "H_PCT10",
               "H_PCT25", "H_MEDIAN", "H_PCT75", "H_PCT90", "A_PCT10", "A_PCT25",
               "A_MEDIAN", "A_PCT75", "A_PCT90"]
NUMERIC = ["TOT_EMP", "EMP_PRSE", "H_MEAN", "A_MEAN", "MEAN_PRSE"]


def original_pipeline_frame(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Reproduce the original preprocessing step by step."""
    raw = merge_raw() if raw is None else raw
    # The original saved the merge to new_data_set.csv and read it back, which
    # determines the column dtypes; do the same in memory.
    buf = io.StringIO()
    raw.to_csv(buf, index=False)
    buf.seek(0)
    df = pd.read_csv(buf)
    df = df.drop(columns=["ANNUAL", "HOURLY"])
    df["H_MEAN"] = pd.to_numeric(df["H_MEAN"].astype(str).str.replace(",", ""), errors="coerce")
    df["A_MEAN"] = pd.to_numeric(df["A_MEAN"].astype(str).str.replace(",", ""), errors="coerce")
    df["TOT_EMP"] = df["TOT_EMP"].apply(lambda x: int(x.replace(",", "")) if isinstance(x, str) else x)
    df[NUMERIC] = SimpleImputer(strategy="mean").fit_transform(df[NUMERIC])
    df[NUMERIC] = StandardScaler().fit_transform(df[NUMERIC])
    pcs = PCA(n_components=2).fit_transform(df[NUMERIC])
    df["PCA1"], df["PCA2"] = pcs[:, 0], pcs[:, 1]
    # Original target definition (note: TOT_EMP is standardised at this point)
    df["Job_Loss_Probability"] = df["TOT_EMP"] * df["Probability"]
    for c in CATEGORICAL:
        df[c] = LabelEncoder().fit_transform(df[c].to_numpy())
    return df


def run_original_models(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    df = original_pipeline_frame(raw)
    X = df.drop(columns=["Job_Loss_Probability"])
    y = df["Job_Loss_Probability"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        cv = np.sqrt(-cross_val_score(model, X_tr, y_tr, scoring="neg_mean_squared_error", cv=5))
        rows.append({
            "model": name,
            "test_MSE": mean_squared_error(y_te, pred),
            "test_MAE": mean_absolute_error(y_te, pred),
            "test_RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
            "test_R2": r2_score(y_te, pred),
            "cv5_RMSE_mean": cv.mean(),
            "cv5_RMSE_std": cv.std(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    print(run_original_models().round(4).to_string(index=False))
