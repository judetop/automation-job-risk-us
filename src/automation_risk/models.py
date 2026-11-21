"""Corrected modelling for research question 4.

Target: the occupation's automation probability (``Probability``, 0-1).
Features: labour-market characteristics from the BLS OES table that do not
contain the target: employment (log), annual wage distribution (log), the
relative standard errors of the employment and wage estimates, and, in the
main feature set, the occupation's SOC major group.

Same model families, split (80/20) and seed (42) as the dissertation;
preprocessing is fitted on the training data only (inside a Pipeline).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import RANDOM_STATE
from .data import WAGE_COLS_ANNUAL

LOG_COLS = ["TOT_EMP"] + WAGE_COLS_ANNUAL
PLAIN_COLS = ["EMP_PRSE", "MEAN_PRSE"]
CAT_COLS = ["SOC_MAJOR_NAME"]

FEATURE_SETS = {
    "labour market + SOC major group": (LOG_COLS + PLAIN_COLS, CAT_COLS),
    "labour market only": (LOG_COLS + PLAIN_COLS, []),
}


def make_preprocessor(num_log, num_plain, cat):
    transformers = [
        ("log", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("scale", StandardScaler())]), num_log),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_plain),
    ]
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def model_zoo():
    return {
        "Baseline (predict mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def split(df: pd.DataFrame):
    X = df.drop(columns=["Probability"])
    y = df["Probability"]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def evaluate(df: pd.DataFrame, feature_set: str = "labour market + SOC major group"):
    """Fit every model; return (metrics table, fitted pipelines, test data)."""
    num_cols, cat_cols = FEATURE_SETS[feature_set]
    X_tr, X_te, y_tr, y_te = split(df)
    num_log = [c for c in num_cols if c in LOG_COLS]
    num_plain = [c for c in num_cols if c in PLAIN_COLS]
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, est in model_zoo().items():
        pipe = Pipeline([("prep", make_preprocessor(num_log, num_plain, cat_cols)), ("model", est)])
        cv_rmse = np.sqrt(-cross_val_score(pipe, X_tr, y_tr, scoring="neg_mean_squared_error", cv=cv))
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        mse = mean_squared_error(y_te, pred)
        rows.append({
            "feature_set": feature_set, "model": name,
            "test_MSE": mse, "test_MAE": mean_absolute_error(y_te, pred),
            "test_RMSE": float(np.sqrt(mse)), "test_R2": r2_score(y_te, pred),
            "cv5_RMSE_mean": cv_rmse.mean(), "cv5_RMSE_std": cv_rmse.std(),
        })
        fitted[name] = pipe
    return pd.DataFrame(rows), fitted, (X_te, y_te)
