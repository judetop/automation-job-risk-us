"""Descriptive analysis for research questions 1-3 and the healthcare section.

Employment for an occupation is the sum of its 51 state columns in the
automation dataset. The Kaggle dataset records occupations with unavailable
data or fewer than 10 workers in a state as 0, and a few occupations (e.g.
Registered Nurses, Physicians and Surgeons) carry 0 employment in every state
because of SOC code mismatches, so employment-weighted figures exclude them.
"""
from __future__ import annotations

import pandas as pd

from . import HIGH_RISK_THRESHOLD
from .data import STATES, add_major_group, load_automation


def automation_with_employment(auto: pd.DataFrame | None = None) -> pd.DataFrame:
    auto = load_automation() if auto is None else auto
    df = add_major_group(auto)
    df["EMPLOYMENT"] = df[STATES].sum(axis=1)
    df["EXPOSED_EMPLOYMENT"] = df["EMPLOYMENT"] * df["Probability"]
    return df


def _risk_summary(g: pd.DataFrame) -> pd.Series:
    emp = g["EMPLOYMENT"].sum()
    return pd.Series({
        "occupations": len(g),
        "employment": emp,
        "mean_probability_unweighted": g["Probability"].mean(),
        "mean_probability_emp_weighted": (g["Probability"] * g["EMPLOYMENT"]).sum() / emp if emp else float("nan"),
        "share_emp_high_risk": g.loc[g["Probability"] > HIGH_RISK_THRESHOLD, "EMPLOYMENT"].sum() / emp if emp else float("nan"),
    })


def major_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("SOC_MAJOR_NAME")[["Probability", "EMPLOYMENT"]].apply(_risk_summary)
    out["occupations"] = out["occupations"].astype(int)
    out["employment"] = out["employment"].astype(int)
    return out.sort_values("mean_probability_emp_weighted", ascending=False)


def state_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Employment-weighted mean automation probability for each state:
    sum_o(emp[state, o] * p[o]) / sum_o(emp[state, o])."""
    p = df["Probability"]
    rows = []
    for s in STATES:
        emp = df[s]
        rows.append({
            "state": s,
            "employment_covered": int(emp.sum()),
            "emp_weighted_probability": (emp * p).sum() / emp.sum(),
            "share_emp_high_risk": emp[p > HIGH_RISK_THRESHOLD].sum() / emp.sum(),
            "workers_in_high_risk_occupations": int(emp[p > HIGH_RISK_THRESHOLD].sum()),
        })
    return pd.DataFrame(rows).sort_values("emp_weighted_probability", ascending=False).reset_index(drop=True)


def original_state_metric(merged: pd.DataFrame) -> pd.Series:
    """What the dissertation's state chart actually plotted: the mean of each
    state's employment column across occupations (a head-count, not a
    probability)."""
    return merged[STATES].mean().sort_values(ascending=False)


HEALTHCARE_EXTRA = {"39-9021": "Personal Care Aides", "21-1093": "Social and Human Service Assistants"}


def healthcare_table(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["SOC_MAJOR"].isin(["29", "31"]) | df["OCC_CODE"].isin(HEALTHCARE_EXTRA)
    cols = ["OCC_CODE", "Occupation", "SOC_MAJOR_NAME", "Probability", "EMPLOYMENT"]
    return df.loc[mask, cols].sort_values("Probability", ascending=False).reset_index(drop=True)
