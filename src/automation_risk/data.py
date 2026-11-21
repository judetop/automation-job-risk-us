"""Loading, merging and cleaning the two source datasets.

Expected files in ``data/`` (see data/README.md for download instructions):

* ``automation_data_by_state.csv``: Frey & Osborne automation probability per
  occupation plus employment counts per US state (Kaggle, "Occupation, Salary
  and Likelihood of Automation").
* ``occupation_salary.xlsx`` (as downloaded from the same Kaggle dataset) or
  ``occupation_salary.csv`` (the same sheet saved as CSV). National BLS
  Occupational Employment Statistics (OES) wage and employment estimates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

AUTOMATION_FILE = "automation_data_by_state.csv"
SALARY_XLSX = "occupation_salary.xlsx"
SALARY_CSV = "occupation_salary.csv"

# 50 states + District of Columbia, as columns in the automation file
STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
    "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
]

# 2010 Standard Occupational Classification (SOC) major groups
SOC_MAJOR_GROUPS = {
    "11": "Management",
    "13": "Business & Financial Operations",
    "15": "Computer & Mathematical",
    "17": "Architecture & Engineering",
    "19": "Life, Physical & Social Science",
    "21": "Community & Social Service",
    "23": "Legal",
    "25": "Education, Training & Library",
    "27": "Arts, Design, Entertainment, Sports & Media",
    "29": "Healthcare Practitioners & Technical",
    "31": "Healthcare Support",
    "33": "Protective Service",
    "35": "Food Preparation & Serving",
    "37": "Building & Grounds Cleaning & Maintenance",
    "39": "Personal Care & Service",
    "41": "Sales & Related",
    "43": "Office & Administrative Support",
    "45": "Farming, Fishing & Forestry",
    "47": "Construction & Extraction",
    "49": "Installation, Maintenance & Repair",
    "51": "Production",
    "53": "Transportation & Material Moving",
}

WAGE_COLS_ANNUAL = ["A_MEAN", "A_PCT10", "A_PCT25", "A_MEDIAN", "A_PCT75", "A_PCT90"]
WAGE_COLS_HOURLY = ["H_MEAN", "H_PCT10", "H_PCT25", "H_MEDIAN", "H_PCT75", "H_PCT90"]
# BLS OES footnotes: '#' = wage at or above the top-code ($100.00/hr or $208,000/yr
# in the 2016 release); '*' = estimate not available.
TOP_CODE = {"annual": 208_000.0, "hourly": 100.0}


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the source data first: "
            "`python scripts/download_data.py` (see data/README.md)."
        )
    return path


def load_automation(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Automation probability + state employment, with SOC renamed to OCC_CODE."""
    df = pd.read_csv(_require(data_dir / AUTOMATION_FILE), encoding="ISO-8859-1")
    return df.rename(columns={"SOC": "OCC_CODE"})


def _excel_csv_format(df: pd.DataFrame) -> pd.DataFrame:
    """Render numeric cells the way Excel's 'Save as CSV' did for the original
    analysis (thousands separators for counts/annual wages, 2 d.p. for hourly
    wages). Only used so that the faithful replication of the original
    notebook sees exactly the same string values it saw in 2023."""
    out = df.copy()

    def fmt(v, kind):
        if isinstance(v, str) or (isinstance(v, float) and np.isnan(v)):
            return v
        if isinstance(v, bool):
            return np.nan
        return f"{v:,.0f}" if kind == "int" else f"{v:,.2f}"

    for c in ["TOT_EMP"] + WAGE_COLS_ANNUAL:
        out[c] = out[c].map(lambda v: fmt(v, "int"))
    for c in WAGE_COLS_HOURLY:
        out[c] = out[c].map(lambda v: fmt(v, "float"))
    for c in ["ANNUAL", "HOURLY"]:
        out[c] = out[c].map(lambda v: "TRUE" if v is True else np.nan)
    return out


def load_salary_raw(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """BLS OES national table as text (thousands separators, '#', '*' kept)."""
    csv_path, xlsx_path = data_dir / SALARY_CSV, data_dir / SALARY_XLSX
    if csv_path.exists():
        return pd.read_csv(csv_path, encoding="ISO-8859-1")
    return _excel_csv_format(pd.read_excel(_require(xlsx_path)))


def merge_raw(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Inner-join automation and salary tables on OCC_CODE (original step)."""
    return pd.merge(load_automation(data_dir), load_salary_raw(data_dir), on="OCC_CODE")


def to_number(series: pd.Series, top_code: float | None = None) -> pd.Series:
    """Parse BLS text values: strip commas, '#' -> top-code, '*' -> NaN."""
    s = series.astype("string").str.replace(",", "", regex=False).str.strip()
    if top_code is not None:
        s = s.mask(s == "#", str(top_code))
    return pd.to_numeric(s, errors="coerce")


def clean_merged(raw: pd.DataFrame) -> pd.DataFrame:
    """Typed, analysis-ready version of the merged table."""
    df = raw.drop(columns=["ANNUAL", "HOURLY"], errors="ignore").copy()
    df["TOT_EMP"] = to_number(df["TOT_EMP"])
    for c in WAGE_COLS_ANNUAL:
        df[c] = to_number(df[c], TOP_CODE["annual"])
    for c in WAGE_COLS_HOURLY:
        df[c] = to_number(df[c], TOP_CODE["hourly"])
    df["SOC_MAJOR"] = df["OCC_CODE"].str[:2]
    df["SOC_MAJOR_NAME"] = df["SOC_MAJOR"].map(SOC_MAJOR_GROUPS)
    return df


def add_major_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SOC_MAJOR"] = df["OCC_CODE"].str[:2]
    df["SOC_MAJOR_NAME"] = df["SOC_MAJOR"].map(SOC_MAJOR_GROUPS)
    return df
