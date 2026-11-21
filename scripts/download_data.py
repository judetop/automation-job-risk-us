"""Download the source data into data/.

Source: Kaggle dataset "Occupation, Salary and Likelihood of Automation"
(andrewmvd/occupation-salary-and-likelihood-of-automation). The files are not
redistributed in this repository because the dataset's licence is not
specified at the source.

If the direct download fails (Kaggle may require you to be signed in), download
the zip manually from
https://www.kaggle.com/datasets/andrewmvd/occupation-salary-and-likelihood-of-automation
and extract it into data/.
"""
from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/occupation-salary-and-likelihood-of-automation"
DATA = Path(__file__).resolve().parents[1] / "data"
SHA256 = {
    "automation_data_by_state.csv": "19beebd269adee09ababda105951045188529d73a8da0c96a17bf3cc466224a5",
    "occupation_salary.xlsx": "6b5fccf8713c56dc9d93176906c73598511250f0eca93a11bfc38614a315ab3a",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    DATA.mkdir(exist_ok=True)
    if not all((DATA / f).exists() for f in SHA256):
        print(f"Downloading {URL} ...")
        req = urllib.request.Request(URL, headers={"User-Agent": "automation-job-risk-us"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            zipfile.ZipFile(io.BytesIO(resp.read())).extractall(DATA)
    ok = True
    for name, expected in SHA256.items():
        path = DATA / name
        if not path.exists():
            print(f"MISSING  {name}"); ok = False
        elif sha256(path) != expected:
            print(f"CHANGED  {name} (checksum differs from the version used in this analysis)")
        else:
            print(f"OK       {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
