# Data

The raw data files are **not committed** to this repository. The Kaggle dataset they come from states "License was not specified at the source", so redistribution rights are unclear. Download them yourself:

```bash
python scripts/download_data.py
```

This fetches the public Kaggle dataset **[Occupation, Salary and Likelihood of Automation](https://www.kaggle.com/datasets/andrewmvd/occupation-salary-and-likelihood-of-automation)** (uploaded by Larxel / `andrewmvd`, version 1, May 2020) and checks the files against the SHA-256 checksums of the versions used in this analysis. If the scripted download fails, download the zip from the Kaggle page (you may need to sign in) and extract it here.

Expected files:

| File | Rows | Description |
|---|---|---|
| `automation_data_by_state.csv` | 702 | `SOC` code, `Occupation`, `Probability` of computerisation (Frey & Osborne), and employment counts for the 50 states + DC. Per the dataset description, state cells with unavailable data or fewer than 10 workers are recorded as 0. |
| `occupation_salary.xlsx` | 1,394 | US national occupational employment and wage estimates (`OCC_CODE`, `OCC_TITLE`, `OCC_GROUP`, `TOT_EMP`, hourly/annual mean and percentile wages, relative standard errors). Values match the BLS Occupational Employment Statistics national release for May 2016 (e.g. all-occupations employment of 140,400,040). `#` = wage at or above the top-code; `*` = not available. |

`occupation_salary.csv` (the same sheet saved from Excel as CSV, which is what the original 2023 notebook used) is also accepted. The loader prefers the CSV if both are present, and converting the xlsx in code reproduces the original merged dataset exactly.

## Original sources

* **Automation probabilities:** Frey, C. B. and Osborne, M. A. (2017). The future of employment: How susceptible are jobs to computerisation? *Technological Forecasting and Social Change*, 114, 254–280. https://doi.org/10.1016/j.techfore.2016.08.019
* **Employment and wages:** U.S. Bureau of Labor Statistics, Occupational Employment Statistics (now OEWS), May 2016 national estimates: https://www.bls.gov/oes/2016/may/oes_nat.htm. BLS data are U.S. government works in the public domain.
