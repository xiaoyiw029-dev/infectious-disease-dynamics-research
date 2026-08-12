"""
Merge weekly cases, climate covariates, and population into one model dataset.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "02_数据" / "01_清洗数据"


def main() -> None:
    cases_path = CLEAN_DIR / "brazil_weekly_chikungunya_cases.csv"
    climate_path = CLEAN_DIR / "brazil_weekly_climate.csv"

    if not cases_path.exists():
        raise FileNotFoundError(cases_path)
    if not climate_path.exists():
        raise FileNotFoundError(climate_path)

    cases = pd.read_csv(cases_path)
    climate = pd.read_csv(climate_path)

    data = cases.merge(climate, on=["year", "epi_week"], how="left")

    # TODO: Replace with confirmed IBGE population value for the selected year.
    data["population"] = 203_000_000
    data["incidence_per_100k"] = data["cases"] / data["population"] * 100_000

    out_path = CLEAN_DIR / "brazil_weekly_model_dataset.csv"
    data.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
