"""
Prepare national weekly chikungunya case counts for Brazil.

This is a first-stage script skeleton. Field names must be confirmed after
reading the OPENDATASUS/SINAN data dictionary.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "02_数据" / "00_原始数据" / "opendatasus_chikungunya"
OUT_DIR = PROJECT_ROOT / "02_数据" / "01_清洗数据"


# TODO: Confirm these names from the official data dictionary.
CONFIG = {
    "year_col": "YEAR",
    "epi_week_col": "EPI_WEEK",
    "case_classification_col": "CLASSIFICATION",
    "date_symptom_col": "DATE_SYMPTOM_ONSET",
    "date_notification_col": "DATE_NOTIFICATION",
    "probable_case_values": [],
    "confirmed_case_values": [],
}


def read_case_files(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Download OPENDATASUS CSV files first."
        )

    frames = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        frame["source_file"] = path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def filter_cases(df: pd.DataFrame, case_definition: str = "probable") -> pd.DataFrame:
    class_col = CONFIG["case_classification_col"]

    if class_col not in df.columns:
        raise KeyError(f"Missing classification column: {class_col}")

    if case_definition == "probable":
        allowed = CONFIG["probable_case_values"]
    elif case_definition == "confirmed":
        allowed = CONFIG["confirmed_case_values"]
    else:
        raise ValueError("case_definition must be 'probable' or 'confirmed'")

    if not allowed:
        raise ValueError("Case classification values are not configured yet.")

    return df[df[class_col].isin(allowed)].copy()


def aggregate_weekly_cases(df: pd.DataFrame) -> pd.DataFrame:
    year_col = CONFIG["year_col"]
    week_col = CONFIG["epi_week_col"]

    missing = [col for col in [year_col, week_col] if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    weekly = (
        df.groupby([year_col, week_col], dropna=False)
        .size()
        .reset_index(name="cases")
        .rename(columns={year_col: "year", week_col: "epi_week"})
        .sort_values(["year", "epi_week"])
    )
    return weekly


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_case_files(RAW_DIR)
    df = filter_cases(df, case_definition="probable")
    weekly = aggregate_weekly_cases(df)
    out_path = OUT_DIR / "brazil_weekly_chikungunya_cases.csv"
    weekly.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
