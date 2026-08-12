"""
Prepare national weekly climate covariates for Brazil.

This skeleton expects INMET data after download and extraction. Column names
must be confirmed from the raw files.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "02_数据" / "00_原始数据" / "inmet"
OUT_DIR = PROJECT_ROOT / "02_数据" / "01_清洗数据"


# TODO: Confirm these names from INMET files.
CONFIG = {
    "date_col": "DATE",
    "state_col": "STATE",
    "station_col": "STATION",
    "temp_col": "TEMP_MEAN",
    "rain_col": "RAINFALL",
    "humidity_col": "HUMIDITY_MEAN",
}


def read_climate_files(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Download INMET CSV files first."
        )

    frames = []
    for path in files:
        frame = pd.read_csv(path, sep=";", decimal=",", encoding="latin1", low_memory=False)
        frame["source_file"] = path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def aggregate_weekly_climate(df: pd.DataFrame) -> pd.DataFrame:
    date_col = CONFIG["date_col"]
    temp_col = CONFIG["temp_col"]
    rain_col = CONFIG["rain_col"]
    humidity_col = CONFIG["humidity_col"]

    missing = [col for col in [date_col, temp_col, rain_col] if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    iso = df[date_col].dt.isocalendar()
    df["year"] = iso.year.astype(int)
    df["epi_week"] = iso.week.astype(int)

    aggregations = {
        temp_col: "mean",
        rain_col: "sum",
    }
    if humidity_col in df.columns:
        aggregations[humidity_col] = "mean"

    weekly = df.groupby(["year", "epi_week"]).agg(aggregations).reset_index()
    weekly = weekly.rename(
        columns={
            temp_col: "temp_mean",
            rain_col: "rainfall_sum",
            humidity_col: "humidity_mean",
        }
    )

    for lag in range(1, 5):
        weekly[f"temp_mean_lag{lag}"] = weekly["temp_mean"].shift(lag)
        weekly[f"rainfall_sum_lag{lag}"] = weekly["rainfall_sum"].shift(lag)
        if "humidity_mean" in weekly.columns:
            weekly[f"humidity_mean_lag{lag}"] = weekly["humidity_mean"].shift(lag)

    return weekly


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_climate_files(RAW_DIR)
    weekly = aggregate_weekly_climate(df)
    out_path = OUT_DIR / "brazil_weekly_climate.csv"
    weekly.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
