"""
Prepare national weekly chikungunya case counts for Brazil.

Primary case definition:
    probable cases = notified records excluding discarded cases
    i.e. CLASSI_FIN != 5.

Time axis:
    symptom onset date (DT_SIN_PRI), converted to Sunday-start
    epidemiological weeks. Week 1 is the week containing January 4.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = (
    PROJECT_ROOT
    / "02_数据"
    / "00_原始数据"
    / "OpenDataSUS_SINAN_Chikungunya"
)
OUT_DIR = PROJECT_ROOT / "02_数据" / "01_清洗数据"
QC_DIR = PROJECT_ROOT / "02_数据" / "04_数据质量控制"

START_DATE = pd.Timestamp("2015-01-01")
CURRENT_DATE = pd.Timestamp("2026-08-25")
ANALYSIS_START_EPI_YEAR = 2015

USE_COLUMNS = [
    "DT_NOTIFIC",
    "SEM_NOT",
    "NU_ANO",
    "DT_SIN_PRI",
    "SEM_PRI",
    "SG_UF",
    "ID_MN_RESI",
    "CLASSI_FIN",
    "CRITERIO",
]


def epi_year_start(year: int) -> date:
    """Return the Sunday starting epidemiological week 1 for a year."""
    jan4 = date(year, 1, 4)
    days_since_sunday = (jan4.weekday() + 1) % 7
    return jan4 - timedelta(days=days_since_sunday)


def epi_week_from_date(value: pd.Timestamp) -> tuple[int, int, date, date]:
    d = value.date()
    start = epi_year_start(d.year)
    next_start = epi_year_start(d.year + 1)

    if d < start:
        epi_year = d.year - 1
        start = epi_year_start(epi_year)
    elif d >= next_start:
        epi_year = d.year + 1
        start = next_start
    else:
        epi_year = d.year

    week = ((d - start).days // 7) + 1
    week_start = start + timedelta(days=(week - 1) * 7)
    week_end = week_start + timedelta(days=6)
    return epi_year, week, week_start, week_end


def source_year_from_path(path: Path) -> int:
    match = re.search(r"CHIKBR(\d{2})\.csv$", path.name)
    if not match:
        raise ValueError(f"Cannot infer source year from {path.name}")
    return 2000 + int(match.group(1))


def read_case_files() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("CHIKBR*/CHIKBR*.csv"))
    if not files:
        raise FileNotFoundError(f"No yearly CSV files found under {RAW_DIR}")

    frames: list[pd.DataFrame] = []
    for path in files:
        source_year = source_year_from_path(path)
        frame = pd.read_csv(path, usecols=USE_COLUMNS, dtype=str, low_memory=False)
        frame["source_year"] = source_year
        frame["source_file"] = path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def add_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["symptom_onset_date"] = pd.to_datetime(out["DT_SIN_PRI"], errors="coerce")
    out["notification_date"] = pd.to_datetime(out["DT_NOTIFIC"], errors="coerce")

    valid_date = (
        out["symptom_onset_date"].notna()
        & (out["symptom_onset_date"] >= START_DATE)
        & (out["symptom_onset_date"] <= CURRENT_DATE)
    )
    out = out.loc[valid_date].copy()

    epi_parts = out["symptom_onset_date"].map(epi_week_from_date)
    out["epi_year"] = epi_parts.map(lambda x: x[0])
    out["epi_week"] = epi_parts.map(lambda x: x[1])
    out["week_start"] = epi_parts.map(lambda x: x[2].isoformat())
    out["week_end"] = epi_parts.map(lambda x: x[3].isoformat())
    out["epi_year_week"] = out.apply(
        lambda row: f"{int(row['epi_year'])}-W{int(row['epi_week']):02d}", axis=1
    )
    out["computed_sem_pri"] = out.apply(
        lambda row: f"{int(row['epi_year'])}{int(row['epi_week']):02d}", axis=1
    )
    out["raw_sem_pri"] = out["SEM_PRI"].fillna("").str.strip()
    out["raw_sem_pri_is_six_digit"] = out["raw_sem_pri"].str.match(r"^\d{6}$")
    out["raw_sem_pri_matches_computed"] = (
        out["raw_sem_pri_is_six_digit"]
        & (out["raw_sem_pri"] == out["computed_sem_pri"])
    )
    return out


def make_weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_discarded"] = df["CLASSI_FIN"].fillna("").str.strip().eq("5")
    df["is_chikungunya_classified"] = df["CLASSI_FIN"].fillna("").str.strip().eq("13")
    df["is_confirmed_chikungunya"] = (
        df["is_chikungunya_classified"]
        & df["CRITERIO"].fillna("").str.strip().isin(["1", "2"])
    )
    df["is_probable"] = ~df["is_discarded"]

    grouped = (
        df.groupby(["epi_year", "epi_week", "epi_year_week", "week_start", "week_end"])
        .agg(
            notifications_total=("CLASSI_FIN", "size"),
            cases=("is_probable", "sum"),
            probable_cases=("is_probable", "sum"),
            discarded_cases=("is_discarded", "sum"),
            chikungunya_classified_cases=("is_chikungunya_classified", "sum"),
            confirmed_chikungunya_cases=("is_confirmed_chikungunya", "sum"),
        )
        .reset_index()
    )

    all_weeks = pd.date_range(
        grouped["week_start"].min(), grouped["week_start"].max(), freq="7D"
    )
    calendar = []
    for week_start in all_weeks:
        epi_year, epi_week, start, end = epi_week_from_date(pd.Timestamp(week_start))
        calendar.append(
            {
                "epi_year": epi_year,
                "epi_week": epi_week,
                "epi_year_week": f"{epi_year}-W{epi_week:02d}",
                "week_start": start.isoformat(),
                "week_end": end.isoformat(),
            }
        )
    calendar_df = pd.DataFrame(calendar)

    weekly = calendar_df.merge(
        grouped,
        on=["epi_year", "epi_week", "epi_year_week", "week_start", "week_end"],
        how="left",
    )
    count_cols = [
        "notifications_total",
        "cases",
        "probable_cases",
        "discarded_cases",
        "chikungunya_classified_cases",
        "confirmed_chikungunya_cases",
    ]
    weekly[count_cols] = weekly[count_cols].fillna(0).astype(int)
    weekly = weekly.loc[weekly["epi_year"] >= ANALYSIS_START_EPI_YEAR].copy()
    return weekly.sort_values(["week_start"]).reset_index(drop=True)


def make_quality_summary(raw: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_year, frame in raw.groupby("source_year"):
        parsed = pd.to_datetime(frame["DT_SIN_PRI"], errors="coerce")
        valid_date = (
            parsed.notna() & (parsed >= START_DATE) & (parsed <= CURRENT_DATE)
        )
        kept = valid.loc[valid["source_year"].eq(source_year)]
        rows.append(
            {
                "source_year": int(source_year),
                "raw_records": int(len(frame)),
                "kept_records": int(len(kept)),
                "excluded_invalid_or_out_of_range_onset_date": int((~valid_date).sum()),
                "missing_DT_SIN_PRI": int(parsed.isna().sum()),
                "missing_DT_NOTIFIC": int(
                    pd.to_datetime(frame["DT_NOTIFIC"], errors="coerce").isna().sum()
                ),
                "raw_SEM_PRI_not_six_digit": int(
                    (~kept["raw_sem_pri_is_six_digit"]).sum()
                ),
                "raw_SEM_PRI_mismatch_computed": int(
                    (
                        kept["raw_sem_pri_is_six_digit"]
                        & ~kept["raw_sem_pri_matches_computed"]
                    ).sum()
                ),
                "resident_uf_values": int(kept["SG_UF"].dropna().nunique()),
                "resident_municipality_values": int(
                    kept["ID_MN_RESI"].dropna().nunique()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("source_year")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    raw = read_case_files()
    valid = add_time_fields(raw)
    weekly = make_weekly_series(valid)
    quality = make_quality_summary(raw, valid)

    weekly_path = OUT_DIR / "brazil_weekly_chikungunya_cases.csv"
    quality_path = QC_DIR / "brazil_weekly_chikungunya_cases_quality_summary.csv"

    weekly.to_csv(weekly_path, index=False, encoding="utf-8-sig")
    quality.to_csv(quality_path, index=False, encoding="utf-8-sig")

    print(f"Wrote {weekly_path}")
    print(f"Wrote {quality_path}")
    print(f"Weeks: {len(weekly)}")
    print(
        f"Start week: {weekly.iloc[0]['epi_year_week']} "
        f"({weekly.iloc[0]['week_start']})"
    )
    print(
        f"End week: {weekly.iloc[-1]['epi_year_week']} "
        f"({weekly.iloc[-1]['week_end']})"
    )
    print(f"Total probable cases: {int(weekly['cases'].sum())}")


if __name__ == "__main__":
    main()
