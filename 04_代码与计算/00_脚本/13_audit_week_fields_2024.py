"""Audit 2024 source week fields against symptom and notification dates."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import runpy

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    root = parser.parse_args().project_root
    source = root / "02_数据/00_原始数据/OpenDataSUS_SINAN_Chikungunya/CHIKBR24/CHIKBR24.csv"
    fit_source = root / "02_数据/01_清洗数据/baseline_fitting_cases_2023W35_2024W45.csv"
    week_script = root / "04_代码与计算/00_脚本/01_prepare_weekly_cases.py"
    week_from_date = runpy.run_path(str(week_script))["epi_week_from_date"]
    year_start = runpy.run_path(str(week_script))["epi_year_start"]

    columns = [
        "DT_SIN_PRI", "SEM_PRI", "DT_NOTIFIC", "SEM_NOT", "CLASSI_FIN",
        "SG_UF_NOT", "TP_SISTEMA",
    ]
    data = pd.read_csv(source, usecols=columns, dtype=str, low_memory=False)

    def expected_codes(field: str) -> tuple[pd.Series, pd.Series]:
        dates = pd.to_datetime(data[field], errors="coerce")
        by_date = {
            date: week_from_date(pd.Timestamp(date)) for date in dates.dropna().unique()
        }
        weeks = dates.map(lambda date: by_date[date] if pd.notna(date) else None)
        codes = weeks.map(
            lambda week: f"{week[0]}{week[1]:02d}" if week is not None else ""
        )
        return dates, codes

    onset, onset_code = expected_codes("DT_SIN_PRI")
    _notification, notification_code = expected_codes("DT_NOTIFIC")
    raw_onset_code = data["SEM_PRI"].fillna("").str.strip()
    symptom_mismatch = raw_onset_code.ne(onset_code)
    notification_mismatch = data["SEM_NOT"].fillna("").str.strip().ne(notification_code)
    bad = data.loc[symptom_mismatch]

    def raw_week_start(code: str):
        if len(code) != 6 or not code.isdigit():
            return pd.NaT
        year, week = int(code[:4]), int(code[4:])
        if not 1 <= week <= 53:
            return pd.NaT
        return pd.Timestamp(year_start(year) + timedelta(weeks=week - 1))

    raw_starts = pd.to_datetime(raw_onset_code.loc[symptom_mismatch].map(raw_week_start))
    computed_starts = pd.to_datetime(
        onset.loc[symptom_mismatch].map(lambda d: week_from_date(d)[2])
    )
    offsets = ((raw_starts - computed_starts).dt.days / 7).value_counts(dropna=False)

    probable = data["CLASSI_FIN"].fillna("").str.strip().ne("5")
    raw_counts = data.loc[probable].groupby(raw_onset_code[probable]).size()
    date_counts = data.loc[probable].groupby(onset_code[probable]).size()
    fit = pd.read_csv(fit_source)
    fit_codes = fit["epi_year"].astype(str) + fit["epi_week"].astype(str).str.zfill(2)
    weekly_shift = fit_codes.map(raw_counts).fillna(0).astype(int) - fit_codes.map(date_counts).fillna(0).astype(int)

    print(f"2024 source records: {len(data)}")
    print(f"SEM_PRI mismatches: {symptom_mismatch.sum()}")
    print(f"Current probable-definition mismatches: {(symptom_mismatch & probable).sum()}")
    print(f"SEM_NOT mismatches: {notification_mismatch.sum()}")
    print(f"Same records mismatch both fields: {(symptom_mismatch & notification_mismatch).sum()}")
    print("Notification state distribution:", bad["SG_UF_NOT"].value_counts(dropna=False).to_dict())
    print("System-code distribution:", bad["TP_SISTEMA"].value_counts(dropna=False).to_dict())
    print("Raw minus official week offset:", offsets.to_dict())
    print(f"Fitting weeks with changed counts: {weekly_shift.ne(0).sum()}")
    print(f"Maximum absolute weekly difference: {weekly_shift.abs().max()}")
    print(f"Official-date 63-week total: {fit['probable_cases'].sum()}")
    print(f"Raw-week 63-week total: {(fit['probable_cases'] + weekly_shift).sum()}")


if __name__ == "__main__":
    main()
