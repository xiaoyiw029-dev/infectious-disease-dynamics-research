"""
Extract the main fitting window for the baseline host-vector model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    PROJECT_ROOT / "02_数据" / "01_清洗数据" / "brazil_weekly_chikungunya_cases.csv"
)
OUT_DIR = PROJECT_ROOT / "02_数据" / "01_清洗数据"
FIG_DIR = PROJECT_ROOT / "05_结果与图表" / "03_导出图片"

START_WEEK = "2023-W35"
END_WEEK = "2024-W45"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["week_end"] = pd.to_datetime(df["week_end"])

    window = df.loc[
        (df["epi_year_week"] >= START_WEEK) & (df["epi_year_week"] <= END_WEEK)
    ].copy()
    if window.empty:
        raise ValueError(f"No records found for {START_WEEK} to {END_WEEK}")

    window = window.sort_values("week_start").reset_index(drop=True)
    window.insert(0, "t_week", range(len(window)))
    window.insert(1, "t_day", window["t_week"] * 7)

    out_path = OUT_DIR / "baseline_fitting_cases_2023W35_2024W45.csv"
    window.to_csv(out_path, index=False, encoding="utf-8-sig")

    peak = window.loc[window["cases"].idxmax()]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.plot(window["week_start"], window["cases"], color="#1f6f8b", linewidth=2)
    ax.fill_between(window["week_start"], window["cases"], color="#8ecae6", alpha=0.35)
    ax.scatter(peak["week_start"], peak["cases"], color="#c1121f", s=36, zorder=5)
    ax.annotate(
        f"Peak: {peak['epi_year_week']}\n{int(peak['cases']):,} cases",
        xy=(peak["week_start"], peak["cases"]),
        xytext=(34, -18),
        textcoords="offset points",
        fontsize=10,
        color="#6a040f",
        arrowprops={"arrowstyle": "->", "color": "#6a040f", "linewidth": 1},
    )
    ax.set_title(
        "Baseline Fitting Window: Brazil Weekly Probable Chikungunya Cases",
        fontsize=14,
        pad=12,
    )
    ax.set_xlabel("Epidemiological week")
    ax.set_ylabel("Weekly probable cases")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    fig_path = FIG_DIR / "baseline_fitting_window_2023W35_2024W45.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"Wrote {out_path}")
    print(f"Wrote {fig_path}")
    print(f"Weeks: {len(window)}")
    print(f"Start: {window.iloc[0]['epi_year_week']} ({window.iloc[0]['week_start'].date()})")
    print(f"End: {window.iloc[-1]['epi_year_week']} ({window.iloc[-1]['week_end'].date()})")
    print(f"Peak: {peak['epi_year_week']} ({int(peak['cases'])} cases)")
    print(f"Cumulative probable cases: {int(window['cases'].sum())}")


if __name__ == "__main__":
    main()
