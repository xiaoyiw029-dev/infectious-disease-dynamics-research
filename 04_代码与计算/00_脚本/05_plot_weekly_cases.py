"""
Plot national weekly chikungunya case trends for Brazil.
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
FIG_DIR = PROJECT_ROOT / "05_结果与图表" / "03_导出图片"
RESULT_DIR = PROJECT_ROOT / "05_结果与图表" / "00_主要结果"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["week_end"] = pd.to_datetime(df["week_end"])
    return df


def plot_primary_trend(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df["week_start"], df["cases"], color="#1f6f8b", linewidth=1.8)
    ax.fill_between(df["week_start"], df["cases"], color="#8ecae6", alpha=0.35)

    peak = df.loc[df["cases"].idxmax()]
    ax.scatter(
        peak["week_start"],
        peak["cases"],
        color="#c1121f",
        s=34,
        zorder=5,
    )
    ax.annotate(
        f"Peak: {peak['epi_year_week']}\n{int(peak['cases']):,} cases",
        xy=(peak["week_start"], peak["cases"]),
        xytext=(24, 18),
        textcoords="offset points",
        fontsize=10,
        color="#6a040f",
        arrowprops={"arrowstyle": "->", "color": "#6a040f", "linewidth": 1},
    )

    ax.set_title(
        "National Weekly Probable Chikungunya Cases in Brazil, 2015-2026",
        fontsize=15,
        pad=14,
    )
    ax.set_xlabel("Epidemiological week")
    ax.set_ylabel("Weekly probable cases")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = FIG_DIR / "brazil_weekly_probable_cases_trend_2015_2026.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_case_definition_comparison(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(
        df["week_start"],
        df["probable_cases"],
        color="#1f6f8b",
        linewidth=1.6,
        label="Probable cases",
    )
    ax.plot(
        df["week_start"],
        df["chikungunya_classified_cases"],
        color="#2a9d8f",
        linewidth=1.3,
        label="CLASSI_FIN = 13",
    )
    ax.plot(
        df["week_start"],
        df["confirmed_chikungunya_cases"],
        color="#e76f51",
        linewidth=1.1,
        label="CLASSI_FIN = 13 and CRITERIO in {1, 2}",
    )

    ax.set_title(
        "Weekly Chikungunya Case Definitions in Brazil, 2015-2026",
        fontsize=15,
        pad=14,
    )
    ax.set_xlabel("Epidemiological week")
    ax.set_ylabel("Weekly cases")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = FIG_DIR / "brazil_weekly_case_definition_comparison_2015_2026.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def write_summary_tables(df: pd.DataFrame) -> tuple[Path, Path]:
    annual = (
        df.groupby("epi_year")[
            [
                "notifications_total",
                "cases",
                "discarded_cases",
                "chikungunya_classified_cases",
                "confirmed_chikungunya_cases",
            ]
        ]
        .sum()
        .reset_index()
    )
    annual_path = RESULT_DIR / "brazil_annual_case_summary_2015_2026.csv"
    annual.to_csv(annual_path, index=False, encoding="utf-8-sig")

    top_weeks = df.nlargest(20, "cases")[
        [
            "epi_year_week",
            "week_start",
            "week_end",
            "cases",
            "chikungunya_classified_cases",
            "confirmed_chikungunya_cases",
        ]
    ]
    top_path = RESULT_DIR / "brazil_top_weekly_case_peaks_2015_2026.csv"
    top_weeks.to_csv(top_path, index=False, encoding="utf-8-sig")
    return annual_path, top_path


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    trend_path = plot_primary_trend(df)
    comparison_path = plot_case_definition_comparison(df)
    annual_path, top_path = write_summary_tables(df)

    print(f"Wrote {trend_path}")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {annual_path}")
    print(f"Wrote {top_path}")


if __name__ == "__main__":
    main()
