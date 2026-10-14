"""
Plot cumulative observed and fitted chikungunya cases for reference-style fits.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import PchipInterpolator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "05_结果与图表" / "00_主要结果"
FIG_DIR = PROJECT_ROOT / "05_结果与图表" / "03_导出图片"


SCENARIOS = [
    {
        "input": RESULT_DIR
        / "reference_infection_cases_free_initial_dual_weekly_fit_v01.csv",
        "output": FIG_DIR
        / "reference_infection_cases_free_initial_dual_cumulative_fit_v01.png",
        "title": "Cumulative Case Fit: Free Initial Conditions",
    },
    {
        "input": RESULT_DIR
        / "reference_infection_cases_fixed_eh189_ih63_dual_weekly_fit_v01.csv",
        "output": FIG_DIR
        / "reference_infection_cases_fixed_eh189_ih63_dual_cumulative_fit_v01.png",
        "title": "Cumulative Case Fit: Fixed E_h(0)=189, I_h(0)=63",
    },
]


def load_weekly_fit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["cum_observed_cases"] = df["observed_I_h_i"].cumsum()
    df["cum_fitted_cases"] = df["fitted_I_h_i"].cumsum()
    return df


def smooth_cumulative_fit(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    x = mdates.date2num(df["week_start"])
    y = df["cum_fitted_cases"].to_numpy()
    x_smooth = pd.Series(pd.date_range(df["week_start"].min(), df["week_start"].max(), periods=500))
    spline = PchipInterpolator(x, y)
    y_smooth = pd.Series(spline(mdates.date2num(x_smooth)))
    return x_smooth, y_smooth


def plot_cumulative_fit(df: pd.DataFrame, title: str, output: Path) -> None:
    x_smooth, y_smooth = smooth_cumulative_fit(df)

    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.scatter(
        df["week_start"],
        df["cum_observed_cases"],
        s=30,
        color="#1f6f8b",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
        label="Observed cumulative probable cases",
        zorder=3,
    )
    ax.plot(
        x_smooth,
        y_smooth,
        color="#c1121f",
        linewidth=2.4,
        label="Fitted cumulative cases",
        zorder=2,
    )

    ax.set_title(title, fontsize=15, pad=13)
    ax.set_xlabel("Epidemiological week")
    ax.set_ylabel("Cumulative cases up to each week")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_formatter(lambda x, _pos: f"{x:,.0f}")
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        df = load_weekly_fit(scenario["input"])
        plot_cumulative_fit(df, scenario["title"], scenario["output"])
        print(scenario["output"])


if __name__ == "__main__":
    main()
