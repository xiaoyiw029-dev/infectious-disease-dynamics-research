"""Select the approved climate candidate without fitting or assigning data splits."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["epi_year_week", "week_start", "week_end"]
ROOT = Path(__file__).resolve().parents[2]


def build_candidate(data_dir: Path, out_dir: Path) -> dict:
    names = {
        "climate": "brazil_weekly_climate.csv",
        "rain": "inmet_rain_completeness_sensitivity_2023_2025.csv",
        "cases": "brazil_weekly_chikungunya_cases.csv",
        "baseline": "baseline_fitting_cases_2023W35_2024W45.csv",
        "station_week": "inmet_station_week_quality_2023_2025.csv",
        "state_week": "inmet_state_week_climate_2023_2025.csv",
    }
    frames = {key: pd.read_csv(data_dir / name) for key, name in names.items()}
    climate, rain, cases = (frames[key] for key in ("climate", "rain", "cases"))
    baseline = frames["baseline"]
    for frame in (climate, rain, cases, baseline):
        if frame[KEYS].isna().any().any() or frame.duplicated("week_start").any():
            raise ValueError("Missing or duplicate week keys")
    pd.testing.assert_frame_equal(climate[KEYS], rain[KEYS])
    expected = cases.loc[
        cases.week_start.between("2023-01-01", "2025-12-21"), KEYS
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(climate[KEYS], expected)
    starts = pd.to_datetime(climate.week_start)
    ends = pd.to_datetime(climate.week_end)
    assert len(climate) == 156
    assert starts.diff().dropna().eq(pd.Timedelta(days=7)).all()
    assert starts.dt.dayofweek.eq(6).all()
    assert (ends - starts).eq(pd.Timedelta(days=6)).all()

    renamed = {
        "temp_mean_population_weighted": "temp_mean_c",
        "humidity_mean_population_weighted": "humidity_mean_pct",
        "temp_states": "temp_valid_uf_count",
        "humidity_states": "humidity_valid_uf_count",
        "temp_mean_population_coverage": "temp_valid_uf_population_share",
        "humidity_mean_population_coverage": "humidity_valid_uf_population_share",
    }
    rain_renamed = {
        "rainfall_complete_population_weighted": "rainfall_168h_mm",
        "complete_states": "rain_valid_uf_count",
        "complete_population_coverage": "rain_valid_uf_population_share",
    }
    selected = climate[KEYS + list(renamed)].rename(columns=renamed).merge(
        rain[KEYS + list(rain_renamed)].rename(columns=rain_renamed),
        on=KEYS, validate="one_to_one", how="left",
    ).merge(
        cases[KEYS + ["probable_cases"]], on=KEYS, validate="one_to_one", how="left",
    )
    selected["in_baseline_window"] = selected.week_start.isin(baseline.week_start)
    selected["rain_coverage_status"] = np.select(
        [selected.rain_valid_uf_count.le(1), selected.rain_valid_uf_count.lt(27)],
        ["not_national", "partial_uf_coverage"], default="all_ufs",
    )
    baseline_selected = selected.loc[selected.in_baseline_window, KEYS + ["probable_cases"]]
    pd.testing.assert_frame_equal(
        baseline_selected.reset_index(drop=True), baseline[KEYS + ["probable_cases"]]
    )
    assert len(baseline_selected) == 63
    assert selected[["probable_cases", "temp_mean_c", "rainfall_168h_mm",
                     "humidity_mean_pct"]].notna().all().all()
    assert selected.probable_cases.ge(0).all()
    assert selected.rainfall_168h_mm.ge(0).all()
    assert selected.humidity_mean_pct.between(0, 100).all()

    # Recompute the selected rain from complete station-weeks, not the 160h column.
    sw, state = frames["station_week"], frames["state_week"]
    assert not sw.duplicated(["station_id", "week_start"]).any()
    population = state[["uf", "population_2024"]].drop_duplicates().set_index("uf")
    assert len(population) == 27 and population.index.is_unique
    assert int(population.population_2024.sum()) == 212583750
    full = sw.loc[sw.rows.eq(168) & sw.rain_hours.eq(168)]
    uf_rain = full.groupby(["week_start", "uf"]).rain_sum.mean().reset_index()
    uf_rain["population"] = uf_rain.uf.map(population.population_2024)
    assert uf_rain.population.notna().all()
    for row in selected.itertuples(index=False):
        group = uf_rain.loc[uf_rain.week_start.eq(row.week_start)]
        np.testing.assert_allclose(
            row.rainfall_168h_mm, np.average(group.rain_sum, weights=group.population),
            rtol=1e-12, atol=1e-10,
        )
        assert row.rain_valid_uf_count == len(group)
        np.testing.assert_allclose(
            row.rain_valid_uf_population_share,
            group.population.sum() / population.population_2024.sum(),
            rtol=1e-12, atol=1e-12,
        )
    for col in [c for c in selected if c.endswith("_population_share")]:
        assert selected[col].between(0, 1).all()

    period_summaries = {}
    masks = {
        "baseline_63_weeks": selected.in_baseline_window,
        "2025_available_weeks": selected.epi_year_week.str.startswith("2025-"),
    }
    for period, mask in masks.items():
        sub = selected.loc[mask]
        period_summaries[period] = {
            "weeks": len(sub),
            "first_week": sub.epi_year_week.iloc[0],
            "last_week": sub.epi_year_week.iloc[-1],
            "rain_min_valid_uf_count": int(sub.rain_valid_uf_count.min()),
            "rain_weeks_all_27_ufs": int(sub.rain_valid_uf_count.eq(27).sum()),
            "rain_min_valid_uf_population_share": float(sub.rain_valid_uf_population_share.min()),
        }
    year_2025 = cases.loc[cases.epi_year.eq(2025), KEYS]
    missing_2025 = year_2025.loc[~year_2025.week_start.isin(selected.week_start)]
    metadata = {
        "decision_date": "2026-10-05",
        "status": "main_candidate_not_model_ready",
        "spatial_aggregation": "within_UF_station_mean_then_fixed_2024_population_weighting",
        "rain_station_week_rule": "168 valid hours and 168 record rows",
        "temp_humidity_station_week_rule": "at least 135 valid hours",
        "population_total_2024": 212583750,
        "missing_UF_rule": "renormalize weights over UFs with eligible stations",
        "coverage_meaning": "whole-UF population share, not actual spatial population coverage",
        "units": {"temp_mean_c": "degC", "rainfall_168h_mm": "mm/week",
                  "humidity_mean_pct": "percent", "probable_cases": "persons/week"},
        "time_basis": "station-local Sunday 00:00 to next Sunday 00:00",
        "split_status": "not_confirmed",
        "2025_prior_model_selection_use": "unknown_confirmed_by_user_2026-10-05",
        "case_calendar_2025_weeks": len(year_2025),
        "missing_climate_weeks_in_2025": missing_2025.to_dict(orient="records"),
        "period_summaries": period_summaries,
        "five_lowest_rain_coverage_weeks": selected.nsmallest(
            5, "rain_valid_uf_population_share"
        )[KEYS + ["rain_valid_uf_count", "rain_valid_uf_population_share"]].to_dict(orient="records"),
        "limitations": [
            "No imputation, lag selection, scaling, split assignment or model fitting performed.",
            "2025-W52 has only one eligible UF: do not use its rain as a national predictor.",
            "2025-W43 reduced coverage requires review; no exclusion threshold is approved.",
            "2025-W53 needs 2026 hourly climate input and is absent here.",
            "Full hourly completeness does not ensure adequate national spatial coverage.",
            "Fixed 2024 weights and revised case snapshots do not recreate real-time availability.",
        ],
        "checks_passed": [
            "156 continuous Sunday-Saturday week keys match the source case calendar",
            "63 baseline week keys and observed counts unchanged",
            "156 selected rain values and coverage independently recomputed from station-weeks",
            "nonnegative counts/rain, bounded humidity/coverage, no missing national aggregates",
        ],
        "inputs_sha256": {
            name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            for name in names.values()
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "brazil_weekly_climate_main_candidate_2023_2025.csv"
    selected.to_csv(output, index=False, encoding="utf-8-sig")
    pd.testing.assert_frame_equal(selected, pd.read_csv(output), check_exact=False,
                                  rtol=1e-12, atol=1e-10)
    metadata["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    (out_dir / "climate_main_candidate_selection_v01.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "02_数据" / "01_清洗数据")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_candidate(args.data_dir, args.out_dir or args.data_dir),
                     ensure_ascii=True, indent=2))
