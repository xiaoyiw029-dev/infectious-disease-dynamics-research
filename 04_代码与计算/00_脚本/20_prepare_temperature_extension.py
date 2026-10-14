"""Audit raw lag-four temperature inputs; no climate model fitting."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def transmission_pair(kappa, ratio, eta, standardized_temperature):
    if kappa <= 0 or ratio <= 0:
        raise ValueError("Positive product and ratio required")
    effect = eta * np.asarray(standardized_temperature, dtype=float)
    return np.exp((np.log(kappa) + np.log(ratio) + effect) / 2), np.exp(
        (np.log(kappa) - np.log(ratio) + effect) / 2)


def run(args):
    source = args.project_root / "02_数据/01_清洗数据"
    case_path = source / "baseline_fitting_cases_2023W35_2024W45.csv"
    weather_path = source / "brazil_weekly_climate_main_candidate_2023_2025_v02.csv"
    cases = pd.read_csv(case_path).sort_values("t_week").reset_index(drop=True)
    climate = pd.read_csv(weather_path)
    cases["temperature_week_start"] = (
        pd.to_datetime(cases.week_start) - pd.Timedelta(weeks=4)).dt.strftime("%Y-%m-%d")
    weather = climate[["week_start", "week_end", "epi_year_week", "temp_mean_c",
                       "temp_valid_uf_count", "temp_valid_uf_population_share"]].rename(columns={
                           "week_start": "temperature_week_start", "week_end": "temperature_week_end",
                           "epi_year_week": "temperature_epi_week"})
    inputs = cases[["t_week", "epi_year_week", "week_start", "week_end",
                    "temperature_week_start"]].merge(weather, on="temperature_week_start",
                                                    validate="one_to_one", how="left")
    assert len(inputs) == 63 and inputs.temp_mean_c.notna().all()
    rows = []
    for n in range(20, 63):
        training_temperature = inputs.temp_mean_c.to_numpy()[:n]
        mean = float(training_temperature.mean())
        scale = float(training_temperature.std())
        assert scale > 0
        for h in (1, 2, 4):
            if n + h > 63:
                continue
            feature, origin = inputs.iloc[n + h - 1], cases.iloc[n - 1]
            available = pd.Timestamp(feature.temperature_week_end) <= pd.Timestamp(origin.week_end)
            assert available
            rows.append({"origin_n": n, "horizon": h, "origin_week": origin.epi_year_week,
                         "origin_week_end": origin.week_end, "target_week": feature.epi_year_week,
                         "temperature_epi_week": feature.temperature_epi_week,
                         "temperature_week_end": feature.temperature_week_end,
                         "temperature_mean_c": float(feature.temp_mean_c),
                         "training_temperature_mean_c": mean, "training_temperature_std_c": scale,
                         "standardized_temperature": (float(feature.temp_mean_c) - mean) / scale,
                         "calendar_available": bool(available)})
    folds = pd.DataFrame(rows)
    assert len(folds) == 125
    # eta=0 must recover any positive constant transmission pair, not necessarily equality.
    for h, v in ((.05, 1.), (.3, .1), (1., .02)):
        got_h, got_v = transmission_pair(h * v, h / v, 0., np.array([-2., 0., 2.]))
        np.testing.assert_allclose(got_h, h, rtol=1e-12)
        np.testing.assert_allclose(got_v, v, rtol=1e-12)
    bh, bv = transmission_pair(.05, .1, .2, np.array([-2., 0., 2.]))
    np.testing.assert_allclose(bh / bv, .1, rtol=1e-12)
    np.testing.assert_allclose(bh * bv, .05 * np.exp(.2 * np.array([-2., 0., 2.])), rtol=1e-12)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "input_audit.json").exists():
        raise FileExistsError("Existing preparation protected")
    inputs.to_csv(args.out_dir / "temperature_lag4_inputs.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(args.out_dir / "temperature_lag4_fold_availability.csv", index=False, encoding="utf-8-sig")
    audit = {
        "date": "2026-10-07", "status": "prepared_not_fitted", "lag_weeks": 4,
        "target_weeks": len(inputs), "forecast_rows": len(folds),
        "first_temperature_week": inputs.iloc[0].temperature_epi_week,
        "last_temperature_week": inputs.iloc[-1].temperature_epi_week,
        "missing_temperature_weeks": int(inputs.temp_mean_c.isna().sum()),
        "minimum_valid_uf_count": int(inputs.temp_valid_uf_count.min()),
        "minimum_valid_uf_population_share": float(inputs.temp_valid_uf_population_share.min()),
        "temperature_min_c": float(inputs.temp_mean_c.min()),
        "temperature_max_c": float(inputs.temp_mean_c.max()),
        "all_calendar_available": bool(folds.calendar_available.all()),
        "nested_eta_zero_and_product_ratio_tests": "PASS",
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (case_path, weather_path)},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "warnings": ["Calendar availability only; actual publication delay unverified.",
                     "No climate coefficient or forecast gain estimated by this script.",
                     "Ratio scenarios, coefficient bounds and profile treatment remain to be fixed before fitting."]}
    (args.out_dir / "input_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    run(parser.parse_args())
