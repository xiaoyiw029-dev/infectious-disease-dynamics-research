"""Paired exploratory climate-information screen; not a mechanistic extension."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rolling = importlib.import_module("17_validate_peak_trend")
CLIMATE = ["temp_mean_c", "humidity_mean_pct", "rainfall_168h_mm"]
FEATURES = ["latest_weekly_count", "recent4_slope"] + CLIMATE
ALPHA = 1.0


def features(y, climate, origin_n):
    _, _, slope = rolling.trend_forecast(y[:origin_n], 1)
    return np.r_[y[origin_n - 1], slope, climate[origin_n - 1]]


def ridge_fit_predict(x, targets, query):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    z = (x - mean) / scale
    center = float(targets.mean())
    coefficient = np.linalg.solve(z.T @ z + ALPHA * np.eye(x.shape[1]),
                                  z.T @ (targets - center))
    raw = float(center + ((query - mean) / scale) @ coefficient)
    return max(0.0, raw), raw, mean, scale, center, coefficient


def forecast_one(y, climate, n, horizon, use_climate):
    # Historical feature origins s can be used only when their targets s+h are observed.
    origins = np.arange(4, n - horizon + 1)
    width = 5 if use_climate else 2
    x = np.array([features(y, climate, int(s))[:width] for s in origins])
    targets = y[origins + horizon - 1]
    query = features(y, climate, n)[:width]
    prediction, raw, mean, scale, intercept, coefficients = ridge_fit_predict(x, targets, query)
    return {"prediction": prediction, "raw_prediction": raw, "training_pairs": len(origins),
            "latest_training_target_week_number": int(origins[-1] + horizon),
            "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
            "target_mean": intercept, "standardized_coefficients": coefficients.tolist()}


def run(args):
    source = args.project_root / "02_数据/01_清洗数据"
    cases_path = source / "baseline_fitting_cases_2023W35_2024W45.csv"
    climate_path = source / "brazil_weekly_climate_main_candidate_2023_2025_v02.csv"
    cases = pd.read_csv(cases_path).sort_values("t_week")
    climate = pd.read_csv(climate_path)
    data = cases.merge(climate[["epi_year_week", "week_start", "week_end", *CLIMATE]],
                       on=["epi_year_week", "week_start", "week_end"], validate="one_to_one",
                       how="left")
    assert len(data) == 63 and data.t_week.tolist() == list(range(63))
    y, weather = data.probable_cases.to_numpy(float), data[CLIMATE].to_numpy(float)
    assert np.isfinite(y).all() and np.isfinite(weather).all()
    prior_summary = json.loads((args.baseline_results / "run_summary.json").read_text())
    assert prior_summary["status"] == "completed"
    assert prior_summary["input_sha256"] == rolling.base.file_hash(cases_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "run_configuration.json").exists():
        raise FileExistsError("Existing climate screen protected")
    config = {
        "date": "2026-10-07", "status": "running",
        "scope": "exploratory matched climate information screen; not a confirmed mechanistic model",
        "initial_train_weeks": prior_summary["initial_train_weeks"],
        "horizons": [1, 2, 4], "ridge_alpha": ALPHA, "features": FEATURES,
        "target": "raw weekly new counts", "negative_forecasts": "clip to zero and retain raw values",
        "preprocessing": "feature mean/std from each horizon's eligible training pairs only",
        "historical_pairs": "origins s>=4 with s+h<=current origin n",
        "climate_availability": "weather through origin week only; assumes available at that origin",
        "hyperparameter_selection": "none; exploratory fixed alpha=1, recent4 trend, three current-week climate inputs",
        "case_sha256": rolling.base.file_hash(cases_path),
        "climate_sha256": rolling.base.file_hash(climate_path),
        "baseline_predictions_sha256": rolling.base.file_hash(args.baseline_results / "rolling_predictions.csv"),
        "runner_sha256": rolling.base.file_hash(__file__),
        "dependency_runner_sha256": rolling.base.file_hash(rolling.__file__),
    }
    (args.out_dir / "run_configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows, fitted = [], []
    for n, horizons in rolling.folds(config["initial_train_weeks"]):
        for h in horizons:
            for model, use_climate in (("ARTrend", False), ("ARTrendClimate", True)):
                result = forecast_one(y, weather, n, h, use_climate)
                origin, target = data.iloc[n - 1], data.iloc[n + h - 1]
                rows.append({"model": model, "origin_n": n, "horizon": h,
                             "prediction": result["prediction"], "raw_prediction": result["raw_prediction"],
                             "observed": float(target.probable_cases),
                             "origin_week": origin.epi_year_week, "origin_week_end": origin.week_end,
                             "target_week": target.epi_year_week, "target_week_start": target.week_start,
                             "error": result["prediction"] - float(target.probable_cases)})
                fitted.append({"model": model, "origin_n": n, "horizon": h, **result})
    predictions = pd.DataFrame(rows)
    baseline = pd.read_csv(args.baseline_results / "rolling_predictions.csv")
    combined = pd.concat([baseline, predictions], ignore_index=True)
    metrics = rolling.score(combined, int(np.argmax(y)) + 1)
    pairs = []
    for (scope, horizon), group in metrics.groupby(["scope", "horizon"]):
        without = group.loc[group.model.eq("ARTrend")].iloc[0]
        with_climate = group.loc[group.model.eq("ARTrendClimate")].iloc[0]
        pairs.append({"scope": scope, "horizon": int(horizon), "n_forecasts": int(without.n_forecasts),
                      "without_climate_mae": float(without.mae), "with_climate_mae": float(with_climate.mae),
                      "without_climate_rmse": float(without.rmse), "with_climate_rmse": float(with_climate.rmse),
                      "mae_reduction_pct": 100 * (1 - with_climate.mae / without.mae),
                      "rmse_reduction_pct": 100 * (1 - with_climate.rmse / without.rmse)})
    pairs = pd.DataFrame(pairs)
    predictions.to_csv(args.out_dir / "climate_screen_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.out_dir / "all_model_metrics.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(args.out_dir / "paired_climate_gain.csv", index=False, encoding="utf-8-sig")
    (args.out_dir / "fold_fits.json").write_text(json.dumps(fitted, indent=2, allow_nan=False), encoding="utf-8")
    summary = {**config, "status": "completed", "forecast_rows": len(predictions),
               "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
               "minimum_training_pairs": min(r["training_pairs"] for r in fitted),
               "negative_predictions_clipped": int(predictions.raw_prediction.lt(0).sum()),
               "paired_metrics": pairs.to_dict("records"),
               "warnings": [
                   "Exploratory development data, not an independent test or a final model decision.",
                   "Only this fixed linear feature specification is tested, not all potential climate mechanisms.",
                   "Same-week climate availability is assumed; historical release vintages are unavailable.",
                   "Few early training pairs and correlated climate features limit coefficient interpretation.",
                   "No causal attribution or statistical significance claim; no uncertainty intervals.",
               ]}
    (args.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    config["status"] = "completed"
    (args.out_dir / "run_configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(pairs.loc[pairs.scope.eq("all_available")].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    run(parser.parse_args())
