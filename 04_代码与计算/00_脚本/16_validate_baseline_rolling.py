"""Expanding-window development validation using the existing A2/A4 simulators."""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import qmc

HORIZONS = (1, 2, 4)
INITIAL_TRAIN = 30
MODEL_FILES = {"A2": "08_fit_dual_transmission_baseline_model.py",
               "A4": "09_fit_dual_fixed_host_initial_model.py"}
FIXED_INITIALS = (189.0, 63.0)
KEYS = ["model", "origin_n", "horizon"]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=2)
def load_model(path):
    name = "rolling_" + Path(path).stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_starts(module, model, observed, count, seed):
    if count < 3:
        raise ValueError("At least three starts required")
    # Generic seeds depend only on the training prefix, never full-window fits.
    n = float(np.clip(2 * observed.sum(), 1.1 * module.LOWER[2], 0.9 * module.UPPER[2]))
    initial = max(float(observed[0]) / 2, 1.1)
    seeds = []
    for bh, bv in ((0.2, 0.2), (0.3, 0.1), (0.1, 0.3)):
        values = ([bh, bv, n, initial, initial, 1e-4] if model == "A2"
                  else [bh, bv, n, 1e-4])
        seeds.append(np.clip(np.log(values), module.LOG_LOWER + 1e-8, module.LOG_UPPER - 1e-8))
    if count > 3:
        unit = qmc.LatinHypercube(d=len(module.PARAMETER_NAMES), seed=seed).random(count - 3)
        seeds.extend(module.LOG_LOWER + unit * (module.LOG_UPPER - module.LOG_LOWER))
    return np.asarray(seeds)


def training_data(module, observed):
    values = np.asarray(observed, dtype=float).copy()
    return module.FitData(
        frame=pd.DataFrame({"probable_cases": values}), observed=values,
        week_edges=np.arange(len(values) + 1, dtype=float) * 7,
        scale=max(float(values.max()), 1.0),
    )


def simulate(module, model, x, edges):
    if model == "A2":
        return module.simulate(x, edges)
    return module.simulate(x, edges, *FIXED_INITIALS)


def fit_task(model_path, model, observed, horizons, seed, starts, max_nfev, refine_nfev):
    start_time = time.perf_counter()
    module = load_model(model_path)
    fit_data = training_data(module, observed)
    n = len(observed)
    initial = make_starts(module, model, fit_data.observed, starts, seed)
    runs = []
    for start_id, x in enumerate(initial, 1):
        extra = FIXED_INITIALS if model == "A4" else ()
        runs.append(module.fit_one(start_id, x, fit_data, max_nfev, *extra))
    finite = [r for r in runs if np.isfinite(r["sse"])]
    if not finite:
        raise RuntimeError(f"No finite training fit: {model}/{n}")
    best = min(finite, key=lambda r: r["sse"])
    extra = FIXED_INITIALS if model == "A4" else ()
    refined = module.fit_one(starts + 1, best["_x"], fit_data, refine_nfev, *extra)
    runs.append(refined)
    best = min((r for r in runs if np.isfinite(r["sse"])), key=lambda r: r["sse"])
    edges = np.arange(n + max(horizons) + 1, dtype=float) * 7
    predicted, states = simulate(module, model, best["_x"], edges)
    if (not np.isfinite(predicted).all() or states[:, :7].min() < -1e-7
            or np.abs(states[:, :4].sum(axis=1) - 1).max() > 1e-7
            or np.abs(states[:, 4:7].sum(axis=1) - 1).max() > 1e-7):
        raise RuntimeError(f"Invalid forecast trajectory: {model}/{n}")
    training_predicted, _ = simulate(module, model, best["_x"], fit_data.week_edges)
    prefix_difference = float(np.abs(training_predicted - predicted[:n]).max())
    np.testing.assert_allclose(training_predicted, predicted[:n], rtol=1e-5, atol=0.05)
    near = [r for r in runs if np.isfinite(r["sse"]) and r["sse"] <= best["sse"] * 1.01]
    clean = [{k: v for k, v in row.items() if not k.startswith("_")} for row in runs]
    diagnostics = {k: v for k, v in best.items() if not k.startswith("_")}
    diagnostics.update({
        "origin_n": n, "model": model, "seed": seed, "training_scale": fit_data.scale,
        "starts_run": len(runs), "converged_starts": sum(bool(r["success"]) for r in runs),
        "near_best_starts_1pct": len(near),
        "forecast_training_prefix_max_difference": prefix_difference,
        "forecast_min_state": float(states[:, :7].min()),
        "elapsed_seconds": time.perf_counter() - start_time,
    })
    for row in clean:
        row.update(model=model, origin_n=n)
    forecasts = [{"model": model, "origin_n": n, "horizon": h,
                  "prediction": float(predicted[n + h - 1]),
                  "fit_converged": bool(best["success"])} for h in horizons]
    return diagnostics, clean, forecasts


def fold_definitions(length=63):
    return [(n, tuple(h for h in HORIZONS if n + h <= length))
            for n in range(INITIAL_TRAIN, length)]


def score_predictions(predictions):
    rows = []
    for scope, frame in (("all_available", predictions),
                         ("common_origins", predictions.loc[predictions.origin_n.le(59)])):
        for (model, horizon), group in frame.groupby(["model", "horizon"]):
            error = group.prediction.to_numpy() - group.observed.to_numpy()
            rows.append({"scope": scope, "model": model, "horizon": int(horizon),
                         "n_forecasts": len(group), "mae": float(np.abs(error).mean()),
                         "rmse": float(np.sqrt(np.square(error).mean())),
                         "mean_error": float(error.mean()),
                         "sse": float(np.square(error).sum())})
    metrics = pd.DataFrame(rows)
    for index, row in metrics.iterrows():
        naive = metrics.loc[metrics.scope.eq(row.scope) & metrics.horizon.eq(row.horizon)
                            & metrics.model.eq("Naive")].iloc[0]
        metrics.loc[index, "mae_improvement_vs_naive_pct"] = 100 * (1 - row.mae / naive.mae)
        metrics.loc[index, "rmse_improvement_vs_naive_pct"] = 100 * (1 - row.rmse / naive.rmse)
    return metrics


def plot_predictions(predictions, output_dir):
    colors = {"A2": "#087e8b", "A4": "#c45128", "Naive": "#777777"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, horizon in zip(axes, HORIZONS, strict=True):
        part = predictions.loc[predictions.horizon.eq(horizon)]
        truth = part.loc[part.model.eq("Naive")]
        ax.scatter(pd.to_datetime(truth.target_week_start), truth.observed,
                   s=22, color="#202020", label="Observed", zorder=5)
        for model in ("A2", "A4", "Naive"):
            group = part.loc[part.model.eq(model)]
            ax.plot(pd.to_datetime(group.target_week_start), group.prediction,
                    color=colors[model], linewidth=1.6,
                    linestyle="--" if model == "Naive" else "-", label=model)
        ax.set_title(f"{horizon}-week ahead | {len(truth)} forecast origins", loc="left", fontsize=11)
        ax.set_ylabel("Weekly new observations")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=4, frameon=False)
    axes[-1].set_xlabel("Target week start date")
    fig.suptitle("Development-period rolling forecasts (no climate inputs)", fontsize=13)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "rolling_forecasts_v01.png", dpi=180)
    fig.savefig(output_dir / "rolling_forecasts_v01.pdf")
    plt.close(fig)


def run(args):
    start_time = time.perf_counter()
    data_path = args.project_root / "02_数据" / "01_清洗数据" / "baseline_fitting_cases_2023W35_2024W45.csv"
    source_dir = args.project_root / "04_代码与计算" / "00_脚本"
    data = pd.read_csv(data_path).sort_values("t_week").reset_index(drop=True)
    assert len(data) == 63 and data.t_week.tolist() == list(range(63))
    starts = pd.to_datetime(data.week_start)
    assert starts.diff().dropna().dt.days.eq(7).all() and starts.dt.dayofweek.eq(6).all()
    assert (pd.to_datetime(data.week_end) - starts).dt.days.eq(6).all()
    assert data.iloc[0].epi_year_week == "2023-W35"
    assert data.iloc[-1].epi_year_week == "2024-W45"
    observed = data.probable_cases.to_numpy(float)
    assert np.isfinite(observed).all() and (observed >= 0).all()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "run_summary.json").exists():
        raise FileExistsError("Use a new output directory; existing run summary is protected")
    configuration = {
        "decision_id": "dev_rolling_30_1_2_4_20261007", "date": "2026-10-07",
        "initial_train_weeks": INITIAL_TRAIN, "horizons": list(HORIZONS),
        "models": ["A2", "A4", "Naive"], "A4_fixed_initials": list(FIXED_INITIALS),
        "seed": args.seed, "starts_per_fit": args.starts, "max_nfev": args.max_nfev,
        "refine_nfev": args.refine_nfev, "workers": args.workers,
        "input_sha256": file_hash(data_path),
        "model_source_sha256": {m: file_hash(source_dir / f) for m, f in MODEL_FILES.items()},
        "runner_sha256": file_hash(__file__),
        "selection": "lowest finite training SSE across starts and one best-start refinement",
        "initialization": "3 generic training-only seeds + log-space Latin hypercube; no saved fit anchors",
        "data_availability": "revised snapshots; retrospective development validation, not real-time",
        "status": "running",
    }
    (args.out_dir / "run_configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    diagnostics, multistart, forecasts = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        for n, horizons in fold_definitions():
            for model, filename in MODEL_FILES.items():
                seed = args.seed + 100 * n + (2 if model == "A2" else 4)
                future = pool.submit(fit_task, str(source_dir / filename), model,
                                     observed[:n].copy(), horizons, seed, args.starts,
                                     args.max_nfev, args.refine_nfev)
                pending[future] = (model, n)
        for future in as_completed(pending):
            diag, trials, prediction = future.result()
            diagnostics.append(diag)
            multistart.extend(trials)
            forecasts.extend(prediction)
            print(f"{len(diagnostics):02d}/66 {diag['model']} n={diag['origin_n']} "
                  f"training_SSE={diag['sse']:.3f} converged={diag['success']} "
                  f"seconds={diag['elapsed_seconds']:.1f}", flush=True)
            pd.DataFrame(diagnostics).sort_values(["model", "origin_n"]).to_csv(
                args.out_dir / "rolling_fits_v01.csv", index=False, encoding="utf-8-sig")
    for n, horizons in fold_definitions():
        for h in horizons:
            forecasts.append({"model": "Naive", "origin_n": n, "horizon": h,
                              "prediction": float(observed[n - 1]), "fit_converged": True})
    predictions = pd.DataFrame(forecasts).sort_values(KEYS).reset_index(drop=True)
    for index, row in predictions.iterrows():
        origin = data.iloc[int(row.origin_n) - 1]
        target = data.iloc[int(row.origin_n + row.horizon) - 1]
        predictions.loc[index, "origin_week"] = origin.epi_year_week
        predictions.loc[index, "origin_week_end"] = origin.week_end
        predictions.loc[index, "target_week"] = target.epi_year_week
        predictions.loc[index, "target_week_start"] = target.week_start
        predictions.loc[index, "observed"] = target.probable_cases
    predictions["error"] = predictions.prediction - predictions.observed
    assert len(predictions) == 285 and not predictions.duplicated(KEYS).any()
    assert np.isfinite(predictions.prediction).all() and predictions.prediction.ge(0).all()
    assert (pd.to_datetime(predictions.target_week_start) >
            pd.to_datetime(predictions.origin_week_end)).all()
    counts = predictions.groupby(["model", "horizon"]).size()
    for model in configuration["models"]:
        assert [counts[model, h] for h in HORIZONS] == [33, 32, 30]
    metrics = score_predictions(predictions)
    predictions.to_csv(args.out_dir / "rolling_predictions_v01.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.out_dir / "rolling_metrics_v01.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(multistart).sort_values(["model", "origin_n", "start_id"]).to_csv(
        args.out_dir / "rolling_multistart_v01.csv", index=False, encoding="utf-8-sig")
    fits = pd.DataFrame(diagnostics)
    plot_predictions(predictions, args.out_dir)
    peak_index = int(np.argmax(observed))
    warnings = [
        "All 63 weeks previously influenced exploratory decisions; not an untouched final test.",
        "Historical parameter bounds and model structures retained; those choices predate this validation.",
        "Main observed peak is before the first forecast origin; no evidence of main-peak early warning.",
        "No forecast intervals; overlapping errors are not independent.",
        "Training-only residual scale does not change the unweighted SSE minimizer.",
        "Finite local multi-start optimization cannot certify global optimality or parameter identifiability.",
    ]
    summary = {
        **configuration, "status": "completed", "elapsed_seconds": time.perf_counter() - start_time,
        "python": sys.version.split()[0], "numpy": np.__version__,
        "scipy": scipy.__version__, "pandas": pd.__version__,
        "fitted_origins": len(fits), "forecast_rows": len(predictions),
        "nonconverged_selected_fits": int((~fits.success).sum()),
        "selected_fits_with_bound_hits": int(fits.any_bound_hit.sum()),
        "fit_diagnostics_by_model": fits.groupby("model").agg(
            origins=("origin_n", "size"), converged=("success", "sum"),
            bound_hits=("any_bound_hit", "sum")).reset_index().to_dict(orient="records"),
        "observed_peak_week_number": peak_index + 1,
        "observed_peak_week": data.iloc[peak_index].epi_year_week,
        "observed_peak_count": float(observed[peak_index]),
        "max_forecast_training_prefix_difference": float(fits.forecast_training_prefix_max_difference.max()),
        "metrics": metrics.to_dict(orient="records"), "warnings": warnings,
    }
    (args.out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    configuration["status"] = "completed"
    (args.out_dir / "run_configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    print(metrics.loc[metrics.scope.eq("all_available")].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--max-nfev", type=int, default=250)
    parser.add_argument("--refine-nfev", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())
