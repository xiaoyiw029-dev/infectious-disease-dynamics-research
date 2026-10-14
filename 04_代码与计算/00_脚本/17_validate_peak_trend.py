"""Earlier-origin retrospective validation, preserving the original 30-week run."""

import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import importlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

base = importlib.import_module("16_validate_baseline_rolling")
MODELS = ("A2", "A4", "Naive", "Trend4")


def reference_simulation(module, model, x, edges, max_step):
    from scipy.integrate import solve_ivp

    parameters = module.unpack(x)
    n = parameters["N_eff"]
    e0, i0 = ((parameters["E_h0"], parameters["I_h0"]) if model == "A2"
              else base.FIXED_INITIALS)
    iv = parameters["i_v0"]
    y0 = [(n - e0 - i0) / n, e0 / n, i0 / n, 0, 1 - iv, 0, iv, 0]
    result = solve_ivp(
        lambda t, y: module.rhs(t, y, parameters["beta_h"], parameters["beta_v"]),
        (float(edges[0]), float(edges[-1])), y0, t_eval=edges, method="DOP853",
        rtol=1e-11, atol=1e-14, max_step=max_step,
    )
    if not result.success:
        raise RuntimeError("Independent precision check failed")
    states = result.y.T
    return module.Q_REPORT * n * np.diff(states[:, 7]), states


def precision_check(module, model, x, training, forecasts):
    n = len(training)
    scale = max(float(np.max(training)), 1.0)
    edges = np.arange(n + len(forecasts) + 1) * 7.0
    full, states = base.simulate(module, model, x, edges)
    trained, _ = base.simulate(module, model, x, np.arange(n + 1) * 7.0)
    reference, reference_states = reference_simulation(module, model, x, edges, 1.0)
    refined, _ = reference_simulation(module, model, x, edges, 0.5)
    difference = float(np.max(np.abs(reference - refined)))
    error = float(np.max(np.abs(np.r_[trained, forecasts] - refined)))
    ref_sse = float(np.square(refined[:n] - training).sum())
    original_sse = float(np.square(trained - training).sum())
    relative_sse_change = abs(ref_sse - original_sse) / max(original_sse, 1.0)
    result = {
        "reference_step_halving_max_difference": difference,
        "original_vs_reference_max_difference": error,
        "reference_sse_relative_difference": relative_sse_change,
        "forecast_training_prefix_max_difference": float(np.abs(trained - full[:n]).max()),
        "forecast_min_state": float(states[:, :7].min()),
        "reference_min_state": float(reference_states[:, :7].min()),
        "precision_count_tolerance": max(.05, .001 * scale),
    }
    # Check errors against a converged independent trajectory, not two adaptive end times.
    if difference > max(1e-5, 1e-6 * scale) or error > result["precision_count_tolerance"]:
        raise RuntimeError(f"Insufficient integration accuracy: {model}/{n}: {result}")
    if relative_sse_change > .005:
        raise RuntimeError(f"Numerical training-objective error too large: {model}/{n}: {result}")
    for s in (states, reference_states):
        if (not np.isfinite(s).all() or s[:, :7].min() < -1e-7
                or np.abs(s[:, :4].sum(axis=1) - 1).max() > 1e-7
                or np.abs(s[:, 4:7].sum(axis=1) - 1).max() > 1e-7):
            raise RuntimeError("Numerical state constraints failed")
    return result


def checked_fit_task(model_path, model, observed, horizons, seed):
    started = time.perf_counter()
    module = base.load_model(model_path)
    from scipy.integrate import solve_ivp

    def accurate_solver(*args, **kwargs):
        kwargs.update(rtol=1e-10, atol=1e-13)
        return solve_ivp(*args, **kwargs)

    # Process-local numerical refinement; original source files remain unchanged.
    module.solve_ivp = accurate_solver
    data = base.training_data(module, observed)
    extra = base.FIXED_INITIALS if model == "A4" else ()
    runs = [module.fit_one(i, x, data, 250, *extra) for i, x in enumerate(
        base.make_starts(module, model, data.observed, 12, seed), 1)]
    best = min((r for r in runs if np.isfinite(r["sse"])), key=lambda r: r["sse"])
    runs.append(module.fit_one(13, best["_x"], data, 1000, *extra))
    best = min((r for r in runs if np.isfinite(r["sse"])), key=lambda r: r["sse"])
    n = len(observed)
    predictions, _ = base.simulate(module, model, best["_x"],
                                  np.arange(n + max(horizons) + 1) * 7.0)
    audit = precision_check(module, model, best["_x"], observed, predictions[n:])
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]
    diag = {k: v for k, v in best.items() if not k.startswith("_")}
    diag.update(audit)
    diag.update(model=model, origin_n=n, seed=seed, training_scale=data.scale,
                starts_run=13, converged_starts=sum(bool(r["success"]) for r in runs),
                near_best_starts_1pct=sum(r["sse"] <= best["sse"] * 1.01 for r in runs),
                effective_rtol=1e-10, effective_atol=1e-13,
                elapsed_seconds=time.perf_counter() - started)
    for row in clean:
        row.update(model=model, origin_n=n)
    forecasts = [{"model": model, "origin_n": n, "horizon": h,
                  "prediction": float(predictions[n + h - 1]),
                  "fit_converged": bool(best["success"])} for h in horizons]
    return diag, clean, forecasts


def trend_forecast(training, horizon):
    values = np.asarray(training, dtype=float)
    if len(values) < 4 or horizon not in base.HORIZONS or not np.isfinite(values).all():
        raise ValueError("Four finite observations and a supported horizon required")
    recent = values[-4:]
    centered_x = np.arange(4, dtype=float) - 1.5
    slope = float(centered_x @ (recent - recent.mean()) / (centered_x @ centered_x))
    raw = float(recent.mean() + slope * (1.5 + horizon))
    return max(0.0, raw), raw, slope


def folds(initial, length=63):
    if not 4 <= initial < 30:
        raise ValueError("Supplemental initial length must be between 4 and 29")
    return [(n, tuple(h for h in base.HORIZONS if n + h <= length))
            for n in range(initial, length)]


def verify_reuse(project, old):
    source = project / "04_代码与计算" / "00_脚本"
    data = project / "02_数据" / "01_清洗数据" / "baseline_fitting_cases_2023W35_2024W45.csv"
    summary = json.loads((old / "run_summary.json").read_text(encoding="utf-8"))
    expected = {"status": "completed", "initial_train_weeks": 30, "horizons": [1, 2, 4],
                "seed": 20261007, "starts_per_fit": 12, "max_nfev": 250,
                "refine_nfev": 1000, "A4_fixed_initials": [189.0, 63.0]}
    for key, value in expected.items():
        if summary[key] != value:
            raise ValueError(f"Incompatible prior setting: {key}")
    if summary["input_sha256"] != base.file_hash(data):
        raise ValueError("Input changed; cannot reuse prior fits")
    for model, filename in base.MODEL_FILES.items():
        if summary["model_source_sha256"][model] != base.file_hash(source / filename):
            raise ValueError(f"Model changed: {model}")
    if summary["runner_sha256"] != base.file_hash(base.__file__):
        raise ValueError("Prior fitting implementation changed")
    return summary


def score(predictions, peak_week):
    targets = predictions.origin_n + predictions.horizon
    subsets = {
        "all_available": predictions,
        "common_origins": predictions.loc[predictions.origin_n.le(59)],
        "original_origins_30plus": predictions.loc[predictions.origin_n.ge(30)],
        "added_origins_before30": predictions.loc[predictions.origin_n.lt(30)],
        "targets_through_observed_peak": predictions.loc[targets.le(peak_week)],
        "targets_after_observed_peak": predictions.loc[targets.gt(peak_week)],
        "last10_target_weeks": predictions.loc[targets.ge(54)],
    }
    rows = []
    for scope, subset in subsets.items():
        for (model, h), group in subset.groupby(["model", "horizon"]):
            error = group.prediction.to_numpy() - group.observed.to_numpy()
            rows.append({"scope": scope, "model": model, "horizon": int(h),
                         "n_forecasts": len(group), "mae": float(np.abs(error).mean()),
                         "rmse": float(np.sqrt(np.square(error).mean())),
                         "mean_error": float(error.mean()), "sse": float(error @ error)})
    result = pd.DataFrame(rows)
    for i, row in result.iterrows():
        for reference in ("Naive", "Trend4"):
            ref = result.loc[result.scope.eq(row.scope) & result.horizon.eq(row.horizon)
                             & result.model.eq(reference)].iloc[0]
            for metric in ("mae", "rmse"):
                result.loc[i, f"{metric}_improvement_vs_{reference}_pct"] = (
                    100 * (1 - row[metric] / ref[metric]) if ref[metric] else np.nan)
    return result


def plot(predictions, out, peak_date):
    colors = {"A2": "#087e8b", "A4": "#c45128", "Naive": "#777777", "Trend4": "#7151a4"}
    for zoom, name in ((False, "expanded_forecasts"), (True, "peak_forecasts")):
        fig, axes = base.plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for ax, horizon in zip(axes, base.HORIZONS, strict=True):
            part = predictions.loc[predictions.horizon.eq(horizon)]
            if zoom:
                part = part.loc[(part.origin_n + part.horizon).le(34)]
            truth = part.loc[part.model.eq("Naive")]
            ax.scatter(pd.to_datetime(truth.target_week_start), truth.observed,
                       color="#202020", s=23, label="Observed", zorder=5)
            for model in MODELS:
                group = part.loc[part.model.eq(model)]
                ax.plot(pd.to_datetime(group.target_week_start), group.prediction,
                        color=colors[model], label=model, linewidth=1.5,
                        linestyle="--" if model in ("Naive", "Trend4") else "-")
            ax.axvline(pd.Timestamp(peak_date), color="#555555", linestyle=":", linewidth=1)
            ax.set_title(f"{horizon}-week ahead | {len(truth)} target weeks", loc="left", fontsize=11)
            ax.set_ylabel("Weekly new observations")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(ncol=5, frameon=False)
        axes[-1].set_xlabel("Target week start date")
        fig.suptitle("Retrospective development forecasts | dotted line: observed peak", fontsize=12)
        fig.autofmt_xdate()
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(out / f"{name}.png", dpi=180)
        fig.savefig(out / f"{name}.pdf")
        base.plt.close(fig)


def run(args):
    started = time.perf_counter()
    project, out = args.project_root, args.out_dir
    source = project / "04_代码与计算" / "00_脚本"
    data_path = project / "02_数据" / "01_清洗数据" / "baseline_fitting_cases_2023W35_2024W45.csv"
    old = project / "05_结果与图表" / "04_滚动验证" / "development_rolling_v01"
    previous = verify_reuse(project, old)
    data = pd.read_csv(data_path).sort_values("t_week").reset_index(drop=True)
    assert len(data) == 63 and data.t_week.tolist() == list(range(63))
    dates = pd.to_datetime(data.week_start)
    assert dates.diff().dropna().dt.days.eq(7).all() and dates.dt.dayofweek.eq(6).all()
    y = data.probable_cases.to_numpy(float)
    assert np.isfinite(y).all() and (y >= 0).all()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_configuration.json").exists():
        raise FileExistsError("Existing run protected; choose another output directory")
    reused_files = ("run_summary.json", "rolling_predictions_v01.csv",
                    "rolling_fits_v01.csv", "rolling_multistart_v01.csv")
    config = {
        "date": "2026-10-07", "status": "running",
        "scope": "supplemental retrospective development validation; no climate",
        "authorization": "User requested earlier origins and a simple trend comparator",
        "setting_status": "20-week start and 4-week trend proposed; not claimed as human-confirmed optimum",
        "initial_train_weeks": args.initial_train, "horizons": list(base.HORIZONS),
        "trend_window": 4, "trend_rule": "OLS in raw counts; max(0, fitted endpoint + h*slope)",
        "models": list(MODELS), "input_sha256": base.file_hash(data_path),
        "model_source_sha256": previous["model_source_sha256"],
        "base_runner_sha256": base.file_hash(base.__file__),
        "runner_sha256": base.file_hash(__file__),
        "reused_files_sha256": {f: base.file_hash(old / f) for f in reused_files},
        "seed": previous["seed"], "starts_per_fit": 12, "max_nfev": 250, "refine_nfev": 1000,
        "A4_fixed_initials": [189, 63], "workers": args.workers,
        "numerical_audit": {
            "new_fit_rtol": 1e-10, "new_fit_atol": 1e-13,
            "reused_fit_rtol": 1e-8, "reused_fit_atol": 1e-10,
            "reference_rtol": 1e-11, "reference_atol": 1e-14, "reference_max_steps_days": [1, .5],
            "step_halving_tolerance": "max(1e-5, 1e-6 * training_max)",
            "original_count_tolerance": "max(.05, .001 * training_max)",
            "training_sse_relative_tolerance": .005,
            "action_if_failed": "abort; no successful result or silent exclusion",
            "reused_fits_and_predictions": "unchanged if independent numerical audit passes",
            "new_fits": "same equations, bounds, starts and SSE, with tighter numerical integration",
        },
    }
    (out / "run_configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    old_predictions = pd.read_csv(old / "rolling_predictions_v01.csv")
    assert len(old_predictions) == 285
    assert not old_predictions.duplicated(base.KEYS).any()
    diagnostics = pd.read_csv(old / "rolling_fits_v01.csv").to_dict("records")
    multistart = pd.read_csv(old / "rolling_multistart_v01.csv").to_dict("records")
    forecasts = old_predictions.loc[old_predictions.model.isin(("A2", "A4")),
                                   base.KEYS + ["prediction", "fit_converged"]].to_dict("records")
    for row in diagnostics:
        row["fit_source"] = "reused_30week_run"
        row["effective_rtol"] = 1e-8
        row["effective_atol"] = 1e-10
    total = 2 * (30 - args.initial_train)
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = []
        for n, hs in folds(args.initial_train):
            if n >= 30:
                continue
            for model, filename in base.MODEL_FILES.items():
                seed = config["seed"] + 100 * n + (2 if model == "A2" else 4)
                pending.append(pool.submit(checked_fit_task, str(source / filename), model,
                                           y[:n].copy(), hs, seed))
        for future in as_completed(pending):
            diag, trials, predicted = future.result()
            diag["fit_source"] = "new_earlier_origin"
            diagnostics.append(diag)
            multistart.extend(trials)
            forecasts.extend(predicted)
            completed += 1
            print(f"{completed}/{total} {diag['model']} n={diag['origin_n']} "
                  f"SSE={diag['sse']:.2f} converged={diag['success']}", flush=True)
            pd.DataFrame(diagnostics).sort_values(["model", "origin_n"]).to_csv(
                out / "rolling_fits.csv", index=False, encoding="utf-8-sig")
    for diag in diagnostics:
        if diag["fit_source"] != "reused_30week_run":
            continue
        model, n = diag["model"], int(diag["origin_n"])
        module = base.load_model(str(source / base.MODEL_FILES[model]))
        x = np.log([diag[name] for name in module.PARAMETER_NAMES])
        maximum_horizon = max(h for h in base.HORIZONS if n + h <= len(y))
        original, _ = base.simulate(module, model, x, np.arange(n + maximum_horizon + 1) * 7.0)
        diag.update(precision_check(module, model, x, y[:n], original[n:]))
    pd.DataFrame(diagnostics).sort_values(["model", "origin_n"]).to_csv(
        out / "rolling_fits.csv", index=False, encoding="utf-8-sig")
    for n, hs in folds(args.initial_train):
        for h in hs:
            forecasts.append({"model": "Naive", "origin_n": n, "horizon": h,
                              "prediction": float(y[n - 1]), "fit_converged": True})
            predicted, raw, slope = trend_forecast(y[:n], h)
            forecasts.append({"model": "Trend4", "origin_n": n, "horizon": h,
                              "prediction": predicted, "raw_prediction": raw,
                              "trend_slope_per_week": slope, "fit_converged": True})
    predictions = pd.DataFrame(forecasts).sort_values(base.KEYS).reset_index(drop=True)
    for i, row in predictions.iterrows():
        origin = data.iloc[int(row.origin_n) - 1]
        target = data.iloc[int(row.origin_n + row.horizon) - 1]
        for column, value in {"origin_week": origin.epi_year_week,
                              "origin_week_end": origin.week_end,
                              "target_week": target.epi_year_week,
                              "target_week_start": target.week_start,
                              "observed": float(target.probable_cases)}.items():
            predictions.loc[i, column] = value
    predictions["error"] = predictions.prediction - predictions.observed
    expected = sum(len(hs) for _, hs in folds(args.initial_train)) * len(MODELS)
    assert len(predictions) == expected and not predictions.duplicated(base.KEYS).any()
    assert np.isfinite(predictions.prediction).all() and predictions.prediction.ge(0).all()
    assert (pd.to_datetime(predictions.target_week_start) >
            pd.to_datetime(predictions.origin_week_end)).all()
    peak = int(np.argmax(y)) + 1
    metrics = score(predictions, peak)
    peak_predictions = predictions.loc[(predictions.origin_n + predictions.horizon).eq(peak)].copy()
    peak_predictions["relative_error_pct"] = 100 * peak_predictions.error / peak_predictions.observed
    predictions.to_csv(out / "rolling_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(out / "rolling_metrics.csv", index=False, encoding="utf-8-sig")
    peak_predictions.to_csv(out / "observed_peak_target.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(multistart).sort_values(["model", "origin_n", "start_id"]).to_csv(
        out / "rolling_multistart.csv", index=False, encoding="utf-8-sig")
    plot(predictions, out, data.iloc[peak - 1].week_start)
    fits = pd.DataFrame(diagnostics)
    summary = {**config, "status": "completed", "elapsed_seconds": time.perf_counter() - started,
               "python": sys.version.split()[0], "numpy": np.__version__,
               "scipy": base.scipy.__version__, "pandas": pd.__version__,
               "new_fits": total, "reused_fits": 66, "total_fits": len(fits),
               "forecast_rows": len(predictions), "multistart_rows": len(multistart),
               "observed_peak_week_number": peak, "observed_peak_week": data.iloc[peak - 1].epi_year_week,
               "observed_peak_count": float(y[peak - 1]),
               "nonconverged_selected_fits": int((~fits.success).sum()),
               "bound_hits": int(fits.any_bound_hit.sum()),
               "trend_clipped_predictions": int(predictions.raw_prediction.lt(0).sum()),
               "metrics": metrics.to_dict("records"),
               "warnings": [
                   "Previously explored development data and revised snapshots, not independent or real-time testing.",
                   "Observed-peak and tail subsets are posthoc diagnostics, not tuned selection criteria.",
                   "Peak-target errors do not quantify predicted peak timing or early-warning accuracy.",
                   "Historical parameter bounds remain unchanged; multi-start cannot certify identifiability.",
                   "Overlapping forecasts are dependent; no significance test or forecast intervals.",
                   "Climate benefit has not been estimated; future observed weather must not leak into forecasts.",
               ]}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    config["status"] = "completed"
    (out / "run_configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(metrics.loc[metrics.scope.eq("all_available")].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--initial-train", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())
