"""Training-only parameter diagnostics for previously saved rolling forecasts."""

import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import importlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

GRID = [300000., 500000., 1000000., 3000000., 10000000., 30000000., 100000000., 212583750.]
THRESHOLDS = [.001, .01, .05]


@lru_cache(maxsize=2)
def environment(project, model):
    source = Path(project) / "04_代码与计算/00_脚本"
    sys.path.insert(0, str(source)) if str(source) not in sys.path else None
    rolling = importlib.import_module("17_validate_peak_trend")
    module = rolling.base.load_model(str(source / rolling.base.MODEL_FILES[model]))
    def accurate(*args, **kwargs):
        kwargs.update(rtol=1e-10, atol=1e-13)
        return solve_ivp(*args, **kwargs)
    module.solve_ivp = accurate
    return rolling, module


def reference(project, model, x, weeks):
    rolling, module = environment(project, model)
    return rolling.reference_simulation(module, model, x, np.arange(weeks + 1) * 7., 1.)


def linear_growth(module, product):
    matrix = np.array([[-module.SIGMA_H, 0, 0, np.sqrt(product)],
                       [module.SIGMA_H, -module.GAMMA_H, 0, 0],
                       [0, np.sqrt(product), -(module.SIGMA_V + module.MU_V), 0],
                       [0, 0, module.SIGMA_V, -module.MU_V]])
    return float(np.linalg.eigvals(matrix).real.max())


def profile_task(project, model, n, observed, n_eff, selected):
    rolling, module = environment(project, model)
    free = np.array([i for i in range(len(module.PARAMETER_NAMES)) if i != 2])
    lower, upper = module.LOG_LOWER[free], module.LOG_UPPER[free]
    scale = max(float(np.max(observed)), 1.)
    def expand(z):
        x = np.empty(len(module.PARAMETER_NAMES))
        x[2], x[free] = np.log(n_eff), z
        return x
    def residual(z):
        simulation = rolling.base.simulate(module, model, expand(z), np.arange(n + 1) * 7.)
        if simulation is None:
            return np.full(n, 1e6)
        return (simulation[0] - observed) / scale
    best_x = np.log([selected[name] for name in module.PARAMETER_NAMES])
    generic = rolling.base.make_starts(module, model, observed, 3, 20261007)
    starts = [best_x[free], generic[0, free], generic[1, free]]
    attempts = []
    for z in starts:
        attempts.append(least_squares(residual, np.clip(z, lower + 1e-8, upper - 1e-8),
                                      bounds=(lower, upper), method="trf", x_scale="jac",
                                      ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=250))
    best = min(attempts, key=lambda r: float(r.fun @ r.fun))
    attempts.append(least_squares(residual, best.x, bounds=(lower, upper), method="trf",
                                  x_scale="jac", ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=1000))
    best = min(attempts, key=lambda r: float(r.fun @ r.fun))
    x = expand(best.x)
    prediction, states = reference(project, model, x, n + 4)
    fine, _ = rolling.reference_simulation(module, model, x, np.arange(n + 5) * 7., .5)
    discrepancy = float(np.max(np.abs(prediction - fine)))
    if discrepancy > max(1e-5, 1e-6 * scale):
        raise RuntimeError("Reference trajectory not converged under step halving")
    row = {"model": model, "origin_n": n, **module.unpack(x),
           "success": bool(best.success), "nfev_selected": best.nfev,
           "training_sse": float(np.square(prediction[:n] - observed).sum()),
           "optimizer_training_sse": float((best.fun @ best.fun) * scale ** 2),
           "forecast_4": float(prediction[n + 3]), "forecast_week27": float(prediction[26]),
           "susceptible_fraction_at_origin": float(states[n, 0]),
           "reference_step_halving_difference": discrepancy}
    trials = [{"model": model, "origin_n": n, "N_eff": n_eff, "start_id": i,
               "success": bool(r.success), "training_sse": float((r.fun @ r.fun) * scale ** 2),
               "nfev": r.nfev} for i, r in enumerate(attempts, 1)]
    return row, trials


def near_summary(trials):
    rows = []
    for (model, n), group in trials.groupby(["model", "origin_n"]):
        best = float(group.reference_training_sse.min())
        for tolerance in THRESHOLDS:
            near = group.loc[group.reference_training_sse.le(best * (1 + tolerance))]
            row = {"model": model, "origin_n": int(n), "sse_tolerance": tolerance,
                   "count_including_refinement": len(near), "converged_count": int(near.success.sum()),
                   "forecast4_min": float(near.forecast_4.min()), "forecast4_max": float(near.forecast_4.max()),
                   "forecast4_span": float(near.forecast_4.max() - near.forecast_4.min()),
                   "beta_product_min": float(near.beta_product.min()),
                   "beta_product_max": float(near.beta_product.max())}
            for name in ("beta_h", "beta_v", "N_eff", "beta_ratio"):
                row[name + "_max_over_min"] = float(near[name].max() / near[name].min())
            rows.append(row)
    return pd.DataFrame(rows)


def local_jacobian(project, model, n, x, training):
    _, module = environment(project, model)
    scale = max(float(training.max()), 1.)
    columns = []
    for j in range(len(x)):
        left, right = x.copy(), x.copy()
        left[j] = max(x[j] - 1e-4, module.LOG_LOWER[j])
        right[j] = min(x[j] + 1e-4, module.LOG_UPPER[j])
        a, _ = reference(project, model, left, n)
        b, _ = reference(project, model, right, n)
        columns.append((b - a) / ((right[j] - left[j]) * scale))
    jac = np.column_stack(columns)
    singular = np.linalg.svd(jac, compute_uv=False)
    cosine = float(jac[:, 0] @ jac[:, 1] / (np.linalg.norm(jac[:, 0]) * np.linalg.norm(jac[:, 1])))
    return {"model": model, "origin_n": n, "parameter_names": module.PARAMETER_NAMES,
            "singular_values": singular.tolist(), "condition_number": float(singular[0] / singular[-1]),
            "beta_sensitivity_column_cosine": cosine,
            "note": "Feasible finite differences of scaled training outputs in log parameters; not a confidence interval."}


def figures(fits, near, profiles, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = {"A2": "#087e8b", "A4": "#c45128"}
    for model in ("A2", "A4"):
        f = fits.loc[fits.model.eq(model)].sort_values("origin_n")
        axes[0, 0].plot(f.origin_n, f.N_eff, "o-", color=colors[model], label=model)
        axes[0, 1].plot(f.origin_n, f.beta_product, "o-", color=colors[model], label=model)
        part = near.loc[near.model.eq(model) & near.sse_tolerance.eq(.01)].sort_values("origin_n")
        ax = axes[1, 0] if model == "A2" else axes[1, 1]
        ax.fill_between(part.origin_n, part.forecast4_min, part.forecast4_max, color=colors[model],
                        alpha=.3, label="Within 1% training SSE (not a prediction interval)")
        ax.plot(f.origin_n, f.forecast_4, "o-", color=colors[model], label="Selected fit")
        ax.plot(f.origin_n, f.observed_4, "k.--", label="Observed target week")
        ax.set_title(f"{model}: 4-week forecasts")
        ax.set_xlabel("Training length (weeks)")
        ax.set_ylabel("Weekly new observations")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, frameon=False)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Effective population scale")
    axes[0, 0].set_title("Selected N_eff")
    axes[0, 1].set_title("Transmission-parameter product")
    axes[0, 1].set_ylabel("beta_h * beta_v (day^-2)")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.axvline(23, color="#888888", linestyle=":", linewidth=1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "parameter_forecast_diagnostics.png", dpi=180)
    fig.savefig(output / "parameter_forecast_diagnostics.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, model in enumerate(("A2", "A4")):
        for n, color in ((23, "#087e8b"), (24, "#c45128")):
            p = profiles.loc[profiles.model.eq(model) & profiles.origin_n.eq(n)].sort_values("N_eff")
            axes[0, col].semilogx(p.N_eff, p.relative_sse_increase_pct, "o-", color=color, label=f"Origin {n}")
            axes[1, col].semilogx(p.N_eff, p.forecast_week27, "o-", color=color, label=f"Origin {n}")
        axes[0, col].set_title(model)
        axes[0, col].set_ylabel("Training SSE increase vs saved fit (%)")
        axes[0, col].legend(frameon=False)
        axes[1, col].axhline(17479, color="#202020", linestyle="--", label="Observed week 27")
        axes[1, col].set_ylabel("Predicted week-27 count")
        axes[1, col].set_xlabel("Fixed N_eff (diagnostic only)")
        axes[1, col].set_ylim(bottom=0)
        axes[1, col].legend(frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "population_profile_diagnostics.png", dpi=180)
    fig.savefig(output / "population_profile_diagnostics.pdf")
    plt.close(fig)


def run(args):
    began = time.perf_counter()
    project, output = args.project_root, args.out_dir
    root = project / "05_结果与图表/04_滚动验证/development_peak_trend_v01"
    casefile = project / "02_数据/01_清洗数据/baseline_fitting_cases_2023W35_2024W45.csv"
    data = pd.read_csv(casefile).sort_values("t_week")
    y = data.probable_cases.to_numpy(float)
    source_fits = pd.read_csv(root / "rolling_fits.csv")
    source_trials = pd.read_csv(root / "rolling_multistart.csv")
    rolling, _ = environment(str(project), "A2")
    previous = json.loads((root / "run_summary.json").read_text())
    assert previous["input_sha256"] == rolling.base.file_hash(casefile)
    for model, name in rolling.base.MODEL_FILES.items():
        assert previous["model_source_sha256"][model] == rolling.base.file_hash(project / "04_代码与计算/00_脚本" / name)
    assert previous["runner_sha256"] == rolling.base.file_hash(rolling.__file__)
    assert len(y) == 63 and np.isfinite(y).all() and (y >= 0).all()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_configuration.json").exists():
        raise FileExistsError("Use a new output directory")
    config = {"date": "2026-10-07", "status": "running",
              "scope": "retrospective parameter and forecast diagnostics; no model change",
              "near_optimal_sse_thresholds": THRESHOLDS, "origins": list(range(20, 31)),
              "profile_origins": [23, 24], "profile_N_eff_grid": GRID,
              "profile_starts": 3, "profile_refinements": 1,
              "profile_max_nfev": 250, "profile_refine_nfev": 1000,
              "source_hashes": {p.name: rolling.base.file_hash(p) for p in
                                [casefile, root / "rolling_fits.csv", root / "rolling_multistart.csv",
                                 root / "run_summary.json"]},
              "runner_sha256": rolling.base.file_hash(__file__),
              "model_source_sha256": previous["model_source_sha256"],
              "reference_source_sha256": rolling.base.file_hash(rolling.__file__),
              "selection": "training SSE only; future outcomes used only for retrospective evaluation"}
    (output / "run_configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    trials = []
    selected = []
    for model in ("A2", "A4"):
        _, module = environment(str(project), model)
        for n in range(20, 31):
            saved = source_fits.loc[source_fits.model.eq(model) & source_fits.origin_n.eq(n)].iloc[0]
            group = source_trials.loc[source_trials.model.eq(model) & source_trials.origin_n.eq(n)]
            for row in group.to_dict("records"):
                x = np.log([row[name] for name in module.PARAMETER_NAMES])
                prediction, states = reference(str(project), model, x, max(n + 4, 27))
                record = {**row, "reference_training_sse": float(np.square(prediction[:n] - y[:n]).sum()),
                          "forecast_4": float(prediction[n + 3]), "observed_4": float(y[n + 3]),
                          "error_4": float(prediction[n + 3] - y[n + 3]),
                          "forecast_week27": float(prediction[26]),
                          "beta_product": row["beta_h"] * row["beta_v"],
                          "beta_ratio": row["beta_h"] / row["beta_v"],
                          "susceptible_fraction_at_origin": float(states[n, 0]),
                          "vector_susceptible_fraction_at_origin": float(states[n, 4]),
                          "early_linear_growth_per_day": linear_growth(module, row["beta_h"] * row["beta_v"])}
                trials.append(record)
                if int(row["start_id"]) == int(saved.start_id):
                    selected.append(record)
    trials, selected = pd.DataFrame(trials), pd.DataFrame(selected)
    assert len(trials) == 286 and len(selected) == 22
    near = near_summary(trials)
    trials.to_csv(output / "replayed_starts.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(output / "selected_origin_diagnostics.csv", index=False, encoding="utf-8-sig")
    near.to_csv(output / "near_optimal_spread.csv", index=False, encoding="utf-8-sig")
    print("Replayed 286 optimization results; beginning 32 fixed-N profiles", flush=True)
    profiles, attempts = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = []
        for model in ("A2", "A4"):
            for n in (23, 24):
                saved = source_fits.loc[source_fits.model.eq(model) & source_fits.origin_n.eq(n)].iloc[0].to_dict()
                for value in GRID:
                    pending.append(pool.submit(profile_task, str(project), model, n, y[:n].copy(), value, saved))
        for future in as_completed(pending):
            row, trial_rows = future.result()
            baseline = selected.loc[selected.model.eq(row["model"]) & selected.origin_n.eq(row["origin_n"])].iloc[0]
            row["saved_fit_reference_sse"] = float(baseline.reference_training_sse)
            row["relative_sse_increase_pct"] = 100 * (row["training_sse"] / baseline.reference_training_sse - 1)
            profiles.append(row)
            attempts.extend(trial_rows)
            print(f"{len(profiles)}/32 {row['model']} n={row['origin_n']} N={row['N_eff']:.0f} "
                  f"SSE increase={row['relative_sse_increase_pct']:.3f}%", flush=True)
    profiles = pd.DataFrame(profiles).sort_values(["model", "origin_n", "N_eff"])
    profiles.to_csv(output / "N_eff_profile.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(attempts).to_csv(output / "profile_attempts.csv", index=False, encoding="utf-8-sig")
    jacobians = []
    for row in selected.loc[selected.origin_n.isin([23, 24])].to_dict("records"):
        _, module = environment(str(project), row["model"])
        x = np.log([row[p] for p in module.PARAMETER_NAMES])
        jacobians.append(local_jacobian(str(project), row["model"], row["origin_n"], x, y[:row["origin_n"]]))
    (output / "local_identifiability.json").write_text(json.dumps(jacobians, indent=2), encoding="utf-8")
    figures(selected, near, profiles, output)
    summary = {**config, "status": "completed", "elapsed_seconds": time.perf_counter() - began,
               "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
               "scipy": rolling.base.scipy.__version__, "replayed_starts": len(trials),
               "profile_points": len(profiles), "profile_optimization_attempts": len(attempts),
               "profile_nonconverged_selected": int((~profiles.success).sum()),
               "max_reference_step_halving_difference": float(profiles.reference_step_halving_difference.max()),
               "origin23_near_optimal": near.loc[near.origin_n.eq(23)].to_dict("records"),
               "local_jacobians": jacobians,
               "warnings": ["Finite multi-start/profile searches do not prove global optimality.",
                            "Near-optimal solution ranges are not prediction or confidence intervals.",
                            "Observed data are revised development snapshots, not independent evaluation.",
                            "Profile N values are diagnostics, not new fixed-population decisions.",
                            "Climate variables and final constraints require project-owner approval."]}
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    config["status"] = "completed"
    (output / "run_configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(near.loc[near.origin_n.eq(23)].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())
