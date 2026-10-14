from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

try:
    from scipy.stats import qmc
except Exception:  # pragma: no cover
    qmc = None


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "02_数据" / "01_清洗数据" / "baseline_fitting_cases_2023W35_2024W45.csv"
MAIN_RESULT_DIR = ROOT / "05_结果与图表" / "00_主要结果"
FIT_RESULT_DIR = ROOT / "05_结果与图表" / "01_拟合结果"
FIGURE_DIR = ROOT / "05_结果与图表" / "03_导出图片"

SIGMA_H = 1.0 / 3.0
GAMMA_H = 1.0 / 6.0
SIGMA_V = 1.0 / 11.0
MU_V = 1.0 / 14.0
Q_REPORT = 1.0

PARAMETER_NAMES = ["beta", "N_eff", "E_h0", "I_h0", "i_v0"]
LOWER = np.array([1.0e-4, 3.0e5, 1.0, 1.0, 1.0e-8], dtype=float)
UPPER = np.array([5.0, 212_583_750.0, 100_000.0, 100_000.0, 0.2], dtype=float)
LOG_LOWER = np.log(LOWER)
LOG_UPPER = np.log(UPPER)


@dataclass(frozen=True)
class FitData:
    frame: pd.DataFrame
    observed: np.ndarray
    week_edges: np.ndarray
    scale: float


def load_fit_data() -> FitData:
    data = pd.read_csv(DATA_PATH)
    data = data.sort_values("t_week").reset_index(drop=True)
    observed = data["probable_cases"].astype(float).to_numpy()
    week_edges = np.arange(len(observed) + 1, dtype=float) * 7.0
    scale = max(float(np.max(observed)), 1.0)
    return FitData(frame=data, observed=observed, week_edges=week_edges, scale=scale)


def unpack(log_params: np.ndarray) -> dict[str, float]:
    values = np.exp(log_params)
    return dict(zip(PARAMETER_NAMES, values, strict=True))


def rhs(_t: float, y: np.ndarray, beta: float, n_eff: float) -> list[float]:
    s_h, e_h, i_h, r_h, s_v, e_v, i_v, a_h = y
    i_h_force = max(i_h, 0.0)
    i_v_force = max(i_v, 0.0)
    s_v_force = max(s_v, 0.0)

    lambda_h = beta * i_v_force
    lambda_v = beta * i_h_force

    ds_h = -lambda_h * s_h
    de_h = lambda_h * s_h - SIGMA_H * e_h
    di_h = SIGMA_H * e_h - GAMMA_H * i_h
    dr_h = GAMMA_H * i_h

    ds_v = MU_V - lambda_v * s_v_force - MU_V * s_v
    de_v = lambda_v * s_v_force - (SIGMA_V + MU_V) * e_v
    di_v = SIGMA_V * e_v - MU_V * i_v

    da_h = SIGMA_H * e_h
    return [ds_h, de_h, di_h, dr_h, ds_v, de_v, di_v, da_h]


def simulate(log_params: np.ndarray, week_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    params = unpack(log_params)
    beta = params["beta"]
    n_eff = params["N_eff"]
    e_h0_count = params["E_h0"]
    i_h0_count = params["I_h0"]
    i_v0 = params["i_v0"]

    s_h0_count = n_eff - e_h0_count - i_h0_count
    s_v0 = 1.0 - i_v0
    if s_h0_count <= 0.0 or s_v0 < 0.0:
        return None

    y0 = np.array(
        [
            s_h0_count / n_eff,
            e_h0_count / n_eff,
            i_h0_count / n_eff,
            0.0,
            s_v0,
            0.0,
            i_v0,
            0.0,
        ],
        dtype=float,
    )

    solution = solve_ivp(
        fun=lambda t, y: rhs(t, y, beta=beta, n_eff=n_eff),
        t_span=(float(week_edges[0]), float(week_edges[-1])),
        y0=y0,
        t_eval=week_edges,
        method="DOP853",
        rtol=1.0e-8,
        atol=1.0e-10,
    )

    if not solution.success or solution.y.shape[1] != len(week_edges):
        return None

    states = solution.y.T
    weekly_cases = Q_REPORT * n_eff * np.diff(states[:, 7])

    if not np.all(np.isfinite(weekly_cases)) or not np.all(np.isfinite(states)):
        return None
    if np.min(weekly_cases) < -1.0:
        return None

    return np.maximum(weekly_cases, 0.0), states


def residuals(log_params: np.ndarray, fit_data: FitData) -> np.ndarray:
    simulated = simulate(log_params, fit_data.week_edges)
    if simulated is None:
        return np.full_like(fit_data.observed, 1.0e6, dtype=float)
    predicted, _states = simulated
    return (predicted - fit_data.observed) / fit_data.scale


def latin_hypercube_starts(n_starts: int, seed: int) -> np.ndarray:
    if n_starts <= 0:
        return np.empty((0, len(PARAMETER_NAMES)))
    if qmc is not None:
        sampler = qmc.LatinHypercube(d=len(PARAMETER_NAMES), seed=seed)
        unit = sampler.random(n=n_starts)
    else:
        rng = np.random.default_rng(seed)
        unit = rng.random((n_starts, len(PARAMETER_NAMES)))
    return LOG_LOWER + unit * (LOG_UPPER - LOG_LOWER)


def build_starting_points(n_starts: int, seed: int) -> np.ndarray:
    anchors = np.array(
        [
            [0.20, 2_000_000.0, 1_000.0, 500.0, 1.0e-4],
            [0.35, 5_000_000.0, 2_000.0, 1_000.0, 1.0e-5],
            [0.15, 1_000_000.0, 500.0, 250.0, 1.0e-3],
            [0.50, 10_000_000.0, 5_000.0, 2_000.0, 1.0e-4],
        ],
        dtype=float,
    )
    anchors = np.log(anchors)
    random_count = max(n_starts - len(anchors), 0)
    starts = latin_hypercube_starts(random_count, seed)
    combined = np.vstack([anchors[: min(len(anchors), n_starts)], starts])
    return np.clip(combined, LOG_LOWER, LOG_UPPER)


def fit_one(start_id: int, start: np.ndarray, fit_data: FitData, max_nfev: int) -> dict[str, object]:
    result = least_squares(
        fun=lambda x: residuals(x, fit_data),
        x0=start,
        bounds=(LOG_LOWER, LOG_UPPER),
        method="trf",
        x_scale="jac",
        ftol=1.0e-9,
        xtol=1.0e-9,
        gtol=1.0e-9,
        max_nfev=max_nfev,
    )

    params = unpack(result.x)
    simulated = simulate(result.x, fit_data.week_edges)
    if simulated is None:
        sse = math.inf
        rmse = math.inf
        mae = math.inf
        predicted = np.full_like(fit_data.observed, np.nan, dtype=float)
        states = np.full((len(fit_data.week_edges), 8), np.nan, dtype=float)
        min_state = math.nan
        max_human_mass_error = math.nan
        max_vector_mass_error = math.nan
        fit_success = False
    else:
        predicted, states = simulated
        residual = predicted - fit_data.observed
        sse = float(np.sum(residual**2))
        rmse = float(np.sqrt(np.mean(residual**2)))
        mae = float(np.mean(np.abs(residual)))
        min_state = float(np.min(states[:, :7]))
        max_human_mass_error = float(np.max(np.abs(np.sum(states[:, :4], axis=1) - 1.0)))
        max_vector_mass_error = float(np.max(np.abs(np.sum(states[:, 4:7], axis=1) - 1.0)))
        fit_success = bool(result.success)

    start_params = unpack(start)
    bound_distance = np.minimum(result.x - LOG_LOWER, LOG_UPPER - result.x)
    bound_hit = bound_distance < 1.0e-4
    r0 = params["beta"] * math.sqrt(SIGMA_V / (GAMMA_H * MU_V * (SIGMA_V + MU_V)))

    row: dict[str, object] = {
        "start_id": start_id,
        "success": fit_success,
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost_scaled": float(result.cost),
        "sse": sse,
        "rmse": rmse,
        "mae": mae,
        "R0": r0,
        "min_state": min_state,
        "max_human_mass_error": max_human_mass_error,
        "max_vector_mass_error": max_vector_mass_error,
        "any_bound_hit": bool(np.any(bound_hit)),
    }
    for name in PARAMETER_NAMES:
        row[name] = params[name]
        row[f"start_{name}"] = start_params[name]
    for name, hit in zip(PARAMETER_NAMES, bound_hit, strict=True):
        row[f"{name}_bound_hit"] = bool(hit)
    row["_x"] = result.x
    row["_jac"] = result.jac
    return row


def fit_multistart(fit_data: FitData, n_starts: int, seed: int, max_nfev: int) -> pd.DataFrame:
    starts = build_starting_points(n_starts, seed)
    rows = []
    for start_id, start in enumerate(starts, start=1):
        row = fit_one(start_id=start_id, start=start, fit_data=fit_data, max_nfev=max_nfev)
        rows.append(row)
        print(
            f"start {start_id:03d}/{len(starts):03d}: "
            f"sse={row['sse']:.3f}, beta={row['beta']:.6g}, "
            f"N_eff={row['N_eff']:.3f}, R0={row['R0']:.6g}, success={row['success']}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["sse", "cost_scaled"], ascending=True).reset_index(drop=True)


def save_outputs(results: pd.DataFrame, fit_data: FitData, args: argparse.Namespace) -> None:
    MAIN_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    clean_results = results.drop(columns=["_x", "_jac"])
    multistart_path = FIT_RESULT_DIR / "constrained_baseline_multistart_results_v01.csv"
    clean_results.to_csv(multistart_path, index=False, encoding="utf-8-sig")

    best = results.iloc[0]
    best_log_params = np.asarray(best["_x"], dtype=float)
    predicted, states = simulate(best_log_params, fit_data.week_edges)
    residual = predicted - fit_data.observed

    weekly_fit = fit_data.frame.copy()
    weekly_fit["model_cases"] = predicted
    weekly_fit["residual"] = residual
    weekly_fit["cumulative_observed"] = np.cumsum(fit_data.observed)
    weekly_fit["cumulative_model"] = np.cumsum(predicted)
    weekly_path = MAIN_RESULT_DIR / "constrained_baseline_weekly_fit_v01.csv"
    weekly_fit.to_csv(weekly_path, index=False, encoding="utf-8-sig")

    parameter_rows = [
        {"parameter": "beta", "estimate": best["beta"], "unit": "day^-1", "role": "constrained beta_h=beta_v"},
        {"parameter": "N_eff", "estimate": best["N_eff"], "unit": "persons", "role": "effective surveillance-scale host population"},
        {"parameter": "E_h0", "estimate": best["E_h0"], "unit": "persons", "role": "initial exposed hosts"},
        {"parameter": "I_h0", "estimate": best["I_h0"], "unit": "persons", "role": "initial infectious hosts"},
        {"parameter": "i_v0", "estimate": best["i_v0"], "unit": "proportion", "role": "initial infectious vector proportion"},
        {"parameter": "sigma_h", "estimate": SIGMA_H, "unit": "day^-1", "role": "fixed"},
        {"parameter": "gamma_h", "estimate": GAMMA_H, "unit": "day^-1", "role": "fixed"},
        {"parameter": "sigma_v", "estimate": SIGMA_V, "unit": "day^-1", "role": "fixed"},
        {"parameter": "mu_v", "estimate": MU_V, "unit": "day^-1", "role": "fixed"},
        {"parameter": "q", "estimate": Q_REPORT, "unit": "dimensionless", "role": "fixed"},
        {"parameter": "R0", "estimate": best["R0"], "unit": "dimensionless", "role": "derived"},
    ]
    param_path = MAIN_RESULT_DIR / "constrained_baseline_parameter_estimates_v01.csv"
    pd.DataFrame(parameter_rows).to_csv(param_path, index=False, encoding="utf-8-sig")

    jac = np.asarray(best["_jac"], dtype=float)
    singular_values = np.linalg.svd(jac, compute_uv=False)
    positive_sv = singular_values[singular_values > 1.0e-12]
    condition_number = float(positive_sv[0] / positive_sv[-1]) if len(positive_sv) else math.inf
    near_best = clean_results[clean_results["sse"] <= float(best["sse"]) * 1.01]
    diagnostics = pd.DataFrame(
        [
            {"metric": "n_weeks", "value": len(fit_data.observed)},
            {"metric": "n_starts", "value": args.starts},
            {"metric": "best_start_id", "value": best["start_id"]},
            {"metric": "best_sse", "value": best["sse"]},
            {"metric": "best_rmse", "value": best["rmse"]},
            {"metric": "best_mae", "value": best["mae"]},
            {"metric": "best_R0", "value": best["R0"]},
            {"metric": "near_best_count_1pct_sse", "value": len(near_best)},
            {"metric": "jacobian_condition_number_log_space", "value": condition_number},
            {"metric": "min_state_best", "value": best["min_state"]},
            {"metric": "max_human_mass_error_best", "value": best["max_human_mass_error"]},
            {"metric": "max_vector_mass_error_best", "value": best["max_vector_mass_error"]},
            {"metric": "any_bound_hit_best", "value": best["any_bound_hit"]},
            {"metric": "observed_total", "value": float(np.sum(fit_data.observed))},
            {"metric": "model_total", "value": float(np.sum(predicted))},
        ]
    )
    diag_path = MAIN_RESULT_DIR / "constrained_baseline_diagnostics_v01.csv"
    diagnostics.to_csv(diag_path, index=False, encoding="utf-8-sig")

    state_columns = ["s_h", "e_h", "i_h", "r_h", "s_v", "e_v", "i_v", "a_h"]
    state_frame = pd.DataFrame(states, columns=state_columns)
    state_frame.insert(0, "t_day", fit_data.week_edges)
    state_path = FIT_RESULT_DIR / "constrained_baseline_state_trajectory_v01.csv"
    state_frame.to_csv(state_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    x = np.arange(len(fit_data.observed))
    ax.plot(x, fit_data.observed, color="#1f2937", linewidth=1.8, marker="o", markersize=3.8, label="Observed probable cases")
    ax.plot(x, predicted, color="#2563eb", linewidth=2.2, label="Constrained SEIR-SEI fit")
    ax.set_xlabel("Epidemiological week")
    ax.set_ylabel("Weekly probable cases")
    ax.set_title("Brazil chikungunya weekly probable cases: constrained baseline fit")
    tick_idx = np.arange(0, len(x), 6)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(weekly_fit.loc[tick_idx, "epi_year_week"], rotation=45, ha="right")
    ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    figure_path = FIGURE_DIR / "constrained_baseline_weekly_fit_v01.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    print("\nSaved outputs:")
    print(f"- {param_path}")
    print(f"- {weekly_path}")
    print(f"- {diag_path}")
    print(f"- {multistart_path}")
    print(f"- {state_path}")
    print(f"- {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the constrained beta_h=beta_v SEIR-SEI baseline model.")
    parser.add_argument("--starts", type=int, default=100, help="Number of multistart initial points.")
    parser.add_argument("--seed", type=int, default=20260825, help="Random seed for Latin hypercube starts.")
    parser.add_argument("--max-nfev", type=int, default=350, help="Maximum function evaluations per start.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit_data = load_fit_data()
    print(f"Loaded {len(fit_data.observed)} weekly observations from {DATA_PATH}")
    print(f"Integration interval: 0 to {fit_data.week_edges[-1]:.0f} days")
    results = fit_multistart(fit_data=fit_data, n_starts=args.starts, seed=args.seed, max_nfev=args.max_nfev)
    save_outputs(results=results, fit_data=fit_data, args=args)


if __name__ == "__main__":
    main()
