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

FREE_NAMES = ["beta_h", "beta_v", "N_eff", "E_h0", "I_h0", "i_v0"]
FIXED_NAMES = ["beta_h", "beta_v", "N_eff", "i_v0"]

FREE_LOWER = np.array([1.0e-4, 1.0e-4, 3.0e5, 1.0, 1.0, 1.0e-8], dtype=float)
FREE_UPPER = np.array([5.0, 5.0, 212_583_750.0, 200_000.0, 200_000.0, 0.2], dtype=float)
FIXED_LOWER = np.array([1.0e-4, 1.0e-4, 3.0e5, 1.0e-8], dtype=float)
FIXED_UPPER = np.array([5.0, 5.0, 212_583_750.0, 0.2], dtype=float)


@dataclass(frozen=True)
class FitData:
    frame: pd.DataFrame
    observed: np.ndarray
    t_eval: np.ndarray
    scale: float


def load_fit_data() -> FitData:
    data = pd.read_csv(DATA_PATH)
    data = data.sort_values("t_week").reset_index(drop=True)
    observed = data["probable_cases"].astype(float).to_numpy()
    # Compare the i-th weekly observation with the model value at the end of that week.
    t_eval = np.arange(1, len(observed) + 1, dtype=float) * 7.0
    scale = max(float(np.max(observed)), 1.0)
    return FitData(frame=data, observed=observed, t_eval=t_eval, scale=scale)


def parameter_names(scenario: str) -> list[str]:
    return FREE_NAMES if scenario == "free" else FIXED_NAMES


def bounds(scenario: str) -> tuple[np.ndarray, np.ndarray]:
    lower = FREE_LOWER if scenario == "free" else FIXED_LOWER
    upper = FREE_UPPER if scenario == "free" else FIXED_UPPER
    return np.log(lower), np.log(upper)


def unpack(log_params: np.ndarray, scenario: str) -> dict[str, float]:
    values = np.exp(log_params)
    return dict(zip(parameter_names(scenario), values, strict=True))


def rhs(_t: float, y: np.ndarray, beta_h: float, beta_v: float) -> list[float]:
    s_h, e_h, i_h, r_h, s_v, e_v, i_v = y
    i_h_force = max(i_h, 0.0)
    i_v_force = max(i_v, 0.0)
    s_v_force = max(s_v, 0.0)

    lambda_h = beta_h * i_v_force
    lambda_v = beta_v * i_h_force

    ds_h = -lambda_h * s_h
    de_h = lambda_h * s_h - SIGMA_H * e_h
    di_h = SIGMA_H * e_h - GAMMA_H * i_h
    dr_h = GAMMA_H * i_h

    ds_v = MU_V - lambda_v * s_v_force - MU_V * s_v
    de_v = lambda_v * s_v_force - (SIGMA_V + MU_V) * e_v
    di_v = SIGMA_V * e_v - MU_V * i_v
    return [ds_h, de_h, di_h, dr_h, ds_v, de_v, di_v]


def initial_counts(params: dict[str, float], scenario: str, fixed_eh0: float, fixed_ih0: float) -> tuple[float, float]:
    if scenario == "free":
        return params["E_h0"], params["I_h0"]
    return fixed_eh0, fixed_ih0


def simulate(
    log_params: np.ndarray,
    fit_data: FitData,
    scenario: str,
    fixed_eh0: float,
    fixed_ih0: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    params = unpack(log_params, scenario)
    beta_h = params["beta_h"]
    beta_v = params["beta_v"]
    n_eff = params["N_eff"]
    e_h0_count, i_h0_count = initial_counts(params, scenario, fixed_eh0, fixed_ih0)
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
        ],
        dtype=float,
    )

    solution = solve_ivp(
        fun=lambda t, y: rhs(t, y, beta_h=beta_h, beta_v=beta_v),
        t_span=(0.0, float(fit_data.t_eval[-1])),
        y0=y0,
        t_eval=fit_data.t_eval,
        method="DOP853",
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    if not solution.success or solution.y.shape[1] != len(fit_data.t_eval):
        return None

    states = solution.y.T
    fitted_cases = Q_REPORT * n_eff * states[:, 2]
    if not np.all(np.isfinite(fitted_cases)) or not np.all(np.isfinite(states)):
        return None
    if np.min(fitted_cases) < -1.0:
        return None
    return np.maximum(fitted_cases, 0.0), states


def residuals(
    log_params: np.ndarray,
    fit_data: FitData,
    scenario: str,
    fixed_eh0: float,
    fixed_ih0: float,
) -> np.ndarray:
    simulated = simulate(log_params, fit_data, scenario, fixed_eh0, fixed_ih0)
    if simulated is None:
        return np.full_like(fit_data.observed, 1.0e6, dtype=float)
    predicted, _states = simulated
    return (predicted - fit_data.observed) / fit_data.scale


def latin_hypercube_starts(n_starts: int, scenario: str, seed: int) -> np.ndarray:
    if n_starts <= 0:
        return np.empty((0, len(parameter_names(scenario))))
    log_lower, log_upper = bounds(scenario)
    if qmc is not None:
        sampler = qmc.LatinHypercube(d=len(log_lower), seed=seed)
        unit = sampler.random(n=n_starts)
    else:
        rng = np.random.default_rng(seed)
        unit = rng.random((n_starts, len(log_lower)))
    return log_lower + unit * (log_upper - log_lower)


def build_starting_points(n_starts: int, scenario: str, seed: int) -> np.ndarray:
    if scenario == "free":
        anchors = np.array(
            [
                [0.0441487783, 1.6014907851, 324_980.0, 29.84, 1.0, 1.0e-8],
                [0.0960236924, 0.5485787093, 338_286.0, 189.0, 63.0, 1.0e-8],
                [0.10, 0.50, 350_000.0, 200.0, 500.0, 1.0e-7],
                [0.25, 0.25, 500_000.0, 500.0, 1_000.0, 1.0e-6],
                [0.50, 0.10, 1_000_000.0, 1_000.0, 2_000.0, 1.0e-6],
                [0.10, 1.50, 1_000_000.0, 100.0, 500.0, 1.0e-7],
            ],
            dtype=float,
        )
    else:
        anchors = np.array(
            [
                [0.0960236924, 0.5485787093, 338_286.0, 1.0e-8],
                [0.10, 0.50, 350_000.0, 1.0e-7],
                [0.25, 0.25, 500_000.0, 1.0e-6],
                [0.50, 0.10, 1_000_000.0, 1.0e-6],
                [0.10, 1.50, 1_000_000.0, 1.0e-7],
                [0.05, 1.00, 500_000.0, 1.0e-8],
            ],
            dtype=float,
        )
    log_lower, log_upper = bounds(scenario)
    random_count = max(n_starts - len(anchors), 0)
    starts = latin_hypercube_starts(random_count, scenario, seed)
    combined = np.vstack([np.log(anchors[: min(len(anchors), n_starts)]), starts])
    return np.clip(combined, log_lower, log_upper)


def compute_r0(beta_h: float, beta_v: float) -> float:
    return math.sqrt(beta_h * beta_v * SIGMA_V / (GAMMA_H * MU_V * (SIGMA_V + MU_V)))


def fit_one(
    start_id: int,
    start: np.ndarray,
    fit_data: FitData,
    scenario: str,
    fixed_eh0: float,
    fixed_ih0: float,
    max_nfev: int,
) -> dict[str, object]:
    log_lower, log_upper = bounds(scenario)
    result = least_squares(
        fun=lambda x: residuals(x, fit_data, scenario, fixed_eh0, fixed_ih0),
        x0=start,
        bounds=(log_lower, log_upper),
        method="trf",
        x_scale="jac",
        ftol=1.0e-9,
        xtol=1.0e-9,
        gtol=1.0e-9,
        max_nfev=max_nfev,
    )

    params = unpack(result.x, scenario)
    simulated = simulate(result.x, fit_data, scenario, fixed_eh0, fixed_ih0)
    if simulated is None:
        predicted = np.full_like(fit_data.observed, np.nan, dtype=float)
        states = np.full((len(fit_data.t_eval), 7), np.nan, dtype=float)
        residual = predicted - fit_data.observed
        sse = math.inf
        rmse = math.inf
        mae = math.inf
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
        min_state = float(np.min(states))
        max_human_mass_error = float(np.max(np.abs(np.sum(states[:, :4], axis=1) - 1.0)))
        max_vector_mass_error = float(np.max(np.abs(np.sum(states[:, 4:7], axis=1) - 1.0)))
        fit_success = bool(result.success)

    e_h0_count, i_h0_count = initial_counts(params, scenario, fixed_eh0, fixed_ih0)
    start_params = unpack(start, scenario)
    bound_distance = np.minimum(result.x - log_lower, log_upper - result.x)
    bound_hit = bound_distance < 1.0e-4

    row: dict[str, object] = {
        "start_id": start_id,
        "scenario": scenario,
        "success": fit_success,
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost_scaled": float(result.cost),
        "loss_sse": sse,
        "rmse": rmse,
        "mae": mae,
        "R0": compute_r0(params["beta_h"], params["beta_v"]),
        "beta_product": params["beta_h"] * params["beta_v"],
        "beta_ratio_h_over_v": params["beta_h"] / params["beta_v"],
        "E_h0": e_h0_count,
        "I_h0": i_h0_count,
        "min_state": min_state,
        "max_human_mass_error": max_human_mass_error,
        "max_vector_mass_error": max_vector_mass_error,
        "any_bound_hit": bool(np.any(bound_hit)),
    }
    for name in parameter_names(scenario):
        row[name] = params[name]
        row[f"start_{name}"] = start_params[name]
    for name, hit in zip(parameter_names(scenario), bound_hit, strict=True):
        row[f"{name}_bound_hit"] = bool(hit)
    row["_x"] = result.x
    row["_jac"] = result.jac
    return row


def fit_multistart(
    fit_data: FitData,
    scenario: str,
    starts: int,
    seed: int,
    fixed_eh0: float,
    fixed_ih0: float,
    max_nfev: int,
) -> pd.DataFrame:
    start_points = build_starting_points(starts, scenario, seed)
    rows = []
    for start_id, start in enumerate(start_points, start=1):
        row = fit_one(start_id, start, fit_data, scenario, fixed_eh0, fixed_ih0, max_nfev)
        rows.append(row)
        print(
            f"{scenario} start {start_id:03d}/{len(start_points):03d}: "
            f"L={row['loss_sse']:.3f}, beta_h={row['beta_h']:.6g}, "
            f"beta_v={row['beta_v']:.6g}, N_eff={row['N_eff']:.3f}, "
            f"E_h0={row['E_h0']:.3f}, I_h0={row['I_h0']:.3f}, "
            f"i_v0={row['i_v0']:.6g}, R0={row['R0']:.6g}, success={row['success']}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["loss_sse", "cost_scaled"], ascending=True).reset_index(drop=True)


def correlation_from_jacobian(jac: np.ndarray, scenario: str) -> tuple[float, float]:
    singular_values = np.linalg.svd(jac, compute_uv=False)
    positive_sv = singular_values[singular_values > 1.0e-12]
    condition_number = float(positive_sv[0] / positive_sv[-1]) if len(positive_sv) else math.inf
    try:
        covariance = np.linalg.pinv(jac.T @ jac)
        diagonal = np.sqrt(np.diag(covariance))
        corr = covariance / np.outer(diagonal, diagonal)
        corr_beta_h_beta_v = float(corr[0, 1])
    except Exception:
        corr_beta_h_beta_v = math.nan
    return condition_number, corr_beta_h_beta_v


def save_outputs(
    results: pd.DataFrame,
    fit_data: FitData,
    scenario: str,
    fixed_eh0: float,
    fixed_ih0: float,
    starts: int,
    tag: str,
) -> dict[str, Path]:
    MAIN_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    prefix = f"reference_infection_cases_{tag}"
    clean_results = results.drop(columns=["_x", "_jac"])
    multistart_path = FIT_RESULT_DIR / f"{prefix}_multistart_results_v01.csv"
    clean_results.to_csv(multistart_path, index=False, encoding="utf-8-sig")

    best = results.iloc[0]
    best_log_params = np.asarray(best["_x"], dtype=float)
    predicted, states = simulate(best_log_params, fit_data, scenario, fixed_eh0, fixed_ih0)
    residual = predicted - fit_data.observed

    weekly_fit = fit_data.frame.copy()
    weekly_fit["observed_I_h_i"] = fit_data.observed
    weekly_fit["fitted_I_h_i"] = predicted
    weekly_fit["residual"] = residual
    weekly_path = MAIN_RESULT_DIR / f"{prefix}_weekly_fit_v01.csv"
    weekly_fit.to_csv(weekly_path, index=False, encoding="utf-8-sig")

    parameter_rows = [
        {"parameter": "beta_h", "estimate": best["beta_h"], "unit": "day^-1", "role": "estimated"},
        {"parameter": "beta_v", "estimate": best["beta_v"], "unit": "day^-1", "role": "estimated"},
        {"parameter": "beta_h_beta_v", "estimate": best["beta_product"], "unit": "day^-2", "role": "derived"},
        {"parameter": "beta_h_over_beta_v", "estimate": best["beta_ratio_h_over_v"], "unit": "dimensionless", "role": "derived"},
        {"parameter": "N_eff", "estimate": best["N_eff"], "unit": "persons", "role": "estimated"},
        {"parameter": "E_h0", "estimate": best["E_h0"], "unit": "persons", "role": "estimated" if scenario == "free" else "fixed"},
        {"parameter": "I_h0", "estimate": best["I_h0"], "unit": "persons", "role": "estimated" if scenario == "free" else "fixed"},
        {"parameter": "i_v0", "estimate": best["i_v0"], "unit": "proportion", "role": "estimated"},
        {"parameter": "sigma_h", "estimate": SIGMA_H, "unit": "day^-1", "role": "fixed"},
        {"parameter": "gamma_h", "estimate": GAMMA_H, "unit": "day^-1", "role": "fixed"},
        {"parameter": "sigma_v", "estimate": SIGMA_V, "unit": "day^-1", "role": "fixed"},
        {"parameter": "mu_v", "estimate": MU_V, "unit": "day^-1", "role": "fixed"},
        {"parameter": "q", "estimate": Q_REPORT, "unit": "dimensionless", "role": "fixed"},
        {"parameter": "R0", "estimate": best["R0"], "unit": "dimensionless", "role": "derived"},
    ]
    parameter_path = MAIN_RESULT_DIR / f"{prefix}_parameter_estimates_v01.csv"
    pd.DataFrame(parameter_rows).to_csv(parameter_path, index=False, encoding="utf-8-sig")

    condition_number, corr_beta_h_beta_v = correlation_from_jacobian(np.asarray(best["_jac"], dtype=float), scenario)
    near_best = clean_results[clean_results["loss_sse"] <= float(best["loss_sse"]) * 1.01]
    diagnostics = pd.DataFrame(
        [
            {"metric": "scenario", "value": scenario},
            {"metric": "n_weeks", "value": len(fit_data.observed)},
            {"metric": "n_starts", "value": starts},
            {"metric": "best_start_id", "value": best["start_id"]},
            {"metric": "best_loss_sse", "value": best["loss_sse"]},
            {"metric": "best_rmse", "value": best["rmse"]},
            {"metric": "best_mae", "value": best["mae"]},
            {"metric": "best_R0", "value": best["R0"]},
            {"metric": "best_beta_product", "value": best["beta_product"]},
            {"metric": "best_beta_ratio_h_over_v", "value": best["beta_ratio_h_over_v"]},
            {"metric": "near_best_count_1pct_sse", "value": len(near_best)},
            {"metric": "jacobian_condition_number_log_space", "value": condition_number},
            {"metric": "approx_corr_log_beta_h_log_beta_v", "value": corr_beta_h_beta_v},
            {"metric": "min_state_best", "value": best["min_state"]},
            {"metric": "max_human_mass_error_best", "value": best["max_human_mass_error"]},
            {"metric": "max_vector_mass_error_best", "value": best["max_vector_mass_error"]},
            {"metric": "any_bound_hit_best", "value": best["any_bound_hit"]},
            {"metric": "observed_mean", "value": float(np.mean(fit_data.observed))},
            {"metric": "fitted_mean", "value": float(np.mean(predicted))},
            {"metric": "observed_peak", "value": float(np.max(fit_data.observed))},
            {"metric": "fitted_peak", "value": float(np.max(predicted))},
        ]
    )
    diagnostics_path = MAIN_RESULT_DIR / f"{prefix}_diagnostics_v01.csv"
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")

    state_columns = ["s_h", "e_h", "i_h", "r_h", "s_v", "e_v", "i_v"]
    state_frame = pd.DataFrame(states, columns=state_columns)
    state_frame.insert(0, "t_day", fit_data.t_eval)
    state_path = FIT_RESULT_DIR / f"{prefix}_state_trajectory_v01.csv"
    state_frame.to_csv(state_path, index=False, encoding="utf-8-sig")

    x = np.arange(1, len(fit_data.observed) + 1)
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    ax.plot(x, fit_data.observed, color="#1f2937", linewidth=1.8, marker="o", markersize=3.8, label="Observed probable cases")
    ax.plot(x, predicted, color="#2563eb", linewidth=2.2, label="Fitted infection cases")
    ax.set_xlabel("Epidemiological week index")
    ax.set_ylabel("Weekly infection cases")
    ax.set_title(f"Reference-style least-squares fit: {tag}")
    tick_idx = np.arange(0, len(x), 6)
    ax.set_xticks(x[tick_idx])
    ax.set_xticklabels(weekly_fit.loc[tick_idx, "epi_year_week"], rotation=45, ha="right")
    ax.grid(True, axis="y", color="#d1d5db", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    figure_path = FIGURE_DIR / f"{prefix}_weekly_fit_v01.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    paths = {
        "parameters": parameter_path,
        "weekly_fit": weekly_path,
        "diagnostics": diagnostics_path,
        "multistart": multistart_path,
        "states": state_path,
        "figure": figure_path,
    }
    print("\nSaved outputs:")
    for path in paths.values():
        print(f"- {path}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit reference-style weekly infection cases with SEIR-SEI model.")
    parser.add_argument("--scenario", choices=["free", "fixed"], required=True)
    parser.add_argument("--starts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--fixed-eh0", type=float, default=189.0)
    parser.add_argument("--fixed-ih0", type=float, default=63.0)
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.tag
    if tag is None:
        tag = "free_initial_dual" if args.scenario == "free" else "fixed_eh189_ih63_dual"

    fit_data = load_fit_data()
    print(f"Loaded {len(fit_data.observed)} weekly observations from {DATA_PATH}")
    print("Loss: L(theta)=sum_i (fitted_I_h(i,theta)-observed_I_h(i))^2")
    print(f"Scenario: {args.scenario}; tag: {tag}")
    if args.scenario == "fixed":
        print(f"Fixed E_h0={args.fixed_eh0:.0f}, I_h0={args.fixed_ih0:.0f}")

    results = fit_multistart(
        fit_data=fit_data,
        scenario=args.scenario,
        starts=args.starts,
        seed=args.seed,
        fixed_eh0=args.fixed_eh0,
        fixed_ih0=args.fixed_ih0,
        max_nfev=args.max_nfev,
    )
    save_outputs(
        results=results,
        fit_data=fit_data,
        scenario=args.scenario,
        fixed_eh0=args.fixed_eh0,
        fixed_ih0=args.fixed_ih0,
        starts=args.starts,
        tag=tag,
    )


if __name__ == "__main__":
    main()
