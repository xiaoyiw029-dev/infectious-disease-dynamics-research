"""
Fit a national baseline host-vector chikungunya model using least squares.

This script is a runnable skeleton once the cleaned weekly dataset exists.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "02_数据" / "01_清洗数据" / "brazil_weekly_model_dataset.csv"
RESULT_DIR = PROJECT_ROOT / "05_结果与图表" / "00_主要结果"


PARAM_FIXED = {
    "mu_h": 1.0 / (75.0 * 365.0),
    "sigma_h": 1.0 / 4.0,
    "gamma_h": 1.0 / 6.0,
    "sigma_v": 1.0 / 10.0,
    "mu_v": 1.0 / 20.0,
}


def rhs(t, y, params):
    s_h, e_h, i_h, r_h, s_v, e_v, i_v, a = y
    n_h = params["n_h"]
    beta_vh = params["beta_vh"]
    beta_hv = params["beta_hv"]
    mu_h = params["mu_h"]
    sigma_h = params["sigma_h"]
    gamma_h = params["gamma_h"]
    lambda_v = params["lambda_v"]
    sigma_v = params["sigma_v"]
    mu_v = params["mu_v"]

    force_h = beta_vh * i_v / n_h
    force_v = beta_hv * i_h / n_h

    ds_h = mu_h * n_h - force_h * s_h - mu_h * s_h
    de_h = force_h * s_h - (sigma_h + mu_h) * e_h
    di_h = sigma_h * e_h - (gamma_h + mu_h) * i_h
    dr_h = gamma_h * i_h - mu_h * r_h

    ds_v = lambda_v - force_v * s_v - mu_v * s_v
    de_v = force_v * s_v - (sigma_v + mu_v) * e_v
    di_v = sigma_v * e_v - mu_v * i_v
    da = sigma_h * e_h

    return [ds_h, de_h, di_h, dr_h, ds_v, de_v, di_v, da]


def simulate(theta, cases, population):
    beta_vh, beta_hv, rho, mosquito_ratio, e_h0_scale, i_h0_scale = np.exp(theta)

    n_h = float(population)
    i_h0 = max(float(cases[0]), 1.0) * i_h0_scale
    e_h0 = max(float(cases[0]), 1.0) * e_h0_scale
    r_h0 = 0.0
    s_h0 = max(n_h - e_h0 - i_h0 - r_h0, 1.0)

    n_v0 = mosquito_ratio * n_h
    i_v0 = 10.0
    e_v0 = 10.0
    s_v0 = max(n_v0 - e_v0 - i_v0, 1.0)
    a0 = 0.0

    params = {
        **PARAM_FIXED,
        "n_h": n_h,
        "beta_vh": beta_vh,
        "beta_hv": beta_hv,
        "lambda_v": PARAM_FIXED["mu_v"] * n_v0,
    }

    y0 = [s_h0, e_h0, i_h0, r_h0, s_v0, e_v0, i_v0, a0]
    weeks = len(cases)
    t_eval = np.arange(0, weeks * 7 + 1, 7)
    sol = solve_ivp(
        lambda t, y: rhs(t, y, params),
        t_span=(0, weeks * 7),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-6,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    cumulative = sol.y[7]
    weekly_incidence = np.diff(cumulative)
    reported = rho * weekly_incidence
    return np.maximum(reported, 0.0)


def residuals(theta, cases, population):
    predicted = simulate(theta, cases, population)
    return predicted - cases


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing cleaned dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "cases" not in df.columns:
        raise KeyError("Cleaned dataset must contain a 'cases' column.")

    population = float(df["population"].dropna().iloc[0]) if "population" in df.columns else 203_000_000
    cases = df["cases"].to_numpy(dtype=float)

    train_mask = df["year"].to_numpy() <= 2023 if "year" in df.columns else np.arange(len(df)) < int(len(df) * 0.75)
    train_cases = cases[train_mask]

    initial = np.log([0.2, 0.2, 0.2, 1.0, 3.0, 1.0])
    lower = np.log([1e-6, 1e-6, 1e-4, 1e-3, 0.1, 0.1])
    upper = np.log([10.0, 10.0, 10.0, 100.0, 100.0, 100.0])

    result = least_squares(
        residuals,
        x0=initial,
        bounds=(lower, upper),
        args=(train_cases, population),
        max_nfev=500,
    )

    full_pred = simulate(result.x, cases, population)
    out = df.copy()
    out["baseline_model_cases"] = full_pred

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / "baseline_model_fit.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    params = pd.DataFrame(
        {
            "parameter": ["beta_vh", "beta_hv", "rho", "mosquito_ratio", "e_h0_scale", "i_h0_scale"],
            "estimate": np.exp(result.x),
        }
    )
    params.to_csv(RESULT_DIR / "baseline_parameter_estimates.csv", index=False, encoding="utf-8-sig")

    print(f"Wrote {out_path}")
    print(result.message)


if __name__ == "__main__":
    main()
