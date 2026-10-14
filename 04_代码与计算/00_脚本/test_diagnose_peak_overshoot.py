import importlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

diagnosis = importlib.import_module("19_diagnose_peak_overshoot")
PROJECT, OUTPUT = Path(sys.argv.pop(1)), Path(sys.argv.pop(1))


class OvershootTests(unittest.TestCase):
    def test_sources_and_counts(self):
        summary = json.loads((OUTPUT / "run_summary.json").read_text())
        rolling, _ = diagnosis.environment(str(PROJECT), "A2")
        self.assertEqual(summary["runner_sha256"], rolling.base.file_hash(diagnosis.__file__))
        root = PROJECT / "05_结果与图表/04_滚动验证/development_peak_trend_v01"
        data = PROJECT / "02_数据/01_清洗数据/baseline_fitting_cases_2023W35_2024W45.csv"
        for p in [data, root / "rolling_fits.csv", root / "rolling_multistart.csv", root / "run_summary.json"]:
            self.assertEqual(summary["source_hashes"][p.name], rolling.base.file_hash(p))
        self.assertEqual(summary["replayed_starts"], 286)
        self.assertEqual(summary["profile_points"], 32)
        self.assertEqual(summary["profile_optimization_attempts"], 128)

    def test_near_solution_spread(self):
        trials = pd.read_csv(OUTPUT / "replayed_starts.csv")
        summary = pd.read_csv(OUTPUT / "near_optimal_spread.csv")
        self.assertEqual(len(summary), 66)
        for row in summary.itertuples():
            group = trials.loc[trials.model.eq(row.model) & trials.origin_n.eq(row.origin_n)]
            near = group.loc[group.reference_training_sse <= group.reference_training_sse.min() * (1 + row.sse_tolerance)]
            self.assertEqual(row.count_including_refinement, len(near))
            self.assertAlmostEqual(row.forecast4_min, float(near.forecast_4.min()), places=7)
            self.assertAlmostEqual(row.forecast4_max, float(near.forecast_4.max()), places=7)
            self.assertAlmostEqual(row.beta_ratio_max_over_min,
                                   float(near.beta_ratio.max() / near.beta_ratio.min()), places=7)

    def test_reference_trajectory_and_target_index(self):
        trials = pd.read_csv(OUTPUT / "replayed_starts.csv")
        data = pd.read_csv(PROJECT / "02_数据/01_清洗数据/baseline_fitting_cases_2023W35_2024W45.csv")
        for model in ("A2", "A4"):
            _, module = diagnosis.environment(str(PROJECT), model)
            row = trials.loc[trials.model.eq(model) & trials.origin_n.eq(23)].iloc[0]
            x = np.log([row[p] for p in module.PARAMETER_NAMES])
            prediction, states = diagnosis.reference(str(PROJECT), model, x, 27)
            self.assertAlmostEqual(row.forecast_4, prediction[26], places=5)
            self.assertEqual(row.observed_4, data.iloc[26].probable_cases)
            self.assertAlmostEqual(row.reference_training_sse,
                                   float(np.square(prediction[:23] - data.probable_cases.to_numpy()[:23]).sum()), delta=.01)
            np.testing.assert_allclose(states[:, :4].sum(axis=1), 1, atol=1e-9)
            np.testing.assert_allclose(states[:, 4:7].sum(axis=1), 1, atol=1e-9)
            self.assertGreaterEqual(states[:, :7].min(), -1e-9)

    def test_profile_bounds_objective_and_precision(self):
        profiles = pd.read_csv(OUTPUT / "N_eff_profile.csv")
        attempts = pd.read_csv(OUTPUT / "profile_attempts.csv")
        for row in profiles.itertuples():
            group = attempts.loc[attempts.model.eq(row.model) & attempts.origin_n.eq(row.origin_n)
                                 & np.isclose(attempts.N_eff, row.N_eff)]
            self.assertEqual(len(group), 4)
            self.assertAlmostEqual(row.optimizer_training_sse, float(group.training_sse.min()), delta=.01)
            self.assertAlmostEqual(row.relative_sse_increase_pct,
                                   100 * (row.training_sse / row.saved_fit_reference_sse - 1), places=7)
            self.assertLess(row.reference_step_halving_difference, 1e-5)
            _, module = diagnosis.environment(str(PROJECT), row.model)
            values = np.array([getattr(row, p) for p in module.PARAMETER_NAMES])
            self.assertTrue((values >= module.LOWER * (1 - 1e-10)).all())
            self.assertTrue((values <= module.UPPER * (1 + 1e-10)).all())

    def test_growth_equation_and_ratio_invariance(self):
        _, m = diagnosis.environment(str(PROJECT), "A2")
        for product in (.02, .05, .08):
            growth = diagnosis.linear_growth(m, product)
            left = ((growth + m.SIGMA_H) * (growth + m.GAMMA_H)
                    * (growth + m.SIGMA_V + m.MU_V) * (growth + m.MU_V))
            self.assertAlmostEqual(left, m.SIGMA_H * m.SIGMA_V * product, places=12)
            for ratio in (.1, 1, 10):
                h, v = np.sqrt(product * ratio), np.sqrt(product / ratio)
                matrix = np.array([[-m.SIGMA_H, 0, 0, h], [m.SIGMA_H, -m.GAMMA_H, 0, 0],
                                   [0, v, -(m.SIGMA_V + m.MU_V), 0], [0, 0, m.SIGMA_V, -m.MU_V]])
                self.assertAlmostEqual(growth, np.linalg.eigvals(matrix).real.max(), places=12)

    def test_future_outcomes_do_not_enter_profile_training(self):
        # The fitting worker receives only the prefix, not the observed evaluation targets.
        import inspect
        code = inspect.getsource(diagnosis.profile_task)
        self.assertNotIn("read_csv", code)
        self.assertIn("prediction[:n] - observed", code)
        source = inspect.getsource(diagnosis.run)
        self.assertIn("n, y[:n].copy(), value, saved", source)


if __name__ == "__main__":
    unittest.main()
