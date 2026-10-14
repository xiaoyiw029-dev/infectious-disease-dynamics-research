import importlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

runner = importlib.import_module("17_validate_peak_trend")
PROJECT = Path(sys.argv.pop(1))
OUTPUT = Path(sys.argv.pop(1)) if len(sys.argv) > 1 else None


class PeakTrendTests(unittest.TestCase):
    def test_trend_exact_line_and_flat(self):
        for horizon in (1, 2, 4):
            value, raw, slope = runner.trend_forecast([1, 3, 5, 7], horizon)
            self.assertAlmostEqual(value, 7 + 2 * horizon)
            self.assertAlmostEqual(raw, value)
            self.assertAlmostEqual(slope, 2)
            self.assertEqual(runner.trend_forecast([5, 5, 5, 5], horizon), (5, 5, 0))

    def test_negative_trend_clipped_and_recent_only(self):
        self.assertEqual(runner.trend_forecast([7, 5, 3, 1], 1), (0, -1, -2))
        self.assertEqual(runner.trend_forecast([999999, 1, 2, 3, 4], 4), (8, 8, 1))

    def test_future_mutation_does_not_change_prediction(self):
        full = np.arange(1, 64, dtype=float)
        changed = full.copy()
        changed[20:] = 1e9
        for h in (1, 2, 4):
            self.assertEqual(runner.trend_forecast(full[:20], h),
                             runner.trend_forecast(changed[:20], h))

    def test_fold_counts_and_peak_coverage(self):
        folds = runner.folds(20)
        self.assertEqual([sum(h in hs for _, hs in folds) for h in (1, 2, 4)], [43, 42, 40])
        for h in (1, 2, 4):
            self.assertIn((27 - h, (1, 2, 4)), folds)
        for n, hs in folds:
            self.assertTrue(all(n <= n + h - 1 < 63 for h in hs))

    def test_reuse_guard_matches_original(self):
        old = PROJECT / "05_结果与图表/04_滚动验证/development_rolling_v01"
        self.assertEqual(runner.verify_reuse(PROJECT, old)["status"], "completed")

    @unittest.skipUnless(OUTPUT and (OUTPUT / "run_summary.json").exists(), "Run outputs unavailable")
    def test_saved_predictions_and_independent_metrics(self):
        predictions = pd.read_csv(OUTPUT / "rolling_predictions.csv")
        metrics = pd.read_csv(OUTPUT / "rolling_metrics.csv")
        data = pd.read_csv(PROJECT / "02_数据/01_清洗数据/baseline_fitting_cases_2023W35_2024W45.csv")
        self.assertEqual(len(predictions), 500)
        self.assertFalse(predictions.duplicated(runner.base.KEYS).any())
        for row in predictions.itertuples():
            self.assertEqual(row.observed, data.iloc[row.origin_n + row.horizon - 1].probable_cases)
            self.assertEqual(row.target_week, data.iloc[row.origin_n + row.horizon - 1].epi_year_week)
            if row.model == "Naive":
                self.assertEqual(row.prediction, data.iloc[row.origin_n - 1].probable_cases)
            elif row.model == "Trend4":
                recent = data.probable_cases.to_numpy()[row.origin_n - 4:row.origin_n]
                coefficients = np.linalg.lstsq(np.column_stack([np.ones(4), np.arange(4)]),
                                               recent, rcond=None)[0]
                raw = coefficients @ np.array([1, 3 + row.horizon])
                self.assertAlmostEqual(row.prediction, max(0, raw), places=7)
        for row in metrics.itertuples():
            part = predictions.loc[predictions.model.eq(row.model) & predictions.horizon.eq(row.horizon)]
            n = part.origin_n
            t = n + part.horizon
            masks = {"all_available": np.ones(len(part), dtype=bool),
                     "common_origins": n.le(59), "original_origins_30plus": n.ge(30),
                     "added_origins_before30": n.lt(30), "targets_through_observed_peak": t.le(27),
                     "targets_after_observed_peak": t.gt(27), "last10_target_weeks": t.ge(54)}
            subset = part.loc[masks[row.scope]]
            error = subset.prediction - subset.observed
            self.assertEqual(len(subset), row.n_forecasts)
            self.assertAlmostEqual(float(error.abs().mean()), row.mae, places=7)
            self.assertAlmostEqual(float(np.sqrt((error ** 2).mean())), row.rmse, places=7)

    @unittest.skipUnless(OUTPUT and (OUTPUT / "run_summary.json").exists(), "Run outputs unavailable")
    def test_reused_results_and_selected_fits(self):
        old = PROJECT / "05_结果与图表/04_滚动验证/development_rolling_v01"
        predictions = pd.read_csv(OUTPUT / "rolling_predictions.csv")
        old_p = pd.read_csv(old / "rolling_predictions_v01.csv").sort_values(runner.base.KEYS)
        reused = predictions.loc[predictions.origin_n.ge(30) &
                                 predictions.model.ne("Trend4")].sort_values(runner.base.KEYS)
        np.testing.assert_allclose(reused.prediction, old_p.prediction, rtol=0, atol=0)
        fits = pd.read_csv(OUTPUT / "rolling_fits.csv")
        trials = pd.read_csv(OUTPUT / "rolling_multistart.csv")
        summary = json.loads((OUTPUT / "run_summary.json").read_text())
        self.assertEqual(len(fits), 86)
        self.assertEqual(len(trials), 1118)
        self.assertEqual(summary["runner_sha256"], runner.base.file_hash(runner.__file__))
        self.assertTrue((fits.original_vs_reference_max_difference <= fits.precision_count_tolerance).all())
        self.assertTrue((fits.reference_sse_relative_difference <= .005).all())
        self.assertTrue((fits.reference_step_halving_max_difference <=
                         np.maximum(1e-5, 1e-6 * fits.training_scale)).all())
        for name, digest in summary["reused_files_sha256"].items():
            self.assertEqual(digest, runner.base.file_hash(old / name))
        for row in fits.itertuples():
            subset = trials.loc[trials.model.eq(row.model) & trials.origin_n.eq(row.origin_n)]
            self.assertEqual(len(subset), 13)
            self.assertAlmostEqual(row.sse, float(subset.sse.min()), delta=1e-5)
            self.assertEqual(row.seed, 20261007 + 100 * row.origin_n + (2 if row.model == "A2" else 4))


if __name__ == "__main__":
    unittest.main()
