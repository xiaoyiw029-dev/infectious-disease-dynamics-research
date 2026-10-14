import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "rolling_validation", Path(__file__).with_name("16_validate_baseline_rolling.py")
)
rolling = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rolling
SPEC.loader.exec_module(rolling)
PROJECT = Path(sys.argv.pop(1)) if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir() else Path(__file__).resolve().parents[2]
RESULTS = Path(sys.argv.pop(1)) if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir() else None


class RollingValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = PROJECT / "04_代码与计算" / "00_脚本"
        cls.a2 = rolling.load_model(str(folder / rolling.MODEL_FILES["A2"]))
        cls.a4 = rolling.load_model(str(folder / rolling.MODEL_FILES["A4"]))

    def test_horizon_counts_and_no_target_in_training(self):
        folds = rolling.fold_definitions()
        self.assertEqual(len(folds), 33)
        self.assertEqual(folds[0], (30, (1, 2, 4)))
        self.assertEqual(folds[-1], (62, (1,)))
        self.assertEqual([sum(h in hs for _, hs in folds) for h in (1, 2, 4)], [33, 32, 30])
        for n, horizons in folds:
            for h in horizons:
                self.assertGreater(n + h - 1, n - 1)
                self.assertLess(n + h - 1, 63)

    def test_future_values_do_not_affect_training_or_starts(self):
        full = np.arange(1, 64, dtype=float)
        changed = full.copy()
        changed[30:] = 1e12
        for module, name in ((self.a2, "A2"), (self.a4, "A4")):
            original = rolling.training_data(module, full[:30])
            altered = rolling.training_data(module, changed[:30])
            np.testing.assert_array_equal(original.observed, altered.observed)
            self.assertEqual(original.scale, 30)
            self.assertEqual(original.scale, altered.scale)
            np.testing.assert_array_equal(
                rolling.make_starts(module, name, original.observed, 12, 42),
                rolling.make_starts(module, name, altered.observed, 12, 42),
            )
            self.assertEqual(len(original.week_edges), 31)

    def test_a4_explicit_initials_match_a2_at_same_conditions(self):
        edges = np.arange(35) * 7.0
        free = np.log([0.2, 0.2, 1e6, 189, 63, 1e-4])
        fixed = np.log([0.2, 0.2, 1e6, 1e-4])
        a, sa = rolling.simulate(self.a2, "A2", free, edges)
        b, sb = rolling.simulate(self.a4, "A4", fixed, edges)
        np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-5)
        np.testing.assert_allclose(sa, sb, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(a, 1e6 * np.diff(sa[:, 7]), rtol=1e-10, atol=1e-8)
        self.assertFalse(np.allclose(a, 1e6 * sa[1:, 2]))

    def test_forecast_extending_horizon_preserves_training_prefix(self):
        x = np.log([0.2, 0.2, 1e6, 189, 63, 1e-4])
        short, _ = rolling.simulate(self.a2, "A2", x, np.arange(31) * 7.0)
        long, _ = rolling.simulate(self.a2, "A2", x, np.arange(35) * 7.0)
        np.testing.assert_allclose(short, long[:30], rtol=1e-5, atol=0.05)

    def test_metrics_and_naive_comparison(self):
        rows = []
        for h in (1, 2, 4):
            for model, pred in (("Naive", [1, 1]), ("A2", [2, 4]), ("A4", [3, 5])):
                for n, truth, prediction in zip((30, 31), (2, 4), pred, strict=True):
                    rows.append(dict(model=model, origin_n=n, horizon=h,
                                     prediction=prediction, observed=truth))
        metrics = rolling.score_predictions(pd.DataFrame(rows))
        naive = metrics.loc[metrics.scope.eq("all_available") & metrics.model.eq("Naive")]
        np.testing.assert_allclose(naive.mae, 2)
        np.testing.assert_allclose(naive.rmse, np.sqrt(5))
        np.testing.assert_allclose(metrics.loc[metrics.model.eq("A2"), "mae_improvement_vs_naive_pct"], 100)


@unittest.skipUnless(RESULTS is not None, "No saved result directory supplied")
class SavedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred = pd.read_csv(RESULTS / "rolling_predictions_v01.csv")
        cls.fits = pd.read_csv(RESULTS / "rolling_fits_v01.csv")
        cls.trials = pd.read_csv(RESULTS / "rolling_multistart_v01.csv")
        cls.metrics = pd.read_csv(RESULTS / "rolling_metrics_v01.csv")
        cls.data = pd.read_csv(PROJECT / "02_数据" / "01_清洗数据" / "baseline_fitting_cases_2023W35_2024W45.csv")
        cls.summary = json.loads((RESULTS / "run_summary.json").read_text(encoding="utf-8"))

    def test_forecast_targets_and_naive(self):
        self.assertEqual(len(self.pred), 285)
        expected = {(m, n, h) for m in ("A2", "A4", "Naive")
                    for n in range(30, 63) for h in (1, 2, 4) if n + h <= 63}
        actual = set(self.pred[["model", "origin_n", "horizon"]].itertuples(index=False, name=None))
        self.assertEqual(actual, expected)
        for row in self.pred.itertuples(index=False):
            self.assertEqual(row.observed, self.data.iloc[row.origin_n + row.horizon - 1].probable_cases)
            self.assertEqual(row.target_week, self.data.iloc[row.origin_n + row.horizon - 1].epi_year_week)
            self.assertEqual(row.origin_week, self.data.iloc[row.origin_n - 1].epi_year_week)
            if row.model == "Naive":
                self.assertEqual(row.prediction, self.data.iloc[row.origin_n - 1].probable_cases)

    def test_independent_metric_recalculation(self):
        for row in self.metrics.itertuples(index=False):
            group = self.pred.loc[self.pred.model.eq(row.model) & self.pred.horizon.eq(row.horizon)]
            if row.scope == "common_origins":
                group = group.loc[group.origin_n.le(59)]
            errors = [p - y for p, y in zip(group.prediction, group.observed, strict=True)]
            self.assertEqual(row.n_forecasts, len(errors))
            self.assertAlmostEqual(row.mae, sum(abs(e) for e in errors) / len(errors), places=7)
            self.assertAlmostEqual(row.rmse, (sum(e * e for e in errors) / len(errors)) ** 0.5, places=7)
            self.assertAlmostEqual(row.mean_error, sum(errors) / len(errors), places=7)

    def test_selection_and_source_integrity(self):
        self.assertEqual(len(self.fits), 66)
        self.assertEqual(len(self.trials), 66 * 13)
        for row in self.fits.itertuples(index=False):
            trials = self.trials.loc[self.trials.model.eq(row.model) & self.trials.origin_n.eq(row.origin_n)]
            self.assertAlmostEqual(row.sse, trials.sse.min(), places=5)
            self.assertEqual(row.training_scale, self.data.iloc[:row.origin_n].probable_cases.max())
            self.assertEqual(row.seed, 20261007 + 100 * row.origin_n + (2 if row.model == "A2" else 4))
        for model, digest in self.summary["model_source_sha256"].items():
            path = PROJECT / "04_代码与计算" / "00_脚本" / rolling.MODEL_FILES[model]
            self.assertEqual(rolling.file_hash(path), digest)
        data_path = PROJECT / "02_数据" / "01_清洗数据" / "baseline_fitting_cases_2023W35_2024W45.csv"
        self.assertEqual(rolling.file_hash(data_path), self.summary["input_sha256"])
        self.assertEqual(self.summary["status"], "completed")
        self.assertEqual(self.summary["runner_sha256"], rolling.file_hash(Path(rolling.__file__)))


if __name__ == "__main__":
    unittest.main()
