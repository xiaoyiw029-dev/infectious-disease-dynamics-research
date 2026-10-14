import importlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

screen = importlib.import_module("18_screen_climate_increment")
PROJECT, OUTPUT = Path(sys.argv.pop(1)), Path(sys.argv.pop(1))


class ClimateScreenTests(unittest.TestCase):
    def test_no_future_targets_or_weather(self):
        y = np.arange(1, 64, dtype=float) ** 2
        weather = np.column_stack([np.arange(63) % 7, np.arange(63) % 5, np.arange(63) % 3])
        changed_y, changed_weather = y.copy(), weather.copy()
        changed_y[20:] = 999999
        changed_weather[20:] = -99999
        for h in (1, 2, 4):
            for use_climate in (False, True):
                a = screen.forecast_one(y, weather, 20, h, use_climate)
                b = screen.forecast_one(changed_y, changed_weather, 20, h, use_climate)
                self.assertEqual(a, b)
                self.assertEqual(a["latest_training_target_week_number"], 20)
                self.assertEqual(a["training_pairs"], 20 - h - 3)

    def test_constant_climate_equals_baseline(self):
        y = np.arange(1, 64, dtype=float) ** 2
        climate = np.ones((63, 3))
        for h in (1, 2, 4):
            a = screen.forecast_one(y, climate, 20, h, False)
            b = screen.forecast_one(y, climate, 20, h, True)
            self.assertAlmostEqual(a["prediction"], b["prediction"], places=10)
            np.testing.assert_allclose(b["standardized_coefficients"][2:], 0, atol=1e-12)

    def test_ridge_against_augmented_least_squares(self):
        x = np.array([[1, 3], [2, 1], [3, 5], [4, 4]], dtype=float)
        y = np.array([3, 8, 5, 7], dtype=float)
        query = np.array([5, 2], dtype=float)
        value, raw, mean, scale, center, coefficient = screen.ridge_fit_predict(x, y, query)
        augmented_x = np.vstack([(x - mean) / scale, np.sqrt(screen.ALPHA) * np.eye(2)])
        augmented_y = np.r_[y - center, np.zeros(2)]
        expected = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
        np.testing.assert_allclose(coefficient, expected, atol=1e-12)
        self.assertAlmostEqual(raw, center + ((query - mean) / scale) @ expected)
        self.assertEqual(value, max(0, raw))

    def test_saved_metrics_and_deterministic_predictions(self):
        predictions = pd.read_csv(OUTPUT / "climate_screen_predictions.csv")
        gains = pd.read_csv(OUTPUT / "paired_climate_gain.csv")
        fits = json.loads((OUTPUT / "fold_fits.json").read_text())
        summary = json.loads((OUTPUT / "run_summary.json").read_text())
        self.assertEqual(len(predictions), 250)
        self.assertEqual(len(fits), 250)
        self.assertEqual(summary["minimum_training_pairs"], 13)
        self.assertEqual(summary["runner_sha256"], screen.rolling.base.file_hash(screen.__file__))
        source = PROJECT / "02_数据/01_清洗数据"
        cases = pd.read_csv(source / "baseline_fitting_cases_2023W35_2024W45.csv")
        weather = pd.read_csv(source / "brazil_weekly_climate_main_candidate_2023_2025_v02.csv")
        merged = cases.merge(weather[["epi_year_week", *screen.CLIMATE]], on="epi_year_week",
                             validate="one_to_one", how="left")
        y = merged.probable_cases.to_numpy(float)
        climate = merged[screen.CLIMATE].to_numpy(float)
        for row in predictions.itertuples():
            expected = screen.forecast_one(y, climate, row.origin_n, row.horizon,
                                           row.model == "ARTrendClimate")
            self.assertAlmostEqual(row.prediction, expected["prediction"], places=7)
            self.assertEqual(row.observed, y[row.origin_n + row.horizon - 1])
        for row in gains.itertuples():
            part = predictions.loc[predictions.horizon.eq(row.horizon)]
            n, t = part.origin_n, part.origin_n + part.horizon
            masks = {"all_available": np.ones(len(part), dtype=bool), "common_origins": n.le(59),
                     "original_origins_30plus": n.ge(30), "added_origins_before30": n.lt(30),
                     "targets_through_observed_peak": t.le(27), "targets_after_observed_peak": t.gt(27),
                     "last10_target_weeks": t.ge(54)}
            part = part.loc[masks[row.scope]]
            errors = {}
            for model in ("ARTrend", "ARTrendClimate"):
                subset = part.loc[part.model.eq(model)]
                self.assertEqual(len(subset), row.n_forecasts)
                errors[model] = float((subset.prediction - subset.observed).abs().mean())
            self.assertAlmostEqual(row.without_climate_mae, errors["ARTrend"], places=7)
            self.assertAlmostEqual(row.with_climate_mae, errors["ARTrendClimate"], places=7)
            self.assertAlmostEqual(row.mae_reduction_pct,
                                   100 * (1 - errors["ARTrendClimate"] / errors["ARTrend"]), places=7)


if __name__ == "__main__":
    unittest.main()
