"""Boundary and missing-value tests for the weekly climate preparation."""

import argparse
import hashlib
import importlib.util
import io
import json
import unittest
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

spec = importlib.util.spec_from_file_location(
    "climate", Path(__file__).with_name("02_prepare_weekly_climate.py")
)
climate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(climate)


class BrasiliaTimezone:
    def timezone_at(self, **kwargs):
        return "America/Sao_Paulo"


def read_fixture(rows):
    metadata = (
        "REGIAO:;CO\nUF:;DF\nESTACAO:;TEST\nCODIGO (WMO):;A001\n"
        "LATITUDE:;-15,7894\nLONGITUDE:;-47,9258\n"
        "ALTITUDE:;1160\nDATA DE FUNDACAO:;01/01/00\n"
    )
    header = (
        "Data;Hora UTC;PRECIPITACAO TOTAL, HORARIO (mm);"
        "TEMPERATURA DO AR - BULBO SECO, HORARIA (C);"
        "UMIDADE RELATIVA DO AR, HORARIA (%)\n"
    )
    body = "".join(
        f"{stamp:%Y/%m/%d};{stamp:%H%M} UTC;{rain};25;70\n"
        for stamp, rain in rows
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("station.csv", metadata + header + body)
    buffer.seek(0)
    accum, stations, counters = {}, {}, defaultdict(int)
    with zipfile.ZipFile(buffer) as archive:
        climate.update_station_weeks(
            archive,
            "station.csv",
            set(pd.to_datetime(["2023-12-24", "2023-12-31", "2024-01-07"])),
            accum,
            stations,
            counters,
            BrasiliaTimezone(),
        )
    return accum, counters


class ClimateBoundaryTests(unittest.TestCase):
    def test_local_sunday_midnight(self):
        times = pd.to_datetime([
            "2023-12-31 02:00", "2023-12-31 03:00",
            "2024-01-07 02:00", "2024-01-07 03:00",
        ])
        accum, _ = read_fixture(list(zip(times, [1, 2, 4, 8])))
        totals = {str(week.date()): row["rain_sum"] for (_, week), row in accum.items()}
        self.assertEqual(totals, {"2023-12-24": 1, "2023-12-31": 6, "2024-01-07": 8})

    def test_complete_cross_year_local_week(self):
        times = pd.date_range("2023-12-31 03:00", periods=168, freq="h")
        accum, counters = read_fixture([(stamp, 1) for stamp in times])
        self.assertEqual(len(accum), 1)
        row = accum[("A001", pd.Timestamp("2023-12-31"))]
        self.assertEqual(row["rows"], 168)
        self.assertEqual(row["rain_hours"], 168)
        self.assertEqual(row["rain_sum"], 168)
        self.assertEqual(counters["invalid_utc_timestamps"], 0)

    def test_missing_rain_is_not_a_valid_zero(self):
        times = pd.date_range("2023-12-31 03:00", periods=168, freq="h")
        rows = [(stamp, "" if index < 9 else 0) for index, stamp in enumerate(times)]
        accum, _ = read_fixture(rows)
        row = accum[("A001", pd.Timestamp("2023-12-31"))]
        self.assertEqual(row["rain_hours"], 159)
        self.assertLess(row["rain_hours"], climate.RAIN_MIN_HOURS)


def verify_outputs(data_dir, utc_dir, cases_path):
    weekly = pd.read_csv(data_dir / "brazil_weekly_climate.csv")
    aligned = pd.read_csv(data_dir / "baseline_climate_case_alignment_2023W35_2024W45.csv")
    station = pd.read_csv(data_dir / "inmet_station_inventory_2023_2025.csv")
    sw = pd.read_csv(data_dir / "inmet_station_week_quality_2023_2025.csv")
    state = pd.read_csv(data_dir / "inmet_state_week_climate_2023_2025.csv")
    cases = pd.read_csv(cases_path)
    baseline = pd.read_csv(cases_path.with_name("baseline_fitting_cases_2023W35_2024W45.csv"))
    keys = ["epi_year_week", "week_start", "week_end"]
    expected = cases.loc[
        (cases.week_start >= "2023-01-01") & (cases.week_end <= "2025-12-31"), keys
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(weekly[keys], expected)
    pd.testing.assert_frame_equal(
        aligned[keys + ["probable_cases"]], baseline[keys + ["probable_cases"]]
    )
    assert len(weekly) == 156 and len(aligned) == 63
    assert not sw.duplicated(["station_id", "week_start"]).any()
    assert sw.rows.between(1, 168).all()
    assert station.timezone.notna().all()
    assert all(sw[v + "_hours"].between(0, sw.rows).all() for v in ("temp", "rain", "humidity"))
    cross = sw[(sw.station_id == "A001") & (sw.week_start == "2023-12-31")]
    assert len(cross) == 1 and cross.iloc[0].rows == 168

    population = state[["uf", "population_2024"]].drop_duplicates().set_index("uf").population_2024
    assert len(population) == 27 and int(population.sum()) == 212583750
    for field in ("temp_mean", "rainfall_sum", "humidity_mean"):
        calculated = []
        for week in weekly.week_start:
            rows = state.loc[(state.week_start == week) & state[field].notna()]
            calculated.append(np.average(rows[field], weights=rows.population_2024))
        np.testing.assert_allclose(
            calculated, weekly[field + "_population_weighted"], rtol=1e-12, atol=1e-10
        )

    # Compare the exploratory 160-hour rain threshold with complete 168-hour weeks.
    complete = sw.loc[(sw.rain_hours == 168) & (sw.rows == 168)]
    full_state = complete.groupby(["week_start", "uf"]).rain_sum.mean().reset_index()
    full_state["population"] = full_state.uf.map(population)
    sensitivity = []
    for week, rows in full_state.groupby("week_start"):
        sensitivity.append({
            "week_start": week,
            "rainfall_complete_population_weighted": np.average(rows.rain_sum, weights=rows.population),
            "complete_states": len(rows),
            "complete_population_coverage": rows.population.sum() / population.sum(),
        })
    sensitivity = weekly[keys + ["rainfall_sum_population_weighted"]].merge(
        pd.DataFrame(sensitivity), on="week_start", how="left", validate="one_to_one"
    )
    sensitivity["complete_minus_160h_mm"] = (
        sensitivity.rainfall_complete_population_weighted - sensitivity.rainfall_sum_population_weighted
    )
    sensitivity.to_csv(
        data_dir / "inmet_rain_completeness_sensitivity_2023_2025.csv",
        index=False, encoding="utf-8-sig",
    )
    base_sensitivity = sensitivity[sensitivity.week_start.isin(aligned.week_start)]
    base_state = state[state.week_start.isin(aligned.week_start)]
    result = {
        "checks_passed": [
            "156 weeks match source case calendar",
            "63 baseline week keys and probable counts unchanged",
            "unique station-week keys and hourly counts within 168",
            "all stations have an IANA timezone",
            "A001 cross-year local week contains 168 hours",
            "all national population-weighted values independently recalculated",
            "three synthetic boundary/missingness tests passed",
        ],
        "timezone_station_counts": station.timezone.value_counts().to_dict(),
        "record_hours": int(sw.rows.sum()),
        "valid_hours": {v: int(sw[v + "_hours"].sum()) for v in ("temp", "rain", "humidity")},
        "eligible_station_weeks": {
            f: int(sw[f].notna().sum()) for f in ("temp_mean", "rainfall_sum", "humidity_mean")
        },
        "baseline_state_weeks_missing": {
            f: {uf: int(n) for uf, n in base_state.groupby("uf")[f].apply(lambda x: x.isna().sum()).items() if n}
            for f in ("temp_mean", "rainfall_sum", "humidity_mean")
        },
        "baseline_rain_168h_sensitivity": {
            "min_states": int(base_sensitivity.complete_states.min()),
            "weeks_all_27_states": int((base_sensitivity.complete_states == 27).sum()),
            "min_population_coverage": float(base_sensitivity.complete_population_coverage.min()),
            "mean_abs_difference_mm": float(base_sensitivity.complete_minus_160h_mm.abs().mean()),
            "max_abs_difference_mm": float(base_sensitivity.complete_minus_160h_mm.abs().max()),
        },
    }
    if utc_dir:
        reference_path = utc_dir / "baseline_climate_case_alignment_2023W35_2024W45.csv"
        reference = pd.read_csv(reference_path)
        pd.testing.assert_frame_equal(aligned[keys], reference[keys])
        result["utc_reference_sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        result["baseline_local_minus_utc"] = {}
        for field in (
            "temp_mean_population_weighted", "rainfall_sum_population_weighted",
            "humidity_mean_population_weighted",
        ):
            difference = (aligned[field] - reference[field]).abs()
            result["baseline_local_minus_utc"][field] = {
                "mean_absolute_difference": float(difference.mean()),
                "max_absolute_difference": float(difference.max()),
                "max_difference_week": aligned.loc[difference.idxmax(), "epi_year_week"],
            }
    (data_dir / "inmet_local_week_verification_2023_2025.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--utc-dir", type=Path)
    parser.add_argument("--cases", type=Path)
    args = parser.parse_args()
    outcome = unittest.TextTestRunner().run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ClimateBoundaryTests)
    )
    if not outcome.wasSuccessful():
        raise SystemExit(1)
    if args.data_dir:
        if not args.cases:
            parser.error("--cases is required with --data-dir")
        verify_outputs(args.data_dir, args.utc_dir, args.cases)
