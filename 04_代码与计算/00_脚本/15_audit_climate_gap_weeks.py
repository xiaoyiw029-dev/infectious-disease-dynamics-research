"""Independently audit raw INMET hours for 2025-W43/W52/W53; never impute."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from importlib.metadata import version
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
from timezonefinder import TimezoneFinder

ROOT = Path(__file__).resolve().parents[2]
WEEKS = {"2025-W43": "2025-10-19", "2025-W52": "2025-12-21", "2025-W53": "2025-12-28"}
BOUNDS = {"rain": (0, 500), "temp": (-30, 60), "humidity": (0, 100)}


def normalize(value):
    return "".join(c for c in unicodedata.normalize("NFKD", value.upper())
                   if not unicodedata.combining(c)).strip()


def parse_value(raw, variable):
    text = raw.strip()
    if not text:
        return None, "blank"
    try:
        number = float(text.replace(",", "."))
    except ValueError:
        return None, "nonnumeric"
    if number == -9999:
        return None, "sentinel"
    low, high = BOUNDS[variable]
    if not math.isfinite(number) or not low <= number <= high:
        return None, "out_of_range"
    return number, "valid"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_hours(paths):
    observations, stations = {}, {}
    counters = Counter()
    sources = []
    start_to_week = {datetime.fromisoformat(start).date(): week for week, start in WEEKS.items()}
    with TimezoneFinder(in_memory=True) as finder:
        for path in paths:
            with zipfile.ZipFile(path) as archive:
                names = sorted(n for n in archive.namelist() if n.lower().endswith(".csv"))
                for number, name in enumerate(names, 1):
                    with io.TextIOWrapper(archive.open(name), encoding="latin1") as file:
                        meta = {}
                        for _ in range(8):
                            label, _, value = next(file).rstrip("\r\n").partition(";")
                            meta[normalize(label).rstrip(":")] = value.strip()
                        station = meta["CODIGO (WMO)"]
                        uf = meta["UF"]
                        tz = finder.timezone_at(
                            lng=float(meta["LONGITUDE"].replace(",", ".")),
                            lat=float(meta["LATITUDE"].replace(",", ".")),
                        )
                        if tz is None:
                            raise ValueError(f"Timezone missing: {name}")
                        info = {"station_id": station, "uf": uf, "timezone": tz}
                        if station in stations and stations[station] != info:
                            raise ValueError(f"Station identity/timezone changed: {station}")
                        stations[station] = info
                        zone = ZoneInfo(tz)
                        reader = csv.reader(file, delimiter=";")
                        headers = [normalize(h) for h in next(reader)]
                        prefixes = {"date": "DATA", "hour": "HORA UTC",
                                    "rain": "PRECIPITACAO TOTAL",
                                    "temp": "TEMPERATURA DO AR - BULBO SECO",
                                    "humidity": "UMIDADE RELATIVA DO AR"}
                        indices = {}
                        for field, prefix in prefixes.items():
                            matches = [i for i, header in enumerate(headers) if header.startswith(prefix)]
                            if len(matches) != 1:
                                raise ValueError(f"Ambiguous {field} in {name}")
                            indices[field] = matches[0]
                        for line_number, row in enumerate(reader, 10):
                            date = row[indices["date"]].replace("/", "-")
                            if not ("2025-10-19" <= date <= "2025-10-26"
                                    or "2025-12-21" <= date <= "2026-01-04"):
                                continue
                            hour = row[indices["hour"]].split()[0]
                            stamp = datetime.fromisoformat(date + "T" + hour[:2] + ":" + hour[2:4])
                            stamp = stamp.replace(tzinfo=timezone.utc)
                            local = stamp.astimezone(zone)
                            week_start = local.date() - timedelta(days=(local.weekday() + 1) % 7)
                            if week_start not in start_to_week:
                                continue
                            week = start_to_week[week_start]
                            raw_values = {v: row[indices[v]] for v in BOUNDS}
                            key = (station, stamp)
                            if key in observations:
                                counters["duplicate_hours"] += 1
                                if observations[key]["raw"] != raw_values:
                                    raise ValueError(f"Conflicting duplicate: {station}/{stamp}")
                                continue
                            observations[key] = {
                                "week": week, "raw": raw_values, "source": path.name,
                                "member": name, "line": line_number,
                            }
                    if number % 150 == 0:
                        print(f"Read {path.name}: {number}/{len(names)} files", flush=True)
            sources.append({"filename": path.name, "bytes": path.stat().st_size,
                            "sha256": sha256(path), "station_files": len(names),
                            "url": f"https://portal.inmet.gov.br/uploads/dadoshistoricos/{path.name}"})
    return observations, stations, counters, sources


def audit(paths, data_dir, out_dir):
    observations, stations, counters, sources = read_hours(paths)
    grouped = defaultdict(dict)
    for (station, stamp), value in observations.items():
        grouped[(station, value["week"])][stamp] = value
    rows, examples = [], []
    hourly = defaultdict(Counter)
    for (station, week), records in sorted(grouped.items()):
        info = stations[station]
        start = datetime.fromisoformat(WEEKS[week]).replace(tzinfo=ZoneInfo(info["timezone"]))
        expected = [start.astimezone(timezone.utc) + timedelta(hours=i) for i in range(168)]
        if not set(records).issubset(expected):
            raise ValueError(f"Unexpected timestamp: {station}/{week}")
        row = {**info, "epi_year_week": week, "week_start": WEEKS[week],
               "rows": len(records), "absent_timestamp_hours": 168 - len(records)}
        counts = Counter()
        sums = Counter()
        missing_examples = []
        for stamp in expected:
            record = records.get(stamp)
            bucket = hourly[(week, stamp.isoformat())]
            bucket["expected_station_hours"] += 1
            if record is None:
                bucket["absent_timestamp_hours"] += 1
                continue
            bucket["observed_station_hours"] += 1
            missing = {}
            for variable in BOUNDS:
                value, status = parse_value(record["raw"][variable], variable)
                counts[variable + "_" + status] += 1
                if status == "valid":
                    sums[variable] += value
                missing[variable] = status != "valid"
            if missing["rain"]:
                bucket["rain_invalid_hours"] += 1
                if missing["temp"] and missing["humidity"]:
                    bucket["rain_temp_humidity_all_invalid"] += 1
                missing_examples.append({
                    "station_id": station, "uf": info["uf"], "epi_year_week": week,
                    "utc_timestamp": stamp.isoformat(), "source_zip": record["source"],
                    "source_member": record["member"], "source_line": record["line"],
                    **{v + "_raw": record["raw"][v] for v in BOUNDS},
                })
        for variable in BOUNDS:
            row[variable + "_hours"] = counts[variable + "_valid"]
            row[variable + "_sum"] = float(sums[variable])
            for status in ("blank", "sentinel", "nonnumeric", "out_of_range"):
                row[variable + "_" + status] = counts[variable + "_" + status]
        row["all_three_invalid_hours"] = sum(
            all(parse_value(r["raw"][v], v)[1] != "valid" for v in BOUNDS)
            for r in records.values()
        )
        # Near-complete stations expose short common gaps without long-outage noise.
        if row["rain_hours"] >= 160:
            examples.extend(missing_examples)
            for item in missing_examples:
                hourly[(week, item["utc_timestamp"])]["near_complete_station_rain_invalid"] += 1
        rows.append(row)
    sw = pd.DataFrame(rows)
    hourly_frame = pd.DataFrame([
        {"epi_year_week": week, "utc_timestamp": stamp, **values}
        for (week, stamp), values in sorted(hourly.items())
    ]).fillna(0)
    for column in hourly_frame.columns[2:]:
        hourly_frame[column] = hourly_frame[column].astype(int)

    previous = pd.read_csv(data_dir / "inmet_station_week_quality_2023_2025.csv")
    old = previous.loc[previous.week_start.isin([WEEKS["2025-W43"], WEEKS["2025-W52"]])]
    compare = old.merge(sw, on=["station_id", "uf", "week_start"], suffixes=("_old", "_audit"),
                        how="outer", indicator=True, validate="one_to_one")
    compare = compare.loc[compare.week_start.ne(WEEKS["2025-W53"])]
    assert compare._merge.eq("both").all()
    for col in ("rows", "rain_hours", "temp_hours", "humidity_hours",
                "rain_sum", "temp_sum", "humidity_sum"):
        np.testing.assert_allclose(compare[col + "_old"], compare[col + "_audit"],
                                   rtol=1e-12, atol=1e-8)
    state = pd.read_csv(data_dir / "inmet_state_week_climate_2023_2025.csv")
    population = state[["uf", "population_2024"]].drop_duplicates().set_index("uf").population_2024
    assert len(population) == 27 and population.index.is_unique
    assert int(population.sum()) == 212583750
    calendar = pd.read_csv(data_dir / "brazil_weekly_chikungunya_cases.csv")
    candidate = pd.read_csv(data_dir / "brazil_weekly_climate_main_candidate_2023_2025.csv")
    national_rows = []
    for week, frame in sw.groupby("epi_year_week"):
        case = calendar.loc[calendar.epi_year_week.eq(week)].iloc[0]
        result = {k: case[k] for k in ("epi_year_week", "week_start", "week_end")}
        result["probable_cases"] = int(case.probable_cases)
        result["in_baseline_window"] = False
        for var, threshold, output in (("temp", 135, "temp_mean_c"),
                                       ("humidity", 135, "humidity_mean_pct"),
                                       ("rain", 168, "rainfall_168h_mm")):
            valid = frame.loc[frame[var + "_hours"].ge(threshold) & frame.rows.le(168)].copy()
            valid["value"] = valid[var + "_sum"]
            if var != "rain":
                valid["value"] /= valid[var + "_hours"]
            uf_values = valid.groupby("uf").value.mean()
            weights = population.reindex(uf_values.index)
            assert weights.notna().all()
            result[output] = float(np.average(uf_values, weights=weights)) if len(valid) else None
            result[var + "_valid_uf_count"] = len(uf_values)
            result[var + "_valid_uf_population_share"] = float(weights.sum() / population.sum())
        count = result["rain_valid_uf_count"]
        result["rain_coverage_status"] = (
            "all_ufs" if count == 27 else "not_national" if count <= 1 else "partial_uf_coverage"
        )
        national_rows.append(result)
    national = pd.DataFrame(national_rows)[candidate.columns]
    numeric_columns = [c for c in national if national[c].dtype.kind in "fiu"]
    for week in ("2025-W43", "2025-W52"):
        np.testing.assert_allclose(
            national.loc[national.epi_year_week.eq(week), numeric_columns].to_numpy(float),
            candidate.loc[candidate.epi_year_week.eq(week), numeric_columns].to_numpy(float),
            rtol=1e-12, atol=1e-8,
        )
    supplement = national.loc[national.epi_year_week.eq("2025-W53")]
    expanded = pd.concat([candidate, supplement], ignore_index=True)
    assert len(expanded) == 157 and not expanded.week_start.duplicated().any()
    expected = calendar.loc[calendar.epi_year.between(2023, 2025),
                            ["epi_year_week", "week_start", "week_end", "probable_cases"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(expanded[expected.columns], expected)
    pd.testing.assert_frame_equal(expanded.iloc[:156].reset_index(drop=True), candidate)
    summaries = {}
    for week, group in sw.groupby("epi_year_week"):
        hour_group = hourly_frame.loc[hourly_frame.epi_year_week.eq(week)]
        summaries[week] = {
            "stations_with_rows": len(group), "rows": int(group.rows.sum()),
            "absent_timestamp_hours": int(group.absent_timestamp_hours.sum()),
            "rain_valid_hours": int(group.rain_hours.sum()),
            "rain_blank_hours": int(group.rain_blank.sum()),
            "rain_sentinel_hours": int(group.rain_sentinel.sum()),
            "rain_nonnumeric_hours": int(group.rain_nonnumeric.sum()),
            "rain_out_of_range_hours": int(group.rain_out_of_range.sum()),
            "all_three_invalid_hours": int(group.all_three_invalid_hours.sum()),
            "complete_rain_stations": int(group.rain_hours.eq(168).sum()),
            "160h_rain_stations": int(group.rain_hours.ge(160).sum()),
            "rain_coverage_by_threshold": {
                str(threshold): {
                    "uf_count": int(group.loc[group.rain_hours.ge(threshold), "uf"].nunique()),
                    "valid_uf_population_share": float(
                        population.reindex(group.loc[group.rain_hours.ge(threshold), "uf"].unique()).sum()
                        / population.sum()
                    ),
                } for threshold in (168, 160)
            },
            "top_common_gaps_near_complete_stations": hour_group.nlargest(
                5, "near_complete_station_rain_invalid"
            ).to_dict(orient="records"),
        }
    report = {
        "audit_date": "2026-10-05", "sources": sources, "counters": dict(counters),
        "weeks": summaries, "national_target_weeks": national.to_dict(orient="records"),
        "checks_passed": [
            "independent csv/zoneinfo parsing reproduces W43/W52 station-week counts and sums",
            "independent state population weighting reproduces W43/W52 main candidate values",
            "original 156 candidate rows unchanged",
            "expanded 157 rows match the complete 2023-2025 case calendar including W53",
        ],
        "processing_scope": "No imputation, threshold changes, fitting or split assignment",
        "readiness": "calendar_complete_but_rain_spatial_coverage_requires_decision",
        "cause_limit": "Blank raw fields establish source-level gaps, not their upstream cause",
        "input_candidate_sha256": sha256(data_dir / "brazil_weekly_climate_main_candidate_2023_2025.csv"),
        "other_inputs_sha256": {
            name: sha256(data_dir / name) for name in (
                "inmet_station_week_quality_2023_2025.csv",
                "inmet_state_week_climate_2023_2025.csv",
                "brazil_weekly_chikungunya_cases.csv",
            )
        },
        "runtime": {
            "python": sys.version.split()[0], "pandas": pd.__version__,
            "timezonefinder": version("timezonefinder"), "tzdata": version("tzdata"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "inmet_2025_target_weeks_station_audit.csv": sw,
        "inmet_2025_target_weeks_hourly_coverage.csv": hourly_frame,
        "inmet_2025_target_weeks_raw_gap_examples.csv": pd.DataFrame(examples),
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, index=False, encoding="utf-8-sig")
    candidate_path = out_dir / "brazil_weekly_climate_main_candidate_2023_2025_v02.csv"
    expanded.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    pd.testing.assert_frame_equal(expanded, pd.read_csv(candidate_path), check_exact=False,
                                  rtol=1e-12, atol=1e-10)
    report["outputs_sha256"] = {name: sha256(out_dir / name) for name in outputs}
    report["outputs_sha256"][candidate_path.name] = sha256(candidate_path)
    (out_dir / "inmet_2025_gap_audit_v01.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-2025", type=Path,
                        default=ROOT / "02_数据" / "00_原始数据" / "inmet" / "2025.zip")
    parser.add_argument("--zip-2026", type=Path,
                        default=ROOT / "02_数据" / "00_原始数据" / "inmet" / "2026.zip")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "02_数据" / "01_清洗数据")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    audit([args.zip_2025, args.zip_2026], args.data_dir, args.out_dir)
