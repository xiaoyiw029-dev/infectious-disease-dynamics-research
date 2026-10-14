"""Audit INMET automatic stations and align weekly climate with case weeks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
import zipfile
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import pandas as pd

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
from timezonefinder import TimezoneFinder


UF_CODES = (
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO",
)
TEMP_MIN_HOURS = 135  # At least 80% of 168 hours.
HUMIDITY_MIN_HOURS = 135
RAIN_MIN_HOURS = 160  # At least 95%: avoid treating missing rain as zero.


def normalized(value: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", value.upper())
        if not unicodedata.combining(c)
    ).strip()


def column(frame: pd.DataFrame, prefix: str) -> str:
    matches = [name for name in frame if normalized(name).startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {prefix!r} column, found {matches}")
    return matches[0]


def numeric(series: pd.Series, low: float, high: float) -> pd.Series:
    values = pd.to_numeric(
        series.astype("string").str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )
    return values.where(values.between(low, high))


def calendar_from_cases(path: Path) -> pd.DataFrame:
    cases = pd.read_csv(path, encoding="utf-8-sig")
    required = {"epi_year_week", "week_start", "week_end"}
    if not required.issubset(cases.columns):
        raise ValueError(f"Case calendar missing {sorted(required - set(cases.columns))}")
    weeks = cases[["epi_year_week", "week_start", "week_end"]].copy()
    weeks["week_start"] = pd.to_datetime(weeks["week_start"], errors="raise")
    weeks["week_end"] = pd.to_datetime(weeks["week_end"], errors="raise")
    weeks = weeks.loc[
        (weeks.week_start >= "2023-01-01") & (weeks.week_end <= "2025-12-31")
    ].sort_values("week_start").reset_index(drop=True)
    if weeks.week_start.duplicated().any():
        raise ValueError("Duplicate week_start in case calendar")
    if not ((weeks.week_end - weeks.week_start).dt.days == 6).all():
        raise ValueError("Case weeks must contain seven calendar dates")
    if not (weeks.week_start.dt.dayofweek == 6).all():
        raise ValueError("Case weeks must start on Sunday")
    if not (weeks.week_start.diff().dropna().dt.days == 7).all():
        raise ValueError("Case calendar is not continuous")
    return weeks


def population_2024(pop_path: Path, states_path: Path) -> pd.DataFrame:
    series = json.loads(pop_path.read_text(encoding="utf-8-sig"))[0]["resultados"][0]["series"]
    by_id = {int(item["localidade"]["id"]): int(item["serie"]["2024"]) for item in series}
    states = json.loads(states_path.read_text(encoding="utf-8-sig"))
    rows = [
        {"uf": item["sigla"], "population_2024": by_id[int(item["id"])]}
        for item in states
    ]
    population = pd.DataFrame(rows)
    if set(population.uf) != set(UF_CODES) or (population.population_2024 <= 0).any():
        raise ValueError("IBGE population input must contain all 27 UFs")
    return population


def file_metadata(raw) -> dict[str, str]:
    meta = {}
    for _ in range(8):
        line = raw.readline().decode("latin1").strip("\r\n")
        label, _, value = line.partition(";")
        meta[normalized(label).rstrip(":")] = value.strip()
    return meta


def update_station_weeks(
    zf: zipfile.ZipFile,
    name: str,
    selected_weeks: set[pd.Timestamp],
    accum: dict,
    stations: dict,
    counters: dict,
    timezone_finder: TimezoneFinder,
) -> None:
    with zf.open(name) as raw:
        meta = file_metadata(raw)
        code = meta.get("CODIGO (WMO)", "").strip()
        uf = meta.get("UF", "").strip().upper()
        if not code or uf not in UF_CODES:
            raise ValueError(f"Invalid station code/UF in {name}: {code}/{uf}")
        if code in stations and stations[code]["uf"] != uf:
            raise ValueError(f"Station {code} changed UF: {stations[code]['uf']} -> {uf}")
        latitude = float(meta.get("LATITUDE", "").replace(",", "."))
        longitude = float(meta.get("LONGITUDE", "").replace(",", "."))
        timezone_name = timezone_finder.timezone_at(lng=longitude, lat=latitude)
        if timezone_name is None:
            raise ValueError(f"Cannot resolve station timezone in {name}")
        if code in stations and stations[code]["timezone"] != timezone_name:
            raise ValueError(f"Station {code} changed timezone across source files")
        station = stations.setdefault(
            code,
            {
                "station_id": code,
                "uf": uf,
                "region": meta.get("REGIAO", ""),
                "station_name": meta.get("ESTACAO", ""),
                "latitude": meta.get("LATITUDE", "").replace(",", "."),
                "longitude": meta.get("LONGITUDE", "").replace(",", "."),
                "timezone": timezone_name,
                "source_files": 0,
            },
        )
        station["source_files"] += 1
        frame = pd.read_csv(raw, sep=";", encoding="latin1", dtype=str, low_memory=False)
    counters["raw_rows"] += len(frame)
    if frame.empty:
        return
    date_col = column(frame, "DATA")
    hour_col = column(frame, "HORA UTC")
    rain_col = column(frame, "PRECIPITACAO TOTAL")
    temp_col = column(frame, "TEMPERATURA DO AR - BULBO SECO")
    humidity_col = column(frame, "UMIDADE RELATIVA DO AR")

    hours = frame[hour_col].str.extract(r"(\d{4})", expand=False)
    utc_text = (
        frame[date_col].str.replace("/", "-", regex=False)
        + " "
        + hours.str.slice(0, 2)
        + ":"
        + hours.str.slice(2, 4)
    )
    utc_times = pd.to_datetime(
        utc_text, format="%Y-%m-%d %H:%M", errors="coerce", utc=True
    )
    counters["invalid_utc_timestamps"] += int(utc_times.isna().sum())
    frame = frame.loc[utc_times.notna()].copy()
    frame["_utc"] = utc_times.loc[frame.index]
    frame["_date"] = (
        frame["_utc"].dt.tz_convert(timezone_name).dt.tz_localize(None).dt.normalize()
    )
    frame = frame.loc[
        (frame["_date"] >= "2023-01-01") & (frame["_date"] <= "2025-12-31")
    ].copy()
    duplicates = frame.duplicated(subset=["_utc"])
    counters["duplicate_hours_within_file"] += int(duplicates.sum())
    frame = frame.loc[~duplicates].copy()
    if frame.empty:
        return

    frame["_week_start"] = frame["_date"] - pd.to_timedelta(
        (frame["_date"].dt.dayofweek + 1) % 7, unit="D"
    )
    frame = frame.loc[frame["_week_start"].isin(selected_weeks)].copy()
    if frame.empty:
        return
    counters["rows_in_case_calendar"] += len(frame)
    frame["_temp"] = numeric(frame[temp_col], -30, 60)
    frame["_rain"] = numeric(frame[rain_col], 0, 500)
    frame["_humidity"] = numeric(frame[humidity_col], 0, 100)

    grouped = frame.groupby("_week_start").agg(
        rows=("_date", "size"),
        temp_sum=("_temp", "sum"),
        temp_hours=("_temp", "count"),
        rain_sum=("_rain", "sum"),
        rain_hours=("_rain", "count"),
        humidity_sum=("_humidity", "sum"),
        humidity_hours=("_humidity", "count"),
    )
    for week_start, row in grouped.iterrows():
        key = (code, week_start)
        target = accum.setdefault(
            key,
            {
                "station_id": code,
                "uf": uf,
                "week_start": week_start,
                "rows": 0,
                "temp_sum": 0.0,
                "temp_hours": 0,
                "rain_sum": 0.0,
                "rain_hours": 0,
                "humidity_sum": 0.0,
                "humidity_hours": 0,
            },
        )
        for field in (
            "rows", "temp_sum", "temp_hours", "rain_sum", "rain_hours",
            "humidity_sum", "humidity_hours",
        ):
            target[field] += row[field]


def build_tables(
    raw_dir: Path, cases_path: Path, pop_path: Path, states_path: Path,
    years: list[int],
):
    calendar = calendar_from_cases(cases_path)
    population = population_2024(pop_path, states_path)
    selected_weeks = set(calendar.week_start)
    accum: dict = {}
    stations: dict = {}
    counters = defaultdict(int)
    sources = []
    with TimezoneFinder(in_memory=True) as finder:
        for year in years:
            path = raw_dir / f"{year}.zip"
            if not path.is_file():
                raise FileNotFoundError(path)
            with zipfile.ZipFile(path) as zf:
                names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
                if not names:
                    raise ValueError(f"No station CSV files in {path}")
                for name in names:
                    update_station_weeks(
                        zf, name, selected_weeks, accum, stations, counters, finder
                    )
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            sources.append(
                {
                    "year": year,
                    "url": f"https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip",
                    "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                    "station_csv_files": len(names),
                }
            )
            print(f"Read {year}: {len(names)} station CSV files", flush=True)

    station_week = pd.DataFrame(accum.values())
    if station_week.empty:
        raise ValueError("No INMET records overlap the case calendar")
    station_week = station_week.sort_values(["week_start", "uf", "station_id"])
    overfull = station_week.rows > 168
    counters["station_weeks_over_168_hours"] = int(overfull.sum())
    for variable, minimum in (
        ("temp", TEMP_MIN_HOURS),
        ("rain", RAIN_MIN_HOURS),
        ("humidity", HUMIDITY_MIN_HOURS),
    ):
        valid = (station_week[f"{variable}_hours"] >= minimum) & ~overfull
        if variable == "rain":
            station_week["rainfall_sum"] = station_week.rain_sum.where(valid)
        else:
            station_week[f"{variable}_mean"] = (
                station_week[f"{variable}_sum"] / station_week[f"{variable}_hours"]
            ).where(valid)

    state_stats = station_week.groupby(["week_start", "uf"], as_index=False).agg(
        temp_mean=("temp_mean", "mean"),
        temp_stations=("temp_mean", "count"),
        rainfall_sum=("rainfall_sum", "mean"),
        rain_stations=("rainfall_sum", "count"),
        humidity_mean=("humidity_mean", "mean"),
        humidity_stations=("humidity_mean", "count"),
    )
    state_week = (
        calendar.assign(_join=1)
        .merge(pd.DataFrame({"uf": UF_CODES, "_join": 1}), on="_join")
        .drop(columns="_join")
        .merge(state_stats, on=["week_start", "uf"], how="left", validate="one_to_one")
        .merge(population, on="uf", validate="many_to_one")
        .sort_values(["week_start", "uf"])
    )
    for name in ("temp_stations", "rain_stations", "humidity_stations"):
        state_week[name] = state_week[name].fillna(0).astype(int)

    national = state_week.groupby("week_start", as_index=False).agg(
        temp_mean=("temp_mean", "mean"),
        temp_states=("temp_mean", "count"),
        temp_stations=("temp_stations", "sum"),
        rainfall_sum=("rainfall_sum", "mean"),
        rain_states=("rainfall_sum", "count"),
        rain_stations=("rain_stations", "sum"),
        humidity_mean=("humidity_mean", "mean"),
        humidity_states=("humidity_mean", "count"),
        humidity_stations=("humidity_stations", "sum"),
    )
    total_population = int(population.population_2024.sum())
    for field in ("temp_mean", "rainfall_sum", "humidity_mean"):
        valid = state_week[field].notna()
        weighted_sum = (
            state_week[field].fillna(0) * state_week.population_2024
        ).groupby(state_week.week_start).sum()
        available_population = (
            state_week.population_2024.where(valid, 0)
            .groupby(state_week.week_start).sum()
        )
        national[f"{field}_population_weighted"] = national.week_start.map(
            weighted_sum / available_population.where(available_population > 0)
        )
        national[f"{field}_population_coverage"] = national.week_start.map(
            available_population / total_population
        )
    station_mean = station_week.groupby("week_start", as_index=False).agg(
        temp_mean_station_equal=("temp_mean", "mean"),
        rainfall_sum_station_equal=("rainfall_sum", "mean"),
        humidity_mean_station_equal=("humidity_mean", "mean"),
    )
    weekly = calendar.merge(national, on="week_start", validate="one_to_one")
    weekly = weekly.merge(station_mean, on="week_start", validate="one_to_one")
    for variable in ("temp", "rain", "humidity"):
        weekly[f"{variable}_all_27_states"] = weekly[f"{variable}_states"] == 27

    station_table = pd.DataFrame(stations.values()).sort_values(["uf", "station_id"])
    valid_weeks = station_week.groupby("station_id", as_index=False).agg(
        weeks_with_records=("week_start", "nunique"),
        weeks_temp_valid=("temp_mean", "count"),
        weeks_rain_valid=("rainfall_sum", "count"),
        weeks_humidity_valid=("humidity_mean", "count"),
        first_week=("week_start", "min"),
        last_week=("week_start", "max"),
    )
    station_table = station_table.merge(
        valid_weeks, on="station_id", validate="one_to_one"
    )

    baseline = pd.read_csv(
        cases_path.parent / "baseline_fitting_cases_2023W35_2024W45.csv",
        encoding="utf-8-sig",
    )
    if len(baseline) != 63:
        raise ValueError(f"Expected 63 baseline weeks, found {len(baseline)}")
    for name in ("week_start", "week_end"):
        baseline[name] = pd.to_datetime(baseline[name])
    aligned = baseline[
        ["epi_year_week", "week_start", "week_end", "probable_cases"]
    ].merge(
        weekly,
        on=["epi_year_week", "week_start", "week_end"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not (aligned._merge == "both").all():
        raise ValueError("Baseline case weeks do not all match climate weeks")
    aligned = aligned.drop(columns="_merge")

    report = {
        "source": "INMET annual automatic-station historical data",
        "time_basis": "INMET UTC timestamp converted to station local civil date before Sunday-Saturday grouping",
        "timezone_lookup": {
            "timezonefinder": version("timezonefinder"),
            "timezonefinder_data": version("timezonefinder-data"),
            "tzdata": version("tzdata"),
        },
        "week_definition": "Sunday through Saturday, keyed to case week_start/week_end",
        "station_week_thresholds_hours": {
            "temperature": TEMP_MIN_HOURS,
            "rainfall": RAIN_MIN_HOURS,
            "humidity": HUMIDITY_MIN_HOURS,
            "expected": 168,
        },
        "sources": sources,
        "raw_record_counts": dict(counters),
        "unique_stations": len(station_table),
        "states_with_a_station": int(station_table.uf.nunique()),
        "stations_by_uf": station_table.groupby("uf").size().to_dict(),
        "population_source": {
            "api": "https://servicodados.ibge.gov.br/api/v3/agregados/6579/periodos/2024/variaveis/9324?localidades=N3%5Ball%5D",
            "reference_year": 2024,
            "total_population": total_population,
            "weights_fixed_across_weeks": True,
            "missing_state_weights_renormalized": True,
        },
        "weekly_rows": len(weekly),
        "baseline_weeks_matched": len(aligned),
        "baseline_coverage": {
            variable: {
                "min_states": int(aligned[f"{variable}_states"].min()),
                "median_states": float(aligned[f"{variable}_states"].median()),
                "max_states": int(aligned[f"{variable}_states"].max()),
                "weeks_all_27_states": int(aligned[f"{variable}_all_27_states"].sum()),
                "weeks_with_no_states": int((aligned[f"{variable}_states"] == 0).sum()),
                "min_population_coverage": float(
                    aligned[
                        {
                            "temp": "temp_mean",
                            "rain": "rainfall_sum",
                            "humidity": "humidity_mean",
                        }[variable] + "_population_coverage"
                    ].min()
                ),
            }
            for variable in ("temp", "rain", "humidity")
        },
    }
    return station_table, station_week, state_week, weekly, aligned, report


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path,
        default=project / "02_数据" / "00_原始数据" / "inmet",
    )
    parser.add_argument(
        "--cases", type=Path,
        default=project / "02_数据" / "01_清洗数据" / "brazil_weekly_chikungunya_cases.csv",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=project / "02_数据" / "01_清洗数据",
    )
    parser.add_argument(
        "--population-json", type=Path,
        default=project / "02_数据" / "00_原始数据" / "ibge_population"
        / "ibge_state_population_2024_api.json",
    )
    parser.add_argument(
        "--states-json", type=Path,
        default=project / "02_数据" / "00_原始数据" / "ibge_population"
        / "ibge_states_api.json",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    station, station_week, state_week, weekly, aligned, report = build_tables(
        args.raw_dir, args.cases, args.population_json, args.states_json,
        [2023, 2024, 2025],
    )
    outputs = {
        "inmet_station_inventory_2023_2025.csv": station,
        "inmet_station_week_quality_2023_2025.csv": station_week,
        "inmet_state_week_climate_2023_2025.csv": state_week,
        "brazil_weekly_climate.csv": weekly,
        "baseline_climate_case_alignment_2023W35_2024W45.csv": aligned,
    }
    for filename, table in outputs.items():
        table.to_csv(args.out_dir / filename, index=False, encoding="utf-8-sig")
        print(f"Wrote {filename}: {len(table)} rows", flush=True)
    (args.out_dir / "inmet_weekly_quality_summary_2023_2025.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["baseline_coverage"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
