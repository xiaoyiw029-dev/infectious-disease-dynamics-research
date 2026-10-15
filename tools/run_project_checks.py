"""Run existing checks with their individual command-line contracts."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().project_root.resolve()
    source = root / "04_代码与计算" / "00_脚本"
    results = root / "05_结果与图表" / "04_滚动验证"
    jobs = [
        ("test_prepare_weekly_climate.py", []),
        ("test_audit_climate_gap_weeks.py", []),
        ("test_validate_baseline_rolling.py", [root, results / "development_rolling_v01"]),
        ("test_validate_peak_trend.py", [root, results / "development_peak_trend_v01"]),
        ("test_screen_climate_increment.py", [root, results / "climate_increment_screen_v01"]),
        ("test_diagnose_peak_overshoot.py", [root, results / "peak_overshoot_diagnostics_v01"]),
    ]
    failed = []
    for name, args in jobs:
        print(f"\n=== {name} ===", flush=True)
        outcome = subprocess.run([sys.executable, "-B", str(source / name), *map(str, args)], cwd=root, check=False)
        if outcome.returncode:
            failed.append(name)
    print(f"\nCheck groups: {len(jobs)}, failed: {len(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
