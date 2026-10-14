import importlib.util
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


spec = importlib.util.spec_from_file_location(
    "audit_climate", Path(__file__).with_name("15_audit_climate_gap_weeks.py")
)
audit_climate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_climate)


class AuditParsingTests(unittest.TestCase):
    def test_missing_rain_is_not_zero(self):
        self.assertEqual(audit_climate.parse_value("", "rain"), (None, "blank"))
        self.assertEqual(audit_climate.parse_value("0", "rain"), (0.0, "valid"))

    def test_comma_decimals_and_missing_marker(self):
        self.assertEqual(audit_climate.parse_value(",6", "rain"), (0.6, "valid"))
        self.assertEqual(audit_climate.parse_value("-9999,0", "rain"), (None, "sentinel"))

    def test_invalid_values_are_classified(self):
        self.assertEqual(audit_climate.parse_value("error", "rain"), (None, "nonnumeric"))
        for value in ("-1", "500,1", "nan", "inf"):
            self.assertEqual(audit_climate.parse_value(value, "rain"), (None, "out_of_range"))

    def test_w53_boundary_and_168_unique_hours(self):
        start = datetime(2025, 12, 28, tzinfo=ZoneInfo("America/Sao_Paulo"))
        hours = [start.astimezone(timezone.utc) + timedelta(hours=i) for i in range(168)]
        self.assertEqual(len(set(hours)), 168)
        self.assertEqual(hours[0], datetime(2025, 12, 28, 3, tzinfo=timezone.utc))
        self.assertEqual(hours[-1], datetime(2026, 1, 4, 2, tzinfo=timezone.utc))
        self.assertEqual(hours[-1].astimezone(start.tzinfo).date().isoformat(), "2026-01-03")


if __name__ == "__main__":
    unittest.main()
