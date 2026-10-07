import unittest
import pandas as pd
from fixtures import candles, metric
from pp_audit.market import audit_market


class MarketAuditTests(unittest.TestCase):
    def test_gap_and_ambiguous_duplicates_are_not_usable_history(self):
        raw = candles(periods=5).drop(index=2)
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
        result = audit_market(raw)
        self.assertEqual(metric(result, "missing_minutes"), 1)
        self.assertEqual(metric(result, "duplicate_open_time_rows"), 1)
        self.assertEqual(metric(result, "ambiguous_open_time_rows"), 2)
        self.assertEqual(len(result.usable), 3)

    def test_invalid_ohlc_and_infinite_volume_are_excluded(self):
        raw = candles(periods=3)
        raw.loc[0, "high"] = 59000
        raw.loc[1, "volume"] = float("inf")
        result = audit_market(raw)
        self.assertEqual(metric(result, "invalid_ohlc_rows"), 1)
        self.assertEqual(metric(result, "numeric_invalid_rows"), 1)
        self.assertEqual(len(result.usable), 1)

    def test_all_invalid_timestamps_do_not_report_a_healthy_zero_gap(self):
        raw = candles(periods=2).astype({"open_time_utc": "object"})
        raw["open_time_utc"] = ["bad", None]
        result = audit_market(raw)
        self.assertTrue(pd.isna(metric(result, "missing_minutes")))
        self.assertEqual(len(result.usable), 0)


if __name__ == "__main__":
    unittest.main()
