import unittest
import pandas as pd
from fixtures import order, candles, metric
from pp_audit.orders import audit_orders
from pp_audit.market import audit_market
from pp_audit.alignment import audit_alignment


class AlignmentAuditTests(unittest.TestCase):
    def test_current_minute_is_not_available_and_stale_prices_are_counted(self):
        orders = audit_orders(pd.DataFrame([
            order(createdAt="2026-04-30 00:01:30"),
            order(createdAt="2026-04-30 00:00:30"),
            order(createdAt="2026-04-30 00:05:00"),
            order(createdAt="bad"),
        ]))
        metrics, _ = audit_alignment(orders, audit_market(candles(periods=4)), {}, [])
        self.assertEqual(metric(metrics, "orders_with_fresh_completed_candle"), 1)
        self.assertEqual(metric(metrics, "orders_without_completed_candle"), 1)
        self.assertEqual(metric(metrics, "orders_with_stale_completed_candle"), 1)
        self.assertEqual(metrics.set_index("metric").loc["orders_with_fresh_completed_candle", "denominator"], 3)

    def test_a_missing_interior_minute_breaks_full_lookback_coverage(self):
        orders = audit_orders(pd.DataFrame([order(createdAt="2026-05-01 00:00:10")]))
        full = candles()
        metrics, _ = audit_alignment(orders, audit_market(full), {}, [])
        self.assertEqual(metric(metrics, "orders_with_complete_1440m_history"), 1)
        broken, _ = audit_alignment(orders, audit_market(full.drop(index=60)), {}, [])
        self.assertEqual(metric(broken, "orders_with_complete_1440m_history"), 0)
        self.assertEqual(metric(broken, "orders_with_complete_240m_history"), 1)
        ambiguous, _ = audit_alignment(orders, audit_market(pd.concat([full, full.iloc[[60]]])), {}, [])
        self.assertEqual(metric(ambiguous, "orders_with_complete_1440m_history"), 0)

    def test_settlement_at_cutoff_is_not_an_available_training_label(self):
        orders = audit_orders(pd.DataFrame([
            order(closedAt="2026-04-30 23:59:59"),
            order(closedAt="2026-05-01 00:00:00"),
            order(closedAt=None, realizedProfit_USDC=None),
            order(createdAt="2026-05-01 00:00:00"),
        ]))
        _, cutoffs = audit_alignment(orders, audit_market(candles()), {}, ["2026-05-01"])
        row = cutoffs.iloc[0]
        self.assertEqual(row.orders_created_before_cutoff, 3)
        self.assertEqual(row.outcome_unavailable_rows, 1)
        self.assertEqual(row.recorded_outcome_not_yet_settled_rows, 1)
        self.assertEqual(row.common_candidates_known_before_cutoff, 1)

    def test_metadata_shortfall_and_unknown_quote_leakage_are_explicit(self):
        orders = audit_orders(pd.DataFrame([order()]))
        metadata = dict(training_rows=1, validation_last_createdAt_utc="2026-05-01 01:00:00")
        metrics, _ = audit_alignment(orders, audit_market(candles()), metadata, [])
        self.assertAlmostEqual(metric(metrics, "validation_market_end_shortfall_seconds"), 3600.001)
        self.assertTrue(pd.isna(metric(metrics, "future_api_quote_match_share")))
        self.assertEqual(metric(metrics, "validation_file_present"), False)

    def test_no_usable_market_produces_zero_coverage_not_a_crash(self):
        raw = candles(periods=2)
        raw["high"] = 1
        metrics, _ = audit_alignment(audit_orders(pd.DataFrame([order()])), audit_market(raw), {}, [])
        self.assertEqual(metric(metrics, "orders_without_completed_candle"), 1)
        self.assertEqual(metric(metrics, "common_candidates_with_complete_1440m_history"), 0)
        self.assertTrue(pd.isna(metric(metrics, "validation_market_end_shortfall_seconds")))

    def test_partial_optional_metadata_keeps_unknown_counts_and_rates_blank(self):
        orders = audit_orders(pd.DataFrame([order()]))
        market = audit_market(candles(periods=2))
        for metadata in [
            {"pre_negative_markup_filter_training_rows": 1},
            {"negative_markup_filter": None},
        ]:
            with self.subTest(metadata=metadata):
                metrics, _ = audit_alignment(orders, market, metadata, [])
                row = metrics.set_index("metric").loc["source_negative_markup_removed_training_rows"]
                self.assertTrue(pd.isna(row.value))
                self.assertTrue(pd.isna(row.rate))

    def test_unavailable_metadata_date_cannot_report_zero_shortfall(self):
        orders = audit_orders(pd.DataFrame([order()]))
        market = audit_market(candles(periods=2))
        for value in ["NaT", "", "invalid date"]:
            with self.subTest(value=value):
                metrics, _ = audit_alignment(orders, market, {"validation_last_createdAt_utc": value}, [])
                self.assertTrue(pd.isna(metric(metrics, "validation_market_end_shortfall_seconds")))


if __name__ == "__main__":
    unittest.main()
