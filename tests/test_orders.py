import unittest
import pandas as pd
from fixtures import order, metric
from pp_audit.orders import audit_orders


class OrderAuditTests(unittest.TestCase):
    def test_missing_payout_is_not_zero_and_totals_use_same_rows(self):
        raw = pd.DataFrame([
            order(),
            order(premiumAmount_USDC=500, orderPrice_per_BTC_USDC=500,
                  markup_per_BTC_USDC=492, realizedProfit_USDC=None, closedAt=None),
            order(realizedProfit_USDC=20, apiBasePrice_per_BTC_USDC=None,
                  markup_per_BTC_USDC=None),
        ])
        result = audit_orders(raw)
        self.assertEqual(metric(result, "outcome_missing_rows"), 1)
        self.assertEqual(metric(result, "zero_outcome_rows"), 1)
        self.assertEqual(metric(result, "positive_outcome_rows"), 1)
        self.assertEqual(metric(result, "labeled_recorded_premium_usdc"), 20)
        self.assertEqual(metric(result, "labeled_outcome_usdc"), 20)
        self.assertEqual(metric(result, "common_candidate_rows"), 1)

    def test_overlapping_time_anomalies_are_counted_once_in_union(self):
        raw = pd.DataFrame([
            order(),
            order(closedAt="2026-04-30 23:00:00", expirationDate="2026-04-30 23:01:00"),
            order(closedAt="2026-04-30 23:00:00"),
        ])
        result = audit_orders(raw)
        self.assertEqual(metric(result, "closed_before_created_rows"), 2)
        self.assertEqual(metric(result, "expired_at_or_before_creation_rows"), 1)
        self.assertEqual(metric(result, "time_order_anomaly_rows"), 2)
        self.assertEqual(metric(result, "technical_candidate_rows"), 1)
        # A normal 20-second creation lag is not a >60-second deviation.
        self.assertEqual(metric(result, "nominal_duration_deviation_over_60s_rows"), 1)

    def test_invalid_numbers_and_price_identity_failures_are_excluded(self):
        raw = pd.DataFrame([
            order(quantity_BTC=0),
            order(realizedProfit_USDC=float("inf")),
            order(orderPrice_per_BTC_USDC=9),
            order(optionType_label="Put"),
            order(apiBasePrice_per_BTC_USDC="broken"),
        ])
        result = audit_orders(raw)
        self.assertEqual(metric(result, "numeric_invalid_rows"), 2)
        self.assertEqual(metric(result, "outcome_invalid_rows"), 1)
        self.assertEqual(metric(result, "technical_candidate_rows"), 1)
        self.assertEqual(metric(result, "common_candidate_rows"), 0)
        self.assertEqual(metric(result, "normalized_price_identity_error_rows"), 1)

    def test_duplicate_rows_do_not_inflate_candidate_count(self):
        result = audit_orders(pd.DataFrame([order(), order()]))
        self.assertEqual(metric(result, "duplicate_rows"), 1)
        self.assertEqual(metric(result, "technical_candidate_rows"), 1)

    def test_empty_input_fails_and_all_missing_outcomes_stay_unavailable(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            audit_orders(pd.DataFrame(columns=order().keys()))
        result = audit_orders(pd.DataFrame([order(realizedProfit_USDC=None)]))
        self.assertTrue(pd.isna(metric(result, "labeled_outcome_usdc")))
        self.assertTrue(pd.isna(metric(result, "top_1pct_outcome_share")))
        self.assertTrue(pd.isna(result.segments.iloc[0].positive_outcome_rate))

    def test_units_and_concentration_are_quantity_aware(self):
        raw = pd.DataFrame([
            order(quantity_BTC=2, premiumAmount_USDC=20, realizedProfit_USDC=30),
            order(realizedProfit_USDC=10),
        ])
        result = audit_orders(raw)
        self.assertEqual(metric(result, "common_base_premium_usdc"), 24)
        self.assertEqual(metric(result, "common_recorded_premium_usdc"), 30)
        self.assertEqual(metric(result, "top_1pct_outcome_share"), .75)

    def test_invalid_premiums_have_a_direct_diagnostic_not_a_false_identity_error(self):
        result = audit_orders(pd.DataFrame([
            order(premiumAmount_USDC=None),
            order(premiumAmount_USDC=-10, orderPrice_per_BTC_USDC=-10),
        ]))
        self.assertEqual(metric(result, "premium_input_invalid_rows"), 2)
        self.assertEqual(metric(result, "normalized_price_identity_error_rows"), 0)
        self.assertEqual(metric(result, "technical_candidate_rows"), 0)


if __name__ == "__main__":
    unittest.main()
