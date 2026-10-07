import unittest
import numpy as np
import pandas as pd
from fixtures import order, candles
from pp_data.features import build_features
from pp_data.pipeline import prepare
from pp_audit.market import audit_market


def dated_order(created, closed=None, **changes):
    at = pd.Timestamp(created, tz="UTC")
    end = at + pd.Timedelta(minutes=30)
    return order(createdAt=at, closedAt=closed or end,
                 expirationDate=end, **changes)


def history_for(raw):
    blocks = []
    for value in pd.to_datetime(raw.createdAt, utc=True):
        blocks.append(candles(value.floor("min") - pd.Timedelta(minutes=1442), 1444))
    return pd.concat(blocks, ignore_index=True).drop_duplicates("open_time_utc")


class PreprocessingTests(unittest.TestCase):
    def test_fixed_time_windows_purge_unsettled_labels_without_reassigning(self):
        raw = pd.DataFrame([
            dated_order("2026-05-31 23:00:00"),
            dated_order("2026-05-31 23:30:00"),  # closes exactly at train end
            dated_order("2026-06-01 00:00:00"),
            dated_order("2026-06-30 23:45:00"),  # crosses validation end
            dated_order("2026-07-01 00:00:00"),
            dated_order("2026-07-31 17:11:47", closed="2026-07-31 17:11:48Z"),
            dated_order("2026-07-31 17:11:48"),
        ])
        result = prepare(raw, history_for(raw))
        self.assertEqual(result.splits["train"].features.order_id.tolist(), ["training_000001"])
        self.assertEqual(result.splits["validation"].features.order_id.tolist(), ["training_000003"])
        self.assertEqual(result.splits["test"].features.order_id.tolist(), ["training_000005"])
        counts = result.exclusions.set_index("reason").rows
        self.assertEqual(counts["label_not_known_before_split_end"], 3)
        self.assertEqual(counts["outside_split_window"], 1)
        self.assertEqual(counts.sum(), len(raw))

    def test_missing_base_is_retained_only_for_risk_and_labels_are_separate(self):
        raw = pd.DataFrame([
            dated_order("2026-05-15 12:00:00", closed="2026-05-15 12:01:00Z",
                        quantity_BTC=2, premiumAmount_USDC=20, realizedProfit_USDC=30),
            dated_order("2026-05-15 12:02:00", apiBasePrice_per_BTC_USDC=np.nan,
                        markup_per_BTC_USDC=np.nan),
            dated_order("2026-05-15 12:03:00", realizedProfit_USDC=np.nan),
        ])
        result = prepare(raw, history_for(raw))
        train = result.splits["train"]
        self.assertEqual(train.metadata.benchmark_eligible.tolist(), [True, False])
        self.assertEqual(train.targets.payout_per_btc_usdc.tolist(), [15, 0])
        self.assertEqual(train.targets.positive_payout.tolist(), [1, 0])
        self.assertTrue(pd.isna(train.features.api_base_price_per_btc_usdc.iloc[1]))
        forbidden = {"closedAt", "closed_at_utc", "realizedProfit_USDC", "payout_usdc",
                     "payout_per_btc_usdc", "recorded_premium_usdc", "markup_per_BTC_USDC"}
        self.assertFalse(forbidden & set(train.features))
        self.assertIn("closed_at_utc", train.metadata)

    def test_future_candles_and_outcomes_do_not_change_features(self):
        raw = pd.DataFrame([dated_order("2026-06-15 12:00:30")])
        history = history_for(raw)
        before, times = build_features(raw, audit_market(history).usable)
        changed = history.copy()
        future = changed.close_time_utc.gt(raw.createdAt.iloc[0])
        changed.loc[future, ["open", "high", "low", "close"]] *= 3
        raw.loc[0, "realizedProfit_USDC"] = 1e6
        raw.loc[0, "closedAt"] = pd.Timestamp("2026-06-15 12:01:00Z")
        raw.loc[0, "premiumAmount_USDC"] = 500
        raw.loc[0, "markup_per_BTC_USDC"] = 492
        after, _ = build_features(raw, audit_market(changed).usable)
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(times.iloc[0], pd.Timestamp("2026-06-15 11:59:59.999Z"))
        self.assertAlmostEqual(before.market_price_usdc_per_btc.iloc[0], 60001)

    def test_returns_and_volatility_use_n_plus_one_consecutive_closed_candles(self):
        history = candles("2026-06-01", 1441)
        history[["open", "high", "low", "close"]] = 100.0
        history.loc[1440, ["open", "high", "low", "close"]] = 110.0
        raw = pd.DataFrame([dated_order("2026-06-02 00:01:00")])
        features, _ = build_features(raw, audit_market(history).usable)
        for n in (60, 240, 1440):
            self.assertAlmostEqual(features[f"log_return_{n}m"].iloc[0], np.log(1.1))
            self.assertAlmostEqual(features[f"realized_vol_{n}m"].iloc[0], np.log(1.1))
        short, _ = build_features(raw, audit_market(history.iloc[1:]).usable)
        self.assertTrue(pd.isna(short.log_return_1440m.iloc[0]))
        gap, _ = build_features(raw, audit_market(history.drop(index=1430)).usable)
        self.assertTrue(pd.isna(gap.realized_vol_60m.iloc[0]))

    def test_duplicate_candles_break_history_and_empty_usable_market_is_counted(self):
        raw = pd.DataFrame([dated_order("2026-06-15 12:00:30")])
        history = history_for(raw)
        duplicate = history.iloc[[-10]]
        result = prepare(raw, pd.concat([history, duplicate], ignore_index=True))
        self.assertEqual(result.exclusions.set_index("reason").rows["insufficient_market_history"], 1)
        bad = history.copy()
        bad["symbol"] = "OTHER"
        result = prepare(raw, bad)
        self.assertTrue(result.splits["validation"].features.empty)
        self.assertEqual(result.exclusions.set_index("reason").rows["insufficient_market_history"], 1)

    def test_invalid_orders_are_excluded_and_equal_timestamps_have_stable_ids(self):
        raw = pd.DataFrame([
            dated_order("2026-05-15 12:00:00", realizedProfit_USDC=5),
            dated_order("2026-05-15 11:00:00", realizedProfit_USDC=6),
            dated_order("2026-05-15 12:00:00", realizedProfit_USDC=7),
            dated_order("2026-05-15 13:00:00", closed="2026-05-15 12:59:59Z"),
        ])
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
        result = prepare(raw, history_for(raw))
        train = result.splits["train"]
        self.assertEqual(train.features.order_id.tolist(), ["training_000002", "training_000001", "training_000003"])
        self.assertEqual(train.features.order_id.tolist(), train.targets.order_id.tolist())
        self.assertEqual(result.exclusions.set_index("reason").rows["invalid_order"], 2)


if __name__ == "__main__":
    unittest.main()
