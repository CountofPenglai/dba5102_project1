import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd


def candles():
    starts = pd.date_range("2026-06-01", periods=7, freq="min", tz="UTC")
    return pd.DataFrame({"open_time_utc": starts,
                         "close_time_utc": starts + pd.Timedelta(milliseconds=59999),
                         "close": 100 * np.exp([0, .01, -.01, .02, -.02, .01, .04])})


class DirectionalFeatureTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("pp_models.directional_features"),
                             "directional feature implementation is missing")
        from pp_models import directional_features
        return directional_features

    def test_semivol_uses_all_minutes_and_preserves_order_ids(self):
        api = self.api()
        created = pd.Series(pd.to_datetime(["2026-06-01 00:04:59.999Z"], utc=True), index=["order"])
        result = api.directional_features(created, candles(), windows=(4,))
        self.assertEqual(result.index.tolist(), ["order"])
        self.assertAlmostEqual(result.loc["order", "up_vol_4m"], np.sqrt(.00025))
        self.assertAlmostEqual(result.loc["order", "down_vol_4m"], np.sqrt(.0005))
        self.assertAlmostEqual(result.loc["order", "total_vol_4m"], np.sqrt(.00075))

    def test_unfinished_and_future_candles_do_not_enter_order_features(self):
        api = self.api()
        created = pd.Series(pd.to_datetime(["2026-06-01 00:04:30Z"], utc=True), index=["order"])
        original = candles()
        first = api.directional_features(created, original, windows=(2,))
        original.loc[4:, "close"] = [1e4, 1e5, 1e6]
        changed = api.directional_features(created, original, windows=(2,))
        pd.testing.assert_frame_equal(first, changed)
        self.assertAlmostEqual(first.up_vol_2m.iloc[0], np.sqrt(.00045))
        self.assertAlmostEqual(first.down_vol_2m.iloc[0], np.sqrt(.0002))

    def test_gap_is_not_silently_filled(self):
        api = self.api()
        created = pd.Series(pd.to_datetime(["2026-06-01 00:04:59.999Z"], utc=True), index=["order"])
        with self.assertRaisesRegex(ValueError, "Missing|contiguous"):
            api.directional_features(created, candles().drop(index=2), windows=(4,))

    def test_prefix_stops_before_future_prices_are_parsed(self):
        api = self.api()
        header = "symbol,interval,open_time_utc,close_time_utc,open,high,low,close,volume,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume\n"
        present = "BTCUSDC,1m,2026-06-30T23:59:00Z,2026-06-30T23:59:59.999Z,100,100,100,100,1,100,1,0,0\n"
        future = "BTCUSDC,1m,2026-07-01T00:00:00Z,NOT_A_DATE,bad,bad,bad,bad,bad,bad,bad,bad,bad\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "market.csv"
            path.write_text(header + present + future + "UNPARSEABLE_FUTURE_TAIL\n")
            usable, info = api.read_market_prefix(path, "2026-07-01T00:00:00Z")
            self.assertEqual(len(usable), 1)
            self.assertEqual(usable.close.iloc[0], 100)
            self.assertEqual(info["prefix_rows"], 1)
            digest = info["prefix_sha256"]
            path.write_text(header + present + future + "DIFFERENT_FUTURE_TAIL\n")
            self.assertEqual(api.read_market_prefix(path, "2026-07-01T00:00:00Z")[1]["prefix_sha256"], digest)
            with self.assertRaisesRegex(ValueError, "validation"):
                api.read_market_prefix(path, "2026-07-02T00:00:00Z")

    def test_augmentation_rejects_mismatched_canonical_features(self):
        from pp_data.io import ModelSplit
        api = self.api()
        starts = pd.date_range("2026-05-31", periods=1442, freq="min", tz="UTC")
        market = pd.DataFrame({"open_time_utc": starts,
                              "close_time_utc": starts + pd.Timedelta(milliseconds=59999),
                              "close": 100 * np.exp(np.arange(1442) * .0001)})
        X = pd.DataFrame({f"realized_vol_{w}m": [.0001 * np.sqrt(w)] for w in (60, 240, 1440)}, index=["x"])
        meta = pd.DataFrame({"created_at_utc": [starts[-1] + pd.Timedelta(minutes=1)],
                             "last_candle_close_utc": [market.close_time_utc.iloc[-1]]}, index=X.index)
        pool = ModelSplit(X, pd.Series([8.], index=X.index), meta)
        extended, stats = api.augment_pool(pool, market)
        pd.testing.assert_frame_equal(extended.X[X.columns], X)
        self.assertAlmostEqual(extended.X.up_vol_1440m.iloc[0], .0001)
        self.assertAlmostEqual(extended.X.down_vol_1440m.iloc[0], 0.)
        self.assertEqual(stats["orders"], 1)
        X.loc["x", "realized_vol_60m"] = 5
        with self.assertRaisesRegex(ValueError, "canonical"):
            api.augment_pool(pool, market)


if __name__ == "__main__":
    unittest.main()
