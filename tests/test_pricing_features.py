import unittest
import numpy as np
import pandas as pd
from fixtures import order, candles
from pp_data.features import build_features
from pp_audit.market import audit_market

def dated_order(created, closed=None, **changes):
    at = pd.Timestamp(created, tz='UTC')
    end = at + pd.Timedelta(minutes=30)
    return order(createdAt=at, closedAt=closed or end, expirationDate=end, **changes)

def history_for(raw):
    blocks = []
    for value in pd.to_datetime(raw.createdAt, utc=True):
        blocks.append(candles(value.floor('min') - pd.Timedelta(minutes=1442), 1444))
    return pd.concat(blocks, ignore_index=True).drop_duplicates('open_time_utc')

class PricingFeatureTests(unittest.TestCase):

    def test_future_candles_and_outcomes_do_not_change_features(self):
        raw = pd.DataFrame([dated_order('2026-06-15 12:00:30')])
        history = history_for(raw)
        before, times = build_features(raw, audit_market(history).usable)
        changed = history.copy()
        future = changed.close_time_utc.gt(raw.createdAt.iloc[0])
        changed.loc[future, ['open', 'high', 'low', 'close']] *= 3
        raw.loc[0, 'realizedProfit_USDC'] = 1000000.0
        raw.loc[0, 'closedAt'] = pd.Timestamp('2026-06-15 12:01:00Z')
        raw.loc[0, 'premiumAmount_USDC'] = 500
        raw.loc[0, 'markup_per_BTC_USDC'] = 492
        after, _ = build_features(raw, audit_market(changed).usable)
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(times.iloc[0], pd.Timestamp('2026-06-15 11:59:59.999Z'))
        self.assertAlmostEqual(before.market_price_usdc_per_btc.iloc[0], 60001)

    def test_returns_and_volatility_use_n_plus_one_consecutive_closed_candles(self):
        history = candles('2026-06-01', 1441)
        history[['open', 'high', 'low', 'close']] = 100.0
        history.loc[1440, ['open', 'high', 'low', 'close']] = 110.0
        raw = pd.DataFrame([dated_order('2026-06-02 00:01:00')])
        features, _ = build_features(raw, audit_market(history).usable)
        for n in (60, 240, 1440):
            self.assertAlmostEqual(features[f'log_return_{n}m'].iloc[0], np.log(1.1))
            self.assertAlmostEqual(features[f'realized_vol_{n}m'].iloc[0], np.log(1.1))
        short, _ = build_features(raw, audit_market(history.iloc[1:]).usable)
        self.assertTrue(pd.isna(short.log_return_1440m.iloc[0]))
        gap, _ = build_features(raw, audit_market(history.drop(index=1430)).usable)
        self.assertTrue(pd.isna(gap.realized_vol_60m.iloc[0]))
