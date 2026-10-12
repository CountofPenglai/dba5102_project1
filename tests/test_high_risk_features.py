"""Price-time features must not use future candles, payouts, or same-time orders."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from pp_data.io import ModelSplit

PATH = Path(__file__).resolve().parents[1] / 'pp_models/high_risk_features.py'


class PeakFeatureTests(unittest.TestCase):
    def api(self):
        self.assertTrue(PATH.exists(), 'high-risk feature implementation missing')
        spec = importlib.util.spec_from_file_location('pp_models.high_risk_features', PATH)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def fixture(self):
        t = pd.date_range('2026-05-01', periods=1700, freq='min', tz='UTC')
        market = pd.DataFrame({'open_time_utc': t, 'close_time_utc': t+pd.Timedelta(milliseconds=59999),
            'close': 100*np.exp(np.arange(len(t))*.0001), 'number_of_trades': 10.,
            'quote_asset_volume': 1000., 'taker_buy_quote_asset_volume': 400.})
        now = t[1601]
        x = pd.DataFrame({'option_type': [0, 1], 'product_duration_seconds': [3600, 3600],
            'remaining_seconds': [3500, 3500], 'quantity_btc': [1., 2.], 'log_moneyness': [.01, .01],
            'log_return_60m': [.006]*2, 'log_return_240m': [.024]*2, 'log_return_1440m': [.144]*2,
            'realized_vol_60m': [np.sqrt(60)*.0001]*2,
            'realized_vol_240m': [np.sqrt(240)*.0001]*2,
            'realized_vol_1440m': [np.sqrt(1440)*.0001]*2})
        meta = pd.DataFrame({'created_at_utc': [now]*2, 'closed_at_utc': [now+pd.Timedelta(hours=1)]*2,
                             'last_candle_close_utc': [market.close_time_utc.iloc[1600]]*2})
        events = pd.DataFrame({'created_at_utc': [now-pd.Timedelta(minutes=m) for m in [1000, 60, 15, 1, 0, 0]],
                               'option_type': [0, 1, 0, 1, 0, 1]})
        return ModelSplit(x, pd.Series([100., 200.]), meta), market, events

    def test_direction_and_counts_are_correct_and_future_mutations_do_not_change_inputs(self):
        api = self.api(); pool, market, events = self.fixture()
        x, dictionary = api.build_high_risk_features(pool, market, events)
        self.assertAlmostEqual(x.loc[0, 'direction_return_60m'], .006)
        self.assertAlmostEqual(x.loc[1, 'direction_return_60m'], -.006)
        self.assertAlmostEqual(x.loc[0, 'risk_rms_60m'], .0001)
        self.assertEqual(x.loc[1, 'risk_rms_60m'], 0.)
        self.assertEqual(x.loc[0, 'platform_same_count_60m'], 1.)
        self.assertEqual(x.loc[1, 'platform_same_count_60m'], 2.)
        self.assertAlmostEqual(x.loc[0, 'market_direction_pressure_60m'], -.2)
        self.assertAlmostEqual(x.loc[1, 'market_direction_pressure_60m'], .2)
        self.assertFalse(any('payout' in c or 'closed' in c or 'threshold' in c for c in x.columns))
        self.assertEqual(set(x.columns), set(dictionary.feature))
        market.loc[1601:, 'close'] = 1e12
        market.loc[1601:, 'number_of_trades'] = 1e12
        pool.y[:] = 1e12; pool.metadata['closed_at_utc'] = pd.Timestamp('2099-01-01', tz='UTC')
        events['realizedProfit_USDC'] = 1e12
        newer, _ = api.build_high_risk_features(pool, market, events)
        pd.testing.assert_frame_equal(x, newer)

    def test_platform_history_gap_is_missing_not_zero_and_real_zero_vol_is_finite(self):
        api = self.api(); pool, market, events = self.fixture()
        # A complete flat market is valid; undefined activity ratios stay missing.
        market['close'] = 100.
        for c in pool.X:
            if c.startswith('log_return') or c.startswith('realized_vol'):
                pool.X[c] = 0.
        events = events[events.created_at_utc.ge(pool.metadata.created_at_utc.iloc[0]-pd.Timedelta(minutes=60))]
        x, _ = api.build_high_risk_features(pool, market, events)
        self.assertTrue(x.platform_same_count_240m.isna().all())
        self.assertFalse(x.platform_same_count_60m.isna().any())
        self.assertTrue(np.isfinite(x.vol_ratio_60m_240m).all())
        self.assertFalse(np.isinf(x.to_numpy()).any())

    def test_canonical_candle_mismatch_and_missing_minute_are_rejected(self):
        api = self.api(); pool, market, events = self.fixture()
        with self.assertRaises(ValueError): api.build_high_risk_features(pool, market.drop(index=1599), events)
        pool.metadata['last_candle_close_utc'] += pd.Timedelta(minutes=1)
        with self.assertRaises(ValueError): api.build_high_risk_features(pool, market, events)


if __name__ == '__main__': unittest.main()
