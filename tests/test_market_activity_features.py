import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd


class MarketActivityFeatureTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.market_activity_features'), 'supplemental features missing')
        from pp_models import market_activity_features
        return market_activity_features

    def market(self):
        starts = pd.date_range('2026-06-01', periods=6, freq='min', tz='UTC')
        return pd.DataFrame({'open_time_utc': starts,
            'close_time_utc': starts + pd.Timedelta(milliseconds=59999),
            'close': 100 * np.exp([0, .1, .3, 0, .4, 9.])})

    def test_dynamic_features_have_hand_computed_scale_and_ordered_drawdowns(self):
        api = self.api()
        created = pd.Series(pd.to_datetime(['2026-06-01 00:04:59.999Z']), index=['x'])
        result = api.price_dynamics(created, self.market(), short_window=2, long_window=4).iloc[0]
        self.assertAlmostEqual(result.vol_change_2m, np.sqrt(.125)-np.sqrt(.025))
        self.assertAlmostEqual(result.vol_ratio_2m_4m, np.sqrt(.125/.075))
        self.assertAlmostEqual(result.max_up_jump_4m, .4)
        self.assertAlmostEqual(result.max_down_jump_4m, .3)
        self.assertAlmostEqual(result.max_drawdown_4m, .3)
        self.assertAlmostEqual(result.max_drawup_4m, .4)

    def test_future_unfinished_prices_cannot_change_features_and_gaps_fail(self):
        api = self.api()
        created = pd.Series(pd.to_datetime(['2026-06-01 00:04:30Z']), index=['x'])
        frame = self.market()
        first = api.price_dynamics(created, frame, short_window=1, long_window=2)
        frame.loc[4:, 'close'] = [1e9, 1e10]
        pd.testing.assert_frame_equal(first, api.price_dynamics(created, frame, short_window=1, long_window=2))
        with self.assertRaisesRegex(ValueError, 'contiguous|Missing'):
            api.price_dynamics(created, frame.drop(index=2), short_window=1, long_window=2)

    def test_flat_market_finite_and_drawdown_requires_peak_before_trough(self):
        api = self.api()
        created = pd.Series(pd.to_datetime(['2026-06-01 00:04:59.999Z']), index=['x'])
        frame = self.market()
        frame['close'] = [100, 101, 102, 103, 104, 105]
        row = api.price_dynamics(created, frame, short_window=2, long_window=4).iloc[0]
        self.assertEqual(row.max_drawdown_4m, 0.)
        self.assertAlmostEqual(row.max_drawup_4m, np.log(1.04))
        frame['close'] = 100.
        row = api.price_dynamics(created, frame, short_window=2, long_window=4).iloc[0]
        self.assertTrue(np.isfinite(row.drop('feature_last_close_utc').astype(float)).all())
        self.assertEqual(row.vol_ratio_2m_4m, 0.)

    def test_order_counts_include_unsettled_and_left_edge_but_exclude_same_time(self):
        api = self.api()
        events = pd.DataFrame({'created_at_utc': pd.to_datetime([
            '2026-06-01 07:59:59Z', '2026-06-01 08:00Z', '2026-06-01 10:59:59Z',
            '2026-06-01 11:00Z', '2026-06-01 11:59:59Z', '2026-06-01 12:00Z',
            '2026-06-01 12:00Z', '2026-06-01 13:00Z'], format='mixed'),
            'option_type': [0, 1, 0, 0, 1, 0, 1, 0], 'closedAt': ['2099-01-01']*8})
        created = pd.Series(pd.to_datetime(['2026-06-01 12:00Z', '2026-06-01 11:00Z']), index=['a', 'b'])
        result = api.platform_activity(created, events)
        self.assertEqual(result.loc['a'].to_dict(), {'platform_call_count_60m': 1, 'platform_put_count_60m': 1,
            'platform_call_count_240m': 2, 'platform_put_count_240m': 2})
        self.assertEqual(result.index.tolist(), ['a', 'b'])
        events['closedAt'] = 'invalid'
        events['realizedProfit_USDC'] = -999
        pd.testing.assert_frame_equal(result, api.platform_activity(created, events))

    def test_order_prefix_never_parses_future_or_outcome_values(self):
        api = self.api()
        head = 'createdAt,optionType,closedAt,realizedProfit_USDC\n'
        past = '2026-06-30 23:00:00,1,not-a-time,not-a-number\n'
        edge = '2026-07-01 00:00:00,BAD,BAD,BAD\n'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'orders.csv'
            path.write_text(head + past + edge + 'UNPARSEABLE FUTURE\n')
            events, info = api.read_order_prefix(path, '2026-07-01T00:00:00Z')
            self.assertEqual(len(events), 1)
            self.assertEqual(events.columns.tolist(), ['created_at_utc', 'option_type'])
            self.assertEqual(events.option_type.iloc[0], 1)
            path.write_text(head + past + edge + 'DIFFERENT FUTURE\n')
            self.assertEqual(api.read_order_prefix(path, '2026-07-01T00:00:00Z')[1]['prefix_sha256'], info['prefix_sha256'])
            with self.assertRaisesRegex(ValueError, 'validation'):
                api.read_order_prefix(path, '2026-07-02T00:00:00Z')
            path.write_text(head + past + '2026-06-29 00:00:00,0,,\n')
            with self.assertRaisesRegex(ValueError, 'chronological'):
                api.read_order_prefix(path, '2026-07-01T00:00:00Z')
