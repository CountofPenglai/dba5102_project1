import unittest
import numpy as np
import pandas as pd

from scripts.macro_event_statistics import crosses, MinuteMarket, summarize_events, matched_event


class MacroStatisticsTests(unittest.TestCase):
    def test_crossing_uses_planned_end_and_exact_boundary(self):
        t = pd.Timestamp('2026-04-10T12:30Z')
        created = pd.Series([t-pd.Timedelta(minutes=1), t, t-pd.Timedelta(minutes=2)])
        remaining = pd.Series([60, 60, 60])
        self.assertEqual(crosses(created, remaining, t).tolist(), [True, False, False])

    def test_market_uses_only_complete_future_minutes_and_missing_is_not_zero(self):
        idx = pd.date_range('2026-04-01T00:00Z', periods=10, freq='min')
        m = pd.DataFrame({'open_time_utc': idx, 'close': np.exp(np.arange(10)*.001)})
        engine = MinuteMarket(m)
        got = engine.window(idx[2]+pd.Timedelta(seconds=20), idx[6]+pd.Timedelta(seconds=20))
        self.assertEqual(got['minutes'], 3)
        self.assertAlmostEqual(got['rms_bp'], 10)
        self.assertAlmostEqual(got['down_rms_bp'], 0)
        broken = MinuteMarket(m.drop(index=4))
        self.assertFalse(broken.window(idx[2], idx[7])['complete'])
        self.assertTrue(np.isnan(broken.window(idx[2], idx[7])['rms_bp']))

    def test_event_ci_not_order_ci(self):
        got = summarize_events([1, 2, 3])
        self.assertTrue(np.isnan(got['ci_low']))
        self.assertEqual(got['events'], 3)
        self.assertEqual(got['leave_one_min'], 1.5)
        self.assertEqual(got['leave_one_max'], 2.5)

    def test_ratio_improvement_is_relative_to_one(self):
        got = summarize_events([.5, 1., 2.], reference=1.)
        self.assertAlmostEqual(got['positive_fraction'], 1/3)

    def test_matching_weights_treated_and_keeps_missing_groups_out(self):
        a = pd.DataFrame({'cell': ['a','a','b'], 'value': [1.,3.,99.]})
        b = pd.DataFrame({'cell': ['a'], 'value': [8.]})
        got = matched_event(a,b,['cell'],['value'])
        self.assertEqual(got['treated_matched'],2)
        self.assertAlmostEqual(got['value_treated'],2)
        self.assertAlmostEqual(got['value_control'],8)
        self.assertIsNone(matched_event(a,b.assign(cell='c'),['cell'],['value']))

