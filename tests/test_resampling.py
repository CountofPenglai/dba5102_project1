import unittest
import numpy as np
import pandas as pd
from pp_models.resampling import draw_weights, paired_daily, differences, LOSS_COLUMNS


class ResamplingTests(unittest.TestCase):
    def test_blocks_preserve_each_calendar_month_even_on_empty_days(self):
        days = pd.date_range('2026-05-01', '2026-07-31', tz='UTC')
        for block in (3, 7, 14):
            w = draw_weights(days, block)
            self.assertEqual(w.shape, (2000, 92))
            for month in (5, 6, 7):
                mask = days.month == month
                np.testing.assert_array_equal(w[:, mask].sum(axis=1), mask.sum())
            np.testing.assert_array_equal(w, draw_weights(days, block))

    def test_macro_is_equal_month_mse_not_order_weighted(self):
        days = pd.DatetimeIndex(['2026-05-01', '2026-06-01'], tz='UTC')
        f = pd.DataFrame(0., index=days, columns=LOSS_COLUMNS)
        f['orders'] = [1, 9]; f['csse'] = [100., 0.]; f['rsse'] = [0., 900.]
        f['csae'] = [10., 0.]; f['rsae'] = [0., 90.]
        d, valid = differences(f, np.ones((1, 2)), days)
        self.assertTrue(valid[0]); self.assertAlmostEqual(d['rmse_delta'][0], 0.)
        self.assertAlmostEqual(d['mae_delta'][0], 0.)
        _, valid = differences(f, np.array([[1., 0.]]), days)
        self.assertFalse(valid[0])

    def test_pairing_rejects_changed_cohorts_or_targets(self):
        days = pd.DatetimeIndex(['2026-05-01'], tz='UTC')
        a = pd.DataFrame(dict(day=[days[0]], model=['a'], actual=[2.], prediction=[1.],
            side=[0], quantity=[.5], tail_group=['tail']), index=['one'])
        b = a.assign(model='b', prediction=3.)
        row = paired_daily(pd.concat([a, b]), 'a', 'b', days).iloc[0]
        self.assertEqual(row.cunder, .5); self.assertEqual(row.runder, 0.)
        with self.assertRaises(ValueError):
            paired_daily(pd.concat([a, b.assign(actual=3.)]), 'a', 'b', days)
