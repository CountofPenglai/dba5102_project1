"""Peak labels must use legal history, respect direction and preserve unknowns."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from pp_data.io import ModelSplit

PATH = Path(__file__).resolve().parents[1] / 'pp_models/high_risk_labels.py'


class PeakLabelTests(unittest.TestCase):
    def api(self):
        self.assertTrue(PATH.exists(), 'high-risk label implementation missing')
        spec = importlib.util.spec_from_file_location('pp_models.high_risk_labels', PATH)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def fixture(self):
        day = pd.Timestamp('2026-04-01', tz='UTC')
        n = 84
        x = pd.DataFrame({'option_type': [1]*n, 'product_duration_seconds': [3600]*n,
                          'quantity_btc': [1.]*n})
        y = pd.Series(np.arange(n, dtype=float))
        meta = pd.DataFrame({'created_at_utc': [day-pd.Timedelta(days=1)]*80 + [day-pd.Timedelta(days=1), day, day+pd.Timedelta(days=1), day-pd.Timedelta(days=31)],
                             'closed_at_utc': [day-pd.Timedelta(hours=1)]*80 + [day, day+pd.Timedelta(hours=1), day+pd.Timedelta(days=2), day-pd.Timedelta(days=30)]})
        y.loc[80:] = 1e9
        return ModelSplit(x, y, meta), day

    def test_excludes_today_unsettled_exact_boundary_and_expired_history(self):
        api = self.api(); pool, day = self.fixture(); c = api.load_settings()
        fitted = api.fit_peak_thresholds(pool, day, c)
        self.assertEqual(fitted['history_ids'].tolist(), list(range(80)))
        labels = api.label_orders(pool, pd.Index([81]), fitted)
        self.assertAlmostEqual(labels.loc[81, 'unit_threshold_q975'], 77.025)
        self.assertAlmostEqual(labels.loc[81, 'unit_threshold_q950'], 75.05)
        pool.y.loc[80:] = 1e15
        after = api.fit_peak_thresholds(pool, day, c)
        pd.testing.assert_frame_equal(fitted['thresholds'], after['thresholds'])

    def test_sparse_tenor_falls_back_to_same_direction_only(self):
        api = self.api(); pool, day = self.fixture()
        pool.X.loc[0:19, 'option_type'] = 0
        pool.y.loc[0:19] = 1e12
        pool.X.loc[81, 'product_duration_seconds'] = 7200
        fitted = api.fit_peak_thresholds(pool, day, api.load_settings())
        labels = api.label_orders(pool, pd.Index([81]), fitted)
        self.assertEqual(labels.loc[81, 'threshold_source'], 'direction_fallback')
        self.assertEqual(labels.loc[81, 'threshold_history_orders'], 60)
        self.assertAlmostEqual(labels.loc[81, 'unit_threshold_q975'], 77.525)

    def test_unknown_and_zero_thresholds_remain_missing_not_negative(self):
        api = self.api(); pool, day = self.fixture()
        pool.X.loc[81, 'option_type'] = 0
        fitted = api.fit_peak_thresholds(pool, day, api.load_settings())
        labels = api.label_orders(pool, pd.Index([81]), fitted)
        self.assertTrue(pd.isna(labels.loc[81, 'high_risk']))
        self.assertEqual(labels.loc[81, 'threshold_source'], 'insufficient_direction_history')
        pool.y.loc[:79] = 0
        labels = api.label_orders(pool, pd.Index([82]), api.fit_peak_thresholds(pool, day, api.load_settings()))
        self.assertTrue(pd.isna(labels.loc[82, 'high_risk']))
        self.assertFalse(labels.loc[82, 'unit_threshold_available'])

    def test_strict_threshold_and_quantity_do_not_change_unit_label(self):
        api = self.api(); pool, day = self.fixture()
        pool.y.loc[:79] = 10.
        fitted = api.fit_peak_thresholds(pool, day, api.load_settings())
        pool.y.loc[81:82] = [10., 11.]
        pool.X.loc[82, 'quantity_btc'] = .001
        labels = api.label_orders(pool, pd.Index([81, 82]), fitted)
        self.assertEqual(labels.high_risk.tolist(), [0, 1])
        self.assertEqual(labels.high_risk_cash_q975.tolist(), [0, 0])
        self.assertEqual(labels.high_risk_unit_q950.tolist(), [0, 1])

    def test_cash_primary_switch_and_invalid_time_are_explicit(self):
        api = self.api(); pool, day = self.fixture(); settings = api.load_settings()
        settings['primary_target'] = 'cash'
        pool.y.loc[81] = 10; pool.X.loc[81, 'quantity_btc'] = 100
        out = api.label_orders(pool, pd.Index([81]), api.fit_peak_thresholds(pool, day, settings))
        self.assertEqual(out.loc[81, 'high_risk'], 1)
        self.assertEqual(out.loc[81, 'high_risk_unit_q975'], 0)
        with self.assertRaises(ValueError): api.fit_peak_thresholds(pool, day+pd.Timedelta(hours=1), settings)
        with self.assertRaises(ValueError): api.fit_peak_thresholds(pool, pd.Timestamp('2026-07-01', tz='UTC'), settings)


if __name__ == '__main__': unittest.main()
