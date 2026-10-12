import importlib.util
import unittest
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE


def market_fixture():
    starts = pd.date_range('2026-05-30', periods=15*140, freq='min', tz='UTC')
    # Flat warmup, then two down bars and recovery, spanning midnight.
    changes = np.zeros(140)
    changes[95:97] = -.001
    prices = 100*np.exp(np.repeat(np.cumsum(changes), 15))
    return pd.DataFrame({'open_time_utc': starts,
        'close_time_utc': starts+pd.Timedelta(milliseconds=59999), 'close': prices})


class FastResponseTests(unittest.TestCase):
    def feature_api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.fast_response_features'))
        from pp_models import fast_response_features
        return fast_response_features

    def model_api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.fast_response_model'))
        from pp_models import fast_response_model
        return fast_response_model

    def test_first_decline_accumulates_then_halves_across_midnight(self):
        api = self.feature_api()
        # Short synthetic warmup; production configuration fixes 96 bars.
        states = api.response_bars(market_fixture(), lookback=4)
        s = states.fast_response_state
        self.assertGreater(s.iloc[95], 0)
        self.assertGreater(s.iloc[96], s.iloc[95])
        self.assertLessEqual(s.max(), 1)
        self.assertAlmostEqual(s.iloc[97], s.iloc[96]/2)
        self.assertAlmostEqual(s.iloc[98], s.iloc[96]/4)
        self.assertEqual(states.fast_response_phase.iloc[95:99].tolist(),
                         ['first_down', 'continued_down', 'recovery', 'recovery'])

    def test_signal_not_visible_until_full_bar_ends_and_future_is_irrelevant(self):
        api = self.feature_api()
        market = market_fixture()
        times = pd.Series(pd.to_datetime(['2026-05-30 23:59:59.999Z', '2026-05-31 00:00:00.000Z'], utc=True), index=['a', 'b'])
        first = api.response_features(times, market, lookback=4)
        self.assertEqual(first.fast_response_state.loc['a'], 0)
        self.assertGreater(first.fast_response_state.loc['b'], 0)
        market.loc[market.open_time_utc.ge(times.iloc[-1]), 'close'] *= 20
        pd.testing.assert_frame_equal(first, api.response_features(times, market, lookback=4))
        # The prior scale cannot include the current shock (floor used for flat past).
        bars = api.response_bars(market_fixture(), lookback=4)
        self.assertAlmostEqual(bars.fast_response_sigma.iloc[95], .0001)

    def test_gap_and_incomplete_latest_bar_fail_without_dropping_orders(self):
        api = self.feature_api()
        market = market_fixture().drop(index=95*15+5)
        times = pd.Series(pd.to_datetime(['2026-05-31 00:00Z'], utc=True), index=['a'])
        with self.assertRaisesRegex(ValueError, 'Missing|contiguous'):
            api.response_features(times, market, lookback=4)
        # Even when there is no bar to join, stale state from 15 minutes ago is forbidden.
        with self.assertRaisesRegex(ValueError, 'Missing|contiguous'):
            api.response_features(times, market.iloc[:95*15], lookback=4)

    def test_closed_form_nonnegative_fit_and_call_invariance(self):
        api = self.model_api()
        X = pd.DataFrame({'option_type': [1, 1, 0], BASE_FEATURE: [10., 20., 100.],
                          'fast_response_state': [1., .5, 1.]})[api.INPUTS]
        y = pd.Series([20., 30., 1.])
        model = api.FastResponseModel(.1).fit(X, y)
        expected = 200/(200+.1*1300)
        self.assertAlmostEqual(model.beta, expected)
        np.testing.assert_allclose(model.predict_raw(X), [10*(1+expected), 20*(1+.5*expected), 100])
        changed = y.copy(); changed.iloc[2] = 1e9
        self.assertEqual(api.FastResponseModel(.1).fit(X, changed).beta, model.beta)
        self.assertLess(api.FastResponseModel(10.).fit(X, y).beta, model.beta)
        self.assertEqual(api.FastResponseModel(.1).fit(X, pd.Series([1., 1., 1.])).beta, 0)
        self.assertEqual(api.FastResponseModel(.1).fit(X, y*0).beta, 0)
        self.assertEqual(api.FastResponseModel(.1).fit(X.iloc[[2]], y.iloc[[2]]).beta, 0)

    def test_input_contract_and_base_restoration(self):
        api = self.model_api()
        X = pd.DataFrame({'option_type': [1], BASE_FEATURE: [10.], 'fast_response_state': [1.]})[api.INPUTS]
        model = api.FastResponseModel(1.)
        with self.assertRaisesRegex(ValueError, 'fitted'):
            model.predict_raw(X)
        model.fit(X, pd.Series([20.]))
        X.fast_response_state = 0.
        self.assertEqual(model.predict_raw(X).tolist(), [10.])
        X.fast_response_state = 1.01
        with self.assertRaises(ValueError):
            model.predict_raw(X)
        with self.assertRaises(ValueError):
            api.FastResponseModel(-1.)
