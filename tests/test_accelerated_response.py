import importlib.util
import unittest
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE
from pp_models.fast_response_features import response_bars, response_features


def candles(returns, start='2026-05-30'):
    starts = pd.date_range(start, periods=5*len(returns), freq='min', tz='UTC')
    return pd.DataFrame({'open_time_utc': starts,
        'close_time_utc': starts+pd.Timedelta(milliseconds=59999),
        'close': 100*np.exp(np.repeat(np.cumsum(returns), 5))})


class AcceleratedResponseTests(unittest.TestCase):
    def test_square_growth_and_half_extra_recovery(self):
        # Flat warmup => sigma floor; normalized excess decline exactly .5.
        market = candles([0.]*6+[-.00006]*3+[0.]*3)
        bars = response_bars(market, lookback=4, interval_minutes=5, power=2)
        state = bars.fast_response_state
        np.testing.assert_allclose(state.iloc[6:9], np.array([.5, 1., 1.5])**2/9, rtol=1e-9)
        self.assertGreater(state.iloc[8]-state.iloc[7], state.iloc[7]-state.iloc[6])
        self.assertAlmostEqual(state.iloc[9], state.iloc[8]/2)
        self.assertAlmostEqual(state.iloc[10], state.iloc[8]/4)
        self.assertEqual(bars.fast_response_phase.iloc[6:10].tolist(),
                         ['first_down','continued_down','continued_down','recovery'])
        linear = response_bars(market, lookback=4, interval_minutes=5, power=1)
        np.testing.assert_allclose(linear.fast_response_state.iloc[6:9], np.array([.5,1.,1.5])/3, rtol=1e-9)
        self.assertAlmostEqual(linear.fast_response_state.iloc[9], linear.fast_response_state.iloc[8]/2)

    def test_five_minute_boundary_future_invariance_and_saturation(self):
        market = candles([0.]*6+[-.02]*3+[0.]*3)
        times = pd.Series(pd.to_datetime(['2026-05-30 00:34:59.999Z','2026-05-30 00:35:00.000Z'],utc=True))
        a = response_features(times,market,lookback=4,interval_minutes=5,power=2)
        self.assertEqual(a.fast_response_state.tolist(), [0.,1.])
        market.loc[market.open_time_utc.ge(times.iloc[-1]),'close'] *= 100
        pd.testing.assert_frame_equal(a,response_features(times,market,lookback=4,interval_minutes=5,power=2))

    def test_default_24h_warmup_gap_rewarm_and_midnight_continuity(self):
        market = candles([0.]*600)
        bars = response_bars(market,lookback=288,interval_minutes=5,power=2)
        self.assertTrue(bars.fast_response_state.iloc[:289].isna().all())
        self.assertEqual(bars.fast_response_state.iloc[289],0.)
        broken = response_bars(market.drop(index=300*5+2),lookback=288,interval_minutes=5,power=2)
        self.assertTrue(broken.fast_response_state.iloc[300:590].isna().all())
        self.assertEqual(broken.fast_response_state.iloc[590],0.)
        # A small synthetic window isolates state persistence across UTC midnight.
        signal = response_bars(candles([0.]*6+[-.00006]*3+[0.]*3,start='2026-05-30 23:20'),
                               lookback=4,interval_minutes=5,power=2)
        self.assertEqual(signal.response_available_at_utc.iloc[7],pd.Timestamp('2026-05-31T00:00Z'))
        self.assertGreater(signal.fast_response_state.iloc[8],signal.fast_response_state.iloc[7])
        with self.assertRaisesRegex(ValueError,'Missing|contiguous'):
            response_features(pd.Series([pd.Timestamp('2026-05-31T01:05Z')]),
                market.drop(index=300*5+2),lookback=288,interval_minutes=5,power=2)

    def model_api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.accelerated_response'))
        from pp_models import accelerated_response
        return accelerated_response

    def test_unregularized_fit_cap_and_calls_unchanged(self):
        api=self.model_api()
        X=pd.DataFrame({'option_type':[1,0],BASE_FEATURE:[10.,100.],'fast_response_state':[1.,1.]})
        model=api.AcceleratedResponseModel(0.).fit(X,pd.Series([1000.,1.]))
        self.assertEqual(model.beta,3.)
        self.assertEqual(model.predict_raw(X).tolist(),[40.,100.])
        self.assertTrue(model.diagnostics()['response_amplitude_capped'])
        y=pd.Series([20.,1.])
        weak=api.AcceleratedResponseModel(0.).fit(X,y)
        self.assertEqual(weak.beta,1.)
        self.assertLess(api.AcceleratedResponseModel(.1).fit(X,y).beta,weak.beta)
        y.iloc[1]=1e12
        self.assertEqual(api.AcceleratedResponseModel(0.).fit(X,y).beta,weak.beta)
        self.assertEqual(api.AcceleratedResponseModel(0.).fit(X,y*0).beta,0.)
        with self.assertRaises(ValueError): api.AcceleratedResponseModel(True)

    def test_legacy_15minute_default_window_remains_causal(self):
        # 9000 complete minutes = 600 old bars.
        market=candles([0.]*1800)
        bars=response_bars(market)
        self.assertTrue(bars.fast_response_state.iloc[:97].isna().all())
        self.assertEqual(bars.fast_response_state.iloc[97],0.)
        broken=response_bars(market.drop(index=100*15+1))
        self.assertTrue(broken.fast_response_state.iloc[100:198].isna().all())
        self.assertEqual(broken.fast_response_state.iloc[198],0.)
