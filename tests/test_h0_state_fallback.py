import unittest
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE
from pp_data.io import ModelData
from pp_models.h0_state_fallback import market_states, match_states, StateGate, history_weight
from pp_models.h0_state_fallback_study import predict_period
from pp_models.h0_time_decay_study import PROTOCOL, predict_period as reference_predict


def fixture():
    dates = pd.date_range('2026-04-01', '2026-06-15', freq='h', tz='UTC', inclusive='left')
    ids = pd.Index([str(i) for i in range(len(dates))]); k = np.arange(len(ids))
    x = pd.DataFrame({BASE_FEATURE: 100.+k%17, 'remaining_seconds': 1800., 'product_duration_seconds': 1800,
        'quantity_btc': .5, 'option_type': k%2, 'realized_vol_60m': .01, 'log_moneyness': (k%9-4)*.001}, index=ids)
    y = pd.Series(np.where(k%3 == 0, 300., 0.), index=ids)
    meta = pd.DataFrame({'created_at_utc': dates, 'closed_at_utc': dates+pd.Timedelta(minutes=30)}, index=ids)
    states = pd.DataFrame({'logv60': np.sin(k/37), 'logv1440': np.cos(k/89),
        'imbalance': .1*np.sin(k/17), 'trend': .2*np.cos(k/29)}, index=dates)
    return ModelData(x, y, meta), states


class StateTests(unittest.TestCase):
    def test_completed_data_only_and_symmetric_low_high_shift(self):
        t = pd.date_range('2026-04-01', periods=3000, freq='min', tz='UTC')
        p = 100*np.exp(np.sin(np.arange(len(t))/60)*.01)
        m = pd.DataFrame({'open_time_utc': t, 'close_time_utc': t+pd.Timedelta(milliseconds=59999), 'close': p})
        before = market_states(m); m.loc[2500:, 'close'] *= 100
        after = market_states(m)
        pd.testing.assert_frame_equal(before.loc[before.index <= t[2500]], after.loc[after.index <= t[2500]])
        pool, history = fixture(); origin = pd.Timestamp('2026-05-01', tz='UTC')
        model = StateGate().fit(history, origin, pd.Timestamp('2026-04-01', tz='UTC'))
        center = np.array(model.state['center']); scale = np.array(model.state['scale'])
        q = pd.DataFrame([center, center+100*scale, center-100*scale], columns=history.columns)
        g, _ = model.predict(q)
        np.testing.assert_array_equal(g, [0., 1., 1.])
        q.iloc[0, 0] = np.nan
        self.assertEqual(model.predict(q)[0][0], 1.)
        missing = StateGate().fit(history.iloc[:5], origin, pd.Timestamp('2026-04-01', tz='UTC'))
        np.testing.assert_array_equal(missing.predict(q)[0], 1.)

    def test_state_matching_cannot_see_next_hour(self):
        _, s = fixture(); created = pd.Series(pd.to_datetime(['2026-04-05 01:59:59Z', '2026-04-05 02:00:00Z']))
        a = match_states(created, s); modified = s.copy(); modified.loc[modified.index >= '2026-04-05 02:00Z'] += 100
        b = match_states(created, modified)
        pd.testing.assert_series_equal(a.iloc[0], b.iloc[0]); self.assertFalse(a.iloc[1].equals(b.iloc[1]))


class WeightTests(unittest.TestCase):
    def test_regularized_solution_and_future_record_rejection(self):
        dates = pd.date_range('2026-04-01', periods=240, freq='h', tz='UTC')
        origin = pd.Timestamp('2026-05-01', tz='UTC')
        p = pd.DataFrame({'h0': 200., 'base': 100., 'actual': 180., 'created': dates,
            'closed': dates+pd.Timedelta(minutes=30), 'forecast_origin': dates.floor('D')})
        par = history_weight(p, origin)
        self.assertTrue(par['supported']); self.assertAlmostEqual(par['alpha'], (240*.8+100*.5)/340)
        self.assertEqual(history_weight(p.iloc[:10], origin)['alpha'], 0.)
        p.loc[0, 'closed'] = origin
        with self.assertRaises(ValueError): history_weight(p, origin)


class FallbackProtocolTests(unittest.TestCase):
    def test_reference_and_weekly_label_isolation(self):
        pool, market = fixture(); protocol = {**PROTOCOL, 'validation_end': '2026-05-15T00:00:00Z'}
        a, sa = predict_period(pool, market, protocol); old, _, _ = reference_predict(pool, protocol)
        np.testing.assert_array_equal(a.loc[a.model.eq('h0'), 'prediction'], old.loc[old.model.eq('h0'), 'prediction'])
        first = '2026-05-01T00:00:00+00:00'; second = '2026-05-08T00:00:00+00:00'
        mask = pool.metadata.created_at_utc.ge('2026-05-01') & pool.metadata.created_at_utc.lt('2026-05-08')
        pool.y.loc[mask] = 1000.
        b, sb = predict_period(pool, market, protocol)
        pd.testing.assert_frame_equal(a.loc[a.origin.eq(first)], b.loc[b.origin.eq(first)])
        self.assertEqual(sa[first], sb[first]); self.assertNotEqual(sa[second], sb[second])
        self.assertTrue(a.h0_weight.between(0, 1).all())
        self.assertTrue(a.month.eq('2026-05').all())
        for st in sa.values():
            for w in st.get('weights', {}).values():
                if w['orders']: self.assertLess(pd.Timestamp(w['latest_close']), pd.Timestamp(st['origin']))

    def test_unsettled_and_future_targets_not_used_in_blending(self):
        pool, market = fixture(); protocol = {**PROTOCOL, 'validation_end': '2026-05-15T00:00:00Z'}
        mask = pool.metadata.created_at_utc.ge('2026-04-24')
        pool.metadata.loc[mask, 'closed_at_utc'] = pd.Timestamp('2026-08-01', tz='UTC')
        a, sa = predict_period(pool, market, protocol)
        pool.y.loc[mask] = 1e8
        b, sb = predict_period(pool, market, protocol)
        pd.testing.assert_frame_equal(a, b); self.assertEqual(sa, sb)


if __name__ == '__main__': unittest.main()
