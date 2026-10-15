import unittest
from pp_models.report_unified_study import score_predictions as score
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE
from pp_data.io import ModelData
from pp_models.h0 import H0
from pp_models.h0 import probability_frame
from pp_models.h0_time_decay import TimeDecayH0, time_weights
from pp_models.h0_time_decay_study import predict_period, PROTOCOL


def fixture():
    times = pd.date_range('2026-03-01', '2026-07-10', freq='3h', tz='UTC', inclusive='left')
    ix = pd.Index([str(i) for i in range(len(times))]); k = np.arange(len(ix))
    created = pd.Series(times, index=ix)
    x = pd.DataFrame({BASE_FEATURE: 100 + k % 17, 'remaining_seconds': 1800 + (k % 5)*100.,
        'product_duration_seconds': 3600, 'quantity_btc': .5, 'option_type': k % 2,
        'realized_vol_60m': .01, 'log_moneyness': (k % 11 - 5)*.001}, index=ix)
    y = pd.Series(np.where(k % 3 == 0, 250. + k % 31, 0.), index=ix)
    return ModelData(x, y, pd.DataFrame({'created_at_utc': created,
        'closed_at_utc': created + pd.Timedelta(minutes=30)}))


class WeightTests(unittest.TestCase):
    def test_half_life_age_and_invalid_time(self):
        origin = pd.Timestamp('2026-05-01', tz='UTC')
        created = pd.Series([origin-pd.Timedelta(days=i) for i in (1, 8, 15)])
        w = time_weights(created, origin, 7)
        np.testing.assert_allclose(w[1:]/w[:-1], [.5, .5])
        with self.assertRaises(ValueError): time_weights(created, origin, 0)
        with self.assertRaises(ValueError): time_weights(pd.Series([origin]), origin, 7)

    def test_unweighted_exactly_original_and_weighted_closed_form(self):
        p = fixture(); origin = pd.Timestamp('2026-05-01', tz='UTC')
        ids = p.X.index[p.metadata.created_at_utc.ge('2026-04-01') & p.metadata.closed_at_utc.lt(origin)]
        x = probability_frame(p.X.loc[ids]); y = p.y.loc[ids]; dates = p.metadata.loc[ids, 'created_at_utc']
        old = H0().fit(x, y, dates)
        new = TimeDecayH0(None).fit(x, y, dates, origin)
        np.testing.assert_array_equal(old.predict_raw(x), new.predict_raw(x))
        model = TimeDecayH0(7).fit(x, y, dates, origin)
        for side in (0, 1):
            mask = x.option_type.eq(side); pos = mask & y.gt(0)
            b = x.loc[pos, BASE_FEATURE].to_numpy(); w = time_weights(dates.loc[pos], origin, 7)
            expected = np.sum(w*b*y.loc[pos]) / np.sum(w*b*b)
            self.assertAlmostEqual(model.parameters[side]['severity'], expected)
            diag = model.weight_diagnostics[side]
            self.assertAlmostEqual(diag['normalized_weight_sum'], int(mask.sum()))
            self.assertLess(diag['effective_orders'], int(mask.sum()))
            np.testing.assert_array_equal(model.parameters[side]['scaler'].mean_, old.parameters[side]['scaler'].mean_)

    def test_degenerate_direction_fallback(self):
        p = fixture(); ids = p.X.index[:60:2]; x = probability_frame(p.X.loc[ids])
        dates = p.metadata.loc[ids, 'created_at_utc']; origin = pd.Timestamp('2026-05-01', tz='UTC')
        for target in (0., 200.):
            m = TimeDecayH0(7).fit(x, pd.Series(target, index=ids), dates, origin)
            pred = m.predict_raw(probability_frame(p.X.iloc[:60]))
            self.assertTrue(np.isfinite(pred).all())
            np.testing.assert_array_equal(pred[1::2], p.X.iloc[1:60:2][BASE_FEATURE])
            self.assertEqual(m.parameters[0]['probability'], float(target > 0))
            if not target: np.testing.assert_array_equal(pred[::2], 0.)


class ProtocolTests(unittest.TestCase):
    def settings(self, end='2026-05-15T00:00:00Z'):
        return {**PROTOCOL, 'validation_end': end}

    def test_april_only_training_and_continuous_week_grid(self):
        p = fixture(); a, states, _ = predict_period(p, self.settings('2026-07-10T00:00:00Z'))
        self.assertEqual(sorted(a.month.unique()), ['2026-05', '2026-06', '2026-07'])
        origins = pd.to_datetime(list(states))
        self.assertTrue(np.all(np.diff(origins.asi8) == pd.Timedelta(days=7).value))
        self.assertIn('2026-05-29T00:00:00+00:00', states)
        self.assertIn('2026-06-05T00:00:00+00:00', states)
        first = states['2026-05-01T00:00:00+00:00']
        self.assertEqual(first['earliest_training_create'], '2026-04-01T00:00:00+00:00')
        p.y.loc[p.metadata.created_at_utc.lt('2026-04-01')] = 1e9
        b, states_b, _ = predict_period(p, self.settings('2026-07-10T00:00:00Z'))
        pd.testing.assert_frame_equal(a, b); self.assertEqual(states, states_b)

    def test_same_week_no_feedback_next_week_updates(self):
        p = fixture(); a, sa, _ = predict_period(p, self.settings())
        mask = p.metadata.created_at_utc.ge('2026-05-01') & p.metadata.created_at_utc.lt('2026-05-08')
        p.y.loc[mask] = 1000.
        b, sb, _ = predict_period(p, self.settings())
        first = '2026-05-01T00:00:00+00:00'; second = '2026-05-08T00:00:00+00:00'
        pd.testing.assert_frame_equal(a.loc[a.origin.eq(first)], b.loc[b.origin.eq(first)])
        self.assertEqual(sa[first], sb[first]); self.assertNotEqual(sa[second], sb[second])
        for name in ('h0', 'h0_decay7', 'h0_decay14'):
            mask = a.origin.eq(second) & a.model.eq(name)
            self.assertFalse(np.allclose(a.loc[mask, 'prediction'], b.loc[mask, 'prediction']))

    def test_unsettled_or_future_labels_cannot_change_predictions(self):
        p = fixture(); mask = p.metadata.created_at_utc.ge('2026-04-23')
        p.metadata.loc[mask, 'closed_at_utc'] = pd.Timestamp('2026-08-01', tz='UTC')
        a, sa, _ = predict_period(p, self.settings())
        p.y.loc[mask] = 1e8; b, sb, _ = predict_period(p, self.settings())
        pd.testing.assert_frame_equal(a, b); self.assertEqual(sa, sb)
        for state in sa.values(): self.assertLess(pd.Timestamp(state['latest_training_close']), pd.Timestamp(state['origin']))

    def test_scoring_cutoff_is_separate_from_prediction(self):
        p = fixture(); pred, _, _ = predict_period(p, self.settings())
        first = pred.loc[pred.model.eq('h0')].index[0]
        p.metadata.loc[first, 'closed_at_utc'] = pd.Timestamp(self.settings()['validation_end'])
        scored, coverage = score(p, pred, self.settings()['validation_end'])
        self.assertNotIn(first, scored.index)
        self.assertEqual(coverage['labels_unavailable'], 1)
        self.assertEqual(scored.groupby('model').size().nunique(), 1)


if __name__ == '__main__': unittest.main()
