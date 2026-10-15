import unittest
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE
from tests.test_h0_state_fallback import fixture
from pp_models.h0_time_decay_study import PROTOCOL, predict_period as reference_predict
from pp_models.h0_blend_ratios import grid_weight
from pp_models.h0_blend_ratios_study import predict_period


def history():
    dates = pd.date_range('2026-04-01', periods=480, freq='h', tz='UTC')
    return pd.DataFrame({'base': 100., 'h0': 200., 'actual': 180., 'created': dates,
        'closed': dates+pd.Timedelta(minutes=30), 'forecast_origin': dates.floor('D')})


class GridWeightTests(unittest.TestCase):
    def test_regularized_grid_choice_and_window_difference(self):
        p = history(); origin = pd.Timestamp('2026-04-25', tz='UTC')
        # With 480 samples the regularized optimum is 434/580, nearest grid .75.
        self.assertEqual(grid_weight(p, origin, 30)['alpha'], .75)
        p.loc[:239, 'actual'] = 100.; p.loc[240:, 'actual'] = 200.
        self.assertEqual(grid_weight(p, origin, 30)['alpha'], .5)
        self.assertEqual(grid_weight(p, origin, 14)['alpha'], .75)
        self.assertEqual(grid_weight(p, origin, 14)['orders'], 240)

    def test_future_or_unsettled_forecasts_rejected(self):
        origin = pd.Timestamp('2026-04-25', tz='UTC')
        for field, value in [('closed', origin), ('created', origin),
                             ('forecast_origin', pd.Timestamp('2026-04-02', tz='UTC'))]:
            p = history(); p.loc[0, field] = value
            with self.assertRaises(ValueError): grid_weight(p, origin, 30)

    def test_low_support_and_identical_anchors_fall_back(self):
        p = history(); origin = pd.Timestamp('2026-04-25', tz='UTC')
        self.assertEqual(grid_weight(p.iloc[:99], origin, 30)['alpha'], 0.)
        p.h0 = p.base
        self.assertEqual(grid_weight(p, origin, 30)['alpha'], 0.)
        self.assertFalse(grid_weight(p, origin, 30)['supported'])


class RatioProtocolTests(unittest.TestCase):
    def test_fixed_direction_weights_and_original_h0(self):
        pool, _ = fixture(); protocol = {**PROTOCOL, 'validation_end': '2026-05-15T00:00:00Z'}
        a, _ = predict_period(pool, protocol); reference, _, _ = reference_predict(pool, protocol)
        hp = a.loc[a.model.eq('h0'), 'prediction']; bp = pool.X.loc[hp.index, BASE_FEATURE]
        np.testing.assert_allclose(hp, reference.loc[reference.model.eq('h0'), 'prediction'], atol=0, rtol=0)
        part = a.loc[a.model.eq('blend_c075_p025')]
        expected = np.where(part.side.eq(0), .75, .25)
        np.testing.assert_allclose(part.prediction, expected*hp+(1-expected)*bp)
        np.testing.assert_array_equal(part.h0_weight, expected)

    def test_current_week_labels_cannot_choose_current_weights(self):
        pool, _ = fixture(); protocol = {**PROTOCOL, 'validation_end': '2026-05-15T00:00:00Z'}
        a, sa = predict_period(pool, protocol)
        mask = pool.metadata.created_at_utc.ge('2026-05-01') & pool.metadata.created_at_utc.lt('2026-05-08')
        pool.y.loc[mask] = 10000.
        b, sb = predict_period(pool, protocol)
        first, second = '2026-05-01T00:00:00+00:00', '2026-05-08T00:00:00+00:00'
        pd.testing.assert_frame_equal(a.loc[a.origin.eq(first)], b.loc[b.origin.eq(first)])
        self.assertEqual(sa[first], sb[first]); self.assertNotEqual(sa[second], sb[second])

    def test_unsettled_and_later_labels_leave_all_predictions_unchanged(self):
        pool, _ = fixture(); protocol = {**PROTOCOL, 'validation_end': '2026-05-15T00:00:00Z'}
        mask = pool.metadata.created_at_utc.ge('2026-04-24')
        pool.metadata.loc[mask, 'closed_at_utc'] = pd.Timestamp('2026-08-01', tz='UTC')
        a, sa = predict_period(pool, protocol); pool.y.loc[mask] = 1e8
        b, sb = predict_period(pool, protocol)
        pd.testing.assert_frame_equal(a, b); self.assertEqual(sa, sb)


if __name__ == '__main__': unittest.main()
