import copy
import unittest
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE
from pp_data.io import ModelSplit
from pp_models.h0_time_decay_study import PROTOCOL
from pp_models.h0_blend_ratios_study import predict_period as coarse_predict
from pp_models.h0_blend_fine_study import predict_period


def fixture():
    dates = pd.date_range('2026-04-01', '2026-07-05', freq='6h', tz='UTC', inclusive='left')
    ids = pd.Index([str(i) for i in range(len(dates))]); k = np.arange(len(ids))
    x = pd.DataFrame({BASE_FEATURE: 100.+k%17, 'remaining_seconds': 1800., 'product_duration_seconds': 1800,
        'quantity_btc': .5, 'option_type': k%2, 'realized_vol_60m': .01, 'log_moneyness': (k%9-4)*.001}, index=ids)
    y = pd.Series(np.where(k%3 == 0, 300., 0.), index=ids)
    meta = pd.DataFrame({'created_at_utc': dates, 'closed_at_utc': dates+pd.Timedelta(minutes=30)}, index=ids)
    pool = ModelSplit(x, y, meta); protocol = {**PROTOCOL, 'validation_end': '2026-07-05T00:00:00Z'}
    coarse, states = coarse_predict(pool, protocol)
    return pool, protocol, coarse, states


class FineBlendTests(unittest.TestCase):
    def test_saved_h0_and_overlapping_ratios_reproduce_coarse(self):
        pool, protocol, coarse, states = fixture(); fine = predict_period(pool, states, protocol)
        for name in ('h0', 'official_raw', 'half_blend', 'blend_c025_p025', 'blend_c050_p025'):
            a, b = (f.loc[f.model.eq(name), 'prediction'] for f in (fine, coarse))
            np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-12)
        p = fine.loc[fine.model.eq('blend_c035_p020')]
        np.testing.assert_array_equal(p.h0_weight, np.where(p.side.eq(0), .35, .2))

    def test_july_boundary_preserves_june26_until_july3(self):
        pool, protocol, _, states = fixture(); a = predict_period(pool, states, protocol)
        before = a.loc[a.day.between('2026-07-01', '2026-07-02')]
        self.assertEqual(set(before.origin), {'2026-06-26T00:00:00+00:00'})
        modified = copy.deepcopy(states)
        modified['2026-07-03T00:00:00+00:00']['h0']['0']['severity'] *= 2
        b = predict_period(pool, modified, protocol)
        pd.testing.assert_frame_equal(a.loc[a.day.lt('2026-07-03')], b.loc[b.day.lt('2026-07-03')])
        self.assertFalse(a.loc[a.day.ge('2026-07-03'), 'prediction'].equals(b.loc[b.day.ge('2026-07-03'), 'prediction']))

    def test_current_and_future_labels_cannot_change_frozen_predictions(self):
        pool, protocol, _, states = fixture(); a = predict_period(pool, states, protocol)
        pool.y.loc[pool.metadata.created_at_utc.ge('2026-06-26')] = 1e8
        b = predict_period(pool, states, protocol)
        # Month-start tail cohorts may legitimately change; they do not change any price.
        cols = ['model', 'prediction', 'h0_weight', 'origin', 'side', 'day']
        pd.testing.assert_frame_equal(a[cols], b[cols])

    def test_missing_week_cannot_silently_use_future_state(self):
        pool, protocol, _, states = fixture(); del states['2026-06-26T00:00:00+00:00']
        with self.assertRaises(ValueError): predict_period(pool, states, protocol)


if __name__ == '__main__': unittest.main()
