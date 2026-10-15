import unittest
import numpy as np
import pandas as pd
from pp_models.two_stage_error_analysis import known_error_signals, error_metrics


class ErrorAnalysisTests(unittest.TestCase):
    def sample(self):
        days = pd.date_range('2026-04-01', periods=5, tz='UTC')
        frame = pd.DataFrame([{'day': day, 'model': 'h0', 'side': 0, 'actual': 10.,
                               'prediction': 8., 'quantity': 2.} for day in days for _ in range(10)])
        metadata = pd.DataFrame({'closed_at_utc': frame.day + pd.Timedelta(hours=1)}, index=frame.index)
        return frame, metadata

    def test_history_uses_only_prior_days_and_already_closed_orders(self):
        frame, metadata = self.sample()
        # These April 1 orders are still unknown on April 4.
        metadata.loc[:9, 'closed_at_utc'] = pd.Timestamp('2026-04-04', tz='UTC')
        first = known_error_signals(frame, metadata)
        april4 = first.loc[first.day.eq(pd.Timestamp('2026-04-04', tz='UTC'))]
        self.assertTrue(april4.history_orders.eq(20).all())
        self.assertTrue(april4.history_creation_days.eq(2).all())
        self.assertTrue(april4.history_error_state.eq('insufficient_history').all())
        # April 5 can use four days of settled predictions.
        april5 = first.loc[first.day.eq(pd.Timestamp('2026-04-05', tz='UTC'))]
        self.assertTrue(april5.history_error_state.eq('prior_underprediction').all())
        self.assertTrue(april5.history_mean_error.eq(-2.).all())
        frame.loc[frame.day.ge(pd.Timestamp('2026-04-04', tz='UTC')), 'actual'] = 100000.
        second = known_error_signals(frame, metadata)
        pd.testing.assert_frame_equal(april4.filter(like='history_'), second.loc[april4.index].filter(like='history_'))

    def test_error_sign_and_quantity_weighted_totals(self):
        frame = pd.DataFrame({'actual': [10., 30.], 'prediction': [6., 36.], 'quantity': [2., 3.],
                              'day': pd.to_datetime(['2026-04-01', '2026-04-02'], utc=True)})
        result = error_metrics(frame)
        self.assertEqual(result['mean_error_per_btc'], 1.)
        self.assertEqual(result['gross_underprediction_usdc'], 8.)
        self.assertEqual(result['gross_overprediction_usdc'], 18.)
        self.assertEqual(result['total_bias_usdc'], 10.)
        self.assertEqual(result['creation_days'], 2)

if __name__ == '__main__': unittest.main()
