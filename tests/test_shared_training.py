"""Shared temporal boundaries and run-scoped training reuse."""
import copy
import unittest

import numpy as np
import pandas as pd

from pp_data.splitting import RollingSplitter, quote_windows
from pp_models.training import TrainingSession
from pp_models.forecast_history import ForwardPredictionHistory
from pp_models.h0 import H0, probability_frame
from pp_models.validation import PROTOCOL, prediction_context
from test_h0_time_decay import fixture


class SharedTrainingTests(unittest.TestCase):
    def setUp(self):
        self.pool = fixture()
        self.protocol = {**PROTOCOL, 'validation_end': '2026-07-10T00:00:00Z'}

    def test_history_boundaries_and_week_clock(self):
        origin = pd.Timestamp('2026-05-01', tz='UTC')
        at_april = self.pool.metadata.index[self.pool.metadata.created_at_utc.eq('2026-04-01')][0]
        closed_at_origin = self.pool.metadata.index[self.pool.metadata.created_at_utc.eq('2026-04-30')][0]
        self.pool.metadata.loc[closed_at_origin, 'closed_at_utc'] = origin
        splitter = RollingSplitter(self.pool, self.protocol)
        windows = list(splitter.windows())
        self.assertIn(at_april, windows[0].history_ids)
        self.assertNotIn(closed_at_origin, windows[0].history_ids)
        self.assertTrue(self.pool.metadata.loc[windows[0].history_ids, 'created_at_utc'].lt(origin).all())
        july = [w for w in windows if w.origin <= pd.Timestamp('2026-07-01', tz='UTC') < w.stop]
        self.assertEqual(july[0].origin, pd.Timestamp('2026-06-26', tz='UTC'))
        self.assertEqual(len(set().union(*(set(w.query_ids) for w in windows))), sum(len(w.query_ids) for w in windows))

    def test_quote_schedule_does_not_require_labels_or_closure(self):
        created = self.pool.metadata.created_at_utc
        actual = list(quote_windows(created, self.protocol))
        self.assertEqual(actual[0].origin, pd.Timestamp('2026-05-01', tz='UTC'))
        bad = {**self.protocol, 'update_days': 0}
        with self.assertRaises(ValueError): list(quote_windows(created, bad))
        with self.assertRaises(ValueError): list(quote_windows(created, self.protocol, warmup_days=3))

    def test_session_reuses_exact_h0_and_rejects_input_mutation(self):
        session = TrainingSession(self.pool, self.protocol)
        window = next(session.splitter.windows())
        model = session.fit(window)
        self.assertIs(model, session.fit(window))
        self.assertEqual(session.fit_counts, {'h0': 1})
        self.assertEqual(session.cache_hits, 1)
        h = window.history_ids
        expected = H0().fit(probability_frame(self.pool.X.loc[h]), self.pool.y.loc[h])
        np.testing.assert_array_equal(model.predict_raw(probability_frame(self.pool.X.loc[window.query_ids])),
                                      expected.predict_raw(probability_frame(self.pool.X.loc[window.query_ids])))
        self.pool.y.iloc[0] += 1
        with self.assertRaises(ValueError): session.fit(window)

    def test_session_rejects_foreign_pool_protocol_and_forged_history(self):
        from dataclasses import replace
        session = TrainingSession(self.pool, self.protocol)
        with self.assertRaises(ValueError): session.require(copy.deepcopy(self.pool), self.protocol)
        with self.assertRaises(ValueError): session.require(self.pool, {**self.protocol, 'history_days': 20})
        window = next(session.splitter.windows())
        with self.assertRaises(ValueError): session.fit(replace(window, history_ids=window.query_ids))

    def test_common_context_has_monthly_tail_reference_only(self):
        splitter = RollingSplitter(self.pool, self.protocol)
        window = next(w for w in splitter.windows() if w.origin == pd.Timestamp('2026-06-26', tz='UTC'))
        result = prediction_context(self.pool, window, splitter)
        self.assertEqual(set(result.month), {'2026-06', '2026-07'})
        self.assertNotIn('actual', result)
        self.assertTrue(result.index.equals(window.query_ids))
        self.assertEqual(set(result.origin), {'2026-06-26T00:00:00+00:00'})

    def test_mutating_session_protocol_cannot_reuse_an_old_fit(self):
        session = TrainingSession(self.pool, self.protocol)
        session.fit(next(session.splitter.windows()))
        session.protocol['history_days'] = 20
        with self.assertRaises(ValueError): session.fit(next(session.splitter.windows()))

    def test_forecast_history_releases_only_settled_previous_predictions(self):
        windows = list(RollingSplitter(self.pool, self.protocol).windows())
        first, second = windows[:2]
        late = first.query_ids[0]
        self.pool.metadata.loc[late, 'closed_at_utc'] = second.origin
        cache = ForwardPredictionHistory(self.pool)
        values = np.full(len(first.query_ids), 100.)
        cache.append(first, values, values)
        past = cache.settled(second.origin)
        self.assertNotIn(late, past.index)
        self.assertTrue(past.closed.lt(second.origin).all())
        self.assertTrue(past.forecast_origin.lt(second.origin).all())
        self.assertIn(late, cache.settled(second.origin+pd.Timedelta(seconds=1)).index)
        with self.assertRaises(ValueError): cache.append(first, values, values)

    def test_study_branches_share_one_h0_fit_per_origin(self):
        from pp_models.h0_time_decay_study import predict_period as decay
        from pp_models.h0_blend_ratios_study import predict_period as coarse
        protocol = {**self.protocol, 'validation_end': '2026-05-15T00:00:00Z'}
        session = TrainingSession(self.pool, protocol)
        a, _, _ = decay(self.pool, protocol, training=session)
        b, states = coarse(self.pool, protocol, training=session)
        np.testing.assert_array_equal(a.loc[a.model.eq('h0'), 'prediction'], b.loc[b.model.eq('h0'), 'prediction'])
        self.assertEqual(session.fit_counts['h0'], sum(s['h0_supported'] for s in states.values()))
        self.assertEqual(session.cache_hits, 2)


if __name__ == '__main__':
    unittest.main()
