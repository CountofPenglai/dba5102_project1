"""Temporal and cohort contracts for the report rerun, independent of private data."""
import importlib.util
import unittest
import numpy as np
import pandas as pd
from pp_data.io import ModelData
from pp_data.features import BASE_FEATURE
from pp_models.h0_time_decay_study import PROTOCOL, predict_period
from test_h0_time_decay import fixture
from test_announcement_overlay import calendar


class UnifiedReportTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.report_unified_study'))
        from pp_models import report_unified_study as study
        self.study = study
        self.pool = fixture()
        self.pool.X['remaining_seconds'] = 4*3600.
        self.protocol = {**PROTOCOL, 'validation_end': '2026-05-22T00:00:00Z'}
        self.reference = predict_period(self.pool, self.protocol)[0].query("model == 'official_raw'")
        self.calendar = calendar()
        template = self.calendar['events'][0]
        self.calendar['events'] = [{**template, 'event_id': str(i), 'event_time_utc': t.isoformat()}
            for i, t in enumerate(pd.date_range('2026-03-02T12:15Z', '2026-07-09T12:15Z', freq='2D'))]

    def test_announcement_rule_is_independent_of_all_labels_and_settlement(self):
        a, _, _ = self.study.predict_announcements(self.pool, self.calendar, self.reference, self.protocol)
        changed = self.pool.y.copy()
        changed[:] = 100000.
        meta = self.pool.metadata.copy(); meta['closed_at_utc'] = pd.NaT
        p = ModelData(self.pool.X, changed, meta)
        b, _, _ = self.study.predict_announcements(p, self.calendar, self.reference, self.protocol)
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue(a.multiplier.gt(1).any())
        self.assertTrue(a.multiplier.le(1.05).all())
        self.assertNotIn('actual', a)

    def test_announcement_reacts_inside_week_only_after_evidence_is_known(self):
        c = calendar(); c['events'][0]['known_by_utc'] = '2026-05-05T00:00Z'
        a, rules, ledger = self.study.predict_announcements(self.pool, c, self.reference, self.protocol)
        at = self.pool.metadata.created_at_utc.reindex(a.index)
        np.testing.assert_array_equal(a.loc[at.lt('2026-05-05'), 'multiplier'], 1.)
        self.assertTrue(a.loc[at.ge('2026-05-05') & at.lt('2026-05-08'), 'multiplier'].gt(1).all())
        self.assertEqual(rules['model_fit_calls'], 0)
        self.assertTrue((pd.to_datetime(ledger.known_by_utc) <= pd.to_datetime(ledger.quote_time)).all())

    def test_missing_query_settlement_is_not_a_prediction_input(self):
        p = self.pool; protocol = {**self.protocol, 'validation_end': '2026-05-08T00:00:00Z'}
        ref = self.reference.loc[self.reference.day.lt(protocol['validation_end'])]
        a, _, _ = self.study.predict_announcements(p, self.calendar, ref, protocol)
        meta = p.metadata.copy();meta.loc[meta.created_at_utc.ge('2026-05-01'), 'closed_at_utc'] = pd.NaT
        b, _, _ = self.study.predict_announcements(ModelData(p.X, p.y, meta), self.calendar, ref, protocol)
        pd.testing.assert_frame_equal(a, b)

    def test_announcement_preserves_cohort_and_week_context_without_support_gate(self):
        p = self.pool; protocol = {**PROTOCOL, 'validation_end': '2026-07-10T00:00:00Z'}
        ref = predict_period(p, protocol)[0].query("model == 'official_raw'")
        a, rules, ledger = self.study.predict_announcements(p, calendar(), ref, protocol)
        july = a.loc[a.day.ge('2026-07-01') & a.day.lt('2026-07-03')]
        self.assertEqual(set(july.origin), {'2026-06-26T00:00:00+00:00'})
        np.testing.assert_array_equal(july.multiplier, 1.)
        self.assertEqual(set(a.model), {'base_announcement_overlay'})
        self.assertTrue(a.multiplier.gt(1).any())
        self.assertEqual(len(a), len(ref))

    def test_combine_deduplicates_agreeing_references_and_rejects_missing_orders(self):
        a = self.reference.copy(); b = a.assign(model='candidate', prediction=a.prediction*.9)
        merged = self.study.merge_predictions([a, pd.concat([a, b])])
        self.assertEqual(len(merged), 2*len(a))
        with self.assertRaises(ValueError): self.study.merge_predictions([a, b.iloc[1:]])

    def test_combine_rejects_reference_or_tail_or_origin_disagreement(self):
        a = self.reference.copy()
        for field, value in [('prediction', 1e8), ('tail_threshold', 1e8), ('origin', '2026-04-01T00:00:00Z')]:
            b = a.copy();b.iloc[0, b.columns.get_loc(field)] = value
            with self.assertRaises(ValueError): self.study.merge_predictions([a, b])
        b = a.assign(model='candidate');b.iloc[0, b.columns.get_loc('tail_threshold')] = 1e8
        with self.assertRaises(ValueError): self.study.merge_predictions([a, b])

    def test_scoring_alone_excludes_unsettled_orders_for_every_model(self):
        a = self.study.merge_predictions([self.reference, self.reference.assign(model='candidate')])
        first = self.reference.index[0]
        self.pool.metadata.loc[first, 'closed_at_utc'] = pd.Timestamp(self.protocol['validation_end'])
        scored, coverage = self.study.score_predictions(self.pool, a, self.protocol['validation_end'])
        self.assertNotIn(first, scored.index)
        self.assertEqual(coverage['labels_unavailable'], 1)
        self.assertEqual(scored.groupby('model').size().nunique(), 1)


if __name__ == '__main__': unittest.main()
