"""Calendar overlay must be causal, additive, bounded and temporary."""
import importlib.util
import unittest

import numpy as np
import pandas as pd


def event(key='cpi', family='CPI', kind='release', release='2026-05-10T12:00Z', known='2026-05-01T12:00Z'):
    return dict(event_id=key, family=family, kind=kind, event_time_utc=release, known_by_utc=known)


def calendar(events=None):
    return dict(events=[event()] if events is None else events, as_of_utc='2026-10-14T12:30Z',
        coverage={f: dict(start='2026-01-01', end_exclusive='2027-01-01') for f in ('CPI', 'NFP', 'FOMC')})


def quotes(times):
    return pd.DataFrame(dict(created=pd.to_datetime(times, utc=True), option_type=0))


class AnnouncementOverlayTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.announcement_overlay'),
                             'Independent calendar overlay is not implemented')
        from pp_models.announcement_overlay import AnnouncementOverlay
        self.model = AnnouncementOverlay()

    def test_preannouncement_gate_ramp_peak_and_exact_expiry(self):
        q = quotes(['2026-05-01T11:59Z', '2026-05-01T12:00Z', '2026-05-08T12:00Z',
                    '2026-05-09T12:00Z', '2026-05-10T12:00Z', '2026-05-10T12:30Z',
                    '2026-05-10T13:00Z'])
        d, ledger = self.model.predict_diagnostics(q, calendar())
        np.testing.assert_allclose(d.multiplier, [1, 1+.01/512, 1.0025, 1.005, 1.01, 1.005, 1], rtol=0, atol=1e-14)
        self.assertEqual(d.status.iloc[0], 'unknown_no_active_evidence')
        self.assertEqual(d.active_events.iloc[-1], 0)
        self.assertTrue((pd.to_datetime(ledger.known_by_utc) <= pd.to_datetime(ledger.quote_time)).all())

    def test_independent_events_sum_and_total_is_capped(self):
        q = quotes(['2026-05-10T12:00Z'])
        d, _ = self.model.predict_diagnostics(q, calendar([event(), event('nfp', 'NFP')]))
        self.assertAlmostEqual(d.multiplier.iloc[0], 1.02)
        c = calendar([event(str(i)) for i in range(7)])
        d, ledger = self.model.predict_diagnostics(q, c)
        self.assertAlmostEqual(d.multiplier.iloc[0], 1.05)
        self.assertEqual(d.active_events.iloc[0], 7)
        self.assertAlmostEqual(ledger.raw_adjustment.sum(), .07)
        self.assertAlmostEqual(ledger.applied_adjustment.sum(), .05)

    def test_unknown_entries_do_not_change_known_predictions_or_diagnostics(self):
        q = quotes(['2026-05-09T12:00Z', '2026-05-10T12:30Z'])
        c = calendar(); expected = self.model.predict_diagnostics(q, c)
        c['events'] += [event('future_evidence', 'NFP', known='2026-10-01T00:00Z'),
                        event('not_preannounced', 'NFP', known='2026-05-10T12:01Z')]
        actual = self.model.predict_diagnostics(q, c)
        for a, b in zip(expected, actual): pd.testing.assert_frame_equal(a, b)
        c = calendar([event(known=None)])
        d, ledger = self.model.predict_diagnostics(q, c)
        np.testing.assert_array_equal(d.multiplier, 1.)
        self.assertTrue(ledger.empty)

    def test_fomc_uses_one_event_and_only_known_anchors(self):
        statement = event('s', 'FOMC', 'statement')
        press = event('p', 'FOMC', 'press_conference', '2026-05-10T12:30Z')
        q = quotes(['2026-05-10T12:00Z', '2026-05-10T12:30Z', '2026-05-10T13:00Z', '2026-05-10T13:30Z'])
        d, ledger = self.model.predict_diagnostics(q, calendar([statement, press]))
        np.testing.assert_allclose(d.multiplier, [1.01, 1.01, 1.005, 1.])
        self.assertTrue(d.active_events.le(1).all())
        self.assertEqual(ledger.event_id.nunique(), 1)
        press['known_by_utc'] = '2026-10-01T00:00Z'
        a = self.model.predict_diagnostics(q, calendar([statement, press]))
        b = self.model.predict_diagnostics(q, calendar([statement]))
        for x, y in zip(a, b): pd.testing.assert_frame_equal(x, y)

    def test_direction_invariance_whitelist_and_coverage(self):
        q = quotes(['2026-05-09T12:00Z']*2); q.loc[1, 'option_type'] = 1
        d, _ = self.model.predict_diagnostics(q, calendar())
        np.testing.assert_array_equal(d.multiplier, [1.005, 1.005])
        for field in ('actual', 'closed_at_utc', 'base'):
            with self.assertRaises(ValueError): self.model.predict_diagnostics(q.assign(**{field: 1}), calendar())
        q.loc[0, 'created'] = pd.NaT
        with self.assertRaises(ValueError): self.model.predict_diagnostics(q, calendar())
        q = quotes(['2028-05-10T12:00Z'])
        d, _ = self.model.predict_diagnostics(q, calendar())
        self.assertEqual(d.status.iloc[0], 'outside_coverage')
        self.assertEqual(d.multiplier.iloc[0], 1.)

    def test_duplicate_events_and_naive_evidence_fail_closed(self):
        q = quotes(['2026-05-10T12:00Z'])
        c = calendar(); c['events'] *= 2
        with self.assertRaises(ValueError): self.model.predict_diagnostics(q, c)
        c = calendar(); c['events'][0]['known_by_utc'] = '2026-05-01T12:00'
        with self.assertRaises(ValueError): self.model.predict_diagnostics(q, c)


if __name__ == '__main__': unittest.main()
