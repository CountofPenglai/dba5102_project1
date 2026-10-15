import unittest
from unittest.mock import patch
import pandas as pd
from scripts.analyze_macro_events import raw_order_study, matched_order_study, OUTCOMES, overlaps_announcements


class MacroAnalysisTests(unittest.TestCase):
    def test_press_conference_is_not_a_second_primary_event(self):
        statement = pd.Timestamp('2026-04-29T18:00Z')
        events = pd.DataFrame({'time': [statement, statement+pd.Timedelta(minutes=30)],
                               'family': ['FOMC', 'FOMC'], 'kind': ['statement', 'press_conference']})
        f = pd.DataFrame({'created': [statement-pd.Timedelta(hours=1)], 'remaining': [7200],
                          'month': ['2026-04'], 'side': ['call'], 'cross_any': [True],
                          'cross_FOMC': [True], 'cross_CPI': [False], 'cross_NFP': [False], 'press_only': [False]})
        with patch('scripts.analyze_macro_events.basic_metrics', return_value={}):
            got = raw_order_study(f, events)
        rows = got.loc[got.period.eq('all') & got.family.isin(['any', 'FOMC'])]
        self.assertEqual(rows.events.tolist(), [1, 1])

    def test_unmatched_event_stays_in_coverage_denominator(self):
        t = pd.Timestamp('2026-04-10T12:30Z')
        f = pd.DataFrame({name: [1.] for name in OUTCOMES})
        f = f.assign(created=t-pd.Timedelta(hours=1), remaining=7200, side='call', tenor=7200,
                     cross_any=True, recent_event_4h=False, position_bin='mid', risk_vol_240_bin='mid')
        events = pd.DataFrame({'time': [t], 'known': [t-pd.Timedelta(days=20)],
                               'event_id': ['CPI'], 'family': ['CPI'], 'month': ['2026-04']})
        per, summary = matched_order_study(f, events)
        self.assertEqual(len(per), 2)
        self.assertTrue(per.treated_orders.eq(1).all())
        self.assertTrue(per.treated_matched.eq(0).all())
        self.assertTrue(summary.coverage.eq(0).all())
        self.assertTrue(summary.eligible_events.eq(1).all())
        self.assertTrue(summary.events.eq(0).all())
        self.assertTrue(overlaps_announcements(t-pd.Timedelta(hours=24), t+pd.Timedelta(hours=4), events))
        self.assertFalse(overlaps_announcements(t+pd.Timedelta(days=1), t+pd.Timedelta(days=2), events))
