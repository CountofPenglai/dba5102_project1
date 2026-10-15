import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from pp_calendar.calendar import build_calendar, query_events
from scripts.macro_phase_statistics import PHASES, phase_mask, phase_overlap_minutes, phase_metrics, aggregate_cells, history_counts
from scripts.analyze_macro_phases import label_for_event
from types import SimpleNamespace


class MacroPhaseTests(unittest.TestCase):
    def test_boundaries_assign_each_quote_to_only_one_stage(self):
        t=pd.Timestamp('2026-04-10T12:30Z')
        times=pd.Series([t+pd.Timedelta(minutes=x) for x in [-61,-60,-1,0,14.99,15,60,239.99,240]])
        masks=[phase_mask(times,t,phase) for phase in PHASES]
        self.assertEqual([x.tolist() for x in masks],[
            [False,True,True,False,False,False,False,False,False],
            [False,False,False,True,True,False,False,False,False],
            [False,False,False,False,False,True,False,False,False],
            [False,False,False,False,False,False,True,True,False]])
        self.assertTrue(np.column_stack(masks).sum(axis=1).max()==1)

    def test_planned_overlap_is_positive_length_and_can_cover_several_stages(self):
        t=pd.Timestamp('2026-04-10T12:30Z')
        created=pd.Series([t-pd.Timedelta(minutes=90),t-pd.Timedelta(minutes=10)])
        remaining=pd.Series([180*60,10*60])
        self.assertEqual([phase_overlap_minutes(created,remaining,t,p).tolist() for p in PHASES],
                         [[60.,10.],[15.,0.],[45.,0.],[30.,0.]])

    def test_no_base_means_unknown_error_not_healthy_zero(self):
        f=pd.DataFrame({'base':[np.nan], 'actual':[100.], 'quantity':[.1], 'remaining':[600.],
                        'cross_primary':[True], 'cross_press':[False], 'known':[True],
                        'future_risk_rms_bp':[np.nan]})
        got=phase_metrics(f)
        self.assertEqual((got['risk_orders'],got['base_orders']),(1,0))
        self.assertTrue(np.isnan(got['bias']) and np.isnan(got['cash_under']))
        self.assertTrue(np.isnan(phase_metrics(f.iloc[:0])['actual_mean_risk']))

    def test_event_weighting_and_empty_cells_do_not_inflate_support(self):
        rows=pd.DataFrame({'event_id':['a','b','c'],'risk_orders':[10,1,0],'base_orders':[10,1,0],
                            'bias':[10.,-100.,np.nan],'mae':[10.,100.,np.nan],'mse':[100.,10000.,np.nan]})
        got=aggregate_cells(rows)
        self.assertEqual((got['calendar_events'],got['risk_events'],got['base_events']),(3,2,2))
        self.assertAlmostEqual(got['bias_order_equal'],0.)
        self.assertAlmostEqual(got['bias_event_equal'],-45.)
        self.assertAlmostEqual(got['largest_event_base_share'],10/11)

    def test_history_requires_settlement_before_origin_and_an_occurred_event(self):
        origin=pd.Timestamp('2026-04-01T00:00Z')
        f=pd.DataFrame({'event_id':['a','a','b'], 'event_time':[origin-pd.Timedelta(days=1)]*2+[origin],
            'created':[origin-pd.Timedelta(hours=2)]*3,
            'closed':[origin-pd.Timedelta(hours=1),origin,origin-pd.Timedelta(minutes=1)],
            'base':[1.,1.,1.], 'known':[True]*3})
        self.assertEqual(history_counts(f,origin),{'risk_orders':1,'base_orders':1,'risk_events':1,'base_events':1})

    def test_revised_february_dates_have_conservative_prior_evidence(self):
        source=Path(__file__).resolve().parents[1]/'external_data/macro_events/sources/2026-10-14/verified_schedule_history.json'
        calendar=build_calendar(json.loads(source.read_text()))
        rows={r['event_id']:r for r in calendar['events']}
        self.assertEqual(len(rows),82)
        self.assertEqual(rows['US_NFP_2026-01']['known_by_utc'],'2026-02-05T05:00:00Z')
        self.assertEqual(rows['US_CPI_2026-01']['known_by_utc'],'2026-02-05T05:00:00Z')
        self.assertEqual(rows['US_NFP_2026-01']['superseded_schedule']['local_date'],'2026-02-06')
        for row in rows.values():
            if row['event_time_utc']<'2026-04-01':
                self.assertLess(pd.Timestamp(row['known_by_utc']),pd.Timestamp(row['event_time_utc'])-pd.Timedelta(days=1))
        with self.assertRaises(ValueError):
            query_events(calendar,'2026-02-11T13:00Z','2026-02-11T14:00Z','2026-02-03T12:00Z')

    def test_post_quote_exposure_does_not_charge_already_elapsed_time(self):
        t=pd.Timestamp('2026-04-29T18:00Z')
        f=pd.DataFrame({'created':[t-pd.Timedelta(minutes=10),t+pd.Timedelta(minutes=10)],'remaining':[1800,1800]})
        event=SimpleNamespace(event_id='fomc',time=t,known_time=t-pd.Timedelta(days=10),family='FOMC')
        got=label_for_event(f,event)
        self.assertEqual(got.future_post_release_minutes.tolist(),[20.,30.])
        self.assertEqual(got.cross_primary.tolist(),[True,False])
        self.assertEqual(got.cross_press.tolist(),[False,True])
        self.assertEqual(got.cross_post_release_minutes.iloc[0],20.)
        self.assertTrue(np.isnan(got.cross_post_release_minutes.iloc[1]))
