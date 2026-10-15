import importlib.util
import itertools
import unittest
import numpy as np
import pandas as pd


def calendar(fomc=False):
    rows=[{'event_id':'release','family':'FOMC' if fomc else 'CPI',
        'kind':'statement' if fomc else 'release','event_time_utc':'2026-03-18T18:00:00Z',
        'known_by_utc':'2026-01-01T00:00:00Z'}]
    if fomc: rows.append({**rows[0],'event_id':'press','kind':'press_conference','event_time_utc':'2026-03-18T18:30:00Z'})
    return {'events':rows,'coverage':{f:{'start':'2026-01-01','end_exclusive':'2027-01-01'} for f in ['CPI','NFP','FOMC']}}


def features_fixture(events=10, per_event=16):
    n=events*per_event
    times=pd.date_range('2026-01-02',periods=events,freq='8D',tz='UTC')
    f=pd.DataFrame({'option_type':np.tile(np.arange(per_event)%2,events),'pre_exposure':0.,'cross_release':1.,
        'post_exposure':.5,'event_id':np.repeat([f'e{i}' for i in range(events)],per_event),
        'event_time':np.repeat(times,per_event),'event_month':np.repeat(times.strftime('%Y-%m'),per_event),
        'calendar_status':'known_event'})
    return f


class MacroMultiplierTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.macro_multiplier_features'))
        self.assertIsNotNone(importlib.util.find_spec('pp_models.macro_multiplier'))
        from pp_models.macro_multiplier_features import build_features
        from pp_models.macro_multiplier import MacroMultiplier
        self.build,self.model=build_features,MacroMultiplier

    def quotes(self, times, minutes):
        return pd.DataFrame({'created':pd.to_datetime(times,utc=True),'remaining_seconds':np.array(minutes)*60,'option_type':0})

    def test_future_exposure_does_not_include_elapsed_jump(self):
        q=self.quotes(['2026-03-18T17:45Z','2026-03-18T19:00Z','2026-03-18T22:00Z'],[75,60,60])
        f=self.build(q,calendar())
        np.testing.assert_allclose(f.pre_exposure,[.25,0,0])
        np.testing.assert_array_equal(f.cross_release,[1,0,0])
        np.testing.assert_allclose(f.post_exposure,[8/15,4/15,0])
        self.assertEqual(f.calendar_status.iloc[2],'no_identified_event')

    def test_fomc_press_is_future_risk_but_same_event_and_bounded(self):
        q=self.quotes(['2026-03-18T17:00Z','2026-03-18T18:15Z'],[300,30])
        f=self.build(q,calendar(True))
        self.assertEqual(f.event_id.nunique(),1)
        np.testing.assert_array_equal(f.cross_release,[1,1])
        self.assertAlmostEqual(f.post_exposure.iloc[0],1.)
        self.assertTrue(0<f.post_exposure.iloc[1]<1)

    def test_unknown_and_multiple_events_cannot_become_adjustable(self):
        q=self.quotes(['2026-03-18T17:45Z'],[75]);c=calendar()
        c['events'][0]['known_by_utc']='2026-03-19T00:00:00Z'
        f=self.build(q,c);self.assertEqual(f.calendar_status.iloc[0],'unknown_evidence')
        np.testing.assert_array_equal(f[['pre_exposure','cross_release','post_exposure']],0)
        c=calendar();c['events'].append({**c['events'][0],'event_id':'second','family':'NFP','event_time_utc':'2026-03-18T18:15:00Z'})
        self.assertEqual(self.build(q,c).calendar_status.iloc[0],'multiple_events')
        outside=self.quotes(['2027-03-18T17:45Z'],[75])
        self.assertEqual(self.build(outside,calendar()).calendar_status.iloc[0],'outside_coverage')

    def test_event_support_not_order_count_controls_identity(self):
        f=features_fixture(4,100);b=pd.Series(100.,index=f.index);y=b*3
        m=self.model(1.).fit(f,b,y,pd.Timestamp('2026-04-01T00:00Z'))
        np.testing.assert_array_equal(m.predict_diagnostics(f).multiplier,1.)
        self.assertFalse(any(m.diagnostics()['shared_enabled']))

    def test_direction_is_shrunk_and_all_cube_corners_obey_caps(self):
        f=features_fixture();b=pd.Series(100.,index=f.index)
        y=pd.Series(np.where(f.option_type.eq(0),180.,20.),index=f.index)
        m=self.model(1.).fit(f,b,y,pd.Timestamp('2026-04-01T00:00Z'))
        d=m.predict_diagnostics(f)
        self.assertGreater(d.loc[f.option_type.eq(0),'direction_adjustment'].mean(),0)
        self.assertLess(d.loc[f.option_type.eq(1),'direction_adjustment'].mean(),0)
        q=f.iloc[:16].copy();q[['pre_exposure','cross_release','post_exposure']]=list(itertools.product([0.,1.],repeat=3))*2
        q['option_type']=[0]*8+[1]*8
        d=m.predict_diagnostics(q)
        self.assertTrue(d.multiplier.between(.95-1e-10,1.05+1e-10).all())
        self.assertTrue(d.direction_adjustment.abs().le(.02+1e-10).all())
        self.assertLessEqual(m.diagnostics()['objective'],m.diagnostics()['null_objective']+1e-9)

    def test_inference_rejects_base_and_targets_and_future_events_cannot_train(self):
        f=features_fixture();b=pd.Series(100.,index=f.index)
        m=self.model(1.).fit(f,b,b*1.5,pd.Timestamp('2026-04-01T00:00Z'))
        for name in ['base','actual','closed_at_utc']:
            with self.assertRaises(ValueError):m.predict_diagnostics(f.assign(**{name:1.}))
        future=f.copy();future['event_time']=pd.Timestamp('2026-04-02T00:00Z')
        z=self.model(1.).fit(future,b,b*5,pd.Timestamp('2026-04-01T00:00Z'))
        np.testing.assert_array_equal(z.predict_diagnostics(future).multiplier,1.)


if __name__=='__main__':unittest.main()
