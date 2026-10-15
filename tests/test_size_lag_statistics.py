import unittest
import numpy as np
import pandas as pd
from scripts.size_lag_statistics import size_thresholds, matched_daily, lag_pairs, select_events


class SizeLagTests(unittest.TestCase):
    def test_threshold_excludes_unsettled_same_day_and_future(self):
        t=pd.Timestamp('2026-04-01',tz='UTC')
        f=pd.DataFrame({'created':[t-pd.Timedelta(days=1)]*31+[t,t+pd.Timedelta(days=1)],
            'closed':[t-pd.Timedelta(hours=1)]*30+[t+pd.Timedelta(hours=1)]*3,
            'side':['put']*33,'tenor':[1800]*33,'quantity':list(range(1,31))+[999,999,999]})
        flags,log=size_thresholds(f,t,t+pd.Timedelta(days=1))
        self.assertAlmostEqual(log.iloc[0].threshold,27.1)
        self.assertTrue(flags.loc[31]);self.assertTrue(pd.isna(flags.loc[32]))

    def test_matching_removes_contract_mix(self):
        f=pd.DataFrame({'day':[pd.Timestamp('2026-04-01',tz='UTC')]*20,
            'tenor':[1]*10+[4]*10,'large':[True]*9+[False]+[True]+[False]*9,
            'actual':[10]*10+[100]*10})
        self.assertLess(f.loc[f.large,'actual'].mean(),f.loc[~f.large,'actual'].mean())
        d=matched_daily(f,'large',['tenor'],['actual'])
        self.assertEqual(d.iloc[0].actual_difference,0)
        self.assertEqual(d.iloc[0].orders,20)

    def test_positive_lag_means_base_later_and_does_not_compress_missing_days(self):
        days=pd.date_range('2026-04-01',periods=6,tz='UTC')
        x=pd.Series([1,5,np.nan,2,4,9],index=days)
        y=x.shift(1)
        p=lag_pairs(x,y,1)
        np.testing.assert_allclose(p.x,p.y)
        self.assertEqual(len(p),4)

    def test_event_selection_uses_first_exceedance_and_separation(self):
        t=pd.date_range('2026-04-01',periods=200,freq='h',tz='UTC')
        signal=pd.Series(0.,index=t);signal.iloc[[30,31,100,103]]=2
        events=select_events(signal,pd.Series(1.,index=t),t[0],t[-1],72)
        self.assertEqual(events,[t[30],t[103]])

    def test_balanced_event_panel_cannot_mix_phase_populations(self):
        from scripts.analyze_size_and_base_lag import balanced_events
        times=pd.date_range('2026-04-03',periods=2,freq='4D',tz='UTC')
        events=pd.DataFrame({'side':['put']*2,'quantile':[.99]*2,'event':[0,1],
            'month':['2026-04']*2,'event_time':times})
        rows=[]
        for i,t in enumerate(times):
            for offset,base in [(-.5,100),(25,200)]:
                if i==1 and offset==25:continue
                for _ in range(2):rows.append({'side':'put','created':t+pd.Timedelta(hours=offset),
                    'base':base,'actual':150,'error':base-150,'base_normalized':base/100,
                    'tenor':1800,'position_bin':'middle'})
        detail,summary,joint=balanced_events(pd.DataFrame(rows),events)
        self.assertEqual(set(detail.event),{0})
        self.assertTrue(summary.events.eq(1).all())
        self.assertTrue(joint.under_then_over.eq(1).all())
        self.assertTrue(joint.normalized_change_mean.eq(1).all())


if __name__=='__main__':unittest.main()
