import importlib.util
import unittest
import numpy as np
import pandas as pd


class ComprehensiveRegimeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.base_regime_statistics'))
        from scripts import base_regime_statistics
        self.api=base_regime_statistics

    def test_error_sign_cash_and_local_slopes(self):
        f=pd.DataFrame({'base':[100.,200.],'actual':[0.,500.],'quantity':[2.,.1],
            'day':pd.to_datetime(['2026-04-01','2026-04-02'],utc=True),'tail90':[False,True]})
        r=self.api.metrics(f)
        self.assertEqual(r['mean_error'],-100.)
        self.assertEqual(r['cash_bias'],170.)
        self.assertEqual(r['cash_under'],30.)
        self.assertEqual(r['cash_over'],200.)
        self.assertEqual(r['mse_slope'],-50000.)
        self.assertEqual(r['mae_slope'],-50.)
        self.assertEqual(r['tail_error_share'],1.)
        eps=1e-5
        losses=lambda m:np.mean((f.base*m-f.actual)**2)
        self.assertAlmostEqual((losses(1+eps)-losses(1-eps))/(2*eps),r['mse_slope'],places=4)

    def test_calendar_resampling_keeps_empty_days_and_constant_bias(self):
        days,w=self.api.calendar('2026-04')
        self.assertEqual(w.shape,(2000,30));np.testing.assert_array_equal(w.sum(axis=1),30)
        f=pd.DataFrame({'base':20.,'actual':10.,'quantity':1.,'tail90':False,
            'day':pd.date_range('2026-04-01',periods=30,tz='UTC')})
        row,draws,linear=self.api.group_statistics(f,days,w)
        np.testing.assert_allclose(draws['bias'],10.)
        np.testing.assert_allclose(linear,0.,atol=1e-14)
        self.assertEqual(row['delete7_bias_min'],10.)

    def test_simultaneous_interval_detects_uncertain_vs_constant(self):
        rng=np.random.default_rng(11)
        rows=pd.DataFrame({'mean_error':[10.,0.01],'supported':[True,True]})
        result,c=self.api.simultaneous(rows,[np.zeros(2000),rng.normal(size=2000)])
        self.assertGreaterEqual(c,1.8)
        self.assertEqual(result.loc[0,'sim_low'],10.)
        self.assertLess(result.loc[1,'sim_low'],0.)
        self.assertGreater(result.loc[1,'sim_high'],0.)

    def test_support_prevents_sparse_four_month_stability(self):
        rows=[]
        for month in ['2026-04','2026-05','2026-06','2026-07']:
            rows.append(dict(side='put',family='vol',group='high',month=month,orders=40,days=10,
                mean_error=10.,day_equal_bias=10.,supported=month!='2026-06',ci_low=5.,ci_high=15.,
                sim_low=4.,sim_high=16.,delete7_bias_min=3.,delete7_bias_max=17.,mse_slope=100.,mae_slope=5.))
        dev=dict(rows[0],month='development',supported=False)
        out=self.api.stability(pd.DataFrame(rows),pd.DataFrame([dev])).iloc[0]
        self.assertFalse(out['four_month_supported']);self.assertEqual(out['tier'],0)

    def test_incomplete_leave_week_cannot_be_called_robust(self):
        rows=[]
        for month in ['2026-04','2026-05','2026-06','2026-07']:
            origin=pd.Timestamp(month+'-01',tz='UTC')
            days=[origin+pd.Timedelta(days=i) for i in range(5) for _ in range(6)]+[origin+pd.Timedelta(days=20)]
            f=pd.DataFrame({'day':days,'base':10.,'actual':[0.]*30+[100.], 'quantity':1.,'tail90':False})
            row,_,_=self.api.group_statistics(f,*self.api.calendar(month))
            row.update(side='put',family='synthetic',group='x',month=month,sim_low=-10.,sim_high=30.)
            rows.append(row)
        out=self.api.stability(pd.DataFrame(rows),pd.DataFrame([dict(rows[0],month='development')])).iloc[0]
        self.assertFalse(out.delete7_same_sign)
        self.assertLess(out.tier,2)

    def test_exact_tenor_control_removes_synthetic_mix_effect(self):
        records=[]
        for month in ['2026-04','2026-05','2026-06','2026-07']:
            for group,tenor,n,e in [('high',1800,20,-10.),('high',14400,100,10.),
                                    ('low',1800,100,-10.),('low',14400,20,10.)]:
                for i in range(n):
                    records.append(dict(side='put',month=month,group=group,tenor=tenor,error=e,
                        day=pd.Timestamp(month+'-01',tz='UTC')+pd.Timedelta(days=i%20)))
        f=pd.DataFrame(records)
        warm=pd.DataFrame({'side':['put']*80,'tenor':[1800]*40+[14400]*40})
        r=self.api.exact_tenor_control(f,warm,[('group',['group'])])
        self.assertTrue((r.loc[r.group.eq('high'),'raw_bias']>0).all())
        np.testing.assert_allclose(r.standardized_bias,0.)
        np.testing.assert_allclose(r.conditional_bias_difference,0.)


if __name__=='__main__':unittest.main()
