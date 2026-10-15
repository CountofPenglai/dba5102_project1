import unittest
import numpy as np
import pandas as pd
from pp_models import weekly_residual as wr
from pp_models import weekly_residual_features as wf

class SignalTests(unittest.TestCase):
    def test_streak_requires_persistence_and_recovers(self):
        f=wf.streak_state(np.array([2.,2.,2.,0.,0.,0.]+[0.]*9),np.ones(15))
        np.testing.assert_allclose(f[:3],[0,0,np.sqrt(3)-1])
        self.assertAlmostEqual(f[3],f[2]/2)
        self.assertAlmostEqual(f[4],f[2]/4)
        self.assertEqual(f[-1],0.)
        self.assertEqual(wf.streak_state(np.array([10.,0.,10.,0.]),np.ones(4)).max(),0.)

    def test_gap_resets_and_wrong_direction_never_triggers(self):
        f=wf.streak_state(np.array([2.,2.,np.nan,2.,2.]),np.ones(5))
        self.assertTrue((f==0).all())
        self.assertTrue((wf.streak_state(-np.ones(10),np.ones(10))==0).all())

    def test_future_market_cannot_change_past_features(self):
        from pp_data.features import BASE_FEATURE
        t=pd.date_range('2026-03-01',periods=1900,freq='min',tz='UTC')
        m=pd.DataFrame({'open_time_utc':t,'close_time_utc':t+pd.Timedelta(milliseconds=59999),
            'close':10000*np.exp(np.cumsum(np.sin(np.arange(1900))*.001))})
        c=pd.Series([t[1700],t[1701]+pd.Timedelta(seconds=10)],index=['a','b'])
        x=pd.DataFrame({BASE_FEATURE:100.,'remaining_seconds':3600.,'market_price_usdc_per_btc':10000.,'option_type':[0,1]},index=c.index)
        a=wf.features(x,c,m)
        m.loc[1701:,'close']*=10
        b=wf.features(x,c,m)
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue(a.available.all())
        m=m.drop(index=1699)
        missing=wf.features(x,c,m)
        self.assertTrue(missing.jump.eq(0).all())

class ExtraTests(unittest.TestCase):
    def fixture(self,y=140.,g=1.):
        return pd.DataFrame({'h':100.,'actual':y,'jump':g,'streak':g,'side':0,'quantity':1.,
            'created':pd.date_range('2026-03-01',periods=200,freq='3h',tz='UTC')})

    def test_constant_shrinkage_and_caps(self):
        m=wr.Extras().fit(self.fixture())
        self.assertEqual(m.state['additive'],5.)
        self.assertEqual(m.state['scale'],.05)
        self.assertEqual(m.state['jump']['0']['beta'],.05)
        self.assertEqual(m.state['streak10']['0']['beta'],.1)
        p=m.predict(self.fixture())
        self.assertAlmostEqual(p['level_add'].iloc[0],105.)
        self.assertAlmostEqual(p['streak05'].iloc[0],105.)
        self.assertAlmostEqual(p['streak10'].iloc[0],110.)

    def test_negative_residual_turns_off_instead_of_positive_only_fit(self):
        f=self.fixture(y=80.);f.loc[0,'actual']=1000
        m=wr.Extras().fit(f)
        self.assertEqual(m.state['jump']['0']['beta'],0.)
        self.assertEqual(m.state['scale'],-.05)
        self.assertTrue(m.predict(f)['jump'].eq(100.).all())

    def test_sparse_support_and_no_signal_fall_back(self):
        f=self.fixture();f.loc[20:,'jump']=0.
        m=wr.Extras().fit(f)
        self.assertEqual(m.state['jump']['0']['beta'],0.)
        self.assertFalse(m.state['jump']['1']['supported'])
        m=wr.Extras().fit(f.iloc[:10])
        self.assertEqual(m.state['scale'],0.)
        self.assertEqual(m.state['additive'],0.)

    def test_uniform_matches_historical_cash_budget(self):
        f=self.fixture();f['jump']=np.linspace(.1,1,len(f));f['quantity']=np.linspace(.1,3,len(f))
        m=wr.Extras().fit(f);p=m.predict(f)
        a=((p['jump']-f.h)*f.quantity).sum();b=((p['uniform_jump']-f.h)*f.quantity).sum()
        self.assertAlmostEqual(a,b)
