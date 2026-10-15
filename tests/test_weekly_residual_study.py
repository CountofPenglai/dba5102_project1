import unittest
import numpy as np
import pandas as pd
from pp_data.io import ModelSplit
from pp_data.features import BASE_FEATURE
from pp_models import weekly_residual_study as ws

class WeeklyProtocolTests(unittest.TestCase):
    def fixture(self):
        times=pd.date_range('2026-02-01','2026-04-15',freq='3h',tz='UTC',inclusive='left')
        ix=pd.Index([str(i) for i in range(len(times))]);created=pd.Series(times,index=ix)
        x=pd.DataFrame({BASE_FEATURE:100.,'remaining_seconds':1800.,'product_duration_seconds':1800,'quantity_btc':1.,
            'option_type':np.arange(len(ix))%2,'market_price_usdc_per_btc':10000.,'strike_price_usdc_per_btc':10000.,
            'realized_vol_60m':.01,'log_moneyness':0.},index=ix)
        y=pd.Series(np.where(np.arange(len(ix))%3==0,300.,0.),index=ix)
        meta=pd.DataFrame({'created_at_utc':created,'closed_at_utc':created+pd.Timedelta(minutes=30)})
        f=pd.DataFrame({'jump':1.,'streak':1.,'jump_available':True,'streak_available':True,'available':True},index=ix)
        monthly={'2026-04-01T00:00:00+00:00':{'models':{'contract_residual':{'family':'contract','columns':ws.CONTRACT,
            'parameters':{str(s):{'supported':False} for s in (0,1)}}}}}
        return ModelSplit(x,y,meta),f,monthly

    def test_current_week_targets_do_not_leak_but_settled_last_week_updates(self):
        p,f,m=self.fixture();args=(f,'2026-04-01T00:00Z','2026-04-15T00:00Z',m)
        a,sa=ws.predict_period(p,*args)
        changed=p.metadata.created_at_utc.ge('2026-04-01')&p.metadata.created_at_utc.lt('2026-04-08')
        p.y.loc[changed]=1000.
        b,sb=ws.predict_period(p,*args)
        first='2026-04-01T00:00:00+00:00';second='2026-04-08T00:00:00+00:00'
        pd.testing.assert_frame_equal(a.loc[a.origin.eq(first)],b.loc[b.origin.eq(first)])
        self.assertEqual(sa[first],sb[first]);self.assertNotEqual(sa[second],sb[second])
        self.assertFalse(np.allclose(a.loc[a.origin.eq(second)&a.model.eq('contract_weekly'),'prediction'],
                                    b.loc[b.origin.eq(second)&b.model.eq('contract_weekly'),'prediction']))

    def test_unsettled_labels_excluded_and_oos_origins_precede_quotes(self):
        p,f,m=self.fixture();mask=p.metadata.created_at_utc.ge('2026-03-20')
        p.metadata.loc[mask,'closed_at_utc']=pd.Timestamp('2026-08-01',tz='UTC')
        a,sa=ws.predict_period(p,f,'2026-04-01T00:00Z','2026-04-15T00:00Z',m)
        p.y.loc[mask]=1e8
        b,sb=ws.predict_period(p,f,'2026-04-01T00:00Z','2026-04-15T00:00Z',m)
        pd.testing.assert_frame_equal(a,b);self.assertEqual(sa,sb)
        for o,s in sa.items():
            self.assertLess(pd.Timestamp(s['latest_training_close']),pd.Timestamp(o))
            self.assertLess(pd.Timestamp(s['latest_oos_close']),pd.Timestamp(o))
            self.assertTrue(s['oos_strict_forward'])
        self.assertTrue(a.groupby('model').size().eq(112).all())
        # Tail thresholds stay at April 1 for the whole month, despite weekly label updates.
        for side in (0,1):self.assertEqual(a.loc[a.side.eq(side),'tail_threshold'].nunique(),1)
