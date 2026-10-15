import unittest
import numpy as np
import pandas as pd
from pp_models.base_residual_components import contract_features, ResidualModel, event_features, EventResidual, market_features
from pp_data.features import BASE_FEATURE


def fixture():
    n=140;ix=pd.Index(range(n));t=pd.Series(pd.date_range('2026-02-01',periods=n,freq='4h',tz='UTC'),index=ix)
    x=pd.DataFrame({BASE_FEATURE:100.,'remaining_seconds':1800.,'option_type':0,
        'market_price_usdc_per_btc':10000.,'strike_price_usdc_per_btc':10000.},index=ix)
    return x,t


class ResidualTests(unittest.TestCase):
    def test_signed_shrunk_fit_and_cap_and_zero_residual(self):
        x,t=fixture();f=contract_features(x)
        for residual,sign in [(50,1),(-50,-1),(0,0)]:
            m=ResidualModel('contract').fit(f,x[BASE_FEATURE],x[BASE_FEATURE]+residual,x.option_type,t)
            p,d=m.predict(f,x[BASE_FEATURE],x.option_type)
            self.assertTrue(np.all(np.sign(p-100)==sign));self.assertTrue(np.all(np.abs(p-100)<=10+1e-10))
            if residual:self.assertLess(abs(p.iloc[0]-100),abs(residual))
        m=ResidualModel('contract').fit(f,x[BASE_FEATURE],x[BASE_FEATURE]+1e6,x.option_type,t)
        p,_=m.predict(f,x[BASE_FEATURE],x.option_type);np.testing.assert_allclose(p,110)

    def test_insufficient_side_and_nonfinite_prediction_fallback(self):
        x,t=fixture();f=contract_features(x)
        m=ResidualModel('contract').fit(f,x[BASE_FEATURE],x[BASE_FEATURE]+50,x.option_type,t)
        p,d=m.predict(f,x[BASE_FEATURE],x.option_type+1);np.testing.assert_allclose(p,100)
        f.iloc[0,0]=np.nan;p,d=m.predict(f,x[BASE_FEATURE],x.option_type)
        self.assertEqual(p.iloc[0],100);self.assertFalse(d.iloc[0])

    def test_inputs_reject_outcome_column(self):
        x,t=fixture();f=contract_features(x);f['closed_at']=1
        with self.assertRaises(ValueError):ResidualModel('contract').fit(f,x[BASE_FEATURE],x[BASE_FEATURE],x.option_type,t)

    def test_reaction_not_visible_before_five_minutes_and_future_invariant(self):
        a=pd.Timestamp('2026-04-10T12:30Z');times=pd.date_range(a-pd.Timedelta(minutes=70),periods=400,freq='min')
        prices=100*np.exp(np.arange(len(times))*.0001)
        m=pd.DataFrame({'open_time_utc':times,'close_time_utc':times+pd.Timedelta(seconds=59.999),'close':prices})
        q=pd.DataFrame({'created':[a-pd.Timedelta(minutes=10),a+pd.Timedelta(minutes=4),a+pd.Timedelta(minutes=5),a+pd.Timedelta(minutes=20)],
            'remaining_seconds':1800.,'option_type':0})
        c={'coverage':{k:{'start':'2026-01-01','end_exclusive':'2027-01-01'} for k in ['CPI','NFP','FOMC']},
           'events':[{'event_id':'CPI_A','family':'CPI','kind':'release','event_time_utc':a.isoformat(),
                      'known_by_utc':'2026-01-01T00:00:00Z'}]}
        f=event_features(q,m,c)
        self.assertTrue((f.loc[:1,['reaction_signed','reaction_abs','reaction_variance']]==0).all().all())
        self.assertTrue(f.loc[2,'reaction_abs']>0)
        m.loc[m.open_time_utc.ge(a+pd.Timedelta(minutes=5)),'close']*=9
        g=event_features(q,m,c);pd.testing.assert_frame_equal(f,g)

    def test_event_support_counts_independent_events_not_orders(self):
        x,t=fixture();f=pd.DataFrame({'option_type':0,'pre_exposure':1.,'cross_release':1.,'post_exposure':0.,
            'reaction_signed':0.,'reaction_abs':0.,'reaction_variance':0.,'event_id':'one',
            'event_time':pd.Timestamp('2026-02-01',tz='UTC'),'event_month':'2026-02','reaction_status':'available','calendar_status':'known_event'},index=x.index)
        model=EventResidual('reaction').fit(f,x[BASE_FEATURE],x[BASE_FEATURE]+50,pd.Timestamp('2026-04-01',tz='UTC'))
        p,d=model.predict(f,x[BASE_FEATURE]);np.testing.assert_allclose(p,100)
        self.assertFalse(d.any())


class FeatureTests(unittest.TestCase):
    def test_market_features_ignore_future_candles(self):
        t=pd.date_range('2026-01-01',periods=1700,freq='min',tz='UTC')
        m=pd.DataFrame({'open_time_utc':t,'close_time_utc':t+pd.Timedelta(seconds=59.999),
                        'close':10000*np.exp(np.sin(np.arange(len(t))/17)*.001),'number_of_trades':10.})
        x,_=fixture();x=x.iloc[:1].copy();x['log_return_60m']=.001;x['realized_vol_60m']=.01
        created=pd.Series([t[1500]+pd.Timedelta(seconds=20)],index=x.index)
        a=market_features(x,created,m)
        m.loc[m.open_time_utc.ge(t[1500]),'close']*=3;m.loc[m.open_time_utc.ge(t[1500]),'number_of_trades']=1e8
        b=market_features(x,created,m);pd.testing.assert_frame_equal(a,b)

    def test_supported_event_fit_changes_predictions_and_respects_cap(self):
        x,t=fixture();f=pd.DataFrame({'option_type':0,'pre_exposure':1.,'cross_release':1.,'post_exposure':1.,
            'reaction_signed':1.,'reaction_abs':1.,'reaction_variance':1.,
            'event_id':['e'+str(i%6) for i in range(len(x))],
            'event_time':pd.Timestamp('2026-02-01',tz='UTC'),
            'event_month':['2026-02' if i%2 else '2026-03' for i in range(len(x))],'reaction_status':'available','calendar_status':'known_event'},index=x.index)
        model=EventResidual('reaction').fit(f,x[BASE_FEATURE],x[BASE_FEATURE]+1e6,pd.Timestamp('2026-04-01',tz='UTC'))
        p,d=model.predict(f,x[BASE_FEATURE]);self.assertTrue(d.all());np.testing.assert_allclose(p,105)
        f['calendar_status']='unknown_evidence';p,d=model.predict(f,x[BASE_FEATURE]);np.testing.assert_allclose(p,100)


class MissingReactionTests(unittest.TestCase):
    def test_missing_observed_reaction_falls_back_to_base_and_is_not_training_support(self):
        a=pd.Timestamp('2026-04-10T12:30Z');times=pd.date_range(a-pd.Timedelta(minutes=70),periods=150,freq='min')
        m=pd.DataFrame({'open_time_utc':times,'close_time_utc':times+pd.Timedelta(seconds=59.999),'close':100*np.exp(np.arange(len(times))*.0001)})
        m=m.loc[m.open_time_utc.ne(a+pd.Timedelta(minutes=2))]
        q=pd.DataFrame({'created':[a+pd.Timedelta(minutes=20)],'remaining_seconds':1800.,'option_type':0})
        c={'coverage':{k:{'start':'2026-01-01','end_exclusive':'2027-01-01'} for k in ['CPI','NFP','FOMC']},
           'events':[{'event_id':'CPI_A','family':'CPI','kind':'release','event_time_utc':a.isoformat(),'known_by_utc':'2026-01-01T00:00:00Z'}]}
        f=event_features(q,m,c)
        model=EventResidual('reaction');model.enabled=np.ones((3,6),bool);model.coef=np.full(18,.01)
        base=pd.Series(100.,index=q.index);p,s=model.predict(f,base)
        self.assertEqual(p.iloc[0],100.);self.assertFalse(s.iloc[0])
        self.assertEqual(f.reaction_status.iloc[0],'missing_market')
        trained=EventResidual('reaction').fit(f,base,base*2,a+pd.Timedelta(days=1))
        self.assertEqual(trained.orders,0)
