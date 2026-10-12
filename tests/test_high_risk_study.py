import unittest
import numpy as np
import pandas as pd
from pp_data.io import ModelSplit
from pp_models.calibration_models import MODEL_INPUTS
from pp_models.high_risk_features import FEATURE_NAMES
from pp_models.high_risk_classifier import load_settings
from pp_models.rolling_protocol import load_protocol
from pp_models.high_risk_study import binary_metrics, forecast_days, classification_tables


class StudyTests(unittest.TestCase):
    def fixture(self):
        n=1004;idx=pd.RangeIndex(n)
        t=pd.Series(list(pd.date_range('2026-03-05',periods=1000,freq='30min',tz='UTC'))+
            [pd.Timestamp(s) for s in ['2026-04-01T10:00Z','2026-04-01T11:00Z','2026-04-02T10:00Z','2026-04-02T11:00Z']])
        x=pd.DataFrame(0.,index=idx,columns=MODEL_INPUTS)
        x['option_type']=np.arange(n)%2;x['api_base_price_per_btc_usdc']=100.
        x['remaining_seconds']=3600.;x['product_duration_seconds']=3600.
        x['realized_vol_60m']=.01;x['quantity_btc']=.1
        # Spread positive events across dates; monotonic outcomes would correctly
        # hit the insufficient-independent-days fallback instead of training.
        y=pd.Series(np.r_[np.repeat(np.random.default_rng(31).permutation(500),2),[100.,200.,300.,400.]],dtype=float,index=idx)
        meta=pd.DataFrame({'created_at_utc':t,'closed_at_utc':t+pd.Timedelta(minutes=30),
            'expiration_at_utc':t+pd.Timedelta(hours=1),'benchmark_eligible':True},index=idx)
        meta.loc[1003,'benchmark_eligible']=False
        f=pd.DataFrame(0.,index=idx,columns=FEATURE_NAMES);f.option_type=x.option_type
        f['direction_return_60m']=np.sin(np.arange(n)*.13)
        risk=ModelSplit(x,y,meta);ids=idx[meta.benchmark_eligible]
        common=ModelSplit(x.loc[ids].copy(),y.loc[ids].copy(),meta.loc[ids].copy())
        return risk,f,common

    def test_false_positive_and_tail_cash_metrics_hand_computed(self):
        y=np.array([1,0,1,0]);p=np.array([.8,.4,.1,.05]);cash=np.array([40.,10.,60.,0.])
        m=binary_metrics(y,p,np.ones(4,dtype=bool),cash,.2)
        self.assertEqual((m['tp'],m['fp'],m['fn'],m['tn']),(1,1,1,1))
        self.assertEqual(m['precision'],.5);self.assertEqual(m['recall'],.5)
        self.assertEqual(m['false_positive_rate'],.5);self.assertEqual(m['tail_cash_recall'],.4)
        self.assertAlmostEqual(m['brier'],.253125)
        zero=binary_metrics(np.zeros(2),np.zeros(2),np.zeros(2,dtype=bool),np.zeros(2),.2)
        self.assertTrue(np.isnan(zero['precision']));self.assertTrue(np.isnan(zero['recall']))

    def test_current_and_future_outcomes_cannot_change_day_one_predictions(self):
        risk,f,b=self.fixture();c=load_settings();protocol=load_protocol()
        before=forecast_days(risk,f,b,protocol,c)
        risk.y.iloc[1000:]=1e9;b.y.loc[b.y.index>=1000]=1e9
        after=forecast_days(risk,f,b,protocol,c)
        for a,z,columns in [(before[0],after[0],['probability','prior','multiplier']),
                             (before[1],after[1],['prediction'])]:
            a=a[a.day.eq('2026-04-01T00:00:00+00:00')]
            z=z[z.day.eq('2026-04-01T00:00:00+00:00')]
            pd.testing.assert_frame_equal(a[columns],z[columns])
        self.assertEqual(before[0].groupby('target').size().tolist(),[4,4])
        self.assertEqual(before[1].groupby('model').size().tolist(),[3]*6)
        self.assertTrue(before[2].trained.any())

    def test_classification_cohorts_and_unavailable_labels_remain_explicit(self):
        risk,f,b=self.fixture();frame,*_=forecast_days(risk,f,b,load_protocol(),load_settings())
        frame.loc[frame.index==1003,'label']=np.nan
        tables=classification_tables(frame,load_settings())
        summary=tables['classification']
        a=summary.query("period == 'all' and cohort == 'risk' and direction == 'all' and method == 'gbdt'")
        self.assertTrue(a.orders.eq(4).all());self.assertTrue(a.available_labels.eq(3).all())

    def test_no_available_labels_still_reports_fallback_coverage(self):
        risk,f,b=self.fixture()
        risk.y.loc[risk.X.option_type.eq(0)]=0.;b.y.loc[b.X.option_type.eq(0)]=0.
        frame,*_=forecast_days(risk,f,b,load_protocol(),load_settings())
        summary=classification_tables(frame,load_settings())['classification']
        calls=summary.query("period == 'all' and cohort == 'risk' and direction == 'Call' and method == 'gbdt'")
        self.assertEqual(len(calls),2)
        self.assertTrue(calls.available_labels.eq(0).all())
        self.assertTrue(calls.fallback_orders.eq(2).all())
        self.assertTrue(calls.brier.isna().all())


if __name__=='__main__':unittest.main()
