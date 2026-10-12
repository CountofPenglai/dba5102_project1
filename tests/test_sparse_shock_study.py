"""Forecasts must retain causal labels, common cohorts and zero-state identity."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from pp_data.io import ModelSplit
from pp_models.calibration_models import MODEL_INPUTS
from pp_models.rolling_protocol import load_protocol
from pp_models.sparse_shock import load_settings

ROOT=Path(__file__).resolve().parents[1]


class SparseStudyTests(unittest.TestCase):
    def api(self):
        p=ROOT/'pp_models/sparse_shock_study.py';self.assertTrue(p.exists(),'sparse study missing')
        spec=importlib.util.spec_from_file_location('pp_models.sparse_shock_study',p)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

    def pool(self):
        t=pd.Series(list(pd.date_range('2026-02-15',periods=240,freq='3h',tz='UTC'))+
                    [pd.Timestamp('2026-04-01T12:00Z'),pd.Timestamp('2026-04-02T12:00Z')])
        x=pd.DataFrame(0.,index=t.index,columns=MODEL_INPUTS)
        x['option_type']=np.arange(len(t))%2;x['api_base_price_per_btc_usdc']=100.
        x['remaining_seconds']=3600.;x['product_duration_seconds']=3600.;x['realized_vol_60m']=.01
        x['quantity_btc']=.1
        y=pd.Series(np.resize([0.,100.,200.,0.],len(t)))
        meta=pd.DataFrame({'created_at_utc':t,'closed_at_utc':t+pd.Timedelta(minutes=30),
                           'expiration_at_utc':t+pd.Timedelta(hours=1)})
        features=pd.DataFrame({'state':.5,'effective_state':.5,'contract_weight':1.,'phase':'strong',
            'event_id':np.arange(len(t))//20+1,'available_at_utc':t.dt.floor('5min')},index=t.index)
        return ModelSplit(x,y,meta),features

    def test_future_or_unsettled_labels_cannot_change_previous_predictions(self):
        m=self.api();p,f=self.pool();p.metadata.loc[240,'closed_at_utc']=pd.Timestamp('2026-04-03',tz='UTC')
        a,info=m.evaluate(p,f,load_protocol(),load_settings())
        p.y.loc[240:]=1e9
        b,_=m.evaluate(p,f,load_protocol(),load_settings())
        pd.testing.assert_series_equal(a['daily'].predicted_total_usdc,b['daily'].predicted_total_usdc)
        overall=a['daily'].query("group == 'overall'")
        self.assertEqual(overall.groupby('model').orders.sum().nunique(),1)
        self.assertEqual(info['unique_scoring_orders'],2)

    def test_zero_state_keeps_both_anchors_exactly_unchanged(self):
        m=self.api();p,f=self.pool();f['effective_state']=0.;f['state']=0.;f['event_id']=0;f['phase']='quiet'
        tables,_=m.evaluate(p,f,load_protocol(),load_settings())
        d=tables['daily'].query("group == 'overall'")
        for anchor,prefix in [('official_raw','official'),('hurdle_base','hurdle')]:
            baseline=d[d.model.eq(anchor)].set_index('day').predicted_total_usdc
            for suffix in ['shock','shock_fixed']:
                other=d[d.model.eq(prefix+'_'+suffix)].set_index('day').predicted_total_usdc
                pd.testing.assert_series_equal(baseline,other)

    def test_budget_reference_has_same_predicted_cash_and_no_label_dependency(self):
        m=self.api();f=pd.DataFrame({'day':['2026-04-01']*2,'month':['2026-04']*2,'side':[0,0],
            'actual':[0.,900.],'quantity':[1.,2.],'anchor':[100.,100.],
            'prediction':[100.,150.],'model':['official_shock']*2})
        a=m.budget_reference(f);f.actual*=2;b=m.budget_reference(f)
        np.testing.assert_allclose(a.uniform_coefficient,4/3)
        np.testing.assert_allclose(a.predicted_total_usdc,400.)
        np.testing.assert_allclose(a.uniform_coefficient,b.uniform_coefficient)


if __name__=='__main__':unittest.main()
