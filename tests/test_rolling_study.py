"""End-to-end causal replay and equal-month aggregation for the rolling study."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from pp_data.io import ModelSplit
from pp_models.calibration_models import MODEL_INPUTS
from pp_models.rolling_protocol import load_protocol

ROOT=Path(__file__).resolve().parents[1]


class RollingStudyTests(unittest.TestCase):
    def api(self):
        path=ROOT/'pp_models/rolling_study.py';self.assertTrue(path.exists(),'rolling study missing')
        spec=importlib.util.spec_from_file_location('pp_models.rolling_study',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

    def pool(self):
        idx=pd.RangeIndex(122);t=pd.Series(list(pd.date_range('2026-03-10',periods=120,freq='h',tz='UTC'))+
            [pd.Timestamp('2026-04-01T12:00Z'),pd.Timestamp('2026-04-02T12:00Z')])
        x=pd.DataFrame(0.,index=idx,columns=MODEL_INPUTS)
        x['option_type']=np.arange(122)%2;x['api_base_price_per_btc_usdc']=100.
        x['remaining_seconds']=3600.;x['product_duration_seconds']=3600.;x['realized_vol_60m']=.01
        x['quantity_btc']=.1;x['distribution_supported']=1.;x['terminal_probability']=.5
        x['terminal_mean_both']=100.
        y=pd.Series(np.resize([0.,100.,200.,0.],122),index=idx)
        meta=pd.DataFrame({'created_at_utc':t,'closed_at_utc':t+pd.Timedelta(minutes=30),
                           'expiration_at_utc':t+pd.Timedelta(hours=1)},index=idx)
        return ModelSplit(x,y,meta)

    def test_future_labels_cannot_change_prior_predictions_and_models_share_ids(self):
        m=self.api();p=self.pool();tables,info=m.evaluate_rolling(p,load_protocol())
        d=tables['daily'];self.assertEqual(d[d.group.eq('overall')].groupby('model').orders.sum().nunique(),1)
        old=d[d.day.eq('2026-04-01T00:00:00+00:00')&d.group.eq('overall')].set_index('model').predicted_total_usdc
        p.y.iloc[-1]=1e12
        other,_=m.evaluate_rolling(p,load_protocol())
        new=other['daily'].query("day == '2026-04-01T00:00:00+00:00' and group == 'overall'").set_index('model').predicted_total_usdc
        pd.testing.assert_series_equal(old,new)

    def test_macro_weights_months_equally_not_orders(self):
        m=self.api();f=pd.DataFrame({'model':['x','x'],'month':['2026-04','2026-05'],
             'group':['overall']*2,'value':['all']*2,'orders':[100,1],
             'rmse_per_btc':[0.,10.],'mae_per_btc':[0.,10.]})
        row=m.macro_metrics(f).iloc[0]
        self.assertAlmostEqual(row.macro_rmse_per_btc,np.sqrt(50))
        self.assertEqual(row.macro_mae_per_btc,5)

    def test_tail_threshold_falls_back_within_direction_and_does_not_drop_orders(self):
        m=self.api();p=self.pool();h=p.X.index[:120];q=p.X.index[120:]
        threshold,source=m.tail_context(p,h,q)
        self.assertEqual(len(threshold),2);self.assertTrue(np.isfinite(threshold).all())
        before=threshold.copy();p.y.loc[q]=1e12
        after,_=m.tail_context(p,h,q)
        pd.testing.assert_series_equal(before,after)


if __name__=='__main__':unittest.main()
