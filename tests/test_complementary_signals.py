"""Protect the information boundary and incremental-alert denominators."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

PATH=Path(__file__).resolve().parents[1]/'scripts/complementary_signals.py'


class ComplementaryTests(unittest.TestCase):
    def api(self):
        self.assertTrue(PATH.exists(), 'complementary signal implementation missing')
        spec=importlib.util.spec_from_file_location('complementary_signals',PATH)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        return m

    def test_asof_respects_publication_delay_staleness_and_input_order(self):
        m=self.api();t=pd.Timestamp('2026-06-01',tz='UTC')
        q=pd.Series([t+pd.Timedelta(minutes=i) for i in [7,4,5,20]],index=['a','b','c','d'])
        f=pd.DataFrame({'time':[t,t+pd.Timedelta(minutes=10)],'v':[11.,99.]})
        out=m.asof_values(q,f,5,3)
        np.testing.assert_allclose(out.v,[11,np.nan,11,np.nan],equal_nan=True)
        f.loc[1,'v']=1e12
        pd.testing.assert_frame_equal(out,m.asof_values(q,f,5,3))

    def test_dvol_close_is_unavailable_until_hour_has_completed(self):
        m=self.api();t=pd.Timestamp('2026-06-01',tz='UTC')
        f=m.dvol_frame([[int(t.timestamp()*1000),40,50,30,45]])
        q=pd.Series([t+pd.Timedelta(minutes=i) for i in [59,60,64,65]])
        out=m.asof_values(q,f,5,65)
        np.testing.assert_allclose(out.dvol,[np.nan,np.nan,np.nan,45],equal_nan=True)

    def test_depth_history_survives_sampling_cadence_change_without_short_warmup(self):
        m=self.api();self.assertTrue(hasattr(m,'normalize_book'))
        times=pd.date_range('2026-05-01',periods=1442,freq='min',tz='UTC')+pd.Timedelta(seconds=10)
        f=pd.DataFrame({'bid':10.,'ask':20.},index=times)
        normalized=m.normalize_book(f)
        self.assertTrue(np.isnan(normalized.bid_prior.iloc[100]))
        self.assertEqual(normalized.bid_prior.iloc[-1],10)
        self.assertEqual(normalized.ask_prior.iloc[-1],20)
        self.assertEqual(normalized.time.iloc[0],pd.Timestamp('2026-05-01T00:01Z'))
        f.loc[f.index[-1],'bid']=999.
        self.assertEqual(m.normalize_book(f).bid_prior.iloc[-1],10)

    def test_forward_return_only_complete_contiguous_window(self):
        m=self.api();t=pd.date_range('2026-06-01',periods=8,freq='min',tz='UTC')
        market=pd.DataFrame({'close_time_utc':t+pd.Timedelta(milliseconds=59999),'close':[100,100,110,121,133.1,146.41,161.051,177.1561]})
        q=pd.Series([t[2],t[4],t[7]])
        f=m.forward_returns(q,market,2)
        np.testing.assert_allclose(f['return'].iloc[:2],[np.log(1.21),np.log(1.21)])
        self.assertTrue(np.isnan(f['return'].iloc[2]))
        self.assertEqual(f.end.iloc[0],market.close_time_utc.iloc[3])
        broken=market.drop(index=2)
        self.assertTrue(np.isnan(m.forward_returns(q.iloc[:1],broken,2)['return'].iloc[0]))

    def test_daily_history_excludes_same_day_unfinished_and_old_origins(self):
        m=self.api();d=pd.Timestamp('2026-06-10',tz='UTC')
        f=pd.DataFrame({'time':[d-pd.Timedelta(days=i) for i in [1,1,31,0]],
                        'end':[d-pd.Timedelta(seconds=1),d,d-pd.Timedelta(days=30),d+pd.Timedelta(hours=1)]})
        self.assertEqual(m.history_mask(f,d).tolist(),[True,False,False,False])

    def test_new_hit_counts_use_full_population_not_union_only(self):
        m=self.api();a=np.array([1,1,1,0,0,0],bool)
        base=np.array([1,0,0,1,0,0],bool);new=np.array([1,1,0,0,1,0],bool)
        v=m.overlap_statistics(a,base,new)
        self.assertEqual(v['new_tail_hits'],1);self.assertEqual(v['lost_tail_hits'],0)
        self.assertEqual(v['new_false_alerts'],1);self.assertEqual(v['removed_false_alerts'],1)
        self.assertEqual(v['alert_jaccard'],.25);self.assertEqual(v['tail_jaccard'],.5)

    def test_feature_group_orientation_and_flow_divides_opposite_depth(self):
        m=self.api();f=pd.DataFrame({'rv60':[.01],'rv240':[.02],'rv1440':[.03],
             'return60':[-.02],'net_buy15':[-100.],'quote15':[1000.],
             'bid':[20.],'ask':[40.],'bid_prior':[40.],'ask_prior':[40.],
             'oi_change1h':[.1],'oi_ratio24h':[1.2],'funding8h':[.0001],
             'dvol':[50.],'dvol_change1h':[2.]})
        put=m.directional_signals(f,1);call=m.directional_signals(f,0)
        self.assertEqual(put.pressure_depth.iloc[0],np.log1p(5))
        self.assertEqual(call.pressure_depth.iloc[0],0)
        self.assertAlmostEqual(put.thin_book.iloc[0],np.log(2))
        self.assertEqual(put.funding_pressure.iloc[0],.0001)
        self.assertEqual(call.funding_pressure.iloc[0],-.0001)


if __name__=='__main__':unittest.main()
