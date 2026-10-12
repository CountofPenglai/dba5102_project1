"""Synthetic counterexamples for the sparse gate and its causal calibration."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


class SparseShockTests(unittest.TestCase):
    def api(self):
        path=ROOT/'pp_models/sparse_shock.py'
        self.assertTrue(path.exists(),'sparse shock implementation missing')
        spec=importlib.util.spec_from_file_location('pp_models.sparse_shock',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

    def signal(self,z,returns=None):
        n=len(z)
        return pd.DataFrame({'z':z,'r5':np.full(n,.001),
            'r15':np.full(n,.004) if returns is None else returns,
            'on':np.full(n,3.),'extreme':np.full(n,6.),'off':np.full(n,2.)},
            index=pd.date_range('2026-04-01',periods=n,freq='5min',tz='UTC'))

    def test_small_or_isolated_moves_do_not_start_but_persistent_extreme_does(self):
        m=self.api();c=m.load_settings()
        f=self.signal([1.,1.,3.,1.,3.,3.])
        r=m.run_state_machine(f,c)
        np.testing.assert_allclose(r.state,[0,0,0,0,0,.5])
        self.assertEqual(r.delay_bars.iloc[-1],1)
        tiny=self.signal([8.,8.],[.0002,.0002])
        np.testing.assert_array_equal(m.run_state_machine(tiny,c).state,0.)

    def test_single_extreme_starts_and_recovery_has_ten_minute_half_life(self):
        m=self.api();f=self.signal([7.,0.,0.,0.,0.],[.007,0,0,0,0])
        r=m.run_state_machine(f,m.load_settings())
        np.testing.assert_allclose(r.state,[1,1,np.sqrt(.5),.5,.5*np.sqrt(.5)])
        self.assertEqual(r.phase.iloc[2],'recovery')
        self.assertEqual(r.event_id.nunique(),1)

    def test_without_strong_renewal_lease_expires_even_in_intermediate_state(self):
        m=self.api();f=self.signal([7.]+[2.5]*10,[.007]+[.002]*10)
        r=m.run_state_machine(f,m.load_settings())
        self.assertEqual(r.phase.iloc[5],'held')
        self.assertEqual(r.phase.iloc[6],'recovery')
        self.assertLess(r.state.iloc[6],1)

    def test_gap_erases_old_evidence_and_requires_new_confirmation(self):
        m=self.api();f=self.signal([3.,np.nan,3.,3.]);r=m.run_state_machine(f,m.load_settings())
        self.assertTrue(np.isnan(r.state.iloc[1]))
        self.assertEqual(r.state.iloc[2],0)
        self.assertEqual(r.state.iloc[3],.5)

    def test_future_market_changes_cannot_change_previous_signal(self):
        m=self.api();n=60*24*14;t=pd.date_range('2026-01-01',periods=n,freq='min',tz='UTC')
        ret=np.sin(np.arange(n)/17)*.0001
        market=pd.DataFrame({'open_time_utc':t,'close_time_utc':t+pd.Timedelta(milliseconds=59999),
                             'close':100*np.exp(np.cumsum(ret))})
        old=m.market_states(market,m.load_settings());cut=pd.Timestamp('2026-01-13',tz='UTC')
        market.loc[market.open_time_utc.ge(cut),'close']*=3
        new=m.market_states(market,m.load_settings())
        pd.testing.assert_frame_equal(old.loc[old.index<=cut],new.loc[new.index<=cut])

    def test_amplitude_event_weighting_shrinkage_and_sparse_fallback(self):
        m=self.api();c=m.load_settings()
        base=pd.Series([100.]*12);state=pd.Series([1.]*12);event=pd.Series(np.repeat([1,2,3],4))
        actual=pd.Series([200.]*12)
        a,detail=m.fit_amplitude(base,state,actual,event,c)
        self.assertAlmostEqual(a,3/8);self.assertEqual(detail['events'],3)
        a,_=m.fit_amplitude(base,state,actual,pd.Series([1]*12),c)
        self.assertEqual(a,0.)
        a,_=m.fit_amplitude(base,state,pd.Series([10000.]*12),event,c)
        self.assertEqual(a,2.)
        a,_=m.fit_amplitude(base,state,pd.Series([0.]*12),event,c)
        self.assertEqual(a,0.)

    def test_contract_direction_changes_relevance_and_zero_state_leaves_anchor_unchanged(self):
        m=self.api();x=pd.DataFrame({'option_type':[0,1,0], 'log_moneyness':[-.03,-.03,0.],
            'realized_vol_60m':[.01]*3,'remaining_seconds':[3600.]*3})
        np.testing.assert_allclose(m.contract_weight(x,m.load_settings()),[0,1,1])
        bad=pd.Series([1.,1.],index=[1,2])
        with self.assertRaises(ValueError):m.fit_amplitude(bad,pd.Series([1.,1.]),bad,bad,m.load_settings())

    def test_duplicate_orders_within_one_event_do_not_inflate_event_weight(self):
        m=self.api();c=m.load_settings()
        f=pd.DataFrame({'base':100.,'state':1.,'y':[300.]*4+[100.]*8,'event':np.repeat([1,2,3],4)})
        a,_=m.fit_amplitude(f.base,f.state,f.y,f.event,c)
        self.assertAlmostEqual(a,.25)
        doubled=pd.concat([f,f.iloc[:4]],ignore_index=True)
        b,_=m.fit_amplitude(doubled.base,doubled.state,doubled.y,doubled.event,c)
        self.assertAlmostEqual(b,.25)

    def test_order_before_bar_end_cannot_see_new_state_and_gap_is_not_filled(self):
        from pp_data.io import ModelSplit
        m=self.api();c=m.load_settings()
        times=pd.Series(pd.to_datetime(['2026-04-01T00:04:59Z','2026-04-01T00:05:00Z']))
        x=pd.DataFrame({'option_type':[1,1],'log_moneyness':[0.,0.],
                        'realized_vol_60m':[.01,.01],'remaining_seconds':[3600.,3600.]})
        pool=ModelSplit(x,pd.Series([0.,0.]),pd.DataFrame({'created_at_utc':times}))
        states=pd.DataFrame(index=pd.date_range('2026-04-01',periods=2,freq='5min',tz='UTC'))
        for direction in ['call','put']:
            states[direction+'_state']=[0.,1.];states[direction+'_event_id']=[0,1]
            states[direction+'_phase']=['quiet','strong'];states[direction+'_since_strong_minutes']=[np.nan,0.]
        np.testing.assert_allclose(m.order_states(pool,states,c).effective_state,[0,1])
        states.iloc[1,states.columns.get_loc('put_state')]=np.nan
        with self.assertRaises(ValueError):m.order_states(pool,states,c)


if __name__=='__main__':unittest.main()
