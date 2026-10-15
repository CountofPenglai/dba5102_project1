"""Guard causal signal definitions and descriptive association arithmetic."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

PATH=Path(__file__).resolve().parents[1]/'scripts/payout_signal_analysis.py'


class PayoutSignalTests(unittest.TestCase):
    def api(self):
        self.assertTrue(PATH.exists(),'signal analysis implementation missing')
        spec=importlib.util.spec_from_file_location('payout_signal_analysis',PATH)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def market(self):
        t=pd.date_range('2026-05-01',periods=1700,freq='min',tz='UTC')
        n=len(t)
        return pd.DataFrame({'open_time_utc':t,'close_time_utc':t+pd.Timedelta(milliseconds=59999),
            'close':100*np.exp(np.arange(n)*.0001),'number_of_trades':np.full(n,10.),
            'quote_asset_volume':np.full(n,1000.),'taker_buy_quote_asset_volume':np.full(n,400.)})

    def test_complete_candles_and_nonoverlapping_activity_denominator(self):
        api=self.api();market=self.market();t=market.close_time_utc.iloc[1600]
        market.loc[1586:1600,'number_of_trades']=20
        f=api.market_signals(pd.Series([t],index=['a']),market).iloc[0]
        self.assertEqual(f.market_trades_per_min_15m,20)
        self.assertEqual(f.market_trade_ratio_15m,2)
        self.assertEqual(f.market_sell_share_15m,.6)
        self.assertEqual(f.market_mean_trade_quote_15m,50)
        self.assertAlmostEqual(f.market_return_15m,.0015)
        market.loc[1601:,'close']=1e10;market.loc[1601:,'number_of_trades']=1e10
        pd.testing.assert_series_equal(f,api.market_signals(pd.Series([t],index=['a']),market).iloc[0])
        with self.assertRaises(ValueError):api.market_signals(pd.Series([t]),market.drop(index=1500))

    def test_platform_count_excludes_same_time_and_normalizes_previous_window(self):
        api=self.api();t=pd.Timestamp('2026-06-01T12:00Z')
        events=pd.DataFrame({'created_at_utc':[t-pd.Timedelta(minutes=m) for m in [301,300,240,61,60,15,1,0,0]],
                            'option_type':[1,1,0,1,1,0,1,0,1]})
        f=api.platform_signals(pd.Series([t],index=['a']),events).iloc[0]
        self.assertEqual(f.platform_put_count_60m,2)
        self.assertEqual(f.platform_call_count_60m,1)
        self.assertEqual(f.platform_put_share_60m,2/3)
        # Three recent events and three in the preceding 240m.
        self.assertEqual(f.platform_activity_ratio_60m,4)
        events['realizedProfit_USDC']=1e10;events['closedAt']='2099-01-01'
        pd.testing.assert_series_equal(f,api.platform_signals(pd.Series([t],index=['a']),events).iloc[0])

    def test_platform_ratio_uses_actual_rates_and_zero_baseline_is_missing(self):
        api=self.api();t=pd.Timestamp('2026-06-01T12:00Z')
        def events(minutes):
            return pd.DataFrame({'created_at_utc':[t-pd.Timedelta(minutes=m) for m in minutes],
                                 'option_type':[1]*len(minutes)})
        query=pd.Series([t],index=['a'])
        row=api.platform_signals(query,events([1000,200,180,120,80,1])).iloc[0]
        self.assertEqual(row.platform_activity_ratio_60m,1)
        for minutes in [[1000],[1000,1]]:
            row=api.platform_signals(query,events(minutes)).iloc[0]
            self.assertTrue(np.isnan(row.platform_activity_ratio_60m))

    def test_past_thresholds_exclude_unsettled_and_same_day_labels(self):
        api=self.api();day=pd.Timestamp('2026-06-10',tz='UTC')
        index=pd.RangeIndex(7)
        X=pd.DataFrame({'option_type':[1]*7,'product_duration_seconds':[1800]*7},index=index)
        meta=pd.DataFrame({'created_at_utc':[day-pd.Timedelta(days=1)]*5+[day,day+pd.Timedelta(days=1)],
                           'closed_at_utc':[day-pd.Timedelta(hours=1)]*4+[day+pd.Timedelta(days=2)]*3},index=index)
        y=pd.Series([0,10,20,30,999,100,200],index=index)
        signals=pd.DataFrame({'activity':[1,2,3,4,999,8,9]},index=index)
        from pp_data.io import ModelSplit
        pool=ModelSplit(X,y,meta)
        context=api.historical_context(pool,signals,day,pd.Index([5]),min_history=3)
        self.assertEqual(context['tail_threshold'].iloc[0],27)
        self.assertAlmostEqual(context['high_threshold'].loc[5,'activity'],3.4)
        y.loc[[4,5,6]]=1e12
        after=api.historical_context(pool,signals,day,pd.Index([5]),min_history=3)
        pd.testing.assert_series_equal(context['tail_threshold'],after['tail_threshold'])
        pd.testing.assert_frame_equal(context['high_threshold'],after['high_threshold'])

    def test_group_rank_adjustment_removes_pure_group_differences(self):
        api=self.api();x=pd.Series([1,1,10,10]);y=pd.Series([2,3,20,30]);g=pd.Series(['a','a','b','b'])
        self.assertGreater(api.rank_correlation(x,y),.8)
        self.assertTrue(np.isnan(api.rank_correlation(x,y,g)))
        self.assertAlmostEqual(api.rank_correlation(pd.Series([1,2,1,2]),pd.Series([4,3,4,3]),g),-1)

    def test_tail_alert_lift_precision_recall_and_ties(self):
        api=self.api()
        actual=pd.Series([False,False,True,True]);alert=pd.Series([False,False,False,True])
        s=api.alert_statistics(actual,alert)
        self.assertEqual(s['precision'],1)
        self.assertEqual(s['lift'],2)
        self.assertEqual(s['recall'],.5)
        self.assertEqual(s['false_positive_rate'],0)
        z=api.alert_statistics(actual,pd.Series([False]*4))
        self.assertEqual(z['alerts'],0);self.assertTrue(np.isnan(z['precision']))

    def test_aggregate_alerts_are_strict_and_period_counts_reconcile(self):
        api=self.api();ids=pd.RangeIndex(4)
        days=['2026-06-01T00:00:00+00:00','2026-06-02T00:00:00+00:00',
              '2026-06-24T00:00:00+00:00','2026-06-25T00:00:00+00:00']
        obs=pd.DataFrame({'phase':['early','early','late','late'],'day':days,'direction':[1]*4,
            'payout':[0.,1,2,3],'cash_payout':[0.,1,2,3],'base':[1.]*4,'base_error':[-1.,0,1,2],
            'day_tenor':days,'tail':[False,False,True,True]},index=ids)
        signals=pd.DataFrame({'activity':[1.,2,3,4]},index=ids)
        high=pd.DataFrame({'activity':[3.]*4},index=ids);low=pd.DataFrame({'activity':[2.]*4},index=ids)
        fallback=pd.DataFrame({'activity':[False]*4},index=ids)
        a,t=api.aggregate_associations(obs,signals,high,low,fallback)
        a=a[a.direction.eq(1)]
        self.assertEqual(a.set_index('period').valid_orders.to_dict(),{'all_june':4,'early':2,'late':2,'without_june24_25':2})
        row=t[t.period.eq('all_june')&t.signal_group.eq('high')].iloc[0]
        self.assertEqual(row.alerts,1);self.assertEqual(row.lift,2);self.assertEqual(row.recall,.5)
        row=t[t.period.eq('all_june')&t.signal_group.eq('low')].iloc[0]
        self.assertEqual(row.alerts,1);self.assertEqual(row.true_positives,0)

    def test_sparse_tenor_uses_direction_history_but_not_opposite_side(self):
        api=self.api();day=pd.Timestamp('2026-06-10',tz='UTC');idx=pd.RangeIndex(6)
        from pp_data.io import ModelSplit
        X=pd.DataFrame({'option_type':[1,1,1,0,0,1],'product_duration_seconds':[1800,1800,7200,7200,7200,7200]},index=idx)
        m=pd.DataFrame({'created_at_utc':[day-pd.Timedelta(days=1)]*5+[day],
                        'closed_at_utc':[day-pd.Timedelta(hours=1)]*5+[day+pd.Timedelta(hours=1)]},index=idx)
        y=pd.Series([0,10,20,999,999,999],index=idx);s=pd.DataFrame({'x':[1,2,3,999,999,8]},index=idx)
        c=api.historical_context(ModelSplit(X,y,m),s,day,pd.Index([5]),min_history=3)
        self.assertEqual(c['tail_threshold'].iloc[0],18)
        self.assertEqual(c['tail_source'].iloc[0],'direction_fallback')
        self.assertTrue(c['feature_fallback'].loc[5,'x'])

    def test_missing_signals_cannot_inflate_population_recall_or_lift(self):
        api=self.api();idx=pd.RangeIndex(6)
        obs=pd.DataFrame({'phase':['early']*6,'day':['2026-06-01T00:00:00+00:00']*6,
            'direction':[1]*6,'payout':[0,0,10,0,20,30],'cash_payout':[0,0,10,0,20,30],
            'base':[1]*6,'base_error':[-1,-1,9,-1,19,29],'day_tenor':['a']*6,
            'tail':[False,False,True,False,True,True]},index=idx)
        sig=pd.DataFrame({'x':[1,2,3,np.nan,5,np.nan]},index=idx)
        threshold=pd.DataFrame({'x':[2,2,2,2,np.nan,2]},index=idx)
        fallback=pd.DataFrame({'x':[False]*6},index=idx)
        _,alerts=api.aggregate_associations(obs,sig,threshold,threshold,fallback)
        r=alerts[alerts.period.eq('all_june')&alerts.signal_group.eq('high')].iloc[0]
        self.assertEqual(r.orders,6);self.assertEqual(r.tail_orders,3)
        self.assertEqual(r.lift,2);self.assertEqual(r.recall,1/3)
        self.assertEqual(r.alert_rate,1/6);self.assertEqual(r.signal_available_orders,3)


if __name__=='__main__':unittest.main()
