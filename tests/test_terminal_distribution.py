import importlib.util
import unittest
import numpy as np
import pandas as pd


def candles(n=9400):
    stamp=pd.date_range('2025-12-31',periods=n,freq='min',tz='UTC')
    changes=.0002*np.sin(np.arange(n)*.19)+.00005*np.cos(np.arange(n)*.071)
    return pd.DataFrame({'open_time_utc':stamp,'close_time_utc':stamp+pd.Timedelta(milliseconds=59999),
                         'close':100*np.exp(np.cumsum(changes))})


class TerminalDistributionTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.terminal_distribution'))
        from pp_models import terminal_distribution
        return terminal_distribution

    def test_empirical_scores_and_option_expectation_by_hand(self):
        api=self.api()
        # Two equiprobable prices; mean price alone is not enough for payoff.
        samples=np.log(np.array([90.,110.])/100)
        self.assertAlmostEqual(api.expected_put(samples,100.,100.),5.)
        scores=api.distribution_scores(np.array([-1.,1.]),0.)
        self.assertAlmostEqual(scores['crps'],.5)
        self.assertAlmostEqual(scores['width_80'],1.6)
        self.assertEqual(scores['covered_80'],1.)
        outside=api.distribution_scores(np.array([-1.,1.]),2.)
        self.assertAlmostEqual(outside['interval_score_80'],13.6)
        self.assertEqual(outside['upper_miss_80'],1.)
        with self.assertRaises(ValueError):api.distribution_scores(np.array([np.nan]),0.)

    def test_history_is_complete_before_day_and_future_changes_do_not_leak(self):
        api=self.api();data=candles();market=api.TerminalMarket(data)
        day=pd.Timestamp('2026-01-04',tz='UTC')
        history=market.history(day,60)
        self.assertTrue((history.index+pd.Timedelta(minutes=60)<day).all())
        self.assertTrue((history.index>=day-pd.Timedelta(days=30)).all())
        query=day+pd.Timedelta(minutes=17)
        before=market.samples(day,60,query,'dynamic')
        changed=data.copy()
        changed.loc[changed.open_time_utc.ge(query.floor('5min')),'close']*=1.5
        after=api.TerminalMarket(changed).samples(day,60,query,'dynamic')
        np.testing.assert_array_equal(before,after)
        self.assertAlmostEqual(before.mean(),0.,places=14)
        self.assertEqual(market.signal_time(query),day+pd.Timedelta(minutes=15))
        with self.assertRaises(ValueError):market.samples(day,60,day-pd.Timedelta(minutes=1),'dynamic')

    def test_fhs_scale_changes_spread_not_the_center(self):
        api=self.api();market=api.TerminalMarket(candles())
        day=pd.Timestamp('2026-01-04',tz='UTC');h=market.history(day,30)
        static=market.samples(day,30,day,'static')
        np.testing.assert_allclose(np.sort(static),np.sort(h['return']-h['return'].mean()))
        dynamic=market.samples(day,30,day,'dynamic')
        z=h['return']/(h.sigma*np.sqrt(30))
        expected=(z-z.mean())*market.sigma_at(day)*np.sqrt(30)
        np.testing.assert_allclose(np.sort(dynamic),np.sort(expected))

    def test_gap_rewarms_and_label_crossing_gap_is_excluded(self):
        api=self.api();data=candles();missing=3000
        market=api.TerminalMarket(data.drop(index=missing))
        origin=data.open_time_utc.iloc[missing]+pd.Timedelta(minutes=1)
        with self.assertRaises(ValueError):market.sigma_at(origin+pd.Timedelta(minutes=100))
        self.assertTrue(np.isfinite(market.sigma_at(origin+pd.Timedelta(minutes=1450))))
        day=pd.Timestamp('2026-01-05',tz='UTC');h=market.history(day,60)
        self.assertFalse(((h.index<=origin)&(h.index+pd.Timedelta(minutes=60)>=origin)).any())

    def test_order_remaining_time_proxy_and_unsupported_fallback(self):
        api=self.api();market=api.TerminalMarket(candles())
        from pp_data.io import ModelSplit
        index=pd.Index(['put','call','long'])
        X=pd.DataFrame({'option_type':[1,0,1],'remaining_seconds':[900,900,86400],
            'strike_price_usdc_per_btc':[100]*3,'market_price_usdc_per_btc':[100]*3},index=index)
        meta=pd.DataFrame({'created_at_utc':pd.to_datetime(['2026-01-04T00:17Z']*3)},index=index)
        p,info=api.augment_pool(ModelSplit(X,pd.Series([10,0,10],index=index),meta),market)
        samples=market.samples(pd.Timestamp('2026-01-04',tz='UTC'),30,meta.created_at_utc.iloc[0],'static')*np.sqrt(.5)
        self.assertAlmostEqual(p.X.loc['put','terminal_static_proxy'],api.expected_put(samples,100,100))
        self.assertEqual(p.X.terminal_supported.tolist(),[1,0,0])
        self.assertEqual(p.X.loc['long','terminal_dynamic_proxy'],0.)
        self.assertEqual(info['orders'],3)

    def test_market_scores_exclude_phase_crossing_labels_and_reconcile(self):
        api=self.api();market=api.TerminalMarket(candles())
        day=pd.Timestamp('2026-01-04',tz='UTC')
        summary,daily=api.evaluate_market(market,{'tuning':[day]},
            {'tuning':[day.isoformat(),(day+pd.Timedelta(days=1)).isoformat()]})
        overall=summary.loc[summary.volatility_group.eq('all')]
        self.assertEqual(overall.origins.sum(),2*(47+46+44+40+32))
        self.assertEqual(overall.excluded_origins.sum(),0)
        self.assertEqual(overall.scheduled_origins.sum(),overall.origins.sum())
        for _,r in overall.iterrows():
            parts=summary.loc[summary.method.eq(r.method)&summary.horizon_minutes.eq(r.horizon_minutes)&summary.volatility_group.ne('all')]
            self.assertEqual(parts.origins.sum(),r.origins)
            self.assertAlmostEqual(np.average(parts.crps,weights=parts.origins),r.crps)
            self.assertAlmostEqual(r.covered_90+r.lower_miss_90+r.upper_miss_90,1.)

    def test_market_missing_origin_is_reported_without_changing_other_horizons(self):
        api=self.api();market=api.TerminalMarket(candles())
        day=pd.Timestamp('2026-01-04',tz='UTC')
        market.forward[30].loc[day]=np.nan
        summary,daily=api.evaluate_market(market,{'tuning':[day]},
            {'tuning':[day.isoformat(),(day+pd.Timedelta(days=1)).isoformat()]})
        overall=summary.loc[summary.volatility_group.eq('all')]
        self.assertEqual(overall.excluded_origins.sum(),2)
        self.assertTrue((overall.scheduled_origins-overall.excluded_origins).eq(overall.origins).all())
        self.assertEqual(overall.loc[overall.horizon_minutes.eq(30),'origins'].tolist(),[46,46])


if __name__=='__main__':unittest.main()
