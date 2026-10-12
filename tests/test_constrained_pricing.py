import importlib.util
import unittest
import numpy as np
import pandas as pd
from test_terminal_distribution import candles


class ConstrainedPricingTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.constrained_pricing'))
        from pp_models import constrained_pricing
        return constrained_pricing

    def frame(self, api, n=120):
        x=np.linspace(-2,2,n)
        return pd.DataFrame({'option_type':np.arange(n)%2, 'api_base_price_per_btc_usdc':20.,
            'constraint_supported':1, 'contract_dynamic':x, 'contract_calibrated':x,
            'contract_momentum':0.,'contract_vol_change':0.,
            'flow_market':x,'flow_call':0.,'flow_put':0.})[api.INPUTS]

    def test_side_calibration_and_all_zero_targets(self):
        api=self.api();X=self.frame(api);y=pd.Series(np.where(X.option_type.eq(0),40.,10.))
        model=api.ConstrainedModel('scaled').fit(X,y)
        np.testing.assert_allclose(model.predict_raw(X),y)
        zero=api.ConstrainedModel('flow',1.).fit(X,y*0)
        np.testing.assert_array_equal(zero.predict_raw(X),np.zeros(len(X)))
        with self.assertRaises(ValueError):model.fit(X,y.iloc[::-1])

    def test_frequency_shrinks_and_never_changes_other_direction(self):
        api=self.api();X=self.frame(api)
        X['contract_dynamic']=X['contract_calibrated']=0.
        y=pd.Series(20*np.exp(.6*X.flow_market))
        weak=api.ConstrainedModel('flow',.1).fit(X,y)
        strong=api.ConstrainedModel('flow',10.).fit(X,y)
        self.assertLess(np.ptp(strong.predict_raw(X)),np.ptp(weak.predict_raw(X)))
        changed=y.copy();changed.loc[X.option_type.eq(1)]*=8
        other=api.ConstrainedModel('flow',.1).fit(X,changed)
        np.testing.assert_allclose(weak.predict_raw(X)[::2],other.predict_raw(X)[::2])
        self.assertLess(weak.predict_raw(X)[0],weak.predict_raw(X)[-2])

    def test_missing_flow_abstains_and_unsupported_has_only_scalar(self):
        api=self.api();X=self.frame(api);X.loc[0,'flow_market']=np.nan
        y=pd.Series(20*np.exp(.3*X.contract_dynamic))
        model=api.ConstrainedModel('flow',.1).fit(X,y)
        query=X.iloc[:2].copy();query.constraint_supported=0
        query.loc[:,'contract_calibrated']=1e6;query.loc[:,'flow_market']=1e6
        pred=model.predict_raw(query)
        np.testing.assert_allclose(pred,[20*model.fits[0]['scale'],20*model.fits[1]['scale']])
        self.assertTrue(np.isfinite(model.predict_raw(X)).all())

    def test_profiled_gradient_matches_finite_difference(self):
        api=self.api();Z=np.array([[-1.,.2],[0.,-.5],[2.,.3]])
        b=np.array([.2,-.3]);base=np.array([10.,20.,30.]);y=np.array([5.,60.,0.]);pen=np.array([.1,1.])
        value,gradient,scale=api.profile_loss(b,Z,base,y,pen)
        step=1e-6
        for j in range(2):
            shift=np.eye(2)[j]*step
            difference=(api.profile_loss(b+shift,Z,base,y,pen)[0]-api.profile_loss(b-shift,Z,base,y,pen)[0])/(2*step)
            self.assertAlmostEqual(gradient[j],difference,places=6)
        w=base*np.exp(Z@b)
        self.assertAlmostEqual(scale,(w@y)/(w@w))

    def test_solver_failure_only_accepted_with_bounded_stationarity_certificate(self):
        api=self.api()
        self.assertTrue(hasattr(api,'stationary_within_bounds'))
        bounds=[(0.,1.),(-1.,1.)]
        self.assertTrue(api.stationary_within_bounds(np.array([.004,.001]),np.array([1.4e-8,3.8e-8]),bounds))
        self.assertFalse(api.stationary_within_bounds(np.array([.004,.001]),np.array([.01,.02]),bounds))
        self.assertTrue(api.stationary_within_bounds(np.array([0.,1.]),np.array([.5,-.5]),bounds))
        self.assertFalse(api.stationary_within_bounds(np.array([-.1,0.]),np.array([0.,0.]),bounds))
        self.assertFalse(api.stationary_within_bounds(np.array([0.,0.]),np.array([np.nan,0.]),bounds))


class ConstrainedDistributionTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.constrained_features'))
        from pp_models import constrained_features
        return constrained_features

    def test_payoff_monotonicity_convexity_and_direction(self):
        api=self.api();samples=np.log(np.array([90.,100.,110.])/100)
        puts=[api.expected_payoff(samples,100.,k,1) for k in (90.,100.,110.)]
        calls=[api.expected_payoff(samples,100.,k,0) for k in (90.,100.,110.)]
        self.assertLessEqual(puts[0],puts[1]);self.assertGreaterEqual(calls[0],calls[1])
        self.assertGreaterEqual(puts[0]+puts[2],2*puts[1])
        self.assertAlmostEqual(puts[1],10/3);self.assertAlmostEqual(calls[1],10/3)

    def test_separate_tail_map_is_ordered_and_can_shrink(self):
        api=self.api();base=np.linspace(-1,1,1001)
        recent=np.where(base<0,base*2,base*.5)
        lower,upper,n=api.tail_factors(base,recent)
        self.assertGreater(lower,1.);self.assertLess(upper,1.)
        out=api.apply_tail_factors(base,lower,upper)
        self.assertTrue((np.diff(out)>=0).all())
        np.testing.assert_allclose(out[base>=0],base[base>=0]*upper)
        self.assertEqual(api.tail_factors(base,recent[:20]),(1.,1.,20))

    def test_forecast_tail_factors_and_samples_ignore_future(self):
        api=self.api();data=candles(26000);market=api.CalibratedMarket(data)
        day=pd.Timestamp('2026-01-15',tz='UTC');stamp=day+pd.Timedelta(minutes=17)
        before=market.samples(day,60,stamp,'calibrated')
        changed=data.copy();changed.loc[changed.open_time_utc.ge(stamp.floor('5min')),'close']*=1.7
        after=api.CalibratedMarket(changed).samples(day,60,stamp,'calibrated')
        np.testing.assert_array_equal(before,after)
        history=market.history(day,60)
        self.assertTrue((history.index+pd.Timedelta(minutes=60)<day).all())


if __name__=='__main__':unittest.main()
