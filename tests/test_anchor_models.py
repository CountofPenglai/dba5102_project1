import importlib.util
import unittest
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE
from pp_models.baselines import RollingMean


def anchor_X(n=4):
    return pd.DataFrame({'option_type': np.arange(n)%2, 'product_duration_seconds': 1800.,
        BASE_FEATURE: np.linspace(1., 4., n), 'realized_vol_60m': .002,
        'realized_vol_1440m': .01, 'log_return_60m': np.linspace(-.01, .01, n),
        'log_moneyness': 0.}, index=[f'o{i}' for i in range(n)])


class AnchorModelTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.anchor_models'), 'anchor models missing')
        from pp_models import anchor_models
        return anchor_models

    def test_raw_official_unchanged_and_scaled_uses_amount_loss_not_ratio_mean(self):
        api = self.api()
        X=anchor_X(2)
        X[BASE_FEATURE]=[1., 2.]
        y=pd.Series([0., 4.], index=X.index)
        np.testing.assert_allclose(api.AnchorModel('official','raw').fit(X,y).predict_raw(X),[1.,2.])
        model=api.AnchorModel('official','scaled').fit(X,y)
        self.assertAlmostEqual(model.diagnostics()['calibration_multiplier'],1.6)
        np.testing.assert_allclose(model.predict_raw(X),[1.6,3.2])

    def test_group_leave_one_out_removes_own_label_and_raw_reproduces_B1(self):
        api=self.api()
        X=anchor_X(3)
        X.option_type=[0,0,1]
        y=pd.Series([2.,6.,10.],index=X.index)
        actual=api.group_leave_one_out(X,y,smoothing=2)
        np.testing.assert_allclose(actual,[22/3,14/3,4.])
        y2=y.copy(); y2.iloc[0]=200.
        self.assertEqual(actual[0],api.group_leave_one_out(X,y2,smoothing=2)[0])
        ref=RollingMean(grouped=True,smoothing=30).fit(X,y).predict_raw(X)
        np.testing.assert_allclose(api.AnchorModel('grouped','raw').fit(X,y).predict_raw(X),ref)

    def test_dynamic_recovers_constant_scale_without_penalizing_intercept(self):
        api=self.api()
        X=anchor_X(40)
        y=2.5*X[BASE_FEATURE]
        model=api.AnchorModel('official','dynamic',lam=10).fit(X,y)
        np.testing.assert_allclose(model.predict_raw(X),y,atol=1e-6)
        self.assertAlmostEqual(model.diagnostics()['calibration_multiplier'],2.5,places=6)

    def test_market_signal_can_raise_and_lower_positive_predictions(self):
        api=self.api()
        X=anchor_X(80)
        X[BASE_FEATURE]=100.
        X.option_type=0
        y=pd.Series(100*np.exp(40*X.log_return_60m),index=X.index)
        model=api.AnchorModel('official','dynamic',lam=.1).fit(X,y)
        pred=model.predict_raw(X)
        self.assertLess(pred[0],90.)
        self.assertGreater(pred[-1],110.)
        self.assertTrue((pred>0).all())
        one=X.iloc[[0]].copy()
        combined=pd.concat([one, X.iloc[[-1]].assign(log_return_60m=1.)])
        self.assertAlmostEqual(model.predict_raw(one)[0],model.predict_raw(combined)[0])

    def test_zero_targets_supported_without_log_labels_and_invalid_base_rejected(self):
        api=self.api()
        X=anchor_X()
        y=pd.Series(0.,index=X.index)
        for anchor in ['official','grouped']:
            for mode in ['scaled','dynamic']:
                np.testing.assert_array_equal(api.AnchorModel(anchor,mode,lam=1).fit(X,y).predict_raw(X),np.zeros(4))
        for value in [0.,-1.,np.nan]:
            bad=X.copy();bad.loc['o0',BASE_FEATURE]=value
            with self.assertRaises(ValueError):
                api.AnchorModel('official','scaled').fit(bad,y)
