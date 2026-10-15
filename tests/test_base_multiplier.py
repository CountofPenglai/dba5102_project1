import importlib.util
import unittest
import numpy as np
import pandas as pd


def fixture():
    n = 240
    rng = np.random.default_rng(91)
    X = pd.DataFrame({'option_type': np.tile([0, 1], n//2),
        'product_duration_seconds': 3600., 'remaining_seconds': rng.uniform(3000, 3600, n),
        'log_moneyness': rng.uniform(-.002, .002, n), 'total_vol_240m': rng.uniform(.0003,.0006,n),
        'down_vol_240m': rng.uniform(.0002,.0004,n), 'log_return_60m': rng.normal(0,.002,n)})
    base = pd.Series(rng.uniform(40,200,n))
    y = pd.Series(np.where(np.arange(n)%3 == 0, 0., base*1.5))
    dates = pd.Series(pd.date_range('2026-02-01', periods=n, freq='4h', tz='UTC'))
    return X, base, y, dates


class BaseMultiplierTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.base_multiplier'))
        from pp_models.base_multiplier import BaseMultiplier
        self.cls = BaseMultiplier

    def test_constant_matches_closed_form_all_labels_and_regularized_loss(self):
        X,b,y,d = fixture()
        m = self.cls('constant', .1, .1).fit(X,b,y,d)
        put = X.option_type.eq(1); s2 = np.mean(y[put]**2)
        expected = np.clip(np.mean(b[put]*(y[put]-b[put])) / (np.mean(b[put]**2)+.1*s2),-.1,.1)
        np.testing.assert_allclose(m.predict_multiplier(X)[put],1+expected,atol=1e-8)
        p = b[put]*m.predict_multiplier(X)[put]
        self.assertAlmostEqual(m.objective_,np.mean((p-y[put])**2)/s2+.1*np.sum(m.coef_**2))

    def test_call_one_range_no_base_input_and_inference_batch_independence(self):
        X,b,y,d = fixture()
        for kind in ('constant','contract','market'):
            m = self.cls(kind, .1, .1).fit(X,b,y,d)
            p=m.predict_multiplier(X)
            self.assertTrue(((p>=.9)&(p<=1.1)).all())
            np.testing.assert_array_equal(p[X.option_type.eq(0)],1.)
            np.testing.assert_allclose(p[::-1],m.predict_multiplier(X.iloc[::-1]))
            np.testing.assert_allclose(p,[m.predict_multiplier(X.iloc[[i]])[0] for i in range(len(X))])
            with self.assertRaisesRegex(ValueError,'whitelist'):
                m.predict_multiplier(X.assign(base_premium=10.))

    def test_unsupported_and_unseen_states_return_one(self):
        X,b,y,d=fixture(); X.loc[1,'product_duration_seconds']=86400
        m=self.cls('market',1.,.1).fit(X,b,y,d)
        self.assertEqual(m.predict_multiplier(X.loc[[1]])[0],1.)
        q=X.loc[[3]].copy();q['log_return_60m']=10.
        self.assertEqual(m.predict_multiplier(q)[0],1.)
        q['log_return_60m']=np.nan
        self.assertEqual(m.predict_multiplier(q)[0],1.)
        small=self.cls('constant',1.,.1).fit(X.iloc[:30],b.iloc[:30],y.iloc[:30],d.iloc[:30])
        np.testing.assert_array_equal(small.predict_multiplier(X),1.)

    def test_strong_training_correction_remains_feasible(self):
        X,b,y,d=fixture();y[:]=10*b
        m=self.cls('market',.1,.1).fit(X,b,y,d)
        self.assertLessEqual(m.max_training_raw_adjustment_,.1+1e-8)
        self.assertTrue(m.optimization_success_)
        self.assertLess(m.objective_,m.null_objective_)

    def test_misalignment_and_invalid_labels_rejected(self):
        X,b,y,d=fixture()
        with self.assertRaises(ValueError):self.cls('constant',1.,.1).fit(X,b,y.iloc[::-1],d)
        y.iloc[0]=np.nan
        with self.assertRaises(ValueError):self.cls('constant',1.,.1).fit(X,b,y,d)


if __name__=='__main__':unittest.main()
