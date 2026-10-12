import unittest
import numpy as np
import pandas as pd
from pp_models.high_risk_features import FEATURE_NAMES
from pp_models.high_risk_classifier import PeakClassifier, load_settings, risk_multiplier


class ClassifierTests(unittest.TestCase):
    def fixture(self, n=240):
        rng=np.random.default_rng(9)
        x=pd.DataFrame(rng.normal(size=(n,len(FEATURE_NAMES))),columns=FEATURE_NAMES)
        x['option_type']=0
        x['platform_activity_ratio_60m']=np.nan
        y=pd.Series((x.direction_return_60m>1.2).astype(int),index=x.index)
        dates=pd.Series(pd.date_range('2026-03-01',periods=n,freq='h',tz='UTC'),index=x.index)
        return x,y,dates

    def test_supported_classifier_learns_without_filling_missing(self):
        x,y,d=self.fixture()
        m=PeakClassifier(load_settings()).fit(x,y,d)
        p=m.predict(x)
        self.assertTrue(m.trained)
        self.assertGreater(p[y.eq(1)].mean(),p[y.eq(0)].mean()+.05)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(x.platform_activity_ratio_60m.isna().all())

    def test_sparse_or_one_event_history_returns_rate_and_never_gates(self):
        x,y,d=self.fixture();y[:]=0;y.iloc[:5]=1
        m=PeakClassifier(load_settings()).fit(x,y,d)
        np.testing.assert_allclose(m.predict(x.iloc[:3]),5.5/241)
        self.assertFalse(m.trained)
        np.testing.assert_array_equal(risk_multiplier(np.array([.1,.2,.9]),False,load_settings()),[1,1,1])
        y.iloc[:30]=1;d[:]=pd.Timestamp('2026-03-01',tz='UTC')
        self.assertFalse(PeakClassifier(load_settings()).fit(x,y,d).trained)

    def test_direction_leakage_and_misalignment_rejected(self):
        x,y,d=self.fixture();x.loc[0,'option_type']=1
        with self.assertRaises(ValueError):PeakClassifier(load_settings()).fit(x,y,d)
        x.loc[0,'option_type']=0
        with self.assertRaises(ValueError):PeakClassifier(load_settings()).fit(x,y.iloc[::-1],d)
        with self.assertRaises(ValueError):PeakClassifier(load_settings()).fit(x.assign(actual_payout=9),y,d)
        m=PeakClassifier(load_settings()).fit(x,y,d)
        with self.assertRaises(ValueError):m.predict(x.assign(option_type=1))

    def test_multiplier_strict_bounds_and_fixed_trigger(self):
        c=load_settings()
        np.testing.assert_array_equal(risk_multiplier(np.array([0,.19999,.2,1]),True,c),[1,1,1.5,1.5])
        for p in [[np.nan],[-.1],[1.1]]:
            with self.assertRaises(ValueError):risk_multiplier(np.array(p),True,c)


if __name__=='__main__':unittest.main()
