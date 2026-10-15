"""Hand-computed payouts and calendar weighting, without production-derived targets."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


class CalibrationTests(unittest.TestCase):
    def api(self):
        path=ROOT/'pp_models/calibration_models.py';self.assertTrue(path.exists(),'calibration models missing')
        spec=importlib.util.spec_from_file_location('pp_models.calibration_models',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

    def frame(self,n=4):
        return pd.DataFrame({'option_type':[0]*n,'api_base_price_per_btc_usdc':[100.]*n,
            'remaining_seconds':[3600.]*n,'product_duration_seconds':[3600.]*n,
            'realized_vol_60m':[.01]*n,'log_moneyness':[0.]*n,
            'terminal_supported':[0.]*n,'terminal_static_proxy':[0.]*n,'terminal_dynamic_proxy':[0.]*n})

    def test_direction_shrink_has_known_solution_and_directions_do_not_mix(self):
        m=self.api();x=self.frame();x['option_type']=[0,0,1,1];y=pd.Series([200.,200.,50.,50.])
        raw=m.CalibrationModel('direction_scaled').fit(x,y).predict_raw(x)
        shrink=m.CalibrationModel('direction_shrunk').fit(x,y).predict_raw(x)
        np.testing.assert_allclose(raw,[200,200,50,50]);np.testing.assert_allclose(shrink,[150,150,75,75])

    def test_day_balancing_invariant_to_duplicate_identical_orders_on_same_day(self):
        m=self.api();x=self.frame(3);x['api_base_price_per_btc_usdc']=1.
        y=pd.Series([0.,0.,30.]);created=pd.Series(pd.to_datetime(['2026-03-01']*2+['2026-03-02'],utc=True))
        model=m.CalibrationModel('day_balanced_shrunk').fit(x,y,created)
        np.testing.assert_allclose(model.predict_raw(x),8.)
        x2=pd.concat([x,x.iloc[[0]]],ignore_index=True);y2=pd.concat([y,pd.Series([0.])],ignore_index=True)
        c2=pd.concat([created,created.iloc[[0]]],ignore_index=True)
        other=m.CalibrationModel('day_balanced_shrunk').fit(x2,y2,c2)
        np.testing.assert_allclose(other.predict_raw(x),8.)

    def test_hurdle_retains_zero_orders_in_probability_and_positive_severity_only(self):
        m=self.api();x=self.frame();y=pd.Series([0.,0.,20.,60.])
        model=m.CalibrationModel('hurdle_base').fit(x,y)
        p,severity=model.predict_components(x)
        np.testing.assert_allclose(p,.5,atol=1e-6);np.testing.assert_allclose(severity,40.)
        np.testing.assert_allclose(model.predict_raw(x),20.,atol=1e-4)

    def test_hurdle_degenerate_history_and_unknown_direction_have_explicit_fallback(self):
        m=self.api();x=self.frame();z=m.CalibrationModel('hurdle_base').fit(x,pd.Series([0.]*4))
        np.testing.assert_array_equal(z.predict_raw(x),0.)
        positive=m.CalibrationModel('hurdle_base').fit(x,pd.Series([40.]*4))
        np.testing.assert_allclose(positive.predict_raw(x),40.)
        x['option_type']=1
        np.testing.assert_allclose(positive.predict_raw(x),100.)

    def test_wrong_target_alignment_and_missing_calendar_fail_closed(self):
        m=self.api();x=self.frame();y=pd.Series([1.]*4,index=[4,5,6,7])
        with self.assertRaises(ValueError):m.CalibrationModel('hurdle_base').fit(x,y)
        with self.assertRaises(ValueError):m.CalibrationModel('day_balanced_shrunk').fit(x,pd.Series([1.]*4))


if __name__=='__main__':unittest.main()
