import unittest
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE
from pp_models.two_stage_severity import (DIRECTIONAL_INPUTS, SEVERITY_INPUTS, TwoStageSeverity,
                                          directional_acceleration)


def sample(rows=80, days=8):
    index = pd.RangeIndex(rows)
    X = pd.DataFrame({
        "option_type": np.arange(rows) % 2,
        BASE_FEATURE: np.linspace(10, 30, rows),
        "remaining_seconds": np.resize([1800., 3600., 7200., 14400.], rows),
        "realized_vol_60m": np.resize([.01, .02, .03], rows),
        "log_moneyness": np.linspace(-.03, .03, rows),
    }, index=index)[SEVERITY_INPUTS]
    created = pd.Series(pd.Timestamp("2026-03-01", tz="UTC")
                        + pd.to_timedelta((np.arange(rows) // 2) % days, unit="D"), index=index)
    y = pd.Series(X[BASE_FEATURE].to_numpy() * (2 + .2 * np.log(X.remaining_seconds / 1800)), index=index)
    return X, y, created


class TwoStageSeverityTests(unittest.TestCase):
    def test_h0_and_g0_match_closed_forms(self):
        X, y, created = sample()
        side = X.option_type.eq(0)
        h0 = TwoStageSeverity("h0").fit(X, y, created)
        g0 = TwoStageSeverity("g0").fit(X, y, created)
        expected_h0 = float(np.dot(X.loc[side, BASE_FEATURE], y.loc[side])
                            / np.dot(X.loc[side, BASE_FEATURE], X.loc[side, BASE_FEATURE]))
        expected_g0 = float((y.loc[side] / X.loc[side, BASE_FEATURE]).mean())
        self.assertAlmostEqual(h0.parameters[0]["coefficient"], expected_h0)
        self.assertAlmostEqual(g0.parameters[0]["coefficient"], expected_g0)

    def test_support_boundary_and_fallback(self):
        X, y, created = sample(80, 5)
        fitted = TwoStageSeverity("g1a").fit(X, y, created)
        self.assertEqual(fitted.parameters[0]["mode"], "gamma")
        unsupported = TwoStageSeverity("g1a", minimum_positive_orders=41).fit(X, y, created)
        self.assertEqual(unsupported.parameters[0]["mode"], "base")
        self.assertEqual(unsupported.parameters[0]["fallback_reason"], "insufficient_positive_support")
        h0 = TwoStageSeverity("h0").fit(X, y, created)
        np.testing.assert_allclose(unsupported.predict_severity(X), h0.predict_severity(X))

    def test_zero_target_and_missing_direction_fallbacks(self):
        X, y, created = sample()
        y.loc[X.option_type.eq(0)] = 0
        zero = TwoStageSeverity("g0").fit(X, y, created)
        self.assertTrue((zero.predict_severity(X.loc[X.option_type.eq(0)]) == 0).all())
        put_only = TwoStageSeverity("g0").fit(X.loc[X.option_type.eq(1)], y.loc[X.option_type.eq(1)],
                                               created.loc[X.option_type.eq(1)])
        call = X.loc[X.option_type.eq(0)].iloc[:2]
        np.testing.assert_allclose(put_only.predict_severity(call), call[BASE_FEATURE])

    def test_gamma_is_positive_and_contract_conditioned(self):
        X, y, created = sample()
        model = TwoStageSeverity("g1a").fit(X, y, created)
        prediction = model.predict_severity(X)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction > 0).all())
        self.assertGreater(np.unique(np.round(prediction / X[BASE_FEATURE], 8)).size, 2)

    def test_tiny_base_is_finite_without_target_clipping(self):
        X, y, created = sample()
        X.loc[0, BASE_FEATURE] = 1e-10
        y.loc[0] = 1e4
        prediction = TwoStageSeverity("g0").fit(X, y, created).predict_severity(X)
        self.assertTrue(np.isfinite(prediction).all())

    def test_illegal_or_misaligned_inputs_raise(self):
        X, y, created = sample()
        with self.assertRaises(ValueError):
            TwoStageSeverity("h0").fit(X, y.rename(index={0: 1000}), created)
        bad = X.copy(); bad.loc[0, BASE_FEATURE] = 0
        with self.assertRaises(ValueError):
            TwoStageSeverity("h0").fit(bad, y, created)

    def test_directional_acceleration_uses_risk_side_and_zero_is_finite(self):
        X = pd.DataFrame({"option_type": [0, 1, 0], "up_vol_60m": [2., 9., 0.],
                          "up_vol_240m": [1., 3., 0.], "down_vol_60m": [8., 4., 0.],
                          "down_vol_240m": [2., 2., 0.]})
        result = directional_acceleration(X)
        self.assertAlmostEqual(result.iloc[0], np.log(2))
        self.assertAlmostEqual(result.iloc[1], np.log(2))
        self.assertEqual(result.iloc[2], 0)
        swapped = X.rename(columns={"up_vol_60m": "down_vol_60m_tmp",
                                    "up_vol_240m": "down_vol_240m_tmp",
                                    "down_vol_60m": "up_vol_60m", "down_vol_240m": "up_vol_240m"}).rename(
            columns={"down_vol_60m_tmp": "down_vol_60m", "down_vol_240m_tmp": "down_vol_240m"})
        swapped["option_type"] = 1 - swapped.option_type
        np.testing.assert_allclose(result, directional_acceleration(swapped))

    def test_g2_adds_signal_without_changing_support_rules(self):
        X, y, created = sample()
        X = X.assign(up_vol_60m=.02, up_vol_240m=np.linspace(.01, .03, len(X)),
                     down_vol_60m=.03, down_vol_240m=np.linspace(.015, .04, len(X)))[DIRECTIONAL_INPUTS]
        model = TwoStageSeverity("g2a").fit(X, y, created)
        self.assertEqual(model.parameters[0]["mode"], "gamma")
        self.assertIn("call_directional_acceleration_coefficient", model.diagnostics())


if __name__ == "__main__":
    unittest.main()
