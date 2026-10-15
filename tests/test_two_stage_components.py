import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pp_data.features import BASE_FEATURE
from pp_models.two_stage_components import INPUTS, AmountRefinement, ProbabilityRefinement


def fixture(n=240):
    X = pd.DataFrame({"option_type": np.arange(n) % 2, BASE_FEATURE: 100.,
        "remaining_seconds": np.exp(np.linspace(6, 10, n)), "realized_vol_60m": .02,
        "log_moneyness": np.sin(np.arange(n)) * .001,
        "up_vol_60m": .001, "up_vol_240m": .002,
        "down_vol_60m": .003, "down_vol_240m": .002})[INPUTS]
    y = pd.Series(np.where(np.arange(n) % 3 == 0, 0., 120.), index=X.index)
    created = pd.Series(pd.date_range("2026-03-01", periods=n, freq="h", tz="UTC"), index=X.index)
    return X, y, created, pd.Series(100., index=X.index)


class ComponentModelTests(unittest.TestCase):
    def test_amount_uses_currency_squared_loss_with_h0_ridge(self):
        X, y, created, anchor = fixture(120)
        X["option_type"] = 0
        X["remaining_seconds"] = 3600.
        X["log_moneyness"] = 0.
        anchor[:] = np.tile([100., 200.], 60)
        y[:] = np.tile([120., 300.], 60)
        model = AmountRefinement("contract", 30).fit(X, y, created, anchor)
        # All covariates are constant: raw-dollar least squares with 30 pseudo observations.
        expected = 1 + float(anchor @ (y-anchor)) / (float(anchor @ anchor) + 30*float(np.mean(anchor**2)))
        np.testing.assert_allclose(model.predict_severity(X, anchor)/anchor, expected, atol=1e-12)

    def test_amount_ignores_zero_payout_features_and_other_direction(self):
        X, y, dates, anchor = fixture()
        before = AmountRefinement("market", 30).fit(X, y, dates, anchor).predict_severity(X, anchor)
        changed = X.copy()
        changed.loc[y.eq(0), "remaining_seconds"] *= 100
        after = AmountRefinement("market", 30).fit(changed, y, dates, anchor).predict_severity(X, anchor)
        np.testing.assert_array_equal(before, after)
        yp = y.copy(); yp.loc[X.option_type.eq(1)] *= 20
        after = AmountRefinement("market", 30).fit(X, yp, dates, anchor).predict_severity(X, anchor)
        np.testing.assert_array_equal(before[X.option_type.eq(0)], after[X.option_type.eq(0)])

    def test_probability_uses_occurrence_only_and_keeps_direction_independent(self):
        X, y, dates, _ = fixture()
        fallback = pd.Series(.4, index=X.index)
        first = ProbabilityRefinement("market", .1).fit(X, y, dates).predict_probability(X, fallback)
        second = ProbabilityRefinement("market", .1).fit(X, y*100, dates).predict_probability(X, fallback)
        np.testing.assert_array_equal(first, second)
        altered = y.copy(); altered.loc[X.option_type.eq(1)] = 0.
        second = ProbabilityRefinement("market", .1).fit(X, altered, dates).predict_probability(X, fallback)
        np.testing.assert_array_equal(first[X.option_type.eq(0)], second[X.option_type.eq(0)])
        np.testing.assert_array_equal(second[X.option_type.eq(1)], fallback.loc[X.option_type.eq(1)])
        self.assertTrue(((first >= 0) & (first <= 1)).all())

    def test_insufficient_support_and_numerical_failure_return_exact_anchor(self):
        X, y, dates, anchor = fixture()
        for ids in (X.index[:20], X.index):
            target = y.loc[ids] * 0
            amount = AmountRefinement("contract", 100).fit(X.loc[ids], target, dates.loc[ids], anchor.loc[ids])
            np.testing.assert_array_equal(amount.predict_severity(X, anchor), anchor)
            probability = ProbabilityRefinement("contract", .03).fit(X.loc[ids], target, dates.loc[ids])
            np.testing.assert_array_equal(probability.predict_probability(X, anchor/200), anchor/200)
        with patch("pp_models.two_stage_components.np.linalg.solve", side_effect=np.linalg.LinAlgError("test")):
            amount = AmountRefinement("contract", 30).fit(X, y, dates, anchor)
        np.testing.assert_array_equal(amount.predict_severity(X, anchor), anchor)
        self.assertTrue(all(d["fallback"] == "numerical_failure" for d in amount.diagnostics()))

    def test_amount_bounds_scale_equivariance_and_zero_anchor(self):
        X, y, dates, anchor = fixture()
        model = AmountRefinement("market", 30).fit(X, y, dates, anchor)
        pred = model.predict_severity(X, anchor)
        scaled = AmountRefinement("market", 30).fit(X, y*10, dates, anchor*10).predict_severity(X, anchor*10)
        np.testing.assert_allclose(scaled, pred*10, rtol=1e-12)
        np.testing.assert_array_equal(model.predict_severity(X, anchor*0), anchor*0)
        extreme = AmountRefinement("contract", 30).fit(X, y*10000, dates, anchor).predict_severity(X, anchor)
        self.assertTrue(np.all(extreme <= 1.5*anchor))
        self.assertTrue(np.all(extreme >= .5*anchor))

    def test_constant_probability_features_stay_finite(self):
        X, y, dates, anchor = fixture()
        for c in X.columns:
            if c != "option_type": X[c] = X[c].iloc[0]
        model = ProbabilityRefinement("market", .1).fit(X, y, dates)
        self.assertTrue(np.isfinite(model.predict_probability(X, anchor/200)).all())

    def test_invalid_features_labels_and_anchors_rejected(self):
        X, y, dates, anchor = fixture()
        with self.assertRaises(ValueError): AmountRefinement("contract", 30).fit(X, y.iloc[::-1], dates, anchor)
        with self.assertRaises(ValueError): ProbabilityRefinement("contract", .1).fit(X.assign(actual=y), y, dates)
        with self.assertRaises(ValueError): AmountRefinement("contract", 30).fit(X, y, dates, anchor*float('nan'))
        with self.assertRaises(ValueError): ProbabilityRefinement("contract", .1).fit(X.assign(option_type=2), y, dates)


if __name__ == "__main__": unittest.main()
