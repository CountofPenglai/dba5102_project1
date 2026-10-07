"""Contract checks only: no real estimator is fitted in this suite."""
import copy
import importlib.util
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd


class ModelTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("pp_models"), "model package is missing")
        from pp_models.config import load_model_config, resolve_config
        from pp_models.estimators import build_model
        return load_model_config, resolve_config, build_model

    def test_config_rejects_leakage_and_unsafe_overrides(self):
        load, resolve, _ = self.api()
        config = load()
        for change in ({"features": ["closed_at_utc"]}, {"window_days": 0},
                       {"seed": True}, {"split": "random"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                resolve(config | change)
        with self.assertRaises(ValueError):
            resolve(config, cohort="risk", use_base=True)
        bad = copy.deepcopy(config)
        bad["models"]["ebm"]["interactions"] = [["closedAt", "remaining_seconds"]]
        with self.assertRaises(ValueError):
            resolve(bad)

    def test_real_constructors_never_fit_or_enable_random_validation(self):
        load, resolve, build = self.api()
        from pygam import LinearGAM
        from interpret.glassbox import ExplainableBoostingRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor
        cfg = resolve(load())
        with patch.object(LinearGAM, "fit", side_effect=AssertionError("training forbidden")), \
             patch.object(ExplainableBoostingRegressor, "fit", side_effect=AssertionError("training forbidden")), \
             patch.object(HistGradientBoostingRegressor, "fit", side_effect=AssertionError("training forbidden")):
            models = {name: build(name, cfg) for name in cfg["selected_models"]}
        self.assertEqual(models["ebm"].estimator.validation_size, 0)
        self.assertEqual(models["ebm"].estimator.early_stopping_rounds, 0)
        self.assertFalse(models["shallow_gbdt"].estimator.early_stopping)
        self.assertEqual(models["ebm"].estimator.interactions, [(4, 1), (0, 6)])
        self.assertLessEqual(models["shallow_gbdt"].estimator.max_depth, 3)

    def test_schema_and_label_alignment_fail_before_fitting(self):
        load, resolve, build = self.api()
        cfg = resolve(load())
        model = build("gam", cfg)
        X = pd.DataFrame([[0, 1800, 0, 60000, .01, .04, .002]] * 2,
                         columns=cfg["features"], index=["a", "b"])
        with self.assertRaises(ValueError):
            model.fit(X, pd.Series([0, 1], index=["b", "a"]))
        with self.assertRaises(ValueError):
            model.fit(X[X.columns[::-1]], pd.Series([0, 1], index=X.index))
        with self.assertRaises(ValueError):
            model.fit(X.assign(realized_vol_60m=np.inf), pd.Series([0, 1], index=X.index))
        with self.assertRaises(ValueError):
            model.predict(X)

    def test_prediction_clipping_preserves_raw_values(self):
        self.api()
        from pp_models.estimators import PayoutRegressor
        class FixedEstimator:
            def predict(self, X):
                return np.array([-2., 8.])
        model = PayoutRegressor(FixedEstimator(), ["remaining_seconds"])
        model.fitted = True  # Only the prediction boundary is exercised, no fit.
        X = pd.DataFrame({"remaining_seconds": [1800., 3600.]})
        np.testing.assert_array_equal(model.predict_raw(X), [-2., 8.])
        np.testing.assert_array_equal(model.predict(X), [0., 8.])


if __name__ == "__main__":
    unittest.main()
