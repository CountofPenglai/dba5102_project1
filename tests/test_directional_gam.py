import importlib.util
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_models.config import load_model_config


class DirectionalGAMTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("pp_models.directional_gam"), "directional GAM is missing")
        from pp_models import directional_gam
        return directional_gam

    def test_direction_can_change_slope_not_only_intercept(self):
        api = self.api()
        cfg = api.variant_config(load_model_config(), "directional", 100)
        x = np.tile(np.linspace(-1, 1, 50), 2)
        option = np.repeat([0., 1.], 50)
        X = pd.DataFrame(1., index=range(100), columns=cfg["features"])
        X["option_type"], X["log_return_60m"] = option, x
        y = pd.Series(100 + 20 * x * (1 - 2 * option), index=X.index)
        with threadpool_limits(limits=1):
            model = api.build_directional_model("gam", cfg).fit(X, y)
            pred = model.predict(X)
        self.assertLess(np.sqrt(np.mean((pred-y)**2)), .1)
        self.assertGreater(pred[49] - pred[0], 39)
        self.assertLess(pred[99] - pred[50], -39)
        self.assertGreater(model.diagnostics()["gam_edof"], 1)
        self.assertLess(model.diagnostics()["gam_final_diff"], 1e-4)

    def test_feature_config_cannot_introduce_answer_columns(self):
        api = self.api()
        cfg = api.variant_config(load_model_config(), "semivol", 100)
        cfg["features"].append("realizedProfit_USDC")
        with self.assertRaisesRegex(ValueError, "features"):
            api.build_directional_model("gam", cfg)

    def test_direction_deviations_do_not_bypass_intercept_penalty(self):
        api = self.api()
        means = []
        for variant in ("multiwindow", "directional", "semivol"):
            cfg = api.variant_config(load_model_config(), variant, 100)
            X = pd.DataFrame(1., index=range(100), columns=cfg["features"])
            X["option_type"] = np.repeat([0., 1.], 50)
            y = 100 + 100 * X.option_type
            with threadpool_limits(limits=1):
                model = api.build_directional_model("gam", cfg).fit(X, y)
                pred = model.predict(X)
            means.append([pred[:50].mean(), pred[50:].mean()])
        # 25/(25+100) times the 100-unit direction effect = 20-unit gap.
        np.testing.assert_allclose(means, [[140., 160.]] * 3, atol=.001)

    def test_original_predictions_match_existing_selected_model(self):
        from pp_models.estimators import build_model
        api = self.api()
        cfg = api.variant_config(load_model_config(), "original", 100)
        X = pd.DataFrame(np.random.default_rng(42).uniform(.1, 1, (50, len(cfg["features"]))), columns=cfg["features"])
        X["option_type"] = np.arange(50) % 2
        y = pd.Series(np.arange(50, dtype=float))
        with threadpool_limits(limits=1):
            old = build_model("gam", cfg).fit(X, y).predict(X)
            new = api.build_directional_model("gam", cfg).fit(X, y).predict(X)
        np.testing.assert_allclose(old, new, rtol=1e-12, atol=1e-12)
