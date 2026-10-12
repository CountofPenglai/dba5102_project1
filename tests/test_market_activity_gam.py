import importlib.util
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_models.config import load_model_config


class MarketActivityGAMTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.market_activity_gam'), 'new GAM variants missing')
        from pp_models import market_activity_gam
        return market_activity_gam

    def test_new_information_can_change_predictions_for_each_direction(self):
        api = self.api()
        for variant, feature in [('price_dynamics', 'vol_change_60m'), ('platform_activity', 'platform_put_count_60m')]:
            cfg = api.variant_config(load_model_config(), variant, 1000)
            X = pd.DataFrame(1., index=range(100), columns=cfg['features'])
            X['option_type'] = np.repeat([0., 1.], 50)
            X[feature] = np.tile(np.linspace(0, 10, 50), 2)
            # Equal direction means isolate slope response from the deliberately penalized intercept.
            y = 100 + 10*(X[feature]-5)*(1-2*X.option_type)
            with threadpool_limits(limits=1):
                model = api.build_activity_model('gam', cfg).fit(X, y)
                pred = model.predict(X)
            self.assertGreater(pred[49]-pred[0], 99.)
            self.assertLess(pred[99]-pred[50], -99.)
            self.assertLess(np.sqrt(np.mean((pred-y)**2)), .1)

    def test_reference_is_identical_and_answer_columns_rejected(self):
        from pp_models.directional_gam import build_directional_model, variant_config
        api = self.api()
        cfg = api.variant_config(load_model_config(), 'semivol', 1000)
        X = pd.DataFrame(np.random.default_rng(3).uniform(.1, 1., (70, len(cfg['features']))), columns=cfg['features'])
        X['option_type'] = np.arange(70) % 2
        y = pd.Series(np.arange(70, dtype=float))
        with threadpool_limits(limits=1):
            old = build_directional_model('gam', variant_config(load_model_config(), 'semivol', 1000)).fit(X, y).predict(X)
            new = api.build_activity_model('gam', cfg).fit(X, y).predict(X)
        np.testing.assert_allclose(new, old, rtol=1e-12, atol=1e-12)
        cfg = api.variant_config(load_model_config(), 'platform_activity', 1000)
        cfg['features'].append('realizedProfit_USDC')
        with self.assertRaisesRegex(ValueError, 'features'):
            api.build_activity_model('gam', cfg)
