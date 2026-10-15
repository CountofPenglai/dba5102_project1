"""Public report components must not import archived experiments."""
import ast
import importlib
import unittest
from pathlib import Path
import numpy as np
import pandas as pd


class ReportArchitectureTests(unittest.TestCase):
    def test_h0_has_only_pricing_inputs_and_no_terminal_placeholders(self):
        from pp_models.h0 import H0, MODEL_INPUTS, probability_frame
        self.assertEqual(len(MODEL_INPUTS), 6)
        self.assertFalse(any('terminal' in c for c in MODEL_INPUTS))
        x = pd.DataFrame(dict(option_type=[0, 1, 0, 1], api_base_price_per_btc_usdc=[100.]*4,
            remaining_seconds=[3600.]*4, product_duration_seconds=[3600.]*4,
            realized_vol_60m=[.01]*4, log_moneyness=[0.]*4))
        model = H0().fit(probability_frame(x), pd.Series([200., 0., 200., 0.]))
        np.testing.assert_allclose(model.predict_raw(x), [200., 0., 200., 0.])
        with self.assertRaises(ValueError): model.predict_raw(x.assign(closed_at=0))

    def test_independent_metrics_and_cash_sign(self):
        from pp_models.metrics import key_metrics
        from pp_models.verification import metrics
        f = pd.DataFrame(dict(actual=[0., 100.], prediction=[10., 70.], quantity=[2., 3.]))
        a = key_metrics(f.actual, f.prediction, f.quantity)
        self.assertAlmostEqual(a['rmse_per_btc'], np.sqrt(500))
        self.assertEqual(a['gross_underprediction_usdc'], 90.)
        self.assertEqual(a['total_bias_usdc'], -70.)
        for k, v in a.items(): self.assertAlmostEqual(metrics(f)[k], v)

    def test_shared_components_import_without_old_experiments(self):
        for name in ['h0', 'metrics', 'resampling', 'runtime', 'verification']:
            m = importlib.import_module('pp_models.'+name)
            tree = ast.parse(Path(m.__file__).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(any(v in node.module for v in
                        ['terminal', 'weekly_residual', 'two_stage', 'frozen_candidate', 'base_multiplier_study']), node.module)


if __name__ == '__main__': unittest.main()
