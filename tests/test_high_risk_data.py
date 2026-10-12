"""Daily prepared data preserves canonical roles and keeps outcomes outside X."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
import pandas as pd
from tests import test_high_risk_labels as label_fixtures

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'pp_models/high_risk_data.py'


class PeakDataTests(unittest.TestCase):
    def api(self):
        self.assertTrue(PATH.exists(), 'high-risk data entry missing')
        spec = importlib.util.spec_from_file_location('pp_models.high_risk_data', PATH)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_daily_data_uses_same_frozen_threshold_for_history_and_scoring(self):
        api = self.api(); pool, day = label_fixtures.PeakLabelTests().fixture()
        x = pd.DataFrame({'log_quantity_btc': 0.}, index=pool.X.index)
        batch = api.prepare_day(pool, x, day, api.load_settings())
        self.assertEqual(batch['train_X'].index.tolist(), list(range(80)))
        self.assertEqual(batch['score_X'].index.tolist(), [81])
        self.assertEqual(int(batch['train_labels'].high_risk.sum()), 2)
        self.assertEqual(batch['score_labels'].loc[81, 'high_risk'], 1)
        self.assertEqual(batch['train_X'].columns.tolist(), ['log_quantity_btc'])
        self.assertEqual(batch['train_labels'].unit_threshold_q975.nunique(), 1)
        self.assertEqual(batch['train_labels'].unit_threshold_q975.iloc[0], batch['score_labels'].unit_threshold_q975.iloc[0])
        self.assertLess(batch['train_labels'].label_available_at_utc.max(), day)

    def test_rejects_misaligned_or_outcome_named_feature_tables(self):
        api = self.api(); pool, day = label_fixtures.PeakLabelTests().fixture()
        x = pd.DataFrame({'log_quantity_btc': 0.}, index=pool.X.index)
        with self.assertRaises(ValueError): api.prepare_day(pool, x.iloc[::-1], day, api.load_settings())
        x['high_risk'] = 0
        with self.assertRaises(ValueError): api.prepare_day(pool, x, day, api.load_settings())

    def test_plan_does_not_create_run_or_start_model_training(self):
        self.api()
        result = subprocess.run([sys.executable, '-m', 'pp_models.high_risk_data', 'plan'],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        import json
        plan = json.loads(result.stdout)
        self.assertFalse(plan['train_classifier'])
        self.assertEqual(plan['primary_quantile'], .975)


if __name__ == '__main__': unittest.main()
