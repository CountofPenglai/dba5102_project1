import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from pp_models.config import load_model_config
from test_model_evaluation import pool_fixture


class MarketActivityStudyTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.market_activity_study'), 'activity study missing')
        from pp_models import market_activity_study
        return market_activity_study

    def test_selection_precedes_followup_fixed_ablation_and_resume(self):
        api = self.api()
        config = api.load_study_config()
        config['periods'] = {'tuning': ['2026-06-01T00:00:00Z', '2026-06-02T00:00:00Z'],
                             'followup': ['2026-06-02T00:00:00Z', '2026-06-03T00:00:00Z']}
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve()/'example'
            def probe(pool, split, resolved, model_factory, period):
                phase = 'tuning' if period == config['periods']['tuning'] else 'followup'
                if phase == 'followup':
                    locked = json.loads((path/'provenance/selected_models.json').read_text())
                    self.assertEqual(locked['price_dynamics']['models']['gam']['lam'], 100)
                    self.assertEqual(locked['platform_activity']['models']['gam']['lam'], 100)
                calls.append(phase)
                lam = resolved['models']['gam']['lam']
                score = {10: 20., 100: 20.1, 1000: 30.}[lam] if phase == 'tuning' else 1./lam
                metrics = {'model': resolved['selected_models'][0], 'cohort': 'risk', 'group': 'overall',
                    'value': 'all', 'orders': 1, 'rmse_per_btc': score}
                return pd.DataFrame([metrics]), pd.DataFrame([metrics | {'day_utc': period[0]}])
            base = load_model_config() | {'min_train_rows': 1}
            kwargs = dict(evaluator=probe)
            result = api.run_activity_study(tmp, 'example', base, config, pool_fixture(), {'synthetic': True}, **kwargs)
            self.assertEqual(result, path)
            self.assertEqual(calls, sorted(calls, key=lambda p: p == 'followup'))
            self.assertEqual(len(calls), 18)
            frame = pd.read_csv(path/'metrics/followup.csv')
            own = frame[frame.variant.eq('platform_activity')]
            self.assertEqual(own.lam.tolist(), [100., 1000.])
            self.assertEqual(own.selected.tolist(), [True, False])
            self.assertEqual(own.fixed_regularization_control.tolist(), [False, True])
            api.run_activity_study(tmp, 'example', base, config, pool_fixture(), {'synthetic': True}, resume=True, **kwargs)
            self.assertEqual(len(calls), 18, 'resume must reuse verified aggregates')
            with self.assertRaisesRegex(ValueError, 'changed'):
                api.run_activity_study(tmp, 'example', base, config, pool_fixture(), {'synthetic': False}, resume=True, **kwargs)
            (path/'metrics/followup.csv').write_text('corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                api.run_activity_study(tmp, 'example', base, config, pool_fixture(), {'synthetic': True}, resume=True, **kwargs)

    def test_test_period_is_rejected(self):
        api = self.api()
        cfg = api.load_study_config()
        cfg['periods']['followup'][1] = '2026-07-02T00:00:00Z'
        with self.assertRaisesRegex(ValueError, 'fixed split'):
            api.validate_study(cfg)
