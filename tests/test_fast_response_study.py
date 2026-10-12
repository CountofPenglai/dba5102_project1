import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from pp_models.config import load_model_config
from test_anchor_study import common_pool


class FastResponseStudyTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.fast_response_study'))
        from pp_models import fast_response_study
        return fast_response_study

    def fixture(self, api):
        cfg = api.load_study_config()
        cfg['periods'] = {'tuning': ['2026-06-01T00:00:00Z', '2026-06-02T00:00:00Z'],
                          'followup': ['2026-06-02T00:00:00Z', '2026-06-03T00:00:00Z']}
        pool = common_pool()
        pool.X['fast_response_state'] = .5
        pool.X['fast_response_phase'] = 'first_down'
        return cfg, pool, load_model_config() | {'min_train_rows': 1}

    def test_selection_frozen_before_followup_and_integrity_on_resume(self):
        api = self.api()
        cfg, pool, base = self.fixture(api)
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve()/'example'
            def probe(pool, split, resolved, model_factory, period):
                phase = 'tuning' if period == cfg['periods']['tuning'] else 'followup'
                if phase == 'followup':
                    self.assertEqual(json.loads((path/'provenance/selected_models.json').read_text())['fast']['response_lambda'], 10.)
                calls.append(phase)
                score = (100. if resolved.get('response_lambda') == .1 else 100.5)
                if phase == 'followup':
                    score = 300. if resolved.get('response_lambda') else 1.
                metrics = dict(model=resolved['selected_models'][0], cohort='benchmark', group='overall',
                               value='all', orders=1, rmse_per_btc=score)
                return pd.DataFrame([metrics]), pd.DataFrame([metrics | {'day_utc': period[0]}])
            result = api.run_response_study(tmp, 'example', base, cfg, pool, {'synthetic': True}, evaluator=probe)
            self.assertEqual(result, path)
            self.assertEqual(calls, ['tuning']*5+['followup']*3)
            table = pd.read_csv(path/'metrics/followup.csv')
            self.assertEqual(set(table.model), {'official_raw', 'official_scaled', 'official_fast'})
            api.run_response_study(tmp, 'example', base, cfg, pool, {'synthetic': True}, evaluator=probe, resume=True)
            self.assertEqual(len(calls), 8)
            with self.assertRaisesRegex(ValueError, 'changed'):
                api.run_response_study(tmp, 'example', base, cfg, pool, {'synthetic': False}, evaluator=probe, resume=True)
            (path/'metrics/followup.csv').write_text('corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                api.run_response_study(tmp, 'example', base, cfg, pool, {'synthetic': True}, evaluator=probe, resume=True)

    def test_daily_asof_same_ids_and_future_label_invariance(self):
        api = self.api()
        from pp_models.evaluation import evaluate
        cfg, pool, base = self.fixture(api)
        # Let fitting data justify a positive Put amplitude.
        pool.X.option_type = 1
        seen = {}
        for mode in ('raw', 'scaled', 'fast'):
            resolved = api.response_config(base, mode, .1 if mode == 'fast' else None)
            history, predictions = [], []
            def factory(name, settings):
                inner = api.build_response_model(name, settings)
                class Probe:
                    def fit(self, X, y):
                        history.append(X.index.tolist()); inner.fit(X, y)
                    def predict_raw(self, X):
                        predictions.append(X.index.tolist()); return inner.predict_raw(X)
                return Probe()
            summary, daily = evaluate(pool, 'validation', resolved, model_factory=factory,
                period=[cfg['periods']['tuning'][0], cfg['periods']['followup'][1]])
            self.assertEqual(history, [['left_edge', 'ready'], ['ready', 'today_early']])
            self.assertEqual(predictions, [['today_early'], ['tomorrow']])
            self.assertEqual(summary.loc[summary.group.eq('fast_response_phase'), 'orders'].sum(), 2)
            self.assertTrue('multiplier_max' in daily)
            seen[mode] = daily.loc[daily.group.eq('overall'), 'predicted_total_usdc'].tolist()
        pool.y.loc['today_early'] = 100.
        _, changed = evaluate(pool, 'validation', api.response_config(base, 'fast', .1),
            model_factory=api.build_response_model, period=[cfg['periods']['tuning'][0], cfg['periods']['followup'][1]])
        values = changed.loc[changed.group.eq('overall'), 'predicted_total_usdc'].tolist()
        self.assertEqual(values[0], seen['fast'][0])
        self.assertNotEqual(values[1], seen['fast'][1])

    def test_test_period_and_nonfixed_grid_rejected(self):
        api = self.api()
        cfg, pool, base = self.fixture(api)
        cfg['periods']['followup'][1] = '2026-07-02T00:00:00Z'
        with self.assertRaises(ValueError): api.plan_study(pool, base, cfg)
        cfg = api.load_study_config(); cfg['lambdas'] = [0.01]
        with self.assertRaises(ValueError): api.validate_study(cfg)
        with self.assertRaises(ValueError): api.response_config(base, 'fast', 0.)
        with self.assertRaises(ValueError): api.response_config(base, 'fast', True)
