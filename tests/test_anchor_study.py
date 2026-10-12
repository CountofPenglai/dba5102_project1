import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from pp_data.io import ModelSplit
from pp_data.features import BASE_FEATURE
from pp_models.config import load_model_config
from test_model_evaluation import pool_fixture


def common_pool():
    pool = pool_fixture()
    ids = pool.metadata.index[pool.metadata.benchmark_eligible]
    X = pool.X.loc[ids].copy()
    X[BASE_FEATURE] = 3.
    return ModelSplit(X, pool.y.loc[ids].copy(), pool.metadata.loc[ids].copy())


class AnchorStudyTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.anchor_study'), 'anchor study missing')
        from pp_models import anchor_study
        return anchor_study

    def short_config(self, api):
        cfg = api.load_study_config()
        cfg['periods'] = {'tuning': ['2026-06-01T00:00:00Z', '2026-06-02T00:00:00Z'],
                          'followup': ['2026-06-02T00:00:00Z', '2026-06-03T00:00:00Z']}
        return cfg

    def test_selection_is_frozen_and_scaled_fallback_is_not_duplicated(self):
        api = self.api()
        cfg, calls = self.short_config(api), []
        base = load_model_config() | {'min_train_rows': 1}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve()/'example'
            def probe(pool, split, resolved, model_factory, period):
                phase = 'tuning' if period == cfg['periods']['tuning'] else 'followup'
                if phase == 'followup':
                    locked = json.loads((path/'provenance/selected_models.json').read_text())
                    self.assertEqual(locked['official']['correction'], 'scaled')
                    self.assertEqual(locked['grouped']['correction_lambda'], 10.)
                calls.append(phase)
                mode, anchor = resolved['correction'], resolved['anchor']
                score = 30. if mode == 'raw' else 20.1 if mode == 'scaled' else 20.
                if anchor == 'grouped' and mode == 'scaled':
                    score = 25.
                if phase == 'followup':
                    score = 1000. if mode == 'dynamic' else 1.
                metrics = dict(model=resolved['selected_models'][0], cohort='benchmark', group='overall',
                    value='all', orders=1, rmse_per_btc=score)
                return pd.DataFrame([metrics]), pd.DataFrame([metrics | {'day_utc': period[0]}])
            kwargs = dict(evaluator=probe)
            result = api.run_anchor_study(tmp, 'example', base, cfg, common_pool(), {'synthetic': True}, **kwargs)
            self.assertEqual(result, path)
            self.assertEqual(calls, sorted(calls, key=lambda phase: phase == 'followup'))
            self.assertEqual(len(calls), 15)
            table = pd.read_csv(path/'metrics/followup.csv')
            self.assertEqual(table.anchor.value_counts().to_dict(), {'grouped': 3, 'official': 2})
            self.assertEqual(table.loc[table.selected, 'model'].tolist(), ['official_scaled', 'grouped_dynamic'])
            api.run_anchor_study(tmp, 'example', base, cfg, common_pool(), {'synthetic': True}, resume=True, **kwargs)
            self.assertEqual(len(calls), 15)
            with self.assertRaisesRegex(ValueError, 'changed'):
                api.run_anchor_study(tmp, 'example', base, cfg, common_pool(), {'synthetic': False}, resume=True, **kwargs)
            (path/'metrics/followup.csv').write_text('corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                api.run_anchor_study(tmp, 'example', base, cfg, common_pool(), {'synthetic': True}, resume=True, **kwargs)

    def test_shared_scoring_rows_and_asof_training(self):
        api = self.api()
        from pp_models.anchor_models import build_anchor_model
        from pp_models.evaluation import evaluate
        cfg, pool = self.short_config(api), common_pool()
        base = load_model_config() | {'min_train_rows': 1}
        seen = {}
        for anchor in ('official', 'grouped'):
            for mode in ('raw', 'scaled', 'dynamic'):
                resolved = api.anchor_config(base, anchor, mode, 1. if mode == 'dynamic' else None)
                history, predictions = [], []
                def factory(name, settings):
                    inner = build_anchor_model(name, settings)
                    class Probe:
                        def fit(self, X, y):
                            history.append(X.index.tolist())
                            return inner.fit(X, y)
                        def predict_raw(self, X):
                            predictions.append(X.index.tolist())
                            return inner.predict_raw(X)
                    return Probe()
                summary, daily = evaluate(pool, 'validation', resolved, model_factory=factory,
                    period=[cfg['periods']['tuning'][0], cfg['periods']['followup'][1]])
                self.assertEqual(history, [['left_edge', 'ready'], ['ready', 'today_early']])
                self.assertEqual(predictions, [['today_early'], ['tomorrow']])
                self.assertEqual(int(summary.loc[summary.group.eq('overall'), 'orders'].iloc[0]), 2)
                seen[(anchor, mode)] = daily[daily.group.eq('overall')].predicted_total_usdc.tolist()
                if anchor == 'official' and mode == 'raw':
                    self.assertEqual(seen[(anchor, mode)], [6., 3.])
        pool.y.loc['today_early'] = 60.
        resolved = api.anchor_config(base, 'grouped', 'scaled')
        _, changed = evaluate(pool, 'validation', resolved, model_factory=build_anchor_model,
            period=[cfg['periods']['tuning'][0], cfg['periods']['followup'][1]])
        values = changed[changed.group.eq('overall')].predicted_total_usdc.tolist()
        self.assertEqual(values[0], seen[('grouped', 'scaled')][0])
        self.assertNotEqual(values[1], seen[('grouped', 'scaled')][1])

    def test_test_period_and_noncommon_pool_rejected(self):
        api = self.api()
        cfg = api.load_study_config()
        cfg['periods']['followup'][1] = '2026-07-02T00:00:00Z'
        with self.assertRaisesRegex(ValueError, 'fixed split'):
            api.validate_study(cfg)
        with self.assertRaisesRegex(ValueError, 'common'):
            api.plan_study(pool_fixture(), load_model_config(), self.short_config(api))

    def test_selection_tolerance_boundary(self):
        api = self.api()
        def candidate(mode, lam, score):
            return {'score': score, 'resolved': {'correction': mode, 'correction_lambda': lam}}
        candidates = [candidate('dynamic', .1, 100.), candidate('dynamic', 1., 100.5),
                      candidate('dynamic', 10., 101.), candidate('scaled', None, 101.01)]
        self.assertEqual(api.choose_correction(candidates)['resolved']['correction_lambda'], 10.)
        candidates[-1]['score'] = 101.
        self.assertEqual(api.choose_correction(candidates)['resolved']['correction'], 'scaled')
