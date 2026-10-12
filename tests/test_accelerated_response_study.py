import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from pp_models.config import load_model_config
from test_anchor_study import common_pool


class AcceleratedStudyTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.accelerated_response_study'))
        from pp_models import accelerated_response_study
        return accelerated_response_study

    def fixture(self,api):
        pool=common_pool()
        pool.X['fast_response_state']=.5;pool.X['fast_response_phase']='first_down'
        for name,value in [('linear5',.5),('power5',.25)]:
            pool.X['response_'+name]=value;pool.X['phase_'+name]='first_down'
        cfg=api.load_study_config()
        cfg['periods']={'tuning':['2026-06-01T00:00:00Z','2026-06-02T00:00:00Z'],
                        'followup':['2026-06-02T00:00:00Z','2026-06-03T00:00:00Z']}
        return pool,cfg,load_model_config()|{'min_train_rows':1}

    def test_exact_minimum_selection_frozen_and_fixed_zero_penalty_control(self):
        api=self.api();pool,cfg,base=self.fixture(api);calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()/'example'
            def probe(pool,split,resolved,period):
                phase='tuning' if period==cfg['periods']['tuning'] else 'followup'
                if phase=='followup':
                    locked=json.loads((path/'provenance/selected_models.json').read_text())
                    self.assertEqual(locked['power5']['response_lambda'],.01)
                    self.assertEqual(locked['linear5']['response_lambda'],0.)
                calls.append((phase,resolved['response_mode'],resolved['response_lambda']))
                score=100.
                if resolved['response_mode']=='power5':
                    score={0.:100.1,.01:100.,.1:100.5}[resolved['response_lambda']]
                elif resolved['response_mode']=='linear5':
                    score={0.:100.,.01:100.1,.1:100.5}[resolved['response_lambda']]
                row=dict(model=resolved['selected_models'][0],cohort='benchmark',group='overall',
                         value='all',orders=1,rmse_per_btc=score)
                return pd.DataFrame([row]),pd.DataFrame([row|{'day_utc':period[0]}])
            api.run_accelerated_study(tmp,'example',base,cfg,pool,{'synthetic':True},evaluator=probe)
            self.assertEqual([x[0] for x in calls],['tuning']*9+['followup']*6)
            self.assertEqual(sum(x==('followup','power5',0.) for x in calls),1)
            api.run_accelerated_study(tmp,'example',base,cfg,pool,{'synthetic':True},evaluator=probe,resume=True)
            self.assertEqual(len(calls),15)
            with self.assertRaisesRegex(ValueError,'changed'):
                api.run_accelerated_study(tmp,'example',base,cfg,pool,{'synthetic':False},evaluator=probe,resume=True)
            (path/'metrics/followup.csv').write_text('corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                api.run_accelerated_study(tmp,'example',base,cfg,pool,{'synthetic':True},evaluator=probe,resume=True)

    def test_variant_routing_daily_targets_and_cap_metrics(self):
        api=self.api();pool,cfg,base=self.fixture(api)
        pool.X.option_type=1
        bounds=[cfg['periods']['tuning'][0],cfg['periods']['followup'][1]]
        old=pool.X.copy(deep=True)
        settings=api.variant_config(base,'power5',0.)
        summary,daily=api.evaluate_variant(pool,'validation',settings,period=bounds)
        before=daily.loc[daily.group.eq('overall'),'predicted_total_usdc'].tolist()
        self.assertIn('multiplier_at_cap_orders',summary)
        self.assertEqual(summary.loc[summary.group.eq('overall'),'orders'].iloc[0],2)
        pd.testing.assert_frame_equal(pool.X,old)
        pool.y.loc['today_early']=1000.
        _,changed=api.evaluate_variant(pool,'validation',settings,period=bounds)
        after=changed.loc[changed.group.eq('overall'),'predicted_total_usdc'].tolist()
        self.assertEqual(before[0],after[0]);self.assertNotEqual(before[1],after[1])
        # Different states must reach the shared model input through the wrapper.
        linear=api.variant_config(base,'linear5',0.)
        _,linear_days=api.evaluate_variant(pool,'validation',linear,period=bounds)
        self.assertNotEqual(after[1],linear_days.loc[linear_days.group.eq('overall'),'predicted_total_usdc'].iloc[1])

    def test_unsupported_configuration_and_test_period_rejected(self):
        api=self.api();pool,cfg,base=self.fixture(api)
        cfg['periods']['followup'][1]='2026-07-02T00:00:00Z'
        with self.assertRaises(ValueError):api.plan_study(pool,base,cfg)
        with self.assertRaises(ValueError):api.variant_config(base,'power5',10.)
        with self.assertRaises(ValueError):api.variant_config(base,'raw',0.)
        with self.assertRaises(ValueError):api.variant_config(base,'fast15',.1)
