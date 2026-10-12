import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from pp_models.config import load_model_config
from test_anchor_study import common_pool


class GuardedStudyTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.guarded_response_study'))
        from pp_models import guarded_response_study
        return guarded_response_study

    def fixture(self,api):
        pool=common_pool();pool.X.option_type=1
        pool.X['response_power5']=.5;pool.X['phase_power5']='first_down'
        cfg=api.load_study_config()
        cfg['periods']={'tuning':['2026-06-01T00:00:00Z','2026-06-02T00:00:00Z'],
                        'followup':['2026-06-02T00:00:00Z','2026-06-03T00:00:00Z']}
        return pool,cfg,load_model_config()|{'min_train_rows':1}

    def test_replayed_rate_model_matches_full_walk_and_future_labels_do_not_leak(self):
        api=self.api();pool,cfg,base=self.fixture(api)
        settings=api.variant_config(base,'calm_rate',.1,.1,'2026-06-01T00:00:00Z')
        full_period=[cfg['periods']['tuning'][0],cfg['periods']['followup'][1]]
        summary,full=api.evaluate_variant(pool,'validation',settings,period=full_period)
        _,part=api.evaluate_variant(pool,'validation',settings,period=cfg['periods']['followup'])
        first=full.loc[full.group.eq('overall')].iloc[0]
        last=full.loc[full.group.eq('overall')].iloc[-1]
        self.assertEqual(last.predicted_total_usdc,part.loc[part.group.eq('overall'),'predicted_total_usdc'].iloc[0])
        self.assertEqual(set(summary.group)&{'pricing_risk_group','payout_tail_group'}, {'pricing_risk_group','payout_tail_group'})
        self.assertEqual(part.loc[part.group.eq('overall'),'unscored_fit_days'].iloc[0],1)
        pool.y.loc['today_early']=1000.
        _,changed=api.evaluate_variant(pool,'validation',settings,period=full_period)
        changed=changed.loc[changed.group.eq('overall')]
        self.assertEqual(first.predicted_total_usdc,changed.predicted_total_usdc.iloc[0])
        self.assertEqual(first.payout_q90,changed.payout_q90.iloc[0])
        self.assertNotEqual(last.predicted_total_usdc,changed.predicted_total_usdc.iloc[1])

    def test_budget_selection_freeze_resume_and_no_followup_reselection(self):
        api=self.api();pool,cfg,base=self.fixture(api);calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()/'example'
            def probe(pool,split,resolved,period):
                phase='tuning' if period==cfg['periods']['tuning'] else 'followup'
                mode=resolved['response_mode'];calls.append(phase)
                if phase=='followup':
                    selected=json.loads((path/'provenance/selected_models.json').read_text())
                    self.assertEqual(selected['calm']['calm_penalty'],.1)
                    self.assertEqual(selected['calm_rate']['upward_step'],.1)
                score,calm_mae=100.,100.
                if mode in ('calm','calm_rate'):
                    param=resolved['calm_penalty'] if mode=='calm' else resolved['upward_step']
                    score=90. if param in (.01,.05) else 95. if param==.1 else 99.
                    calm_mae=120. if param in (.01,.05) else 100.
                if phase=='followup':score=500.
                row=dict(model=resolved['selected_models'][0],cohort='benchmark',orders=1,
                    rmse_per_btc=score,mae_per_btc=100.,gross_underprediction_usdc=100.)
                summary=pd.DataFrame([row|{'group':'overall','value':'all'},
                    row|{'group':'pricing_risk_group','value':'calm','mae_per_btc':calm_mae},
                    row|{'group':'payout_tail_group','value':'tail'}])
                return summary,pd.DataFrame([row|{'group':'overall','value':'all','day_utc':period[0]}])
            api.run_guarded_study(tmp,'example',base,cfg,pool,{'synthetic':True},evaluator=probe)
            self.assertEqual(calls,['tuning']*8+['followup']*4)
            decisions=json.loads((path/'provenance/selection_checks.json').read_text())
            self.assertTrue(decisions['calm']['selected_passes_all_budgets'])
            api.run_guarded_study(tmp,'example',base,cfg,pool,{'synthetic':True},evaluator=probe,resume=True)
            self.assertEqual(len(calls),12)
            (path/'metrics/followup.csv').write_text('corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                api.run_guarded_study(tmp,'example',base,cfg,pool,{'synthetic':True},evaluator=probe,resume=True)

    def test_missing_groups_do_not_pass_budgets_and_test_dates_rejected(self):
        api=self.api();pool,cfg,base=self.fixture(api)
        row=dict(cohort='benchmark',group='overall',value='all',rmse_per_btc=100.,mae_per_btc=100.,gross_underprediction_usdc=100.)
        record={'summary':pd.DataFrame([row]),'score':100.,'resolved':{'calm_penalty':.1,'upward_step':None},'id':'x'}
        chosen,info=api.choose_guarded([record],record,record,cfg['budgets'])
        self.assertFalse(info['selected_passes_all_budgets'])
        self.assertEqual(chosen,record)
        cfg['periods']['followup'][1]='2026-07-02T00:00:00Z'
        with self.assertRaises(ValueError):api.plan_study(pool,base,cfg)
        with self.assertRaises(ValueError):api.variant_config(base,'calm_rate',.1,.1,None)
