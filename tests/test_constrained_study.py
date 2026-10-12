import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from pp_models.config import load_model_config
from test_anchor_study import common_pool


class ConstrainedStudyTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.constrained_study'))
        from pp_models import constrained_study
        return constrained_study

    def fixture(self,api):
        from pp_models.constrained_pricing import INPUTS
        p=common_pool()
        for f in INPUTS:
            if f not in p.X:p.X[f]=1.
        p.X['terminal_supported']=p.X.option_type.eq(1).astype(int)
        p.X['terminal_static_proxy']=p.X['terminal_dynamic_proxy']=10.
        cfg=api.load_study_config()
        cfg['periods']={'tuning':['2026-06-01T00:00:00Z','2026-06-02T00:00:00Z'],
                       'followup':['2026-06-02T00:00:00Z','2026-06-03T00:00:00Z']}
        return p,cfg,load_model_config()|{'min_train_rows':1}

    def test_future_targets_do_not_change_current_prediction_and_test_rejected(self):
        api=self.api();p,cfg,base=self.fixture(api)
        resolved=api.variant_config(base,'flow',1.)
        bounds=[cfg['periods']['tuning'][0],cfg['periods']['followup'][1]]
        _,first=api.evaluate_variant(p,'validation',resolved,period=bounds)
        p.y.loc['today_early']=1000.
        _,second=api.evaluate_variant(p,'validation',resolved,period=bounds)
        a=first.loc[first.group.eq('overall')];b=second.loc[second.group.eq('overall')]
        self.assertEqual(a.predicted_total_usdc.iloc[0],b.predicted_total_usdc.iloc[0])
        self.assertNotEqual(a.predicted_total_usdc.iloc[1],b.predicted_total_usdc.iloc[1])
        with self.assertRaises(ValueError):api.evaluate_variant(p,'test',resolved)

    def test_selection_persisted_before_followup_and_resume_verifies_hash(self):
        api=self.api();p,cfg,base=self.fixture(api);calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'trial'
            def evaluator(pool,split,resolved,period):
                phase='tuning' if period==cfg['periods']['tuning'] else 'followup';calls.append(phase)
                if phase=='followup':
                    selected=json.loads((root/'provenance/selected_models.json').read_text())
                    self.assertEqual(selected['flow']['constraint_lambda'],1.)
                score={None:100.,.1:95.,1.:90.,10.:99.}[resolved.get('constraint_lambda')]
                row={'model':resolved['selected_models'][0],'cohort':'benchmark','group':'overall',
                     'value':'all','orders':1,'rmse_per_btc':score}
                return pd.DataFrame([row]),pd.DataFrame([row|{'day_utc':period[0]}])
            api.run_study(tmp,'trial',base,cfg,p,{'synthetic':True},evaluator=evaluator)
            self.assertEqual(calls,['tuning']*12+['followup']*6)
            api.run_study(tmp,'trial',base,cfg,p,{'synthetic':True},evaluator=evaluator,resume=True)
            self.assertEqual(len(calls),18)
            (root/'metrics/followup.csv').write_text('broken')
            with self.assertRaisesRegex(ValueError,'checksum'):
                api.run_study(tmp,'trial',base,cfg,p,{'synthetic':True},evaluator=evaluator,resume=True)
        cfg['periods']['followup'][1]='2026-07-02T00:00:00Z'
        with self.assertRaises(ValueError):api.plan_study(p,base,cfg)


if __name__=='__main__':unittest.main()
