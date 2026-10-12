import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from pp_models.config import load_model_config
from test_anchor_study import common_pool


class TerminalPayoutTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.terminal_payout'))
        from pp_models import terminal_payout
        return terminal_payout

    def test_penalized_fit_by_hand_calls_and_long_tenor_unchanged(self):
        api=self.api()
        X=pd.DataFrame({'option_type':[1,1,0,1],'api_base_price_per_btc_usdc':[10.,20.,30.,40.],
            'terminal_supported':[1,1,0,0],'terminal_static_proxy':[20.,0.,0.,0.],
            'terminal_dynamic_proxy':[20.,0.,0.,0.]})[api.INPUTS]
        y=pd.Series([20.,10.,10000.,20000.])
        model=api.TerminalPayoutModel('dynamic',.1).fit(X,y)
        self.assertAlmostEqual(model.a,20/37);self.assertAlmostEqual(model.b,24/37)
        pred=model.predict_raw(X)
        np.testing.assert_array_equal(pred[2:],[30.,40.])
        stronger=api.TerminalPayoutModel('dynamic',1.).fit(X,y)
        self.assertLess(stronger.b,model.b)
        scaled=api.TerminalPayoutModel('scaled').fit(X,y)
        self.assertAlmostEqual(scaled.a,.8)
        empty=X.copy();empty.terminal_supported=0
        np.testing.assert_array_equal(api.TerminalPayoutModel('dynamic',.1).fit(empty,y).predict_raw(empty),X.api_base_price_per_btc_usdc)
        zeros=y.copy();zeros.iloc[:2]=0
        np.testing.assert_array_equal(api.TerminalPayoutModel('dynamic',.1).fit(X,zeros).predict_raw(X),[0,0,30,40])
        with self.assertRaises(ValueError):api.TerminalPayoutModel('dynamic',0.)
        with self.assertRaises(ValueError):model.fit(X,y.iloc[::-1])


class TerminalStudyTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.terminal_distribution_study'))
        from pp_models import terminal_distribution_study
        return terminal_distribution_study

    def fixture(self,api):
        p=common_pool();p.X.option_type=1
        p.X['terminal_supported']=1
        p.X['terminal_static_proxy']=5.
        p.X['terminal_dynamic_proxy']=10.
        cfg=api.load_study_config()
        cfg['periods']={'tuning':['2026-06-01T00:00:00Z','2026-06-02T00:00:00Z'],
                       'followup':['2026-06-02T00:00:00Z','2026-06-03T00:00:00Z']}
        return p,cfg,load_model_config()|{'min_train_rows':1}

    def test_current_targets_do_not_affect_same_day_prediction(self):
        api=self.api();p,cfg,base=self.fixture(api)
        resolved=api.variant_config(base,'dynamic',.1)
        bounds=[cfg['periods']['tuning'][0],cfg['periods']['followup'][1]]
        _,original=api.evaluate_variant(p,'validation',resolved,period=bounds)
        p.y.loc['today_early']=1000.
        _,changed=api.evaluate_variant(p,'validation',resolved,period=bounds)
        a=original.loc[original.group.eq('overall')];b=changed.loc[changed.group.eq('overall')]
        self.assertEqual(a.predicted_total_usdc.iloc[0],b.predicted_total_usdc.iloc[0])
        self.assertNotEqual(a.predicted_total_usdc.iloc[1],b.predicted_total_usdc.iloc[1])
        with self.assertRaises(ValueError):api.evaluate_variant(p,'test',resolved)

    def test_selection_frozen_before_followup_resume_and_identity(self):
        api=self.api();p,cfg,base=self.fixture(api);calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'example'
            def probe(pool,split,resolved,period):
                phase='tuning' if period==cfg['periods']['tuning'] else 'followup'
                calls.append(phase)
                if phase=='followup':
                    selected=json.loads((root/'provenance/selected_models.json').read_text())
                    self.assertEqual(selected['static']['distribution_lambda'],.1)
                    self.assertEqual(selected['dynamic']['distribution_lambda'],.1)
                score={None:100.,.01:95.,.1:90.,1.:99.}[resolved['distribution_lambda']]
                r={'model':resolved['selected_models'][0],'cohort':'benchmark','group':'overall','value':'all',
                   'orders':1,'rmse_per_btc':score if phase=='tuning' else 500.}
                return pd.DataFrame([r]),pd.DataFrame([r|{'day_utc':period[0]}])
            api.run_terminal_study(tmp,'example',base,cfg,p,{'synthetic':True},evaluator=probe)
            self.assertEqual(calls,['tuning']*8+['followup']*4)
            api.run_terminal_study(tmp,'example',base,cfg,p,{'synthetic':True},evaluator=probe,resume=True)
            self.assertEqual(len(calls),12)
            (root/'metrics/followup.csv').write_text('corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                api.run_terminal_study(tmp,'example',base,cfg,p,{'synthetic':True},evaluator=probe,resume=True)
        cfg['periods']['followup'][1]='2026-07-02T00:00:00Z'
        with self.assertRaises(ValueError):api.plan_study(p,base,cfg)


if __name__=='__main__':unittest.main()
