import importlib.util
import unittest
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE
from pp_data.io import ModelSplit
from pp_models.config import load_model_config,resolve_config
from pp_models.evaluation import evaluate
from test_model_evaluation import pool_fixture


class GuardedResponseTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.guarded_response'))
        from pp_models import guarded_response
        return guarded_response

    def inputs(self):
        return pd.DataFrame({'option_type':[1,1,0],BASE_FEATURE:[10.,20.,100.],
            'fast_response_state':[1.,.5,1.],'realized_vol_60m':[1.,1.,100.],
            'log_return_60m':[0.,0.,-100.]})

    def test_calm_penalty_closed_form_and_call_independence(self):
        api=self.api();X=self.inputs();y=pd.Series([20.,30.,1.])
        model=api.GuardedResponseModel(.1).fit(X,y)
        self.assertAlmostEqual(model.beta,200/281.25)
        self.assertEqual(model.predict_raw(X)[2],100.)
        y.iloc[2]=1e12;X.loc[2,'realized_vol_60m']=1e6
        self.assertAlmostEqual(api.GuardedResponseModel(.1).fit(X,y).beta,model.beta)
        self.assertGreater(api.GuardedResponseModel(.01).fit(X,y).beta,model.beta)
        self.assertEqual(api.GuardedResponseModel(.1).fit(X,y*0).beta,0.)

    def test_only_upward_updates_are_limited(self):
        api=self.api();X=self.inputs().iloc[[0]]
        model=api.GuardedResponseModel(.01,.1)
        model.fit(X,pd.Series([10.]));self.assertEqual(model.beta,0.)
        model.fit(X,pd.Series([100.]));self.assertAlmostEqual(model.beta,.1)
        self.assertTrue(model.diagnostics()['response_upward_limited'])
        model.fit(X,pd.Series([100.]));self.assertAlmostEqual(model.beta,.2)
        model.fit(X,pd.Series([0.]));self.assertEqual(model.beta,0.)
        self.assertEqual(model.diagnostics()['response_fit_updates'],4)

    def test_pricing_context_uses_features_and_historical_targets_only_for_tail_threshold(self):
        api=self.api();X=self.inputs();y=pd.Series([20.,30.,1.])
        context=api.fit_risk_context(X,y)
        self.assertEqual(context['calm_vol_q80'],1.)
        self.assertEqual(context['calm_downside_q80'],0.)
        self.assertEqual(context['payout_q90'],29.)
        self.assertEqual(api.calm_mask(X,context).tolist(),[True,True,False])
        changed=api.fit_risk_context(X,pd.Series([200.,300.,1.]))
        self.assertEqual(context['calm_vol_q80'],changed['calm_vol_q80'])
        self.assertEqual(context['calm_downside_q80'],changed['calm_downside_q80'])
        no_put=api.fit_risk_context(X.iloc[[2]],y.iloc[[2]])
        self.assertFalse(api.calm_mask(X,no_put).any())
        self.assertEqual(api.GuardedResponseModel(.1).fit(X.iloc[[2]],y.iloc[[2]]).beta,0.)

    def test_invalid_inputs_and_hyperparameters_fail(self):
        api=self.api();X=self.inputs();y=pd.Series([20.,30.,1.])
        for penalty,step in [(True,None),(-1,None),(.1,True),(.1,0),(.1,1.)]:
            with self.assertRaises(ValueError):api.GuardedResponseModel(penalty,step)
        X.loc[0,'realized_vol_60m']=-1
        with self.assertRaises(ValueError):api.GuardedResponseModel(.1).fit(X,y)

    def test_stateful_evaluation_replays_before_phase_and_updates_unscored_days(self):
        # Predict on Jun1 and Jun3; Jun2 still requires a real daily fit.
        pool=pool_fixture();ids=['left_edge','ready','today_early','tomorrow']
        pool=ModelSplit(pool.X.loc[ids].copy(),pool.y.loc[ids].copy(),pool.metadata.loc[ids].copy())
        pool.metadata.loc['tomorrow','created_at_utc']=pd.Timestamp('2026-06-03T10:00Z')
        pool.metadata.loc['tomorrow','closed_at_utc']=pd.Timestamp('2026-06-03T11:00Z')
        cfg=resolve_config(load_model_config()|{'min_train_rows':1},cohort='benchmark',models=['gam'])
        cfg.update(sequential_start='2026-06-01T00:00:00Z',daily_direction_metrics=True)
        histories=[]
        class Counter:
            def __init__(self):self.count=0
            def fit(self,X,y):self.count+=1;histories.append(X.index.tolist())
            def predict_raw(self,X):return np.full(len(X),self.count)
        bounds=['2026-06-01T00:00:00Z','2026-06-04T00:00:00Z']
        _,full=evaluate(pool,'validation',cfg,lambda *_:Counter(),period=bounds)
        self.assertEqual(full.loc[full.group.eq('overall'),'predicted_total_usdc'].tolist(),[2.,3.])
        self.assertEqual(histories,[['left_edge','ready'],['ready','today_early'],['ready','today_early']])
        _,part=evaluate(pool,'validation',cfg,lambda *_:Counter(),period=['2026-06-03T00:00:00Z',bounds[1]])
        self.assertEqual(part.loc[part.group.eq('overall'),'predicted_total_usdc'].tolist(),[3.])
        self.assertEqual(part.loc[part.group.eq('overall'),'unscored_fit_days'].iloc[0],2)
        with self.assertRaises(ValueError):
            evaluate(pool,'validation',cfg|{'sequential_start':'2026-06-02T00:00:00Z'},lambda *_:Counter(),period=bounds)
