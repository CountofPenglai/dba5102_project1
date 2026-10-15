import importlib.util
import unittest
import numpy as np
import pandas as pd
from pp_data.io import ModelSplit
from research_fixtures import fixture


class MultiplierStudyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.base_multiplier_study'))
        from pp_models import base_multiplier_study
        self.study=base_multiplier_study
        self.pool=fixture()
        self.pool.X['total_vol_240m']=.03
        self.pool.X['log_return_60m']=0.

    def test_no_feedback_future_and_unsettled_targets_do_not_change_predictions(self):
        p=self.pool; start=pd.Timestamp('2026-04-01T00:00Z')
        idx=p.metadata.index[p.metadata.created_at_utc.eq(pd.Timestamp('2026-03-25T12:00Z'))][0]
        p.metadata.loc[idx,'closed_at_utc']=pd.Timestamp('2026-04-03T00:00Z')
        specs={'constant':('constant',.1,.1),'market':('market',1.,.1)}
        a,_,_=self.study.predict_period(p,start,pd.Timestamp('2026-04-04T00:00Z'),specs)
        self.assertNotIn('actual',a)
        y=p.y.copy(); y.loc[p.metadata.created_at_utc.ge(start)]=np.nan;y.loc[idx]=1e9
        b,_,_=self.study.predict_period(ModelSplit(p.X,y,p.metadata),start,pd.Timestamp('2026-04-04T00:00Z'),specs)
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue((pd.to_datetime(a.latest_training_close)<pd.to_datetime(a.origin)).all())
        for name in specs:
            np.testing.assert_array_equal(a.query('model == @name and side == 0').multiplier,1.)

    def test_selection_only_development_and_stronger_regularization_tiebreak(self):
        macro=pd.DataFrame([{'model':n,'macro_rmse_per_btc':r,'macro_mae_per_btc':a}
            for n,r,a in [('official_raw',100.,80.),('constant_l0.1',90.,90.),('constant_l1',95.,79.),('constant_l10',95.,79.)]])
        selected=self.study.choose_specs(macro,{'constant_l0.1':('constant',.1,.1),
            'constant_l1':('constant',1.,.1),'constant_l10':('constant',10.,.1)})
        self.assertEqual(selected['constant']['strength'],10.)
        self.assertTrue(selected['constant']['mae_vs_base_pass'])

    def test_access_guard_blocks_official_always_and_test_until_selection(self):
        guard=self.study.AccessGuard()
        for name in ['pp_transactions_validation.csv','outputs/preprocessing/datasets/test/targets.csv']:
            with self.assertRaises(PermissionError):guard.check('open',(name,'r',0))
        guard.allow_pressure=True
        guard.check('open',('outputs/preprocessing/datasets/test/targets.csv','r',0))
        with self.assertRaises(PermissionError):guard.check('open',('pp_transactions_validation.csv','r',0))

    def test_tail_frozen_and_cash_accounting(self):
        p=self.pool
        a,_,_=self.study.predict_period(p,'2026-04-01T00:00Z','2026-04-04T00:00Z',{'constant':('constant',1.,.1)})
        scored=self.study.score(p,a)
        self.assertTrue(scored.groupby(['side','tenor']).tail_threshold.nunique().eq(1).all())
        tables=self.study.tables(scored,'2026-04-01T00:00Z','2026-04-04T00:00Z',['constant'],uncertainty=False)
        s=tables['summary']
        np.testing.assert_allclose(s.total_bias_usdc,s.gross_overprediction_usdc-s.gross_underprediction_usdc,atol=1e-8)
        self.assertFalse(tables['assessment'].risk_pass.any()) # Insufficient tail support.

    def test_continuous_freeze_has_one_origin_and_no_later_label_feedback(self):
        p=self.pool
        for col in p.metadata:p.metadata[col]+=pd.Timedelta(days=30)
        specs={'constant':('constant',1.,.1)}
        a,_,_=self.study.predict_period(p,'2026-04-01T00:00Z','2026-05-04T00:00Z',specs,True)
        monthly,_,_=self.study.predict_period(p,'2026-04-01T00:00Z','2026-05-04T00:00Z',specs)
        self.assertEqual(a.origin.nunique(),1)
        self.assertEqual(monthly.origin.nunique(),2)
        y=p.y.copy();y.loc[p.metadata.created_at_utc.ge(pd.Timestamp('2026-04-01T00:00Z'))]=np.nan
        b,_,_=self.study.predict_period(ModelSplit(p.X,y,p.metadata),'2026-04-01T00:00Z','2026-05-04T00:00Z',specs,True)
        pd.testing.assert_frame_equal(a,b)

    def test_execution_config_matches_fixed_implementation(self):
        settings=self.study.load_settings()
        self.assertEqual(settings['lambdas'],[.1,1.,10.])


if __name__=='__main__':unittest.main()
