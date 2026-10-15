import importlib.util
import unittest
import numpy as np
import pandas as pd
from pp_data.io import ModelSplit
from research_fixtures import fixture
from test_macro_multiplier import calendar


class MacroStudyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('pp_models.macro_multiplier_study'))
        from pp_models import macro_multiplier_study as study
        self.study=study
        self.pool=fixture()
        self.calendar=calendar()
        template=self.calendar['events'][0]
        self.calendar['events']=[{**template,'event_id':str(i),'event_time_utc':t.isoformat()}
            for i,t in enumerate(pd.date_range('2026-02-02T12:30Z','2026-04-03T12:30Z',freq='3D'))]

    def test_frozen_predictions_ignore_future_and_unsettled_targets(self):
        p=self.pool;start=pd.Timestamp('2026-04-01T00:00Z')
        idx=p.metadata.index[p.metadata.created_at_utc.eq(pd.Timestamp('2026-03-25T12:00Z'))][0]
        p.metadata.loc[idx,'closed_at_utc']=pd.Timestamp('2026-04-03T00:00Z')
        a,_,_=self.study.predict_period(p,self.calendar,start,'2026-04-04T00:00Z',{'announcement':1.})
        y=p.y.copy();y.loc[p.metadata.created_at_utc.ge(start)]=np.nan;y.loc[idx]=1e12
        b,_,_=self.study.predict_period(ModelSplit(p.X,y,p.metadata),self.calendar,start,'2026-04-04T00:00Z',{'announcement':1.})
        pd.testing.assert_frame_equal(a,b)
        self.assertNotIn('actual',a)
        self.assertTrue((pd.to_datetime(a.latest_training_close)<pd.to_datetime(a.origin)).all())
        self.assertTrue(a.query("model == 'announcement'").supported.any())

    def test_selection_uses_development_mae_and_stronger_tie(self):
        table=pd.DataFrame({'model':['official_raw','l1','l10','l100'],
            'macro_rmse_per_btc':[100.,90.,95.,95.],'macro_mae_per_btc':[80.,81.,79.,79.]})
        s=self.study.choose(table,{'l1':1.,'l10':10.,'l100':100.})
        self.assertEqual(s['strength'],100.)
        self.assertTrue(s['mae_vs_base_pass'])

    def test_predictions_do_not_require_query_settlement_times(self):
        p=self.pool;start=pd.Timestamp('2026-04-01T00:00Z')
        a,_,_=self.study.predict_period(p,self.calendar,start,'2026-04-04T00:00Z',{'announcement':1.})
        metadata=p.metadata.copy()
        metadata.loc[metadata.created_at_utc.ge(start),'closed_at_utc']=pd.NaT
        b,_,_=self.study.predict_period(ModelSplit(p.X,p.y,metadata),self.calendar,start,'2026-04-04T00:00Z',{'announcement':1.})
        pd.testing.assert_frame_equal(a,b)

    def test_scoring_keeps_call_corrections_and_shared_tail_thresholds(self):
        p,_,_=self.study.predict_period(self.pool,self.calendar,'2026-04-01T00:00Z','2026-04-04T00:00Z',{'announcement':1.})
        f=self.study.score(self.pool,p)
        self.assertTrue(f.groupby(['side','tenor']).tail_threshold.nunique().eq(1).all())
        from pp_models.macro_multiplier_statistics import tables
        out=tables(f,'2026-04-01T00:00Z','2026-04-04T00:00Z','announcement',True)
        self.assertIn('call',set(out['paired_intervals'].scope))
        self.assertFalse(out['assessment'].adoption_point_pass.any())
        np.testing.assert_allclose(out['summary'].total_bias_usdc,
                                  out['summary'].gross_overprediction_usdc-out['summary'].gross_underprediction_usdc,atol=1e-8)


if __name__=='__main__':unittest.main()
