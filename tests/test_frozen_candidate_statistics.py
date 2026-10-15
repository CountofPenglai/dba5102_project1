import unittest
import numpy as np
import pandas as pd
from pp_models.frozen_candidate_statistics import tables, comparisons


class StatisticsTests(unittest.TestCase):
    def test_mae_increase_is_cost_not_accuracy_veto(self):
        rows=[]
        for name,r,a in [('official_raw',100,10),('h0',90,12),('contract30',80,15),('g1a',85,14),('static',95,11)]:
            rows.append(dict(model=name,macro_rmse_per_btc=r,macro_mae_per_btc=a))
        result=comparisons(pd.DataFrame(rows))
        p=result.query("candidate == 'contract30' and reference == 'h0'").iloc[0]
        self.assertTrue(p.rmse_improved);self.assertGreater(p.mae_delta,0)
        self.assertNotIn('mae_pass',result)

    def test_cash_accounting_and_missing_tail_support(self):
        rows=[]
        for model in ['official_raw','h0','contract30','g1a','static']:
            for i in range(8):
                rows.append(dict(order_id=i,model=model,day=pd.Timestamp('2026-04-01',tz='UTC')+pd.Timedelta(days=i),
                    month='2026-04',side=i%2,tenor=3600,quantity=2.,actual=100.,prediction=80.,
                    tail_group='tail' if i<4 else 'body',multiplier=.8))
        f=pd.DataFrame(rows).set_index('order_id')
        t=tables(f,'2026-04-01T00:00Z','2026-04-10T00:00Z',uncertainty=False)
        s=t['summary'];np.testing.assert_allclose(s.total_bias_usdc,s.gross_overprediction_usdc-s.gross_underprediction_usdc)
        self.assertFalse(t['risk_constraints'].support_sufficient.any())
        self.assertFalse(t['risk_constraints'].direction_risk_pass.any())


if __name__=='__main__':unittest.main()
