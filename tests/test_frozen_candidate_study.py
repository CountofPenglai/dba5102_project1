import unittest
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from pp_data.io import ModelSplit
from research_fixtures import fixture
from pp_models import frozen_candidate_study as study


def sample():
    p = fixture()
    p.X['terminal_supported'] = p.X.option_type
    p.X['terminal_static_proxy'] = 100. * p.X.option_type
    p.X['terminal_dynamic_proxy'] = 0.
    return p


class FrozenCandidateTests(unittest.TestCase):
    def test_canonical_pressure_identity_detects_replaced_manifest_or_test_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);artifacts={}
            for kind in ('features','targets','metadata'):
                name=f'datasets/test/{kind}.csv';p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text('original');artifacts[name]={'sha256':study.file_digest(p)}
            (root/'manifest.json').write_text(json.dumps({'artifacts':artifacts}))
            identity=study.canonical_identity(root)
            full=study.canonical_identity(root,True,identity['manifest_sha256'])
            self.assertEqual(len(full['test_artifacts']),3)
            p=root/'datasets/test/targets.csv';p.write_text('replacement')
            with self.assertRaises(ValueError):study.canonical_identity(root,True,identity['manifest_sha256'])
            artifacts['datasets/test/targets.csv']['sha256']=study.file_digest(p)
            (root/'manifest.json').write_text(json.dumps({'artifacts':artifacts}))
            with self.assertRaises(ValueError):study.canonical_identity(root,True,identity['manifest_sha256'])

    def test_manual_proxy_matches_original_after_market_gap_warmup(self):
        from pp_models.terminal_distribution import TerminalMarket,expected_put
        path=study.PROJECT/'scripts/verify_frozen_candidates.py'
        spec=importlib.util.spec_from_file_location('verify_frozen_candidates',path)
        verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
        times=pd.date_range('2026-02-01','2026-03-03',freq='min',inclusive='left',tz='UTC')
        values=100*np.exp(np.cumsum(np.random.default_rng(21).normal(0,.001,len(times))))
        market=pd.DataFrame(dict(open_time_utc=times,close_time_utc=times+pd.Timedelta(minutes=1)-pd.Timedelta(milliseconds=1),close=values))
        market=market.loc[~market.open_time_utc.between('2026-02-17T12:00Z','2026-02-17T12:05Z')]
        stamp=pd.Timestamp('2026-03-01T10:00Z');terminal=TerminalMarket(market)
        proxy=expected_put(terminal.samples(stamp.floor('D'),60,stamp,'static'),100.,100.)
        X=pd.DataFrame(dict(terminal_supported=[1],remaining_seconds=[3600.],market_price_usdc_per_btc=[100.],
                            strike_price_usdc_per_btc=[100.],terminal_static_proxy=[proxy]))
        result=verifier.manual_proxy(X,pd.Series([stamp]),market)
        self.assertLess(result['max_proxy_absolute_error'],1e-8)

    def test_parameter_replay_matches_all_three_original_recipes(self):
        path=study.PROJECT/'scripts/verify_frozen_candidates.py'
        spec=importlib.util.spec_from_file_location('verify_frozen_candidates',path)
        verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
        p=sample();n=len(p.X)
        p.X['remaining_seconds']=np.resize([1800.,3600.,14400.],n)
        p.X['log_moneyness']=np.sin(np.arange(n))*.01
        f,states,_=study.predict_period(p,'2026-04-01T00:00Z','2026-04-04T00:00Z')
        ids=f.query("model == 'h0'").index
        replay=verifier.replay(p.X.loc[ids],next(iter(states.values())))
        for name in study.MODELS:
            np.testing.assert_allclose(replay[name],f.query('model == @name').prediction,rtol=1e-12,atol=1e-10)

    def test_next_month_can_use_prior_settled_month_but_not_same_month(self):
        p=sample();offset=p.metadata.copy()
        for col in offset:offset[col]+=pd.Timedelta(days=30)
        p=ModelSplit(pd.concat([p.X,p.X],ignore_index=True),pd.concat([p.y,p.y],ignore_index=True),
                     pd.concat([p.metadata,offset],ignore_index=True))
        a,_,_=study.predict_period(p,'2026-04-01T00:00Z','2026-05-04T00:00Z')
        change=p.metadata.created_at_utc.ge(pd.Timestamp('2026-04-01T00:00Z'))&p.metadata.created_at_utc.lt(pd.Timestamp('2026-05-01T00:00Z'))
        y=p.y.copy();y.loc[change]*=2
        b,_,_=study.predict_period(ModelSplit(p.X,y,p.metadata),'2026-04-01T00:00Z','2026-05-04T00:00Z')
        pd.testing.assert_frame_equal(a.query("month == '2026-04'"),b.query("month == '2026-04'"))
        self.assertFalse(a.query("month == '2026-05' and model == 'h0'").prediction.equals(
                         b.query("month == '2026-05' and model == 'h0'").prediction))

    def test_monthly_no_feedback_and_unknown_query_closing_time(self):
        p = sample(); start = pd.Timestamp('2026-04-01T00:00Z')
        pending = p.metadata.index[p.metadata.created_at_utc.eq(pd.Timestamp('2026-03-25T12:00Z'))][0]
        p.metadata.loc[pending, 'closed_at_utc'] = start
        a, states, cohorts = study.predict_period(p, start, '2026-04-04T00:00Z')
        q = p.metadata.created_at_utc.ge(start)
        y = p.y.copy(); y.loc[q] = np.nan; y.loc[pending] = 1e10
        meta = p.metadata.copy(); meta.loc[q, 'closed_at_utc'] = pd.NaT
        b, other, _ = study.predict_period(ModelSplit(p.X,y,meta),start,'2026-04-04T00:00Z')
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(states,other)
        self.assertEqual(len(states),1)
        self.assertNotIn('actual',a)
        self.assertEqual(len(a),q.sum()*5)
        self.assertLess(pd.Timestamp(cohorts.iloc[0].latest_training_close),start)

    def test_original_probability_and_unmodified_static_call(self):
        from pp_models.calibration_models import CalibrationModel
        from pp_models.two_stage_study import probability_frame
        p=sample(); start=pd.Timestamp('2026-04-01T00:00Z')
        f,_,_=study.predict_period(p,start,'2026-04-04T00:00Z')
        train=study.history(p,start,30); q=f.query("model == 'h0'").index
        h=CalibrationModel('hurdle_base').fit(probability_frame(p.X.loc[train]),p.y.loc[train])
        prob,sev=h.predict_components(probability_frame(p.X.loc[q]))
        for name in ('h0','contract30','g1a'):
            np.testing.assert_allclose(f.query('model == @name').probability,prob)
        np.testing.assert_allclose(f.query("model == 'h0'").prediction,prob*sev)
        np.testing.assert_array_equal(f.query("model == 'static' and side == 0").multiplier,1.)

    def test_scoring_only_removes_unavailable_labels_after_prediction(self):
        p=sample(); end=pd.Timestamp('2026-04-04T00:00Z')
        idx=p.metadata.index[-1];p.metadata.loc[idx,'closed_at_utc']=end
        f,_,_=study.predict_period(p,'2026-04-01T00:00Z',end)
        self.assertEqual((f.index==idx).sum(),5)
        scored,coverage=study.score(p,f,end)
        self.assertNotIn(idx,scored.index)
        self.assertEqual(coverage['labels_unavailable'],1)

    def test_access_guard_and_config(self):
        g=study.AccessGuard()
        with self.assertRaises(PermissionError):g.check('open',('pp_transactions_validation.csv','r',0))
        with self.assertRaises(PermissionError):g.check('open',('outputs/preprocessing/datasets/test/targets.csv','r',0))
        self.assertEqual(study.load_settings()['amount_strength'],30)


if __name__=='__main__':unittest.main()
