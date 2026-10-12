"""Rebuild anchor forecasts and amplitude formulas; reconcile retained aggregates.

The tested market-state module is reused; anchor and amplitude estimators and
study aggregation functions are not. No row predictions are saved.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT, DEFAULT_OUTPUT, DEFAULT_DATA
from pp_data.features import BASE_FEATURE
from pp_models.directional_features import read_market_prefix
from pp_models.directional_study import install_development_guard
from pp_models.rolling_protocol import load_rolling_pool, load_protocol
from pp_models.scoped_data import development_snapshot
from pp_models.sparse_shock import load_settings, market_states, order_states
from pp_models.runs import fingerprint
from pp_models.study_storage import atomic_json, file_digest
from scripts.verify_rolling_study import close, manual_metrics, design


def manual_hurdle(train,y,query):
    output=np.empty(len(query))
    for side in [0,1]:
        hs=train.option_type.eq(side);qs=query.option_type.eq(side)
        if not qs.any():continue
        b=train.loc[hs,BASE_FEATURE].to_numpy();target=y.loc[hs].to_numpy()
        qb=query.loc[qs,BASE_FEATURE].to_numpy()
        if not len(b):output[qs]=qb;continue
        positive=target>0;p=np.full(qs.sum(),positive.mean())
        if 0<positive.sum()<len(b):
            ht,qt=design(train.loc[hs]),design(query.loc[qs]);center=ht.mean(0);scale=ht.std(0)
            scale[scale==0]=1
            model=LogisticRegression(C=.1,max_iter=1000,tol=1e-9,solver='lbfgs')
            model.fit(np.clip((ht-center)/scale,-3,3),positive.astype(int))
            p=model.predict_proba(np.clip((qt-center)/scale,-3,3))[:,1]
        severity=(b[positive]@target[positive])/(b[positive]@b[positive]) if positive.any() else 0.
        output[qs]=qb*p*severity
    return output


def verify(root):
    guard=install_development_guard(DEFAULT_OUTPUT)
    manifest=json.loads((root/'provenance/manifest.json').read_text());assert manifest['status']=='complete'
    for name,digest in manifest['files'].items():assert file_digest(root/name)==digest,name
    source=json.loads((root/'provenance/source.json').read_text())
    for name,digest in source['source']['code_sha256'].items():assert file_digest(PROJECT/name)==digest,name
    for name,digest in source['config_sha256'].items():assert file_digest(PROJECT/name)==digest,name
    identity,_=development_snapshot(DEFAULT_OUTPUT)
    assert fingerprint(identity)==source['source']['dataset_sha256']
    protocol,c=load_protocol(),load_settings();pool=load_rolling_pool()
    market,info=read_market_prefix(DEFAULT_DATA/'btc_market_history.csv',protocol['validation_end'])
    assert info==source['market_prefix']
    states=market_states(market,c);features=order_states(pool,states,c)
    read=lambda n:pd.read_csv(root/f'metrics/{n}.csv',float_precision='round_trip',dtype={'value':str})
    daily,summary,parameters,cohorts,replays=(read(n) for n in ['daily','summary','daily_parameters','daily_cohorts','anchor_replay_cohorts'])
    ledger={n:pd.Series(np.nan,index=pool.X.index) for n in ['official_raw','hurdle_base']}
    created,closed=pool.metadata.created_at_utc,pool.metadata.closed_at_utc
    scored=[];checked_metrics=checked_amplitudes=checked_replays=0
    for day in pd.date_range('2026-03-02','2026-06-30',tz='UTC'):
        q=pool.X.index[created.ge(day)&created.lt(day+pd.Timedelta(days=1))]
        if not len(q):continue
        h=pool.X.index[created.ge(day-pd.Timedelta(days=30))&created.lt(day)&closed.lt(day)]
        row=replays[replays.day.eq(day.isoformat())].iloc[0]
        assert len(h)==row.training_orders and len(q)==row.predicted_orders
        assert fingerprint(list(map(str,h)))==row.train_ids_sha256
        assert fingerprint(list(map(str,q)))==row.prediction_ids_sha256
        assert pd.Timestamp(row.latest_training_close)==closed.loc[h].max()<day
        ledger['official_raw'].loc[q]=pool.X.loc[q,BASE_FEATURE]
        ledger['hurdle_base'].loc[q]=manual_hurdle(pool.X.loc[h],pool.y.loc[h],pool.X.loc[q])
        checked_replays+=1
        if day<pd.Timestamp('2026-04-01',tz='UTC'):continue
        q=q[closed.loc[q].lt('2026-07-01')]
        cohort=cohorts[cohorts.day.eq(day.isoformat())].iloc[0]
        assert fingerprint(list(map(str,h)))==cohort.train_ids_sha256
        assert fingerprint(list(map(str,q)))==cohort.score_ids_sha256
        scored.extend(map(str,q))
        for anchor,prefix in [('official_raw','official'),('hurdle_base','hurdle')]:
            assert ledger[anchor].loc[h].notna().all()
            amplitudes={}
            for side in [0,1]:
                ids=h[(pool.X.loc[h,'option_type']==side)&(features.loc[h,'effective_state']>0)&(ledger[anchor].loc[h]>0)]
                events=features.loc[ids,'event_id'];n_events=events.nunique()
                expected=raw=0.;fallback=n_events<3 or len(ids)<10
                if not fallback:
                    numerator=denominator=0.
                    for event in events.unique():
                        members=events.index[events.eq(event)]
                        b=ledger[anchor].loc[members].to_numpy();g=features.loc[members,'effective_state'].to_numpy()
                        y=pool.y.loc[members].to_numpy()
                        numerator+=np.mean(b*g*(y-b));denominator+=np.mean((b*g)**2)
                    raw=max(0.,numerator/denominator);expected=min(2.,raw*n_events/(n_events+5))
                p=parameters[(parameters.day==day.isoformat())&(parameters.anchor==anchor)&(parameters.direction==side)].iloc[0]
                close(p.amplitude,expected);close(p.raw_amplitude,raw)
                assert (p.events,p.orders,p.fallback)==(n_events,len(ids),fallback)
                amplitudes[side]=expected;checked_amplitudes+=1
            b=ledger[anchor].loc[q].to_numpy();g=features.loc[q,'effective_state'].to_numpy()
            side=pool.X.loc[q,'option_type'];a=side.map(amplitudes).to_numpy()
            predictions={anchor:b,prefix+'_shock':b*(1+a*g),prefix+'_shock_fixed':b*(1+.5*g)}
            for method,pred in predictions.items():
                # This also verifies exact identity of every untriggered order.
                np.testing.assert_array_equal(pred[g==0],b[g==0])
                for group,value,mask in [('overall','all',np.ones(len(q),bool))]+[('direction',str(s),side.eq(s).to_numpy()) for s in [0,1]]:
                    if not mask.any():continue
                    observed=daily[(daily.day==day.isoformat())&(daily.model==method)&(daily.group==group)&(daily.value==value)]
                    assert len(observed)==1
                    expected=manual_metrics(pool.y.loc[q].to_numpy()[mask],pred[mask],pool.X.loc[q,'quantity_btc'].to_numpy()[mask])
                    for key,value in expected.items():close(observed[key].iloc[0],value)
                    checked_metrics+=1
    assert len(scored)==len(set(scored))==source['execution']['unique_scoring_orders']
    assert fingerprint(scored)==source['execution']['scoring_ids_sha256']
    assert checked_replays==len(replays)==source['execution']['anchor_replay_days']
    additive=['orders','actual_total_usdc','predicted_total_usdc','total_bias_usdc','gross_overprediction_usdc','gross_underprediction_usdc']
    reconciled=0
    for row in summary[summary.group.isin(['overall','direction'])].itertuples():
        part=daily[(daily.model==row.model)&(daily.group==row.group)&(daily.value==row.value)]
        if row.month!='all':part=part[part.month==row.month]
        for key in additive:close(part[key].sum(),getattr(row,key))
        close(np.sqrt(np.average(part.rmse_per_btc**2,weights=part.orders)),row.rmse_per_btc)
        close(np.average(part.mae_per_btc,weights=part.orders),row.mae_per_btc)
        reconciled+=1
        if row.group=='direction':
            for group in ['direction_tail','direction_settlement','direction_recent_loss']:
                sub=summary[(summary.month==row.month)&(summary.model==row.model)&(summary.group==group)&summary.value.str.startswith(row.value+'|')]
                for key in additive:close(sub[key].sum(),getattr(row,key))
                close(np.sqrt(np.average(sub.rmse_per_btc**2,weights=sub.orders)),row.rmse_per_btc)
    for row in read('macro_monthly').itertuples():
        part=summary[(summary.month!='all')&(summary.model==row.model)&(summary.group=='overall')]
        close(np.sqrt(np.mean(part.rmse_per_btc**2)),row.macro_rmse_per_btc)
        close(part.mae_per_btc.mean(),row.macro_mae_per_btc)
    budget=read('budget_matched_diagnostic')
    for row in budget.itertuples():
        counterpart=summary[(summary.month==row.month)&(summary.model==row.overlay)&(summary.group=='direction')&(summary.value==str(row.direction))].iloc[0]
        close(row.predicted_total_usdc,counterpart.predicted_total_usdc)
    assert guard['blocked_test_opens']==0
    result={'status':'passed','scoring_orders':len(scored),'anchor_replay_days':checked_replays,
        'independent_amplitude_checks':checked_amplitudes,'independent_daily_metric_rows':checked_metrics,
        'daily_to_summary_checks':reconciled,'equal_budget_cash_checks':len(budget),
        'state_validation':'reused tested causal state module; no independent state-machine claim',
        'test_csv_read':False,'july_prices_read':False,'original_manifest_sha256':file_digest(root/'provenance/manifest.json'),
        'verification_code_sha256':{str(p.relative_to(PROJECT)):file_digest(p) for p in [Path(__file__).resolve(),PROJECT/'scripts/verify_rolling_study.py']}}
    atomic_json(root/'provenance/verification.json',result)
    for p in [Path(__file__).resolve(),PROJECT/'scripts/verify_rolling_study.py']:
        (root/'provenance'/p.name).write_bytes(p.read_bytes())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args()
    with threadpool_limits(limits=1):print(json.dumps(verify(args.run_dir.resolve()),indent=2))
