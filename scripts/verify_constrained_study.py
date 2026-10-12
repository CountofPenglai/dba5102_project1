"""Reconcile aggregate results and independently check sampled as-of features/optima."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pp_data.config import PROJECT, DEFAULT_DATA, DEFAULT_OUTPUT
from pp_data.features import BASE_FEATURE
from pp_models.constrained_study import variant_config, load_study_config
from pp_models.constrained_pricing import ConstrainedModel, INPUTS, FLOW
from pp_models.constrained_features import CalibratedMarket, augment_constraints
from pp_models.config import load_model_config
from pp_models.data import daily_batches
from pp_models.directional_features import read_market_prefix
from pp_models.directional_study import install_development_guard
from pp_models.market_activity_features import read_order_prefix
from pp_models.scoped_data import load_validation_pool
from pp_models.study import study_snapshot
from pp_models.study_storage import atomic_json, file_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=PROJECT/'outputs/model_studies/constrained_pricing_20261009_verified')
    run=parser.parse_args().run_dir; guard=install_development_guard(DEFAULT_OUTPUT)
    manifest=json.loads((run/'provenance/manifest.json').read_text())
    assert manifest['status']=='complete'
    for name,h in manifest['files'].items(): assert file_digest(run/name)==h,name
    identity=manifest['identity'];source=identity['source']
    assert study_snapshot(DEFAULT_OUTPUT)==source['development']
    for name,h in source['auxiliary_code'].items():assert file_digest(PROJECT/name)==h,name
    trial_count=0
    for path in run.glob('trials/*/*/record.json'):
        record=json.loads(path.read_text())
        for name,h in record['files'].items():assert file_digest(path.parent/name)==h
        trial_count+=1
    assert trial_count==18
    old=PROJECT/'outputs/model_studies/terminal_distribution_20261009'
    old_plan=json.loads((old/'provenance/plan.json').read_text())
    for phase in ('tuning','followup'):
        for key in ('prediction_ids_sha256','training_ids_by_day_sha256'):
            assert identity['plan']['periods'][phase][key]==old_plan['periods'][phase][key]
    read=lambda p:pd.read_csv(p,dtype={'value':str},float_precision='round_trip')
    baseline_rows,aggregate_rows=0,0
    for phase in ('tuning','followup'):
        s,d=read(run/f'metrics/{phase}.csv'),read(run/f'metrics/{phase}_daily.csv')
        prior=read(old/f'metrics/{phase}.csv')
        for variant,prior_mode in [('raw','raw'),('static_reference','static')]:
            a=s[s.variant.eq(variant)].set_index(['group','value'])
            b=prior[prior.distribution_mode.eq(prior_mode)]
            if prior_mode=='static':b=b[b.distribution_lambda.eq(.01)]
            b=b.set_index(['group','value'])
            common=a.index.intersection(b.index)
            columns=['orders','rmse_per_btc','mae_per_btc','actual_total_usdc','predicted_total_usdc']
            np.testing.assert_allclose(a.loc[common,columns],b.loc[common,columns],rtol=1e-12,atol=1e-9)
            baseline_rows+=len(common)
        for (candidate,group,value),parts in d.groupby(['candidate_id','group','value']):
            total=s[s.candidate_id.eq(candidate)&s.group.eq(group)&s.value.eq(value)].iloc[0]
            n=parts.orders.sum();assert n==total.orders
            np.testing.assert_allclose(np.sqrt(np.average(parts.rmse_per_btc**2,weights=parts.orders)),total.rmse_per_btc,rtol=1e-12)
            np.testing.assert_allclose(np.average(parts.mae_per_btc,weights=parts.orders),total.mae_per_btc,rtol=1e-12)
            for c in ('actual_total_usdc','predicted_total_usdc','gross_overprediction_usdc','gross_underprediction_usdc'):
                np.testing.assert_allclose(parts[c].sum(),total[c],rtol=1e-12,atol=1e-9)
            aggregate_rows+=1
    m,md=read(run/'metrics/market_summary.csv'),read(run/'metrics/market_daily.csv')
    previous=read(old/'metrics/market_summary.csv')
    keys=['phase','method','horizon_minutes','volatility_group']
    columns=['origins','covered_90','lower_miss_90','upper_miss_90','width_90','interval_score_90','crps']
    np.testing.assert_allclose(m[m.method.ne('calibrated')].set_index(keys)[columns].sort_index(),previous.set_index(keys)[columns].sort_index(),rtol=1e-12,atol=1e-10)
    for level in (80,90,95):
        np.testing.assert_allclose(md[f'covered_{level}']+md[f'lower_miss_{level}']+md[f'upper_miss_{level}'],1.,atol=1e-14)
    for key,parts in md.groupby(keys):
        total=m.set_index(keys).loc[key];assert total.origins==parts.origins.sum()
        for c in columns[1:]:np.testing.assert_allclose(np.average(parts[c],weights=parts.origins),total[c],rtol=1e-12)
    assert md.excluded_origins.sum()==0
    pool=load_validation_pool(variant_config(load_model_config(),'raw'))
    usable,mi=read_market_prefix(DEFAULT_DATA/'btc_market_history.csv','2026-07-01T00:00:00Z')
    events,ei=read_order_prefix(DEFAULT_DATA/'pp_transactions_training.csv','2026-07-01T00:00:00Z')
    assert mi==source['market_prefix'] and ei==source['order_prefix']
    market=CalibratedMarket(usable);pool,_=augment_constraints(pool,market,usable,events)
    ids=pool.X.index[pool.X.constraint_supported.eq(1)]
    sampled=ids[np.linspace(0,len(ids)-1,20,dtype=int)]
    for idx in sampled:
        stamp=pool.metadata.at[idx,'created_at_utc'];snap=stamp.floor('5min');X=pool.X.loc[idx]
        available=usable.loc[usable.close_time_utc.le(snap)]
        np.testing.assert_allclose(X.flow_market,np.log1p(available.tail(60).number_of_trades.mean()),atol=1e-14)
        for side,label in [(0,'call'),(1,'put')]:
            n=((events.created_at_utc.ge(snap-pd.Timedelta(minutes=60)))&events.created_at_utc.lt(snap)&events.option_type.eq(side)).sum()
            if snap-pd.Timedelta(minutes=60)>=events.created_at_utc.min():
                np.testing.assert_allclose(X[f'flow_{label}'],np.log1p(n),atol=1e-14)
        duration=X.remaining_seconds/60;h=next(h for h in (30,60,120,240,480) if h>=duration)
        day=stamp.floor('D');hist=market.history(day,h)
        assert (hist.index+pd.Timedelta(minutes=h)<day).all()
        actual=np.log(market.price.reindex(hist.index+pd.Timedelta(minutes=h)).to_numpy()/market.price.reindex(hist.index).to_numpy())
        np.testing.assert_allclose(actual,hist['return'],atol=2e-15)
        z=actual/(hist.sigma.to_numpy()*np.sqrt(h));centered=z-z.mean()
        recent=z[hist.index>=day-pd.Timedelta(days=7)]
        lo,hi=np.quantile(centered,[.05,.95]);rl,rh=np.quantile(recent,[.05,.95])
        factors=np.exp(len(recent)/(len(recent)+200)*np.log(np.clip([max(-rl,0)/max(-lo,1e-8),max(rh,0)/max(hi,1e-8)],.5,2))) if len(recent)>=100 else np.ones(2)
        np.testing.assert_allclose(market.parameters(day,h)[:2],factors,atol=1e-12)
        returns=centered*market.sigma_at(stamp)*np.sqrt(duration)
        returns*=np.where(returns<0,factors[0],factors[1])
        gap=X.market_price_usdc_per_btc*np.exp(returns)-X.strike_price_usdc_per_btc
        payout=np.maximum(gap if X.option_type==0 else -gap,0).mean()
        np.testing.assert_allclose(X.contract_calibrated,np.log1p(payout)-np.log1p(X[BASE_FEATURE]),atol=1e-11)
    selected=json.loads((run/'provenance/selected_models.json').read_text());objective_gaps=[];optima=0
    for day,tr,_ in daily_batches(pool,'validation',30,load_study_config()['periods']['followup']):
        if day.day not in (21,24,30):continue
        X,y=pool.X.loc[tr,INPUTS],pool.y.loc[tr]
        for mode,cfg in selected.items():
            model=ConstrainedModel(mode,cfg['constraint_lambda']).fit(X,y)
            for side,fit in model.fits.items():
                sub=X.loc[X.option_type.eq(side)];target=y.loc[sub.index].to_numpy()
                values=sub[model.features].to_numpy(float)
                Z=np.clip((np.where(np.isfinite(values),values,fit['center'])-fit['center'])/fit['spread'],-3,3)
                Z[sub.constraint_supported.eq(0)]=0
                base=sub[BASE_FEATURE].to_numpy();pen=cfg['constraint_lambda']*np.array([10 if f in FLOW else 1 for f in model.features])
                def objective(beta):
                    w=base*np.exp(Z@beta);scale=np.dot(w,target)/np.dot(w,w)
                    return np.sum((scale*w-target)**2)/np.sum(target**2)+np.sum(pen*beta**2)
                result=minimize(objective,np.zeros(len(model.features)),method='SLSQP',
                    bounds=[(0,1)]+[(-1,1)]*(len(model.features)-1),options={'maxiter':500,'ftol':1e-12})
                assert result.success,result.message
                gap=abs(objective(fit['beta'])-result.fun);assert gap<1e-8,gap
                objective_gaps.append(gap);optima+=1
    result={'passed':True,'checked_trial_checkpoints':trial_count,'reproduced_baseline_summary_rows':baseline_rows,
        'reconciled_payout_summary_rows':aggregate_rows,'reconciled_market_summary_rows':len(m),
        'independent_feature_samples':len(sampled),'independent_slsqp_optima':optima,
        'maximum_solver_objective_difference':max(objective_gaps),'test_csv_read':False,
        'blocked_test_opens':guard['blocked_test_opens'],'no_july_price_parsed':True}
    output=run/'provenance/verification.json';atomic_json(output,result)
    script=run/'provenance/verification_source.py';script.write_bytes(Path(__file__).read_bytes())
    manifest['files'].update({str(p.relative_to(run)):file_digest(p) for p in (output,script)})
    atomic_json(run/'provenance/manifest.json',manifest)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
