"""Three-month rolling calibration study: python -m pp_models.rolling_study plan|run."""
import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
import warnings
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT, DEFAULT_DATA, DEFAULT_OUTPUT
from pp_data.io import ModelSplit
from .calibration_models import CalibrationModel, MODEL_INPUTS, MODEL_NAMES
from .directional_features import read_market_prefix
from .directional_study import install_development_guard
from .evaluation import key_metrics
from .rolling_protocol import load_protocol, load_rolling_pool, rolling_batches, protocol_coverage, PROTOCOL_PATH
from .runs import fingerprint, run_destination
from .study import study_snapshot
from .study_storage import file_digest, atomic_json
from .terminal_distribution import TerminalMarket, HORIZONS

CONFIG_PATH=PROJECT/'config/rolling_calibration_v1.json'


def load_settings():
    c=json.loads(CONFIG_PATH.read_text())
    if (c['version']!='rolling_calibration_v1' or c['protocol']!='config/validation_protocol_v2.json'
        or c['controls']+c['candidates']!=MODEL_NAMES or c['group_smoothing']!=30
        or c['static_proxy_lambda']!=.01 or c['hurdle_logistic_C']!=.1
        or c['hurdle_standardized_clip']!=3 or c['calibration_shrink_lambda']!=1
        or c['tail_quantile']!=.9 or c['tail_minimum_history']!=30 or c['recovery_days']!=3
        or c['recovery_daily_loss_quantile']!=.9 or c['recovery_minimum_history_days']!=10
        or c['external_signals'] or c['commercial_margin']):raise ValueError('Unsupported fixed rolling study settings')
    return c


def settlement_features(pool,market):
    """Only created time, contractual duration and current spot/strike enter scenarios."""
    X=pool.X.copy();supported=X.remaining_seconds.le(480*60)
    X['distribution_supported']=supported.astype(int)
    X['terminal_probability']=np.nan;X['terminal_mean_both']=np.nan
    X['terminal_supported']=(supported&X.option_type.eq(1)).astype(int)
    X['terminal_static_proxy']=0.;X['terminal_dynamic_proxy']=0.
    for idx in X.index[supported]:
        stamp=pool.metadata.at[idx,'created_at_utc'];tau=X.at[idx,'remaining_seconds']/60
        horizon=next(h for h in HORIZONS if h>=tau)
        r=market.samples(stamp.floor('D'),horizon,stamp,'static')*np.sqrt(tau/horizon)
        difference=X.at[idx,'market_price_usdc_per_btc']*np.exp(r)-X.at[idx,'strike_price_usdc_per_btc']
        payoff=np.maximum(difference if X.at[idx,'option_type']==0 else -difference,0.)
        X.at[idx,'terminal_probability']=float((payoff>0).mean())
        X.at[idx,'terminal_mean_both']=float(payoff.mean())
        if X.at[idx,'option_type']==1:X.at[idx,'terminal_static_proxy']=float(payoff.mean())
    return ModelSplit(X,pool.y.copy(),pool.metadata.copy())


def tail_context(pool,history,today):
    h=pool.X.loc[history];q=pool.X.loc[today];threshold=pd.Series(np.nan,index=today)
    source=pd.Series('unavailable',index=today)
    for (side,tenor),f in q.groupby(['option_type','product_duration_seconds']):
        side_ids=h.index[h.option_type.eq(side)]
        cell=h.index[h.option_type.eq(side)&h.product_duration_seconds.eq(tenor)]
        ids=cell if len(cell)>=30 else side_ids
        if len(ids)>=30:
            threshold.loc[f.index]=pool.y.loc[ids].quantile(.9)
            source.loc[f.index]='direction_tenor' if len(cell)>=30 else 'direction_fallback'
    return threshold,source


def recent_loss_state(pool,history,today,day):
    """Known high-loss history, not a claim that current market risk has recovered."""
    flags=pd.Series(False,index=today)
    for side in [0,1]:
        ids=history[pool.X.loc[history].option_type.eq(side)]
        f=pd.DataFrame({'y':pool.y.loc[ids],'day':pool.metadata.loc[ids].created_at_utc.dt.floor('D')})
        daily=f.groupby('day').y.mean()
        old=daily[daily.index<day-pd.Timedelta(days=3)]
        recent=daily[daily.index>=day-pd.Timedelta(days=3)]
        high=len(old)>=10 and len(recent)>0 and recent.max()>old.quantile(.9)
        flags.loc[pool.X.loc[today].option_type.eq(side)]=high
    return flags


def settlement_buckets(meta):
    duration=(meta.expiration_at_utc-meta.created_at_utc).dt.total_seconds()
    ratio=(meta.closed_at_utc-meta.created_at_utc).dt.total_seconds()/duration
    return pd.Series(np.select([ratio.lt(.5),ratio.lt(.95),ratio.le(1.05)],
                               ['before_half','early','near_expiry'],default='after_expiry'),index=meta.index)


def groups(frame):
    yield 'overall','all',frame
    for side,f in frame.groupby('side'):
        yield 'direction',str(side),f
        for name,column in [('direction_tail','tail_group'),('direction_settlement','settlement'),
                            ('direction_recent_loss','recent_loss')]:
            for value,part in f.groupby(column):yield name,f'{side}|{value}',part
    for tenor,f in frame.groupby('tenor'):yield 'tenor',str(tenor),f


def macro_metrics(monthly):
    rows=[]
    for name,f in monthly[monthly.group.eq('overall')].groupby('model'):
        rows.append({'model':name,'months':len(f),'orders':int(f.orders.sum()),
            'macro_rmse_per_btc':float(np.sqrt(np.mean(f.rmse_per_btc**2))),
            'macro_mae_per_btc':float(f.mae_per_btc.mean())})
    return pd.DataFrame(rows)


def payout_aggregates(frame):
    rows=[];daily=[]
    for period in ['all']+sorted(frame.month.unique()):
        selected=frame if period=='all' else frame[frame.month.eq(period)]
        for model,f in selected.groupby('model'):
            for group,value,part in groups(f):
                rows.append({'month':period,'model':model,'group':group,'value':value,'days':part.day.nunique(),
                    **key_metrics(part.actual,part.prediction,part.quantity)})
    for (day,model),f in frame.groupby(['day','model']):
        for group,value,part in groups(f):
            if group not in ['overall','direction']:continue
            daily.append({'day':day,'month':f.month.iloc[0],'model':model,'group':group,'value':value,
                          **key_metrics(part.actual,part.prediction,part.quantity)})
    summary=pd.DataFrame(rows);daily=pd.DataFrame(daily)
    macro=macro_metrics(summary[summary.month.ne('all')])
    return {'summary':summary,'daily':daily,'macro_monthly':macro}


def probability_aggregates(frame):
    rows=[]
    for period in ['all']+sorted(frame.month.unique()):
        f=frame if period=='all' else frame[frame.month.eq(period)]
        for scope in ['all_orders','distribution_supported']:
            g=f if scope=='all_orders' else f[f.distribution_supported.eq(1)]
            for side,sub in g.groupby('side'):
                for name in ['historical_probability','hurdle_probability']+(['terminal_probability'] if scope=='distribution_supported' else []):
                    p=sub[name].to_numpy();a=sub.actual.gt(0).to_numpy()
                    rows.append({'month':period,'scope':scope,'direction':side,'method':name,'orders':len(sub),
                        'positive_orders':int(a.sum()),'actual_positive_rate':float(a.mean()),'mean_probability':float(p.mean()),
                        'brier':brier_score_loss(a,p),'log_loss':log_loss(a,p,labels=[False,True])})
    return pd.DataFrame(rows)


def evaluate_rolling(pool,protocol,progress=None):
    payouts=[];probabilities=[];parameters=[];cohorts=[];batches=list(rolling_batches(pool,protocol))
    for day,h,q in batches:
        if len(h)<protocol['minimum_history_orders']:raise ValueError(f'Insufficient history at {day}')
        Xt=pool.X.loc[h,MODEL_INPUTS];Xq=pool.X.loc[q,MODEL_INPUTS];y=pool.y.loc[h]
        threshold,source=tail_context(pool,h,q)
        current=pd.DataFrame({'day':day.isoformat(),'month':day.strftime('%Y-%m'),
            'actual':pool.y.loc[q],'quantity':pool.X.loc[q,'quantity_btc'],'side':Xq.option_type.astype(int),
            'tenor':Xq.product_duration_seconds,'settlement':settlement_buckets(pool.metadata.loc[q]),
            'recent_loss':recent_loss_state(pool,h,q,day).map({True:'recent_high_loss',False:'other'}),
            'tail_group':np.where(threshold.isna(),'unavailable',np.where(pool.y.loc[q].gt(threshold),'tail','body'))},index=q)
        pframe=current.copy()
        pframe['distribution_supported']=pool.X.loc[q,'distribution_supported']
        pframe['terminal_probability']=pool.X.loc[q,'terminal_probability']
        pframe['terminal_mean_both']=pool.X.loc[q,'terminal_mean_both']
        for side in [0,1]:
            target=y[Xt.option_type.eq(side)]
            pframe.loc[Xq.option_type.eq(side),'historical_probability']=(target.gt(0).sum()+.5)/(len(target)+1)
        cohorts.append({'day':day.isoformat(),'month':day.strftime('%Y-%m'),'training_orders':len(h),'scoring_orders':len(q),
            'train_ids_sha256':fingerprint([str(i) for i in h]),'score_ids_sha256':fingerprint([str(i) for i in q]),
            'latest_training_close':pool.metadata.loc[h,'closed_at_utc'].max().isoformat(),
            'tail_threshold_fallbacks':int(source.eq('direction_fallback').sum()),'tail_threshold_unavailable':int(source.eq('unavailable').sum())})
        for name in MODEL_NAMES:
            started=perf_counter()
            with threadpool_limits(limits=1),warnings.catch_warnings():
                warnings.simplefilter('error')
                model=CalibrationModel(name).fit(Xt,y,pool.metadata.loc[h,'created_at_utc'])
                prediction=model.predict_raw(Xq)
            parameters.append({'day':day.isoformat(),'month':day.strftime('%Y-%m'),'model':name,
                'fit_predict_seconds':perf_counter()-started,**model.diagnostics()})
            payouts.append(current.assign(model=name,prediction=prediction))
            if name=='hurdle_base':
                prob,severity=model.predict_components(Xq)
                pframe['hurdle_probability']=prob;pframe['hurdle_severity']=severity
        probabilities.append(pframe)
        if progress and (day.day==1 or day.day%10==0):progress(f'{day.date()}: {len(h)} history, {len(q)} scored, seven fixed methods')
    frame=pd.concat(payouts);pframe=pd.concat(probabilities);tables=payout_aggregates(frame)
    tables['probability']=probability_aggregates(pframe)
    tables['daily_parameters']=pd.DataFrame(parameters);tables['daily_cohorts']=pd.DataFrame(cohorts)
    gaps=[]
    for month in ['all']+sorted(pframe.month.unique()):
        g=pframe if month=='all' else pframe[pframe.month.eq(month)]
        for (side,settlement),part in g[g.distribution_supported.eq(1)].groupby(['side','settlement']):
            positive=part.actual.gt(0)
            row={'month':month,'direction':side,'settlement':settlement,'positive_orders':int(positive.sum()),
                'actual_positive_rate':float(positive.mean()),'terminal_probability_mean':part.terminal_probability.mean(),
                'hurdle_probability_mean':part.hurdle_probability.mean(),
                **key_metrics(part.actual,part.terminal_mean_both,part.quantity)}
            if positive.any():
                row['hurdle_positive_amount_mae']=float(np.mean(np.abs(part.loc[positive,'hurdle_severity']-part.loc[positive,'actual'])))
            gaps.append(row)
    tables['settlement_proxy_gap']=pd.DataFrame(gaps)
    # Exact old June cohort for checking reference reproduction, not for selection.
    meta=pool.metadata.loc[frame.index]
    old=meta.created_at_utc.ge('2026-06-01')&((meta.created_at_utc.lt('2026-06-21')&meta.closed_at_utc.lt('2026-06-21'))|
                                         meta.created_at_utc.ge('2026-06-21'))
    legacy=frame.loc[old.to_numpy()].copy()
    if len(legacy):
        legacy['month']=np.where(pd.to_datetime(legacy.day).dt.day<=20,'early','late')
        tables['legacy_june']=payout_aggregates(legacy)['summary']
    return tables,{'models':MODEL_NAMES,'daily_model_evaluations':len(parameters),'unique_scoring_orders':len(pframe),
        'scoring_ids_sha256':fingerprint([str(i) for i in pframe.index]),'scoring_days':len(batches),
        'no_july':True,'cohort':'benchmark','parameters_tuned':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','run'])
    parser.add_argument('--run-id');args=parser.parse_args(argv)
    if args.action=='run' and not args.run_id:parser.error('run requires --run-id')
    started=perf_counter();guard=install_development_guard(DEFAULT_OUTPUT)
    protocol=load_protocol();settings=load_settings();source=study_snapshot(DEFAULT_OUTPUT)
    hashes={str(p.relative_to(PROJECT)):file_digest(p) for p in [PROTOCOL_PATH,CONFIG_PATH]}
    pool=load_rolling_pool();risk=load_rolling_pool('risk')
    coverage=protocol_coverage(pool,protocol);risk_coverage=protocol_coverage(risk,protocol)
    plan={'protocol':protocol,'models':settings,'common_coverage':coverage.to_dict('records'),
          'risk_coverage':risk_coverage.to_dict('records'),'planned_daily_evaluations':int(coverage.scoring_days.sum())*len(MODEL_NAMES),
          'input_storage':'v1 immutable artifacts; evaluation roles use v2; old excluded rows remain excluded',
          'training_performed':False,'test_used':False}
    if args.action=='plan':print(json.dumps(plan,indent=2));return 0
    root=run_destination(PROJECT/'outputs/model_studies',args.run_id);root.mkdir(parents=True,exist_ok=False)
    for name in ['metrics','reports','provenance']:(root/name).mkdir()
    atomic_json(root/'provenance/plan.json',plan);atomic_json(root/'provenance/manifest.json',{'status':'running'})
    market,market_info=read_market_prefix(DEFAULT_DATA/'btc_market_history.csv',protocol['validation_end'])
    feature_pool=settlement_features(pool,TerminalMarket(market))
    tables,info=evaluate_rolling(feature_pool,protocol,progress=lambda s:print(s,flush=True))
    tables['coverage']=coverage;tables['risk_coverage']=risk_coverage
    if study_snapshot(DEFAULT_OUTPUT)!=source or any(file_digest(PROJECT/k)!=v for k,v in hashes.items()):
        raise ValueError('Source changed during rolling study')
    for name,f in tables.items():f.to_csv(root/f'metrics/{name}.csv',index=False,float_format='%.17g')
    provenance={'study':'rolling_calibration_v1','source':source,'config_sha256':hashes,'market_prefix':market_info,
        'execution':info,'elapsed_seconds':perf_counter()-started,
        'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        'test_csv_read':False,'blocked_test_opens':guard['blocked_test_opens'],
        'opened_development_files':sorted(guard['opened_development_files']),
        'limitations':['Apr--May previously supplied historical training data, June repeatedly studied; exploratory retrospective replay',
            'No future labels used in daily fits; fixed candidate designs already informed by earlier June experiments',
            'v1 retained cohort unchanged, including its original boundary exclusions; no raw-data reselection',
            'Different settlement groups are outcome diagnostics, never prediction inputs',
            'Calendar weighting/shrinkage changes statistical tradeoffs and may worsen tail underprediction',
            'No demand, commercial margin, external signals, July or model promotion']}
    atomic_json(root/'provenance/source.json',provenance)
    with ZipFile(root/'provenance/source_snapshot.zip','w',ZIP_DEFLATED) as z:
        files=set(source['code_sha256'])|set(hashes)|{'tests/test_rolling_protocol.py','tests/test_calibration_models.py',
             'tests/test_rolling_study.py','docs/rolling_validation.md'}
        files.update(str(p.relative_to(PROJECT)) for d in ['pp_data','pp_audit'] for p in (PROJECT/d).glob('*.py'))
        for name in sorted(files):z.write(PROJECT/name,name)
    paths=list((root/'metrics').iterdir())+[root/'provenance/plan.json',root/'provenance/source.json',root/'provenance/source_snapshot.zip']
    atomic_json(root/'provenance/manifest.json',{'status':'complete','files':{str(p.relative_to(root)):file_digest(p) for p in paths}})
    print(json.dumps({'output':str(root),**info,'elapsed_seconds':provenance['elapsed_seconds']},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
