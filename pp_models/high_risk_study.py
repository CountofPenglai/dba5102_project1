"""Fixed two-label prequential GBDT study: python -m pp_models.high_risk_study plan|run."""
import argparse
import json
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from pp_data.config import PROJECT, DEFAULT_DATA, DEFAULT_OUTPUT
from .calibration_models import CalibrationModel
from .directional_features import read_market_prefix
from .directional_study import install_development_guard
from .evaluation import key_metrics
from .high_risk_classifier import PeakClassifier, load_settings, risk_multiplier
from .high_risk_data import prepare_day
from .high_risk_features import build_high_risk_features, FEATURE_NAMES
from .high_risk_labels import load_settings as label_settings
from .market_activity_features import read_order_prefix
from .rolling_protocol import load_protocol, load_rolling_pool, rolling_batches
from .rolling_study import payout_aggregates, tail_context, settlement_buckets, recent_loss_state
from .runs import fingerprint, run_destination
from .scoped_data import development_snapshot
from .sparse_shock_study import pricing_inputs, check_anchors
from .study_storage import atomic_json, file_digest

ANCHORS=['official_raw','hurdle_base']


def binary_metrics(y,p,enabled,cash,threshold):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);cash=np.asarray(cash,dtype=float)
    enabled=np.asarray(enabled,dtype=bool)
    if (y.ndim!=1 or p.shape!=y.shape or cash.shape!=y.shape or enabled.shape!=y.shape
            or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any()
            or not np.isfinite(cash).all() or (cash<0).any()):raise ValueError('Invalid classification metrics inputs')
    if not len(y):
        return {**{name:np.nan for name in ['prevalence','mean_probability','brier','log_loss','average_precision',
            'roc_auc','alarm_fraction','precision','precision_lift','recall','false_positive_rate','tail_cash_recall']},
            **{name:0 for name in ['positive_orders','alarms','tp','fp','fn','tn','positive_payout_usdc','caught_positive_payout_usdc']},
            'alarm_probability':threshold}
    positive=y==1;alarm=enabled&(p>=threshold)
    tp=int((alarm&positive).sum());fp=int((alarm&~positive).sum())
    fn=int((~alarm&positive).sum());tn=int((~alarm&~positive).sum())
    precision=tp/(tp+fp) if tp+fp else np.nan;prevalence=float(y.mean())
    tail_cash=float(cash[positive].sum())
    return {'positive_orders':int(positive.sum()),'prevalence':prevalence,'mean_probability':float(p.mean()),
        'brier':float(np.mean((p-y)**2)),'log_loss':log_loss(y,p,labels=[0,1]),
        'average_precision':average_precision_score(y,p) if positive.any() else np.nan,
        'roc_auc':roc_auc_score(y,p) if 0<positive.sum()<len(y) else np.nan,
        'alarm_probability':threshold,'alarms':int(alarm.sum()),'alarm_fraction':float(alarm.mean()),
        'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,
        'precision_lift':precision/prevalence if prevalence else np.nan,
        'recall':tp/(tp+fn) if tp+fn else np.nan,'false_positive_rate':fp/(fp+tn) if fp+tn else np.nan,
        'positive_payout_usdc':tail_cash,'caught_positive_payout_usdc':float(cash[alarm&positive].sum()),
        'tail_cash_recall':float(cash[alarm&positive].sum()/tail_cash) if tail_cash else np.nan}


def forecast_days(pool,features,benchmark,protocol,c,progress=None):
    """Produce transient predictions; fits receive only settled historical labels."""
    if list(features)!=FEATURE_NAMES or not features.index.equals(pool.X.index):
        raise ValueError('Feature schema or ordering mismatch')
    ids=benchmark.X.index
    if (not ids.isin(pool.X.index).all() or not pool.metadata.loc[ids,'benchmark_eligible'].all()
            or not np.array_equal(benchmark.y.to_numpy(),pool.y.loc[ids].to_numpy())
            or not benchmark.metadata.created_at_utc.equals(pool.metadata.loc[ids,'created_at_utc'])
            or not benchmark.metadata.closed_at_utc.equals(pool.metadata.loc[ids,'closed_at_utc'])):
        raise ValueError('Risk and benchmark cohorts are inconsistent')
    benchmark_batches={d:(h,q) for d,h,q in rolling_batches(benchmark,protocol)}
    Xb=pricing_inputs(benchmark);classification=[];payouts=[];parameters=[];cohorts=[]
    for day,h,q in rolling_batches(pool,protocol):
        batch=prepare_day(pool,features,day,label_settings())
        if not h.equals(batch['train_X'].index) or not q.equals(batch['score_X'].index):
            raise ValueError('Daily classifier roles differ from canonical rolling protocol')
        common=pd.DataFrame({'day':day.isoformat(),'month':day.strftime('%Y-%m'),'actual':pool.y.loc[q],
            'quantity':pool.X.loc[q,'quantity_btc'],'side':pool.X.loc[q,'option_type'].astype(int),
            'benchmark_eligible':pool.metadata.loc[q,'benchmark_eligible'],
            'peak_q975':batch['score_labels'].high_risk_unit_q975.astype(float)},index=q)
        forecasts={}
        for target in c['targets']:
            frame=common.copy();frame['target']=target
            frame['probability']=np.nan;frame['prior']=np.nan;frame['trained']=False
            for side in [0,1]:
                train=batch['train_labels'][target]
                ht=train.index[train.notna() & pool.X.loc[h,'option_type'].eq(side)]
                qt=q[pool.X.loc[q,'option_type'].eq(side)]
                started=perf_counter()
                model=PeakClassifier(c).fit(features.loc[ht],train.loc[ht],pool.metadata.loc[ht,'created_at_utc'])
                frame.loc[qt,'probability']=model.predict(features.loc[qt])
                frame.loc[qt,'prior']=model.prior;frame.loc[qt,'trained']=model.trained
                parameters.append({'day':day.isoformat(),'month':day.strftime('%Y-%m'),'target':target,'direction':side,
                    'scoring_orders':len(qt),'fit_predict_seconds':perf_counter()-started,**model.detail})
            frame['multiplier']=risk_multiplier(frame.probability,frame.trained,c)
            frame['alarm']=frame.multiplier.gt(1)
            # Current outcomes enter evaluation only after probabilities and gates exist.
            frame['label']=batch['score_labels'][target].astype(float)
            classification.append(frame);forecasts[target]=frame
        cohort={'day':day.isoformat(),'risk_training_orders':len(h),'risk_scoring_orders':len(q),
            'risk_train_ids_sha256':fingerprint(list(map(str,h))),'risk_score_ids_sha256':fingerprint(list(map(str,q))),
            'latest_training_close':pool.metadata.loc[h,'closed_at_utc'].max().isoformat(),
            'benchmark_training_orders':0,'benchmark_scoring_orders':0}
        if day in benchmark_batches:
            hb,qb=benchmark_batches[day]
            if len(hb)<protocol['minimum_history_orders'] or not qb.isin(q).all():
                raise ValueError('Invalid common-cohort anchor history or score IDs')
            cohort.update(benchmark_training_orders=len(hb),benchmark_scoring_orders=len(qb),
                benchmark_train_ids_sha256=fingerprint(list(map(str,hb))),benchmark_score_ids_sha256=fingerprint(list(map(str,qb))))
            threshold,_=tail_context(benchmark,hb,qb)
            part=common.loc[qb].copy()
            part['tenor']=benchmark.X.loc[qb,'product_duration_seconds']
            part['settlement']=settlement_buckets(benchmark.metadata.loc[qb])
            part['recent_loss']=recent_loss_state(benchmark,hb,qb,day).map({True:'recent_high_loss',False:'other'})
            part['tail_group']=np.where(threshold.isna(),'unavailable',np.where(benchmark.y.loc[qb].gt(threshold),'tail','body'))
            for name in ANCHORS:
                anchor=CalibrationModel(name).fit(Xb.loc[hb],benchmark.y.loc[hb]).predict_raw(Xb.loc[qb])
                payouts.append(part.assign(model=name,anchor=anchor,prediction=anchor,multiplier=1.,alarm=False))
                for target,forecast in forecasts.items():
                    multiplier=forecast.loc[qb,'multiplier'].to_numpy()
                    payouts.append(part.assign(model=name+'_'+target.split('_')[-1],anchor=anchor,
                        prediction=anchor*multiplier,multiplier=multiplier,alarm=multiplier>1))
        cohorts.append(cohort)
        if progress and (day.day==1 or day.day%10==0):
            progress(f'{day.date()}: {len(h)} settled history; {len(q)} risk orders; both labels scored')
    return pd.concat(classification),pd.concat(payouts),pd.DataFrame(parameters),pd.DataFrame(cohorts)


def classification_slices(frame):
    for period in ['all']+sorted(frame.month.unique()):
        selected=frame if period=='all' else frame[frame.month.eq(period)]
        for target,f in selected.groupby('target'):
            for cohort in ['risk','benchmark_common']:
                sub=f if cohort=='risk' else f[f.benchmark_eligible]
                for direction in ['all','Call','Put']:
                    part=sub if direction=='all' else sub[sub.side.eq(0 if direction=='Call' else 1)]
                    if not len(part):continue
                    yield {'period':period,'target':target,'cohort':cohort,'direction':direction},part


def classification_tables(frame,c):
    rows=[];curve=[];bins=[];daily=[]
    for key,f in classification_slices(frame):
        available=f.label.notna();part=f[available];cash=part.actual*part.quantity
        for method,column in [('gbdt','probability'),('historical_probability','prior')]:
            enabled=part.trained if method=='gbdt' else np.ones(len(part),dtype=bool)
            rows.append(key|{'method':method,'orders':len(f),'available_labels':len(part),
                'fallback_orders':int((~f.trained).sum()) if method=='gbdt' else 0,
                **binary_metrics(part.label,part[column],enabled,cash,c['alarm_probability'])})
        for probability in c['diagnostic_probabilities']:
            curve.append(key|{'orders':len(part),**binary_metrics(part.label,part.probability,part.trained,cash,probability)})
        cut=pd.cut(part.probability,[-1e-12,.025,.05,.1,.2,.4,1.0000001],right=False)
        for interval,g in part.groupby(cut,observed=True):
            bins.append(key|{'probability_bin':str(interval),'orders':len(g),'positive_orders':int(g.label.sum()),
                'mean_probability':g.probability.mean(),'actual_rate':g.label.mean()})
    for (day,target,side),f in frame.groupby(['day','target','side']):
        part=f[f.label.notna()]
        if len(part):daily.append({'day':day,'month':f.month.iloc[0],'target':target,'direction':side,
            'orders':len(f),'available_labels':len(part),**binary_metrics(part.label,part.probability,part.trained,
                part.actual*part.quantity,c['alarm_probability'])})
    return {'classification':pd.DataFrame(rows),'threshold_diagnostic':pd.DataFrame(curve),
            'probability_bins':pd.DataFrame(bins),'classification_daily':pd.DataFrame(daily)}


def overlay_tables(frame):
    rows=[];slices=[]
    for period in ['all']+sorted(frame.month.unique()):
        selected=frame if period=='all' else frame[frame.month.eq(period)]
        for model,f in selected.groupby('model'):
            for direction in ['all','Call','Put']:
                part=f if direction=='all' else f[f.side.eq(0 if direction=='Call' else 1)]
                if not len(part):continue
                key={'period':period,'model':model,'direction':direction}
                alarm=part.alarm;peak=part.peak_q975.eq(1);cash=part.actual*part.quantity
                base_error=part.anchor-part.actual;error=part.prediction-part.actual
                rows.append(key|{'orders':len(part),'adjusted_orders':int(alarm.sum()),'adjusted_fraction':float(alarm.mean()),
                    'maximum_multiplier':part.multiplier.max(),'added_predicted_cash':float(np.dot(part.prediction-part.anchor,part.quantity)),
                    'reduced_underprediction_cash':float(np.dot(np.maximum(-base_error,0)-np.maximum(-error,0),part.quantity)),
                    'added_overprediction_cash':float(np.dot(np.maximum(error,0)-np.maximum(base_error,0),part.quantity)),
                    'peak_q975_orders':int(peak.sum()),'caught_peak_q975_orders':int((peak&alarm).sum()),
                    'peak_q975_cash':float(cash[peak].sum()),'caught_peak_q975_cash':float(cash[peak&alarm].sum())})
                for group,mask in [('peak_q975',peak),('body_q975',part.peak_q975.eq(0)),('label_unavailable',part.peak_q975.isna())]:
                    sub=part[mask]
                    if len(sub):slices.append(key|{'group':group,**key_metrics(sub.actual,sub.prediction,sub.quantity)})
    return {'overlay':pd.DataFrame(rows),'fixed_peak_slices':pd.DataFrame(slices)}


def evaluate(pool,features,benchmark,protocol,c,progress=None):
    classification,payouts,parameters,cohorts=forecast_days(pool,features,benchmark,protocol,c,progress)
    tables=payout_aggregates(payouts)|classification_tables(classification,c)|overlay_tables(payouts)
    tables['training_daily']=parameters;tables['daily_cohorts']=cohorts
    tables['training_support']=parameters.groupby(['target','direction']).agg(
        windows=('day','size'),trained_windows=('trained','sum'),
        minimum_positives=('positive_orders','min'),median_positives=('positive_orders','median'),
        maximum_positives=('positive_orders','max'),fallback_scoring_orders=('scoring_orders',lambda s:int(s[~parameters.loc[s.index,'trained']].sum()))).reset_index()
    info={'risk_scoring_orders':int(cohorts.risk_scoring_orders.sum()),'risk_scoring_days':len(cohorts),
        'benchmark_scoring_orders':int(cohorts.benchmark_scoring_orders.sum()),
        'benchmark_scoring_days':int(cohorts.benchmark_scoring_orders.gt(0).sum()),
        'classifier_fits':int(parameters.trained.sum()),'fallbacks':int((~parameters.trained).sum()),
        'parameters_tuned':False,'test_used':False,'persisted_row_predictions':False,'models':sorted(payouts.model.unique())}
    return tables,info


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','run'])
    parser.add_argument('--run-id');args=parser.parse_args(argv);c=load_settings();protocol=load_protocol()
    if args.action=='plan':
        print(json.dumps({'settings':c,'protocol':protocol,'features':FEATURE_NAMES,'training_performed':False},indent=2));return 0
    if not args.run_id:parser.error('run requires --run-id')
    destination=run_destination(PROJECT/'outputs/model_studies',args.run_id)
    started=perf_counter();guard=install_development_guard(DEFAULT_OUTPUT)
    identity,_=development_snapshot();pool=load_rolling_pool('risk');benchmark=load_rolling_pool('benchmark')
    paths=[DEFAULT_DATA/'btc_market_history.csv',DEFAULT_DATA/'pp_transactions_training.csv']
    stats={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
    names=sorted({str(p.relative_to(PROJECT)) for directory in ['pp_models','pp_data'] for p in (PROJECT/directory).glob('*.py')}
        |{'scripts/payout_signal_analysis.py','config/high_risk_gbdt_v1.json','config/high_risk_features_v1.json',
          'config/validation_protocol_v2.json','docs/plans/2026-10-09-high-risk-gbdt.md',
          'tests/test_high_risk_classifier.py','tests/test_high_risk_study.py'})
    hashes={name:file_digest(PROJECT/name) for name in names}
    market,market_source=read_market_prefix(paths[0],protocol['validation_end'])
    events,event_source=read_order_prefix(paths[1],protocol['validation_end'])
    features,dictionary=build_high_risk_features(pool,market,events)
    tables,info=evaluate(pool,features,benchmark,protocol,c,progress=lambda s:print(s,flush=True))
    tables['feature_dictionary']=dictionary;reference=check_anchors(tables)
    if (identity!=development_snapshot()[0] or any((p.stat().st_size,p.stat().st_mtime_ns)!=s for p,s in stats.items())
            or any(file_digest(PROJECT/name)!=sha for name,sha in hashes.items())):raise ValueError('Source changed during training')
    source={'settings':c,'protocol':protocol,'development':identity,'code_sha256':hashes,
        'market_prefix':market_source,'order_prefix':event_source,'execution':info,'anchor_reproduction':reference,
        'elapsed_seconds':perf_counter()-started,
        'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        'test_csv_read':False,'july_prices_parsed':False,'blocked_test_opens':guard['blocked_test_opens'],
        'opened_development_files':sorted(guard['opened_development_files']),
        'limitations':['Exploratory Apr--Jun periods previously inspected; not independent holdout evidence',
            'Orders and rolling windows overlap; they are not independent shock events',
            'Fixed 1.5 multiplier is a targeting probe, not fitted optimal pricing or revenue',
            'Different labels have different prevalence; AP values alone do not select a winner',
            'Classifier uses risk cohort, anchors use benchmark cohort; missing Base coverage is explicit']}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.high_risk_gbdt_',dir=destination.parent) as temp:
        root=Path(temp)
        for name in ['metrics','provenance','reports']:(root/name).mkdir()
        for name,table in tables.items():table.to_csv(root/f'metrics/{name}.csv',index=False,float_format='%.17g')
        atomic_json(root/'provenance/source.json',source);atomic_json(root/'metrics/execution.json',info)
        with ZipFile(root/'provenance/source_snapshot.zip','w',ZIP_DEFLATED) as archive:
            for name in names:archive.write(PROJECT/name,name)
        atomic_json(root/'provenance/manifest.json',{'status':'complete','files':{
            str(p.relative_to(root)):file_digest(p) for p in root.rglob('*') if p.is_file()}})
        root.rename(destination)
    print(json.dumps({'output':str(destination),**info,'elapsed_seconds':source['elapsed_seconds']},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
