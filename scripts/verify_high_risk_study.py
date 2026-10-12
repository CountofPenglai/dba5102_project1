"""Independently rebuild daily thresholds/fits and check saved aggregate outputs.

Reuses audited source-feature construction and the previously verified hurdle
anchor; does not use the new classifier wrapper, label fitter, study replay or
aggregation functions. Row-level data remains in memory.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT,DEFAULT_DATA,DEFAULT_OUTPUT
from pp_models.high_risk_features import build_high_risk_features
from pp_models.directional_features import read_market_prefix
from pp_models.market_activity_features import read_order_prefix
from pp_models.directional_study import install_development_guard
from pp_models.rolling_protocol import load_rolling_pool
from pp_models.calibration_models import CalibrationModel
from pp_models.sparse_shock_study import pricing_inputs
from pp_models.scoped_data import development_snapshot
from pp_models.study_storage import file_digest,atomic_json


def close(a,b):
    np.testing.assert_allclose(float(a),float(b),rtol=1e-9,atol=1e-8,equal_nan=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    root=parser.parse_args(argv).run_dir.resolve();source=json.loads((root/'provenance/source.json').read_text())
    manifest=json.loads((root/'provenance/manifest.json').read_text())
    for name,sha in manifest['files'].items():assert file_digest(root/name)==sha,name
    for name,sha in source['code_sha256'].items():assert file_digest(PROJECT/name)==sha,name
    guard=install_development_guard(DEFAULT_OUTPUT);identity,_=development_snapshot()
    assert identity==source['development']
    pool=load_rolling_pool('risk');benchmark=load_rolling_pool('benchmark')
    c=source['settings'];protocol=source['protocol']
    market,ms=read_market_prefix(DEFAULT_DATA/'btc_market_history.csv',protocol['validation_end'])
    events,es=read_order_prefix(DEFAULT_DATA/'pp_transactions_training.csv',protocol['validation_end'])
    assert ms==source['market_prefix'] and es==source['order_prefix']
    features,_=build_high_risk_features(pool,market,events)
    meta=pool.metadata;bm=benchmark.metadata;xb=pricing_inputs(benchmark)
    records=[];prices=[];supports=[]
    saved_support=pd.read_csv(root/'metrics/training_daily.csv')
    for day in pd.date_range(protocol['validation_start'],protocol['validation_end'],freq='D',inclusive='left'):
        h=pool.X.index[(meta.created_at_utc>=day-pd.Timedelta(days=30))&(meta.created_at_utc<day)&(meta.closed_at_utc<day)]
        q=pool.X.index[(meta.created_at_utc>=day)&(meta.created_at_utc<day+pd.Timedelta(days=1))&(meta.closed_at_utc<protocol['validation_end'])]
        if len(q)==0:continue
        current={}
        for target,quantile in [('high_risk_unit_q975',.975),('high_risk_unit_q950',.95)]:
            chunks=[]
            for side in [0,1]:
                hs=h[pool.X.loc[h,'option_type'].eq(side)];qs=q[pool.X.loc[q,'option_type'].eq(side)]
                def labels(ids):
                    out=pd.Series(np.nan,index=ids)
                    if len(hs)<60:return out
                    for tenor,group in pool.X.loc[ids].groupby('product_duration_seconds'):
                        cell=hs[pool.X.loc[hs,'product_duration_seconds'].eq(tenor)]
                        threshold=np.quantile(pool.y.loc[cell if len(cell)>=80 else hs].to_numpy(),quantile)
                        if threshold>0:out.loc[group.index]=(pool.y.loc[group.index]>threshold).astype(float)
                    return out
                train=labels(hs).dropna();score=labels(qs);n=len(train);positive=int(train.sum())
                days=meta.loc[train.index[train.eq(1)],'created_at_utc'].dt.floor('D').nunique()
                supported=n>=100 and positive>=10 and n-positive>=20 and days>=3
                prior=(positive+.5)/(n+1)
                p=np.full(len(qs),prior)
                if supported:
                    with threadpool_limits(limits=1):
                        model=HistGradientBoostingClassifier(**c['classifier']).fit(features.loc[train.index],train.astype(int))
                        if len(qs):p=model.predict_proba(features.loc[qs])[:,1]
                row=saved_support[(saved_support.day==day.isoformat())&(saved_support.target==target)&(saved_support.direction==side)].iloc[0]
                for key,value in [('training_orders',n),('positive_orders',positive),('positive_days',days),
                                  ('historical_probability',prior),('trained',supported),('scoring_orders',len(qs))]:close(row[key],value)
                supports.append((day,target,side))
                chunks.append(pd.DataFrame({'day':day.isoformat(),'month':day.strftime('%Y-%m'),'target':target,
                    'side':side,'label':score,'probability':p,'prior':prior,'trained':supported,
                    'actual':pool.y.loc[qs],'quantity':pool.X.loc[qs,'quantity_btc'],
                    'benchmark_eligible':meta.loc[qs,'benchmark_eligible']},index=qs))
            part=pd.concat(chunks).reindex(q);part['alarm']=part.trained&part.probability.ge(.2)
            records.append(part);current[target]=part
        hb=benchmark.X.index[(bm.created_at_utc>=day-pd.Timedelta(days=30))&(bm.created_at_utc<day)&(bm.closed_at_utc<day)]
        qb=benchmark.X.index[(bm.created_at_utc>=day)&(bm.created_at_utc<day+pd.Timedelta(days=1))&(bm.closed_at_utc<protocol['validation_end'])]
        if len(qb):
            for name in ['official_raw','hurdle_base']:
                with threadpool_limits(limits=1):base=CalibrationModel(name).fit(xb.loc[hb],benchmark.y.loc[hb]).predict_raw(xb.loc[qb])
                part=current['high_risk_unit_q975'].loc[qb,['day','month','side','actual','quantity']].copy()
                part['peak_q975']=current['high_risk_unit_q975'].loc[qb,'label'];part['anchor']=base
                prices.append(part.assign(model=name,prediction=base,alarm=False))
                for target,g in current.items():
                    alarm=g.loc[qb,'alarm'];prices.append(part.assign(model=name+'_'+target.split('_')[-1],
                        prediction=base*np.where(alarm,1.5,1),alarm=alarm))
    frame=pd.concat(records);pf=pd.concat(prices)
    class_checks=0
    for filename in ['classification','threshold_diagnostic']:
        table=pd.read_csv(root/f'metrics/{filename}.csv')
        for _,row in table.iterrows():
            f=frame[frame.target.eq(row.target)]
            if row.period!='all':f=f[f.month.eq(row.period)]
            if row.cohort=='benchmark_common':f=f[f.benchmark_eligible]
            if row.direction!='all':f=f[f.side.eq(0 if row.direction=='Call' else 1)]
            if filename=='classification':
                close(row.orders,len(f));close(row.available_labels,f.label.notna().sum())
                close(row.fallback_orders,(~f.trained).sum() if row.method=='gbdt' else 0)
            f=f[f.label.notna()];y=f.label.to_numpy();positive=y==1
            historical=filename=='classification' and row.method=='historical_probability'
            p=f.prior.to_numpy() if historical else f.probability.to_numpy()
            gate=(p>=row.alarm_probability)&(True if historical else f.trained.to_numpy())
            tp=(gate&positive).sum();fp=(gate&~positive).sum();cash=(f.actual*f.quantity).to_numpy()
            values={'positive_orders':positive.sum(),'prevalence':y.mean(),'mean_probability':p.mean(),
                'brier':np.mean((p-y)**2),'log_loss':log_loss(y,p,labels=[0,1]),
                'average_precision':average_precision_score(y,p) if positive.any() else np.nan,
                'roc_auc':roc_auc_score(y,p) if np.unique(y).size==2 else np.nan,
                'alarms':gate.sum(),'alarm_fraction':gate.mean(),'tp':tp,'fp':fp,
                'fn':(~gate&positive).sum(),'tn':(~gate&~positive).sum(),
                'precision':tp/gate.sum() if gate.any() else np.nan,
                'recall':tp/positive.sum() if positive.any() else np.nan,
                'false_positive_rate':fp/(~positive).sum() if (~positive).any() else np.nan,
                'positive_payout_usdc':cash[positive].sum(),'caught_positive_payout_usdc':cash[positive&gate].sum(),
                'tail_cash_recall':cash[positive&gate].sum()/cash[positive].sum() if cash[positive].sum()>0 else np.nan}
            for name,value in values.items():close(row[name],value)
            class_checks+=1
    payout_checks=0
    for filename in ['summary','daily','fixed_peak_slices']:
        table=pd.read_csv(root/f'metrics/{filename}.csv',dtype={'value':str})
        if filename!='fixed_peak_slices':table=table[table.group.isin(['overall','direction'])]
        for _,row in table.iterrows():
            f=pf[pf.model.eq(row.model)]
            if filename=='daily':f=f[f.day.eq(row.day)]
            else:
                period=row.period if filename=='fixed_peak_slices' else row.month
                if period!='all':f=f[f.month.eq(period)]
            if filename=='fixed_peak_slices':
                if row.direction!='all':f=f[f.side.eq(0 if row.direction=='Call' else 1)]
                f=f[f.peak_q975.eq(1 if row.group=='peak_q975' else 0)]
            elif row.group=='direction':f=f[f.side.eq(int(row.value))]
            e=(f.prediction-f.actual).to_numpy();w=f.quantity.to_numpy()
            values={'orders':len(f),'rmse_per_btc':np.sqrt(np.mean(e**2)),'mae_per_btc':np.mean(abs(e)),
                'actual_total_usdc':np.dot(f.actual,w),'predicted_total_usdc':np.dot(f.prediction,w),
                'total_bias_usdc':np.dot(e,w),'gross_overprediction_usdc':np.dot(np.maximum(e,0),w),
                'gross_underprediction_usdc':np.dot(np.maximum(-e,0),w)}
            for name,value in values.items():close(row[name],value)
            payout_checks+=1
    for _,row in pd.read_csv(root/'metrics/macro_monthly.csv').iterrows():
        f=pf[pf.model.eq(row.model)];mse=[];mae=[]
        for _,g in f.groupby('month'):
            e=g.prediction-g.actual;mse.append((e**2).mean());mae.append(abs(e).mean())
        close(row.macro_rmse_per_btc,np.sqrt(np.mean(mse)));close(row.macro_mae_per_btc,np.mean(mae))
    ranges=frame.groupby(['target','month','side']).agg(minimum_probability=('probability','min'),
        median_probability=('probability','median'),p95_probability=('probability',lambda s:s.quantile(.95)),
        maximum_probability=('probability','max')).reset_index()
    ranges.to_csv(root/'metrics/probability_ranges.csv',index=False,float_format='%.17g')
    peaks=[]
    for (target,day),f in frame.groupby(['target','day']):
        peaks.append({'target':target,'day':day,'orders':len(f),'actual_total_usdc':float(np.dot(f.actual,f.quantity)),
            'positive_orders':int(f.label.sum()),'alarms':int(f.alarm.sum()),
            'maximum_probability':f.probability.max(),'positive_orders_caught':int((f.alarm&f.label.eq(1)).sum())})
    peak=pd.DataFrame(peaks).sort_values(['target','actual_total_usdc'],ascending=[True,False]).groupby('target').head(5)
    peak.to_csv(root/'metrics/largest_payout_days.csv',index=False,float_format='%.17g')
    assert development_snapshot()[0]==identity
    for name,sha in source['code_sha256'].items():assert file_digest(PROJECT/name)==sha,name
    result={'status':'passed','independent_daily_classifier_slots':len(supports),
        'classification_and_threshold_rows':class_checks,'payout_aggregate_rows':payout_checks,
        'macro_monthly_rows':6,'independent_labels_and_fits':True,
        'reused_components':['previously verified feature construction','previously verified hurdle anchor'],
        'test_csv_read':False,'july_prices_parsed':False,'blocked_test_opens':guard['blocked_test_opens'],
        'verification_script_sha256':file_digest(Path(__file__))}
    atomic_json(root/'provenance/verification.json',result);print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
