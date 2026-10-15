"""Replay saved coefficients with explicit arithmetic; never fit an estimator."""
import argparse
import json
from pathlib import Path
import sys
from zipfile import ZipFile
import hashlib
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from pp_data.config import DEFAULT_DATA
from pp_data.features import BASE_FEATURE
from pp_models.frozen_candidate_study import AccessGuard, prepare, MODELS, canonical_identity
from pp_models.directional_features import read_market_prefix
from pp_models.put_final_io import read_test_market_prefix
from pp_models.scoped_data import load_validation_pool
from pp_models.data import load_pool
from pp_models.study_storage import atomic_json, file_digest


def replay(X,state):
    base=X[BASE_FEATURE].to_numpy(float);side=X.option_type.to_numpy(int)
    tau=X.remaining_seconds.to_numpy(float)
    uncertainty=np.maximum(X.realized_vol_60m.to_numpy(float)*np.sqrt(tau/3600),1e-6)
    position=np.clip((1-2*side)*X.log_moneyness.to_numpy(float)/uncertainty,-5,5)
    design=np.column_stack([np.log(base),np.log(tau),position])
    out={'official_raw':base.copy(),'h0':np.zeros(len(X)),'contract30':np.zeros(len(X)),
         'g1a':np.zeros(len(X)),'static':base.copy()}
    for d in (0,1):
        mask=side==d;h=state['h0'][str(d)]
        if 'coef' in h:
            z=np.clip((design[mask]-h['mean'])/h['scale'],-3,3)
            p=1/(1+np.exp(-(z@np.asarray(h['coef'])+h['intercept'])))
        else:p=np.full(mask.sum(),h['probability'])
        anchor=base[mask]*h['severity'];out['h0'][mask]=p*anchor
        a=state['contract30'][str(d)];mult=1.
        if a:
            z=np.clip((design[mask,1:]-a['mean'])/a['scale'],-3,3)
            mult=np.clip(1+np.column_stack([np.ones(mask.sum()),z])@np.asarray(a['beta']),.5,1.5)
        out['contract30'][mask]=p*anchor*mult
        g=state['g1a'][str(d)]
        if g['mode']=='gamma':
            z=np.clip((design[mask,1:]-g['mean'])/g['scale'],-3,3)
            conditional=np.exp(z@np.asarray(g['coef'])+g['intercept'])
        else:conditional=np.full(mask.sum(),g['coefficient'])
        out['g1a'][mask]=p*base[mask]*conditional
    active=X.terminal_supported.eq(1).to_numpy()
    out['static'][active]=state['static']['a']*base[active]+state['static']['b']*X.loc[active,'terminal_static_proxy']
    return out


def manual_proxy(X,created,market):
    """Independently reconstruct daily 30-day complete-window zero-mean returns."""
    price=pd.Series(market.close.to_numpy(float),index=pd.DatetimeIndex(market.open_time_utc)+pd.Timedelta(minutes=1))
    price=price.reindex(pd.date_range(price.index[0],price.index[-1],freq='min'))
    v=price.to_numpy();valid_returns=np.isfinite(v)&np.isfinite(np.r_[np.nan,v[:-1]])
    positions=np.arange(len(v))
    # Sigma is also required by the original static-history frame. A missing
    # return resets its 1440-return warmup, even though static ignores its value.
    valid_sigma=(positions-np.maximum.accumulate(np.where(valid_returns,-1,positions)))>=1440
    cache={};largest=0.;checked=0
    for i in X.index[X.terminal_supported.eq(1)]:
        day=created.loc[i].floor('D');tau=X.at[i,'remaining_seconds']/60
        h=next(v for v in (30,60,120,240,480) if v>=tau);key=day,h
        if key not in cache:
            origins=price.index[(price.index>=day-pd.Timedelta(days=30))&
                (price.index+pd.Timedelta(minutes=h)<day)&(price.index.minute%30==0)]
            pos=price.index.get_indexer(origins);good=np.isfinite(v)
            cs=np.r_[0,np.cumsum(good)]
            keep=((cs[pos+h+1]-cs[pos])==h+1)&valid_sigma[pos]
            pos=pos[keep];returns=np.log(v[pos+h])-np.log(v[pos])
            cache[key]=returns-returns.mean()
        scenarios=cache[key]*np.sqrt(tau/h)
        proxy=np.maximum(X.at[i,'strike_price_usdc_per_btc']-
            X.at[i,'market_price_usdc_per_btc']*np.exp(scenarios),0).mean()
        diff=abs(proxy-X.at[i,'terminal_static_proxy']);largest=max(largest,diff);checked+=1
        np.testing.assert_allclose(proxy,X.at[i,'terminal_static_proxy'],rtol=1e-11,atol=1e-8)
    return dict(orders_checked=checked,max_proxy_absolute_error=largest,distribution_cells=len(cache))


def metrics(p):
    e=p.prediction.to_numpy()-p.actual.to_numpy();q=p.quantity.to_numpy()
    y=p.actual.to_numpy();pred=p.prediction.to_numpy()
    return dict(orders=len(p),rmse_per_btc=np.sqrt(np.mean(e**2)),mae_per_btc=np.mean(abs(e)),
        actual_total_usdc=float(y@q),predicted_total_usdc=float(pred@q),total_bias_usdc=float(e@q),
        gross_underprediction_usdc=float(np.maximum(-e,0)@q),gross_overprediction_usdc=float(np.maximum(e,0)@q))


def verify(root):
    root=Path(root);manifest=json.loads((root/'provenance/manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('Incomplete run')
    pre=json.loads((root/'provenance/preflight.json').read_text());settings=json.loads((root/'provenance/config.json').read_text())
    canonical_identity(expected_manifest=pre['canonical']['manifest_sha256'])
    for name,digest in manifest['artifacts'].items():assert file_digest(root/name)==digest,name
    with ZipFile(root/'provenance/source.zip') as z:
        for name,digest in pre['sources'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
            assert file_digest(PROJECT/name)==digest,name
    for name,digest in pre['protected'].items():assert file_digest(PROJECT/name)==digest,name
    guard=AccessGuard();sys.addaudithook(guard.check)
    checks={};fit_calls=0
    def deny_fit(frame,event,arg):
        nonlocal fit_calls
        if event=='call' and frame.f_code.co_name in ('fit','fit_transform','predict','predict_proba','predict_raw','predict_severity','predict_components'):
            fit_calls+=1;raise AssertionError('Verification cannot fit or call model prediction')
    sys.setprofile(deny_fit)
    try:
        for phase,start,end in [('development',settings['development_start'],settings['development_end']),
                               ('pressure',settings['development_end'],settings['pressure_end'])]:
            if phase=='development':
                pool=load_validation_pool({'cohort':'benchmark','use_base':True})
                market,mi=read_market_prefix(DEFAULT_DATA/'btc_market_history.csv',end)
            else:
                guard.allow_pressure=True
                assert canonical_identity(include_test=True,expected_manifest=pre['canonical']['manifest_sha256'])==manifest['phases']['pressure']['canonical']
                pool=load_pool('test',{'cohort':'benchmark','use_base':True})
                market,mi=read_test_market_prefix(DEFAULT_DATA/'btc_market_history.csv')
            assert mi==manifest['phases'][phase]['market']
            pool,_=prepare(pool,market)
            states=json.loads((root/f'provenance/{phase}_parameters.json').read_text())
            frames=[]
            for stamp,state in states.items():
                origin=pd.Timestamp(stamp);stop=min(pd.Timestamp(end),origin+pd.offsets.MonthBegin(1))
                created=pool.metadata.created_at_utc;closed=pool.metadata.closed_at_utc
                ids=pool.X.index[created.ge(origin)&created.lt(stop)]
                preds=replay(pool.X.loc[ids],state)
                history=pool.X.index[created.lt(origin)&closed.lt(origin)]
                threshold=pd.Series(np.nan,index=ids)
                for (side,tenor),part in pool.X.loc[ids].groupby(['option_type','product_duration_seconds']):
                    h=pool.X.loc[history];cell=h.index[h.option_type.eq(side)&h.product_duration_seconds.eq(tenor)]
                    if len(cell)<30:cell=h.index[h.option_type.eq(side)]
                    if len(cell)>=30:threshold.loc[part.index]=np.quantile(pool.y.loc[cell],.9)
                for name,pred in preds.items():
                    f=pd.DataFrame(dict(prediction=pred,actual=pool.y.loc[ids],quantity=pool.X.loc[ids,'quantity_btc'],
                        side=pool.X.loc[ids,'option_type'].astype(int),tenor=pool.X.loc[ids,'product_duration_seconds'].astype(int),
                        day=created.loc[ids].dt.floor('D'),month=created.loc[ids].dt.strftime('%Y-%m'),model=name),index=ids)
                    f['tail_group']=np.where(threshold.isna(),'unavailable',np.where(f.actual>threshold,'tail','body'))
                    frames.append(f.loc[closed.loc[ids].lt(pd.Timestamp(end))])
            full=pd.concat(frames);query_ids=full.loc[full.model.eq('h0')].index
            proxy_check=manual_proxy(pool.X.loc[query_ids],pool.metadata.created_at_utc,market)
            saved=pd.read_csv(root/f'metrics/{phase}/summary.csv',dtype={'value':str})
            errors=[]
            for row in saved.itertuples():
                p=full.loc[full.model.eq(row.model)]
                if row.month!='all':p=p.loc[p.month.eq(row.month)]
                if row.group=='direction':p=p.loc[p.side.eq(int(row.value))]
                elif row.group=='direction_tail':
                    s,g=row.value.split('|');p=p.loc[p.side.eq(int(s))&p.tail_group.eq(g)]
                elif row.group=='tenor':p=p.loc[p.tenor.eq(int(row.value))]
                elif row.group!='overall':raise ValueError(row.group)
                for key,v in metrics(p).items():
                    np.testing.assert_allclose(v,getattr(row,key),rtol=1e-11,atol=1e-7)
                    errors.append(abs(v-getattr(row,key)))
            macro=pd.read_csv(root/f'metrics/{phase}/macro_monthly.csv')
            for row in macro.itertuples():
                p=full.loc[full.model.eq(row.model)]
                m=[metrics(g) for _,g in p.groupby('month')]
                np.testing.assert_allclose([np.sqrt(np.mean([v['rmse_per_btc']**2 for v in m])),np.mean([v['mae_per_btc'] for v in m])],
                    [row.macro_rmse_per_btc,row.macro_mae_per_btc],rtol=1e-11,atol=1e-8)
            checks[phase]={'orders':len(query_ids),'summary_cells':len(saved),'maximum_metric_error':max(errors),**proxy_check}
    finally:sys.setprofile(None)
    result=dict(status='passed',phases=checks,model_fit_or_predict_calls=fit_calls,
                official_validation_open_attempts=guard.official_attempts,premature_july_open_attempts=guard.premature_test_attempts,
                source_and_artifact_hashes_verified=True,protected_files=len(pre['protected']))
    atomic_json(root/'provenance/verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_directory')
    verify(parser.parse_args().run_directory)
