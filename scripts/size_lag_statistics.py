"""Descriptive size contrasts and time alignment; no prediction-model fitting."""
import numpy as np
import pandas as pd


def size_thresholds(f,start,end):
    flag=pd.Series(pd.NA,index=f.index,dtype='boolean');records=[]
    for day in pd.date_range(start,end.floor('D'),freq='D'):
        today=f.loc[f.created.ge(day)&f.created.lt(min(day+pd.Timedelta(days=1),end))]
        h=f.loc[f.created.ge(day-pd.Timedelta(days=30))&f.created.lt(day)&f.closed.lt(day)]
        for (side,tenor),g in today.groupby(['side','tenor']):
            ref=h.loc[h.side.eq(side)&h.tenor.eq(tenor)];source='direction_tenor'
            if len(ref)<30:ref=h.loc[h.side.eq(side)];source='direction'
            cut=ref.quantity.quantile(.9) if len(ref)>=30 else np.nan
            if np.isfinite(cut):flag.loc[g.index]=g.quantity.gt(cut)
            records.append({'day':day,'side':side,'tenor':tenor,'threshold':cut,'history':len(ref),'source':source,
                'payout_q90':ref.actual.quantile(.9) if 'actual' in ref and len(ref)>=30 else np.nan})
    return flag,pd.DataFrame(records)


def matched_daily(f,flag,controls,metrics):
    rows=[]
    f=f.loc[f[flag].notna()]
    for keys,g in f.groupby(['day',*controls],observed=True,dropna=False):
        a=g.loc[g[flag].astype(bool)];b=g.loc[~g[flag].astype(bool)]
        if not len(a) or not len(b):continue
        row={'day':keys[0],'orders':len(g),'large_orders':len(a),'cells':1}
        for metric in metrics:
            valid=a[metric].notna().any() and b[metric].notna().any()
            row[metric+'_large']=a[metric].mean() if valid else np.nan
            row[metric+'_other']=b[metric].mean() if valid else np.nan
        rows.append(row)
    if not rows:return pd.DataFrame()
    cells=pd.DataFrame(rows);daily=[]
    for day,g in cells.groupby('day'):
        row={'day':day,**{k:g[k].sum() for k in ['orders','large_orders','cells']}}
        for metric in metrics:
            good=g[metric+'_large'].notna()&g[metric+'_other'].notna()
            for label in ['large','other']:
                row[metric+'_'+label]=np.average(g.loc[good,metric+'_'+label],weights=g.loc[good,'orders']) if good.any() else np.nan
            row[metric+'_difference']=row[metric+'_large']-row[metric+'_other']
        daily.append(row)
    return pd.DataFrame(daily).set_index('day')


def block_weights(n,seed=20261014,repeats=2000):
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(repeats,int(np.ceil(n/7))))
    indices=((starts[:,:,None]+np.arange(7))%n).reshape(repeats,-1)[:,:n]
    weights=np.zeros((repeats,n),dtype=int)
    np.add.at(weights,(np.repeat(np.arange(repeats),n),indices.ravel()),1)
    return weights


def contrast_summary(d,metric,start,end):
    days=pd.date_range(start,end.floor('D'),inclusive='left' if end==end.floor('D') else 'both')
    s=d[metric+'_difference'].reindex(days);v=s.to_numpy(float);ok=np.isfinite(v)
    weights=block_weights(len(days));den=weights@ok
    draws=np.divide(weights@np.nan_to_num(v),den,out=np.full(2000,np.nan),where=den>0)
    finite=draws[np.isfinite(draws)]
    lo,hi=np.quantile(finite,[.025,.975]) if len(finite)>=1900 else [np.nan,np.nan]
    deletion=[]
    for j in range(len(days)-6):
        mask=ok.copy();mask[j:j+7]=False
        if mask.sum()>=10:deletion.append(v[mask].mean())
    row={'days':int(ok.sum()),'orders':int(d.orders.sum()),'large_orders':int(d.large_orders.sum()),
        'large_mean':d[metric+'_large'].mean(),'other_mean':d[metric+'_other'].mean(),
        'difference':float(np.nanmean(v)),'ci_low':lo,'ci_high':hi,'valid_draws':len(finite),
        'delete7_min':min(deletion) if deletion else np.nan,'delete7_max':max(deletion) if deletion else np.nan,
        'delete7_valid':len(deletion),'delete7_expected':max(len(days)-6,0)}
    influence=np.where(ok,v-row['difference'],0)
    linear=weights@influence/max(ok.sum(),1)
    return row,linear


def lag_pairs(x,y,lag):
    # x(t) versus y(t+lag); shift in calendar units, never compress empty dates.
    shifted=y.copy();shifted.index=shifted.index-pd.Timedelta(days=lag)
    return pd.concat([x.rename('x'),shifted.rename('y')],axis=1).dropna()


def select_events(signal,threshold,start,end,separation=72):
    candidates=signal.index[(signal>threshold)&(signal.index>=start+pd.Timedelta(hours=24))&(signal.index<end-pd.Timedelta(hours=48))]
    result=[]
    for t in candidates:
        if not result or t-result[-1]>=pd.Timedelta(hours=separation):result.append(t)
    return result
