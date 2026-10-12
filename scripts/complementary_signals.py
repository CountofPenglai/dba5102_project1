"""Forward exploration of liquidity, crowding and implied-volatility information.

python -m scripts.complementary_signals --external PATH --run-id ID
Only aggregate outputs; raw public inputs are separately checksum-addressed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from pp_data.config import PROJECT, DEFAULT_DATA, DEFAULT_OUTPUT
from pp_models.directional_features import read_market_prefix
from pp_models.directional_study import install_development_guard
from pp_models.scoped_data import load_validation_pool
from pp_models.study import study_snapshot
from pp_models.study_storage import atomic_json, file_digest
from pp_models.runs import run_destination
from scripts.payout_signal_analysis import build_observations, aggregate_associations, alert_statistics, rank_correlation

END=pd.Timestamp('2026-07-01T00:00:00Z')
START=pd.Timestamp('2026-05-01T00:00:00Z')
BASE=['log_rv60','log_rv240','log_rv1440','direction_return60']
GROUPS={'background':BASE,
 'liquidity':BASE+['pressure_depth','thin_book','flow_imbalance','pressure_thin'],
 'crowding':BASE+['oi_change1h','log_oi_ratio24h','funding_pressure','vol_funding'],
 'implied':BASE+['log_dvol','dvol_change1h','log_iv_rv','vol_iv_gap']}
GROUPS['all']=list(dict.fromkeys(sum(GROUPS.values(),[])))


def asof_values(query,frame,delay_minutes,stale_minutes):
    if not query.index.is_unique or query.dt.tz is None or query.isna().any():raise ValueError('Invalid query times')
    right=frame.sort_values('time').copy()
    if right.time.duplicated().any():raise ValueError('Duplicate external time')
    right['time']=right.time+pd.Timedelta(minutes=delay_minutes)
    left=pd.DataFrame({'query':query.array,'order':np.arange(len(query))})
    joined=pd.merge_asof(left.sort_values('query'),right,left_on='query',right_on='time',
                         direction='backward',tolerance=pd.Timedelta(minutes=stale_minutes))
    return joined.sort_values('order').drop(columns=['query','order','time']).set_axis(query.index)


def dvol_frame(rows):
    raw=pd.DataFrame(rows,columns=['stamp','open','high','low','dvol']).sort_values('stamp')
    if raw.stamp.duplicated().any():raise ValueError('Duplicate DVOL candle')
    raw['time']=pd.to_datetime(raw.stamp,unit='ms',utc=True)+pd.Timedelta(hours=1)
    raw['dvol_change1h']=raw.dvol.diff().where(raw.time.diff().eq(pd.Timedelta(hours=1)))
    return raw[['time','dvol','dvol_change1h']]


def normalize_book(book):
    """One completed-minute observation; do not weight dense sampling more heavily."""
    frame=book.resample('1min',closed='left',label='right').last()
    for name in ['bid','ask']:
        prior=frame[name].shift(1).rolling(1440,min_periods=1296).mean()
        frame[name+'_prior']=prior.where(frame.index-frame.index[0]>=pd.Timedelta(days=1))
    return frame.dropna(subset=['bid','ask']).rename_axis('time').reset_index()


def read_external(root):
    manifest=json.loads((root/'manifest.json').read_text());pieces={k:[] for k in ['metrics','bookDepth','fundingRate','dvol']}
    for record in manifest['files']:
        path=root/record['path']
        if file_digest(path)!=record['sha256']:raise ValueError('External input checksum changed')
        kind=path.parent.name
        if kind=='dvol':pieces[kind].extend(json.loads(path.read_text())['result']['data']);continue
        with ZipFile(path) as z:
            f=pd.read_csv(z.open(z.namelist()[0]))
        if kind=='bookDepth':f=f.loc[f.percentage.abs().eq(.2)]
        pieces[kind].append(f)
    metric=pd.concat(pieces['metrics'],ignore_index=True).sort_values('create_time')
    if set(metric.symbol)!= {'BTCUSDC'}:raise ValueError('Unexpected OI symbol')
    metric['time']=pd.to_datetime(metric.create_time,utc=True)
    oi=metric.set_index('time').sum_open_interest
    prior=oi.rolling('24h',closed='left',min_periods=280).mean()
    change=oi/oi.reindex(oi.index-pd.Timedelta(hours=1)).to_numpy()-1
    metrics=pd.DataFrame({'time':oi.index,'oi_change1h':change.to_numpy(),'oi_ratio24h':(oi/prior).to_numpy()})
    depth=pd.concat(pieces['bookDepth'],ignore_index=True)
    depth['time']=pd.to_datetime(depth.timestamp,utc=True)
    if depth.duplicated(['time','percentage']).any():raise ValueError('Duplicate depth snapshot/band')
    book=depth.pivot(index='time',columns='percentage',values='notional').rename(columns={-.2:'bid',.2:'ask'}).sort_index()
    book=normalize_book(book)
    fund=pd.concat(pieces['fundingRate'],ignore_index=True).sort_values('calc_time')
    funding=pd.DataFrame({'time':pd.to_datetime(fund.calc_time,unit='ms',utc=True),
                        'funding8h':fund.last_funding_rate*8/fund.funding_interval_hours})
    iv=dvol_frame(pieces['dvol'])
    tables={'book':book,'oi':metrics,'funding':funding,'iv':iv};audit=[]
    for name,f in tables.items():
        if f.time.duplicated().any() or f.time.isna().any() or not f.time.is_monotonic_increasing:
            raise ValueError(f'Duplicate or unordered {name}')
        # Last pre-July DVOL candle is complete exactly at the boundary; it is never joined to June.
        if (f.time.gt(END) if name in ['iv','book'] else f.time.ge(END)).any():raise ValueError('External data beyond development period')
        if np.isinf(f.drop(columns='time').to_numpy()).any():raise ValueError('Infinite external features')
        audit.append({'source':name,'rows':len(f),'first':f.time.min().isoformat(),'last':f.time.max().isoformat(),
                      'max_gap_minutes':f.time.diff().dt.total_seconds().max()/60,
                      'feature_incomplete_rows':int(f.drop(columns='time').isna().any(axis=1).sum())})
    if (book[['bid','ask']]<=0).any().any() or (oi<=0).any() or (iv.dvol<=0).any():raise ValueError('Nonpositive market state')
    return tables,pd.DataFrame(audit),manifest


def market_frame(market):
    logp=np.log(market.close);r=logp.diff()
    out=pd.DataFrame({'time':market.close_time_utc})
    for w in [60,240,1440]:
        contiguous=market.close_time_utc.diff(w).eq(pd.Timedelta(minutes=w))
        out[f'rv{w}']=np.sqrt(r.pow(2).rolling(w).mean()).where(contiguous)
    out['return60']=(logp-logp.shift(60)).where(market.close_time_utc.diff(60).eq(pd.Timedelta(hours=1)))
    out['net_buy15']=(2*market.taker_buy_quote_asset_volume-market.quote_asset_volume).rolling(15).sum()
    out['quote15']=market.quote_asset_volume.rolling(15).sum()
    if not market.close_time_utc.diff().dropna().eq(pd.Timedelta(minutes=1)).all():
        raise ValueError('Market grid has gaps; do not bridge flow windows')
    return out


def joined_signals(query,market,tables,lag):
    return pd.concat([asof_values(query,market,0,1),asof_values(query,tables['book'],lag,2),
        asof_values(query,tables['oi'],lag,10),asof_values(query,tables['funding'],lag,540),
        asof_values(query,tables['iv'],lag,65)],axis=1)


def directional_signals(f,side):
    sign=1 if side==0 else -1
    depth=f.ask if side==0 else f.bid;prior=f.ask_prior if side==0 else f.bid_prior
    out=pd.DataFrame(index=f.index)
    for w in [60,240,1440]:out[f'log_rv{w}']=np.log(f[f'rv{w}'].where(f[f'rv{w}']>0))
    out['direction_return60']=sign*f.return60/(f.rv60*np.sqrt(60))
    out['pressure_depth']=np.log1p((sign*f.net_buy15).clip(lower=0)/depth)
    out['thin_book']=np.log(prior/depth)
    out['flow_imbalance']=sign*f.net_buy15/f.quote15
    out['pressure_thin']=out.flow_imbalance*out.thin_book
    out['oi_change1h']=f.oi_change1h
    out['log_oi_ratio24h']=np.log(f.oi_ratio24h)
    out['funding_pressure']=-sign*f.funding8h
    annual=f.rv1440*np.sqrt(365*1440)
    out['vol_funding']=annual*out.funding_pressure
    out['log_dvol']=np.log(f.dvol/100)
    out['dvol_change1h']=f.dvol_change1h
    out['log_iv_rv']=np.log((f.dvol/100)/annual)
    out['vol_iv_gap']=annual*out.log_iv_rv
    return out.replace([np.inf,-np.inf],np.nan)


def forward_returns(query,market,horizon):
    t=market.close_time_utc.dt.as_unit('ns').astype('int64').to_numpy()
    q=query.dt.as_unit('ns').astype('int64').to_numpy();p=np.searchsorted(t,q,side='right')-1
    endpos=p+horizon;valid=(p>=0)&(endpos<len(t));safe=np.clip(endpos,0,len(t)-1);p=np.clip(p,0,len(t)-1)
    valid &= (t[safe]-t[p]==horizon*60_000_000_000)&(q-t[p]<=60_000_000_000)
    values=np.log(market.close.to_numpy()[safe]/market.close.to_numpy()[p]);values[~valid]=np.nan
    return pd.DataFrame({'return':values,'end':pd.to_datetime(t[safe],utc=True).where(valid)},index=query.index)


def history_mask(frame,day):
    return frame.time.ge(day-pd.Timedelta(days=30))&frame.time.lt(day)&frame.end.lt(day)


def overlap_statistics(actual,baseline,new):
    a,b,n=map(lambda x:np.asarray(x,bool),(actual,baseline,new))
    union=(b|n).sum();tu=((b|n)&a).sum()
    return {'new_tail_hits':int((a&n&~b).sum()),'lost_tail_hits':int((a&b&~n).sum()),
        'new_false_alerts':int((~a&n&~b).sum()),'removed_false_alerts':int((~a&b&~n).sum()),
        'alert_jaccard':float((b&n).sum()/union) if union else np.nan,
        'tail_jaccard':float((a&b&n).sum()/tu) if tu else np.nan}


def probe_market(query,raw,market,lag):
    rows=[];coverage=[];correlations=[]
    for side in [0,1]:
        X=directional_signals(raw,side)
        valid=X.notna().all(axis=1)
        june=query.ge('2026-06-01')
        for feature in X:
            correlations.append({'lag_minutes':lag,'direction':side,'feature':feature,
                'june_valid':int(X.loc[june,feature].notna().sum()),
                'rho_with_log_rv60':rank_correlation(X.loc[june,feature],X.loc[june,'log_rv60'])})
        for horizon in [60,240,480]:
            outcome=forward_returns(query,market,horizon);frame=pd.DataFrame({'time':query,'end':outcome.end})
            movement=(1 if side==0 else -1)*outcome['return']
            for day in pd.date_range('2026-06-01','2026-06-30',tz='UTC'):
                phase_end=pd.Timestamp('2026-06-21T00:00:00Z') if day.day<=20 else END
                history=history_mask(frame,day)&valid&movement.notna()
                scoring=query.ge(day)&query.lt(day+pd.Timedelta(days=1))&outcome.end.lt(phase_end)&movement.notna()
                keep=scoring&valid
                coverage.append({'lag_minutes':lag,'direction':side,'horizon':horizon,'day':day.isoformat(),
                                 'eligible':int(scoring.sum()),'scored':int(keep.sum()),'history':int(history.sum())})
                if history.sum()<500 or not keep.any():raise ValueError('Insufficient common history/scoring data')
                threshold=movement.loc[history].quantile(.9);ytrain=movement.loc[history].gt(threshold).astype(int)
                actual=movement.loc[keep].gt(threshold).astype(int)
                for name,features in GROUPS.items():
                    scaler=StandardScaler().fit(X.loc[history,features])
                    train=np.clip(scaler.transform(X.loc[history,features]),-5,5)
                    today=np.clip(scaler.transform(X.loc[keep,features]),-5,5)
                    model=LogisticRegression(C=.1,solver='lbfgs',max_iter=500,tol=1e-8)
                    with warnings.catch_warnings():
                        warnings.simplefilter('error');model.fit(train,ytrain)
                    p=model.predict_proba(today)[:,1];trainp=model.predict_proba(train)[:,1]
                    flag=p>np.quantile(trainp,.8)
                    rows.append(pd.DataFrame({'lag_minutes':lag,'direction':side,'horizon':horizon,'model':name,
                        'time':query.loc[keep],'day':day.isoformat(),'phase':'early' if day.day<=20 else 'late',
                        'actual':actual,'p':p,'alert':flag,'label_threshold':threshold},index=query.index[keep]))
            print(f'Market probe lag={lag} side={side} horizon={horizon} complete',flush=True)
    return pd.concat(rows),pd.DataFrame(coverage),pd.DataFrame(correlations)


def market_aggregates(pred):
    daily=[];summary=[];overlaps=[]
    def scores(f):
        result={'brier':brier_score_loss(f.actual,f.p),'log_loss':log_loss(f.actual,f.p,labels=[0,1]),
            'average_precision':average_precision_score(f.actual,f.p) if f.actual.sum() else np.nan,
            **alert_statistics(f.actual,f.alert),'days':f.day.nunique(),
            'hit_days':f.loc[f.actual.eq(1)&f.alert,'day'].nunique()}
        return result
    for keys,f in pred.groupby(['lag_minutes','direction','horizon','model','day']):
        daily.append(dict(zip(['lag_minutes','direction','horizon','model','day'],keys))|scores(f))
    for period in ['all_june','early','late','without_june24_25','nonoverlap']:
        f=pred
        if period in ['early','late']:f=f[f.phase.eq(period)]
        if period=='without_june24_25':f=f[~f.time.dt.day.isin([24,25])]
        if period=='nonoverlap':f=f[(f.time.dt.minute.eq(0))&(f.time.dt.hour*60%f.horizon==0)]
        for keys,g in f.groupby(['lag_minutes','direction','horizon']):
            baseline=g[g.model.eq('background')].sort_values('time')
            for name,h in g.groupby('model'):
                h=h.sort_values('time')
                if not h.time.equals(baseline.time):raise ValueError('Probe cohort mismatch')
                common=dict(zip(['lag_minutes','direction','horizon'],keys))|{'period':period,'model':name}
                summary.append(common|scores(h))
                overlaps.append(common|overlap_statistics(h.actual,baseline.alert,h.alert))
    return pd.DataFrame(summary),pd.DataFrame(daily),pd.DataFrame(overlaps)


def order_analysis(pool,raw,base):
    signals=directional_signals(raw,0)
    put=pool.X.option_type.eq(1);signals.loc[put]=directional_signals(raw.loc[put],1)
    obs,high,low,fallback,coverage=build_observations(pool,signals,base)
    associations,alerts=aggregate_associations(obs,signals,high,low,fallback)
    records=[];daily=[]
    for side in [0,1]:
        for period in ['all_june','early','late','june24_25']:
            selected=obs.direction.eq(side)&obs['tail'].notna()
            if period in ['early','late']:selected&=obs.phase.eq(period)
            if period=='june24_25':selected&=pd.to_datetime(obs.day).dt.day.isin([24,25])
            f=obs[selected];idx=f.index;a=f['tail'].astype(bool)
            b=signals.loc[idx,'log_rv60'].gt(high.loc[idx,'log_rv60'])
            for feature in signals:
                new=signals.loc[idx,feature].gt(high.loc[idx,feature])
                records.append({'direction':side,'period':period,'feature':feature,
                    **overlap_statistics(a,b,new),'tail_orders':int(a.sum()),
                    'new_hit_days':f.loc[a&new&~b,'day'].nunique(),
                    'new_tail_cash':f.loc[a&new&~b,'cash_payout'].sum()})
        for (phase,day),f in obs[obs.direction.eq(side)].groupby(['phase','day']):
            a=f['tail'].fillna(False).astype(bool)
            for feature in ['log_rv60','pressure_depth','thin_book','oi_change1h','funding_pressure','log_dvol','dvol_change1h']:
                flag=signals.loc[f.index,feature].gt(high.loc[f.index,feature])
                daily.append({'phase':phase,'day':day,'direction':side,'feature':feature,
                              'orders':len(f),'tail_orders':int(a.sum()),'alerts':int(flag.sum()),
                              'hits':int((a&flag).sum()),'tail_cash':f.loc[a,'cash_payout'].sum(),
                              'hit_tail_cash':f.loc[a&flag,'cash_payout'].sum()})
    return {'order_associations':associations,'order_alerts':alerts,'order_overlap':pd.DataFrame(records),
            'order_daily':pd.DataFrame(daily)},coverage


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--external',required=True,type=Path)
    parser.add_argument('--run-id',required=True);args=parser.parse_args();started=perf_counter()
    guard=install_development_guard(DEFAULT_OUTPUT);source=study_snapshot(DEFAULT_OUTPUT)
    root=run_destination(PROJECT/'outputs/diagnostics',args.run_id)
    if root.exists():raise ValueError('Output already exists')
    external,audit,manifest=read_external(args.external.resolve())
    print('External files checked; reading pre-July market prefix',flush=True)
    market,market_info=read_market_prefix(DEFAULT_DATA/'btc_market_history.csv',END.isoformat())
    market_features=market_frame(market)
    query=pd.Series(pd.date_range(START,END,inclusive='left',freq='30min'))
    tables={'external_coverage':audit};predictions=[];coverages=[];corr=[]
    for lag in [5,60]:
        raw=joined_signals(query,market_features,external,lag)
        pred,cov,c=probe_market(query,raw,market,lag);predictions.append(pred);coverages.append(cov);corr.append(c)
    pred=pd.concat(predictions)
    tables['market_summary'],tables['market_daily'],tables['market_overlap']=market_aggregates(pred)
    tables['market_coverage']=pd.concat(coverages);tables['signal_background_correlation']=pd.concat(corr)
    pool=load_validation_pool({'cohort':'risk','use_base':False})
    common=load_validation_pool({'cohort':'benchmark','use_base':True})
    raw=joined_signals(pool.metadata.created_at_utc,market_features,external,5)
    order_tables,order_coverage=order_analysis(pool,raw,common.X.api_base_price_per_btc_usdc)
    tables.update(order_tables)
    if study_snapshot(DEFAULT_OUTPUT)!=source:raise ValueError('Canonical source changed during run')
    root.mkdir(parents=True)
    for folder in ['metrics','provenance','reports']:(root/folder).mkdir()
    for name,table in tables.items():table.to_csv(root/f'metrics/{name}.csv',index=False,float_format='%.12g')
    provenance={'version':'complementary_signals_v1','market_prefix':market_info,'canonical':source,
        'external_manifest_sha256':file_digest(args.external/'manifest.json'),'external_manifest':manifest,
        'order_coverage':order_coverage,'source_groups':GROUPS,'logistic_C':.1,'parameters_tuned':False,
        'lags_minutes':[5,60],'scoring_start':'2026-06-01','scoring_end_exclusive':END.isoformat(),
        'test_csv_read':False,'guard':{k:sorted(v) if isinstance(v,set) else v for k,v in guard.items()},
        'elapsed_seconds':perf_counter()-started,
        'limitations':['June already inspected; both phases are exploratory',
            'Event-time plus lag scenario; historical archive does not prove point-in-time publication',
            'Overlapping market windows and correlated orders are not independent tail events',
            'DVOL is 30-day IV, not short-tenor skew; OI is open contracts, not leverage or net position',
            'Market probe targets directional terminal return tails; actual early-settlement payout differs',
            'Signal tests do not establish superior pricing or profitability']}
    atomic_json(root/'provenance/source.json',provenance)
    with ZipFile(root/'provenance/source_snapshot.zip','w',ZIP_DEFLATED) as z:
        files=set(source['code_sha256'])|{'scripts/complementary_signals.py','scripts/fetch_complementary_data.py',
            'scripts/payout_signal_analysis.py','tests/test_complementary_signals.py','docs/plans/2026-10-09-complementary-signals.md'}
        for name in sorted(files):z.write(PROJECT/name,name)
    atomic_json(root/'provenance/manifest.json',{'status':'complete','files':{
        str(p.relative_to(root)):file_digest(p) for p in list((root/'metrics').iterdir())+list((root/'provenance').iterdir())}})
    print(json.dumps({'output':str(root),'elapsed_seconds':provenance['elapsed_seconds'],'order_coverage':order_coverage},indent=2))


if __name__=='__main__':main()
