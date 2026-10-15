"""As-of descriptive signal exploration; save aggregates, never order rows.

Run from project root: python -m scripts.payout_signal_analysis --run-id ID
"""
import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from pp_data.config import PROJECT, DEFAULT_DATA, DEFAULT_OUTPUT
from pp_models.data import daily_batches
from pp_models.directional_features import read_market_prefix
from pp_models.directional_study import install_development_guard
from pp_models.market_activity_features import read_order_prefix
from pp_models.runs import run_destination, fingerprint
from pp_models.scoped_data import load_validation_pool
from pp_models.study import study_snapshot
from pp_models.study_storage import atomic_json, file_digest

PERIODS={'early':['2026-06-01T00:00:00Z','2026-06-21T00:00:00Z'],
         'late':['2026-06-21T00:00:00Z','2026-07-01T00:00:00Z']}


def _times(s):
    if s.empty or not s.index.is_unique or s.isna().any() or s.dt.tz is None:
        raise ValueError('Unique, nonempty, timezone-aware query series required')
    return s.dt.as_unit('ns').astype('int64').to_numpy()


def _divide(a,b):
    a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    return np.divide(a,b,out=np.full(a.shape,np.nan),where=b>0)


def market_signals(created, usable):
    times=_times(created);h=usable
    if (h.empty or h.open_time_utc.duplicated().any() or not h.open_time_utc.is_monotonic_increasing
            or not h.open_time_utc.eq(h.open_time_utc.dt.floor('min')).all()
            or not np.isfinite(h[['close','number_of_trades','quote_asset_volume','taker_buy_quote_asset_volume']]).all().all()
            or h.close.le(0).any()):
        raise ValueError('Expected audited minute market history')
    opens=_times(h.open_time_utc);closes=_times(h.close_time_utc)
    pos=np.searchsorted(closes,times,side='right')-1
    if ((pos<1500).any() or (times-closes[pos]>60_000_000_000).any()
            or (opens[pos]-opens[pos-1500]!=1500*60_000_000_000).any()):
        raise ValueError('Missing fresh continuous market history')
    out={};logp=np.log(h.close);ret=logp.diff()
    def at(s):return s.to_numpy()[pos]
    for w in (5,15,60,240):
        out[f'market_trades_per_min_{w}m']=at(h.number_of_trades.rolling(w).mean())
    for w in (15,60):
        prior=at(h.number_of_trades.shift(w).rolling(1440).mean())
        out[f'market_trade_ratio_{w}m']=_divide(out[f'market_trades_per_min_{w}m'],prior)
        quote=at(h.quote_asset_volume.rolling(w).sum());trades=at(h.number_of_trades.rolling(w).sum())
        buy=at(h.taker_buy_quote_asset_volume.rolling(w).sum())
        out[f'market_quote_per_min_{w}m']=quote/w
        out[f'market_mean_trade_quote_{w}m']=_divide(quote,trades)
        out[f'market_sell_share_{w}m']=1-_divide(buy,quote)
    for w in (15,60,240):
        out[f'market_return_{w}m']=at(logp-logp.shift(w))
        out[f'market_up_rms_{w}m']=at(np.sqrt(ret.clip(lower=0).pow(2).rolling(w).mean()))
        out[f'market_down_rms_{w}m']=at(np.sqrt(ret.clip(upper=0).pow(2).rolling(w).mean()))
    rms={w:np.sqrt(ret.pow(2).rolling(w).mean()) for w in (15,60,240)}
    for w in (60,240):out[f'market_total_rms_{w}m']=at(rms[w])
    out['market_vol_ratio_15m_60m']=_divide(at(rms[15]),at(rms[60]))
    out['market_vol_change_60m']=at(rms[60]-rms[60].shift(60))
    out['market_max_up_jump_60m']=at(ret.clip(lower=0).rolling(60).max())
    out['market_max_down_jump_60m']=at((-ret).clip(lower=0).rolling(60).max())
    paths=logp.to_numpy()[pos[:,None]-np.arange(240,-1,-1)]
    out['market_drawdown_240m']=(np.maximum.accumulate(paths,axis=1)-paths).max(axis=1)
    out['market_drawup_240m']=(paths-np.minimum.accumulate(paths,axis=1)).max(axis=1)
    frame=pd.DataFrame(out,index=created.index)
    if np.isinf(frame.to_numpy()).any():raise ValueError('Infinite market signals')
    return frame


def platform_signals(created,events):
    times=_times(created)
    if events.empty or events.created_at_utc.isna().any() or not events.option_type.isin([0,1]).all():
        raise ValueError('Invalid platform events')
    first=events.created_at_utc.min().value
    streams={label:np.sort(events.loc[events.option_type.eq(side),'created_at_utc'].dt.as_unit('ns').astype('int64'))
             for side,label in [(0,'call'),(1,'put')]}
    def count(stamps,start,end):
        return (np.searchsorted(stamps,end,side='left')-np.searchsorted(stamps,start,side='left')).astype(float)
    out={}
    for w in (15,60,240):
        start=times-w*60_000_000_000
        for label,stamps in streams.items():
            values=count(stamps,start,times);values[start<first]=np.nan
            out[f'platform_{label}_count_{w}m']=values
    total=out['platform_put_count_60m']+out['platform_call_count_60m']
    out['platform_total_count_60m']=total
    out['platform_put_share_60m']=_divide(out['platform_put_count_60m'],total)
    prior=sum(count(stamps,times-300*60_000_000_000,times-60*60_000_000_000) for stamps in streams.values())
    ratio=_divide(4*total,prior)
    ratio[times-300*60_000_000_000<first]=np.nan
    out['platform_activity_ratio_60m']=ratio
    return pd.DataFrame(out,index=created.index)


def historical_context(pool,signals,day,ids,min_history=30):
    """Thresholds use only already settled orders created in the past 30 days."""
    if not pool.X.index.equals(signals.index):raise ValueError('Misaligned signals')
    meta=pool.metadata;day=pd.Timestamp(day)
    if day.tzinfo is None or day!=day.floor('D'):raise ValueError('Expected UTC daily origin')
    mask=meta.created_at_utc.ge(day-pd.Timedelta(days=30))&meta.created_at_utc.lt(day)&meta.closed_at_utc.lt(day)
    history=pool.X.loc[mask];today=pool.X.loc[ids]
    threshold=pd.Series(np.nan,index=ids);source=pd.Series('unavailable',index=ids)
    high=pd.DataFrame(np.nan,index=ids,columns=signals.columns);low=high.copy()
    fallback=pd.DataFrame(False,index=ids,columns=signals.columns)
    for (side,tenor),query in today.groupby(['option_type','product_duration_seconds']):
        direction=history.index[history.option_type.eq(side)]
        cell=history.index[history.option_type.eq(side)&history.product_duration_seconds.eq(tenor)]
        chosen=cell if len(cell)>=min_history else direction
        if len(chosen)>=min_history:
            threshold.loc[query.index]=pool.y.loc[chosen].quantile(.9)
            source.loc[query.index]='direction_tenor' if len(cell)>=min_history else 'direction_fallback'
        for feature in signals:
            values=signals.loc[cell,feature].dropna()
            if len(values)<min_history:
                values=signals.loc[direction,feature].dropna();fallback.loc[query.index,feature]=True
            if len(values)>=min_history:
                high.loc[query.index,feature]=values.quantile(.8)
                low.loc[query.index,feature]=values.quantile(.2)
    return {'tail_threshold':threshold,'tail_source':source,'high_threshold':high,'low_threshold':low,'feature_fallback':fallback}


def rank_correlation(x,y,groups=None,control=None):
    frame=pd.DataFrame({'x':x,'y':y})
    if groups is not None:frame['group']=groups
    if control is not None:frame['control']=control
    frame=frame.replace([np.inf,-np.inf],np.nan).dropna()
    if len(frame)<3:return float('nan')
    names=['x','y']+(['control'] if control is not None else [])
    ranks=frame[names].rank(method='average')
    if groups is not None:ranks-=ranks.groupby(frame['group']).transform('mean')
    else:ranks-=ranks.mean()
    if control is not None:
        c=ranks.control.to_numpy();denom=c@c
        if denom>1e-12:
            for key in ['x','y']:ranks[key]-=c*(ranks[key].to_numpy()@c/denom)
    a,b=ranks.x.to_numpy(),ranks.y.to_numpy();scale=np.sqrt((a@a)*(b@b))
    return float(np.clip(a@b/scale,-1,1)) if scale>1e-12 else float('nan')


def alert_statistics(actual,alert):
    a=np.asarray(actual,bool);flag=np.asarray(alert,bool)
    if len(a)!=len(flag) or not len(a):raise ValueError('Nonempty aligned alerts required')
    tp=int((a&flag).sum());tails=int(a.sum());n=len(a);alerts=int(flag.sum())
    rate=tails/n;precision=tp/alerts if alerts else float('nan')
    return {'orders':n,'tail_orders':tails,'alerts':alerts,'true_positives':tp,'tail_rate':rate,
            'alert_rate':alerts/n,'precision':precision,'lift':precision/rate if rate else float('nan'),
            'recall':tp/tails if tails else float('nan'),
            'false_positive_rate':int((flag&~a).sum())/(n-tails) if n>tails else float('nan')}


def build_observations(pool,signals,base):
    rows=[];highs=[];lows=[];fallbacks=[];coverage={}
    for phase,bounds in PERIODS.items():
        ids_in_phase=[];threshold_sources=[]
        for day,train,ids in daily_batches(pool,'validation',30,bounds):
            context=historical_context(pool,signals,day,ids)
            X=pool.X.loc[ids]
            frame=pd.DataFrame({'phase':phase,'day':day.isoformat(),'direction':X.option_type,
                'tenor':X.product_duration_seconds,'payout':pool.y.loc[ids],
                'cash_payout':pool.y.loc[ids]*X.quantity_btc,'base':base.reindex(ids),
                'tail_threshold':context['tail_threshold'],'tail_source':context['tail_source']},index=ids)
            frame['tail']=frame.payout.gt(frame.tail_threshold).where(frame.tail_threshold.notna())
            frame['base_error']=frame.payout-frame.base
            frame['day_tenor']=frame.day+'|'+frame.tenor.astype(str)
            rows.append(frame);highs.append(context['high_threshold']);lows.append(context['low_threshold']);fallbacks.append(context['feature_fallback'])
            ids_in_phase.extend(str(i) for i in ids);threshold_sources.extend(context['tail_source'])
        coverage[phase]={'orders':len(ids_in_phase),'scoring_ids_sha256':fingerprint(ids_in_phase),
                         'tail_threshold_sources':pd.Series(threshold_sources).value_counts().to_dict()}
    return pd.concat(rows),pd.concat(highs),pd.concat(lows),pd.concat(fallbacks),coverage


def aggregate_associations(observations,signals,high,low,fallback):
    associations=[];alerts=[]
    masks={'all_june':pd.Series(True,index=observations.index),'early':observations.phase.eq('early'),
           'late':observations.phase.eq('late'),
           'without_june24_25':~pd.to_datetime(observations.day).dt.day.isin([24,25])}
    for period,mask in masks.items():
        for side in (0,1):
            obs=observations.loc[mask&observations.direction.eq(side)]
            for feature in signals:
                values=signals.loc[obs.index,feature];valid=values.notna();f=obs.loc[valid];x=values.loc[valid]
                raw=rank_correlation(x,f.payout);within=rank_correlation(x,f.payout,f.day_tenor)
                row={'period':period,'direction':side,'feature':feature,'orders':len(obs),'valid_orders':len(f),
                     'missing_fraction':float((~valid).mean()),'zero_fraction':float(x.eq(0).mean()),'unique_values':x.nunique(),
                     'spearman_payout':raw,'spearman_cash':rank_correlation(x,f.cash_payout),
                     'date_tenor_adjusted':within,'base_available_orders':int(f.base.notna().sum()),
                     'spearman_base_error':rank_correlation(x,f.base_error),
                     'date_tenor_base_adjusted':rank_correlation(x,f.payout,f.day_tenor,f.base),
                     'paired_days':f.day.nunique()}
                if period=='all_june':
                    daily=[];loo=[]
                    for day in sorted(f.day.unique()):
                        same=f.day.eq(day)
                        if same.sum()>=8:daily.append(rank_correlation(x.loc[same],f.loc[same,'payout']))
                        loo.append(rank_correlation(x.loc[~same],f.loc[~same,'payout']))
                    d=np.array(daily);d=d[np.isfinite(d)];leave=np.array(loo);leave=leave[np.isfinite(leave)]
                    row.update(daily_valid_days=len(d),daily_positive_fraction=float((d>0).mean()) if len(d) else np.nan,
                               daily_median_spearman=float(np.median(d)) if len(d) else np.nan,
                               leave_one_day_min=float(leave.min()) if len(leave) else np.nan,
                               leave_one_day_max=float(leave.max()) if len(leave) else np.nan)
                associations.append(row)
                for label,threshold,op in [('high',high,np.greater),('low',low,np.less)]:
                    selected=obs.loc[obs['tail'].notna()]
                    if selected.empty:continue
                    available=values.loc[selected.index].notna()&threshold.loc[selected.index,feature].notna()
                    flag=available&pd.Series(op(values.loc[selected.index],threshold.loc[selected.index,feature]),index=selected.index)
                    stats=alert_statistics(selected['tail'].astype(bool),flag)
                    alerts.append({'period':period,'direction':side,'feature':feature,'signal_group':label,
                        **stats,'signal_available_orders':int(available.sum()),
                        'signal_unavailable_orders':int((~available).sum()),
                        'unavailable_tail_orders':int(selected.loc[~available,'tail'].sum()),
                        'days_with_alerts':selected.loc[flag,'day'].nunique(),
                        'days_with_true_positives':selected.loc[flag&selected['tail'].astype(bool),'day'].nunique(),
                        'threshold_fallback_fraction':float(fallback.loc[selected.index[available],feature].mean())})
    return pd.DataFrame(associations),pd.DataFrame(alerts)


def feature_dictionary(names):
    records=[]
    for name in names:
        family='platform_activity' if name.startswith('platform_') else ('contract_control' if name in ['log_moneyness','remaining_seconds','quantity_btc'] else ('market_activity' if any(k in name for k in ['trades','trade_ratio','quote','sell_share']) else 'price_dynamics'))
        unit=('orders' if 'count' in name else 'trades/minute' if 'trades_per_min' in name else
              'USDC/minute' if 'quote_per_min' in name else 'USDC/trade' if 'mean_trade' in name else
              'seconds' if name=='remaining_seconds' else 'BTC' if name=='quantity_btc' else
              'ratio' if any(k in name for k in ['ratio','share']) else 'log_return')
        records.append({'feature':name,'family':family,'unit':unit})
    return pd.DataFrame(records)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    args=parser.parse_args(argv);started=perf_counter();guard=install_development_guard(DEFAULT_OUTPUT)
    root=run_destination(PROJECT/'outputs/diagnostics',args.run_id)
    source=study_snapshot(DEFAULT_OUTPUT);own=file_digest(Path(__file__))
    pool=load_validation_pool({'cohort':'risk','use_base':False})
    common=load_validation_pool({'cohort':'benchmark','use_base':True})
    base=common.X.api_base_price_per_btc_usdc
    market,market_info=read_market_prefix(DEFAULT_DATA/'btc_market_history.csv','2026-07-01T00:00:00Z')
    events,event_info=read_order_prefix(DEFAULT_DATA/'pp_transactions_training.csv','2026-07-01T00:00:00Z')
    print('Building completed-market and preceding-platform signals',flush=True)
    signals=pd.concat([market_signals(pool.metadata.created_at_utc,market),platform_signals(pool.metadata.created_at_utc,events),
                       pool.X[['log_moneyness','remaining_seconds','quantity_btc']]],axis=1)
    if len(signals.columns)!=41 or not signals.index.equals(pool.X.index):raise ValueError('Unexpected signal set')
    obs,high,low,fallback,coverage=build_observations(pool,signals,base)
    print(f'Aggregating {len(obs)} orders, 41 signals, two directions; no model fitting',flush=True)
    associations,alerts=aggregate_associations(obs,signals,high,low,fallback)
    daily=[]
    for (phase,day,side),f in obs.groupby(['phase','day','direction']):
        eligible=f['tail'].notna()
        daily.append({'phase':phase,'day':day,'direction':side,'orders':len(f),'tail_eligible':int(eligible.sum()),
            'tail_orders':int(f.loc[eligible,'tail'].sum()),'zero_tail_threshold_orders':int(f.tail_threshold.eq(0).sum()),
            'tail_threshold_median':f.tail_threshold.median(),'payout_mean_per_btc':f.payout.mean(),
            'cash_total_usdc':f.cash_payout.sum(),'tail_cash_usdc':f.loc[f['tail'].eq(True),'cash_payout'].sum(),
            'base_available_orders':int(f.base.notna().sum()),'direction_fallback_orders':int(f.tail_source.eq('direction_fallback').sum())})
    if study_snapshot(DEFAULT_OUTPUT)!=source or file_digest(Path(__file__))!=own:raise ValueError('Source changed during analysis')
    root.mkdir(parents=True,exist_ok=False)
    for name in ['metrics','provenance','reports','figures']:(root/name).mkdir()
    tables={'associations':associations,'tail_alerts':alerts,'daily_tail_summary':pd.DataFrame(daily),
            'signal_dictionary':feature_dictionary(signals.columns)}
    for name,table in tables.items():table.to_csv(root/f'metrics/{name}.csv',index=False,float_format='%.12g')
    provenance={'version':'payout_signals_v1','periods':PERIODS,'source':source,'script_sha256':own,
        'market_prefix':market_info,'order_event_prefix':event_info,'coverage':coverage,
        'target':'payout/BTC; tail greater than prior30d settled same direction/tenor q90, fallback direction if n<30',
        'signal_alerts':'strict >historicalq80 or <historicalq20; same history/fallback; all June thresholds replayed before subset diagnostics',
        'rank_controls':'global average ranks demeaned within day x nominal tenor; optional residualization on similarly demeaned Base rank',
        'test_csv_read':False,'blocked_test_opens':guard['blocked_test_opens'],'opened_development_files':sorted(guard['opened_development_files']),
        'elapsed_seconds':perf_counter()-started,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        'limitations':['Descriptive multiple-signal exploration; no independent-order p values or causal/profit claim',
            'June repeatedly observed; phases and June24/25 removal are sensitivity checks, not fresh confirmation',
            'Overlapping market/order windows and clustered payouts reduce effective event count',
            'Market trade counts are fills, not submitted/cancelled orders or order-book depth',
            'Platform export excludes negative-markup rows and lacks IDs; frequency is observed-record proxy, not total demand',
            'Signal ranking does not establish incremental out-of-sample model value; no model trained',
            'Risk cohort differs from prior common-Base model studies; only Base diagnostics use available common rows']}
    atomic_json(root/'provenance/source.json',provenance)
    with ZipFile(root/'provenance/source_snapshot.zip','w',ZIP_DEFLATED) as archive:
        files=set(source['code_sha256'])|{str(Path(__file__).relative_to(PROJECT)),
            'tests/test_payout_signal_analysis.py','docs/plans/2026-10-09-payout-signals.md'}
        files.update(str(p.relative_to(PROJECT)) for directory in ['pp_data','pp_audit'] for p in (PROJECT/directory).glob('*.py'))
        for name in sorted(files):archive.write(PROJECT/name,name)
    paths=list((root/'metrics').iterdir())+list((root/'provenance').iterdir())
    atomic_json(root/'provenance/manifest.json',{'producer':'payout_signal_analysis','status':'complete',
        'files':{str(p.relative_to(root)):file_digest(p) for p in paths}})
    print(json.dumps({'output':str(root),'coverage':coverage,'elapsed_seconds':provenance['elapsed_seconds']},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
