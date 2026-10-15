"""Comprehensive descriptive Base regimes; old data only, never train or price candidates."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pp_data.config import PROJECT,DEFAULT_DATA,DEFAULT_OUTPUT
from pp_data.features import BASE_FEATURE
from pp_models.data import load_pool
from pp_models.directional_features import augment_pool
from pp_models.put_final_io import read_test_market_prefix
from pp_models.two_stage_study import tail_context
from scripts.analyze_base_regimes import install_guards,digest,check_previous
from scripts.payout_signal_analysis import market_signals,platform_signals,rank_correlation
from scripts.base_regime_statistics import MONTHS,calendar,metrics,group_statistics,simultaneous,stability,exact_tenor_control

START=pd.Timestamp('2026-04-01T00:00Z')
END=pd.Timestamp('2026-07-31T17:11:48Z')
DEFAULT_RESULT=PROJECT/'outputs/diagnostics/base_regimes_comprehensive_20261014'


def old_events():
    path=DEFAULT_DATA/'pp_transactions_training.csv'
    expected='3f046081325c40eaee5139429dbf85e668f98b281c26f2f7bb523eb9781ccbd5'
    if digest(path)!=expected:raise ValueError('Old training file changed')
    raw=pd.read_csv(path,usecols=['createdAt','optionType'])
    times=pd.to_datetime(raw.createdAt,utc=True,format='mixed',errors='raise')
    side=pd.to_numeric(raw.optionType,errors='coerce')
    valid=times.notna()&side.isin([0,1])&times.lt(END)
    return pd.DataFrame({'created_at_utc':times[valid],'option_type':side[valid].astype(int)}),{
        'sha256':expected,'source_rows':len(raw),'events':int(valid.sum()),'columns':['createdAt','optionType'],
        'meaning':'all supplied old creation records; not guaranteed full platform demand'}


def five_minute_run(created,market):
    bars=market.set_index('open_time_utc').close.resample('5min').agg(['last','count'])
    prices=bars['last'].where(bars['count'].eq(5))
    ret=np.log(prices).diff().to_numpy()
    out={}
    available=(bars.index+pd.Timedelta(minutes=5)).as_unit('ns').astype('int64')
    pos=np.searchsorted(available,created.dt.as_unit('ns').astype('int64'),side='right')-1
    if (pos<1).any():raise ValueError('Missing five-minute run history')
    for side,sign in [('call',1),('put',-1)]:
        streak=0;values=[]
        for value in ret:
            streak=streak+1 if np.isfinite(value) and sign*value>0 else 0
            values.append(streak)
        out[side]=np.asarray(values)[pos]
    return out


def feature_frame(pool,market,events):
    augmented,info=augment_pool(pool,market)
    x=augmented.X;meta=pool.metadata;created=meta.created_at_utc
    f=pd.DataFrame({'base':x[BASE_FEATURE],'actual':pool.y,'quantity':x.quantity_btc,
        'day':created.dt.floor('D'),'month':created.dt.strftime('%Y-%m'),
        'side':x.option_type.map({0:'call',1:'put'}),'tenor':x.product_duration_seconds.astype(int),
        'created':created,'closed':meta.closed_at_utc},index=x.index)
    f['error']=f.base-f.actual
    activity=market_signals(created,market);platform=platform_signals(created,events)
    runs=five_minute_run(created,market);sign=1-2*x.option_type
    descriptors=[]
    def add(name,value,family,definition):
        f[name]=value;descriptors.append({'feature':name,'family':family,'definition':definition})
    add('remaining_fraction',x.remaining_seconds/x.product_duration_seconds,'contract','remaining / nominal seconds')
    add('position_raw',sign*x.log_moneyness,'contract','direction * log(spot/strike)')
    add('position',sign*x.log_moneyness/np.maximum(x.total_vol_240m*np.sqrt(x.remaining_seconds/60),1e-6),
        'contract','directional log moneyness / past total RMS240 / sqrt(remaining minutes)')
    add('quantity_level',x.quantity_btc,'exposure','insured BTC quantity; diagnostic, not automatic volume pricing')
    add('base_level',f.base,'base_diagnostic','observed Base per BTC; not a permitted multiplier predictor')
    add('base_scale',f.base/np.maximum(x.market_price_usdc_per_btc*x.total_vol_240m*np.sqrt(x.remaining_seconds/60),1e-6),
        'base_diagnostic','Base / (spot * total RMS240 * sqrt(remaining minutes)); diagnostic only')
    for w in [15,60,240,1440]:
        if w==15:
            up,down=activity.market_up_rms_15m,activity.market_down_rms_15m
            total=np.sqrt(up**2+down**2);ret=activity.market_return_15m
        else:
            up,down=x[f'up_vol_{w}m'],x[f'down_vol_{w}m'];total=x[f'total_vol_{w}m'];ret=x[f'log_return_{w}m']
        risk=np.where(x.option_type.eq(0),up,down)
        add(f'risk_vol_{w}',risk,'volatility',f'risk-direction RMS over all {w} minutes')
        add(f'trend_{w}',sign*ret/np.maximum(total*np.sqrt(w),1e-6),'trend',f'direction return over {w} minutes / contemporaneous historical RMS sqrt(w)')
    add('vol_ratio_15_60',activity.market_vol_ratio_15m_60m,'dynamics','total RMS15 / RMS60')
    add('vol_ratio_60_240',x.total_vol_60m/np.maximum(x.total_vol_240m,1e-12),'dynamics','total RMS60 / RMS240')
    add('vol_change',activity.market_vol_change_60m/np.maximum(x.total_vol_240m,1e-12),'dynamics','(RMS60 minus previous nonoverlapping RMS60) / RMS240')
    add('direction_share',f.risk_vol_240**2/np.maximum(x.total_vol_240m**2,1e-24),'dynamics','risk-direction share of total squared returns240')
    add('risk_jump',np.where(x.option_type.eq(0),activity.market_max_up_jump_60m,activity.market_max_down_jump_60m),
        'path','largest completed risk-direction minute log jump over60m')
    add('risk_excursion',np.where(x.option_type.eq(0),activity.market_drawup_240m,activity.market_drawdown_240m),
        'path','largest ordered risk-direction log excursion over240m')
    add('risk_run',np.where(x.option_type.eq(0),runs['call'],runs['put']),'path','consecutive completed5m bars moving in payout-risk direction')
    for w in [15,60]:
        add(f'market_trades_{w}',activity[f'market_trades_per_min_{w}m'],'activity',f'actual executed market trades/min over{w}m')
    add('market_activity_ratio',activity.market_trade_ratio_60m,'activity','recent60m executions/min divided by previous nonoverlapping24h')
    add('market_pressure',sign*(1-2*activity.market_sell_share_60m),'activity','risk-direction signed net taker flow / total quote volume60m')
    for w in [15,60,240]:
        add(f'platform_same_{w}',np.where(x.option_type.eq(0),platform[f'platform_call_count_{w}m'],platform[f'platform_put_count_{w}m']),
            'platform',f'same-direction source creations [t-{w}m,t), no same-time records')
    add('platform_ratio',platform.platform_activity_ratio_60m,'platform','recent60m total order rate / previous240m rate; missing if denominator zero')
    f['utc_session']=created.dt.hour.floordiv(6).map({0:'00-06',1:'06-12',2:'12-18',3:'18-24'})
    f['weekend']=np.where(created.dt.dayofweek>=5,'weekend','weekday')
    f['tenor_band']=pd.cut(x.product_duration_seconds/3600,[0,1,4,np.inf],labels=['<=1h','2-4h','>=8h']).astype(str)
    f['tail90']=False;f['tail95']=False
    thresholds=[]
    for month in MONTHS:
        origin=pd.Timestamp(month+'-01',tz='UTC')
        h=x.index[created.lt(origin)&meta.closed_at_utc.lt(origin)]
        q=x.index[f.month.eq(month)]
        t90,source=tail_context(pool,h,q)
        f.loc[q,'tail90']=pool.y.loc[q].gt(t90)
        for (side,tenor),sub in x.loc[q].groupby(['option_type','product_duration_seconds']):
            cell=h[x.loc[h,'option_type'].eq(side)&x.loc[h,'product_duration_seconds'].eq(tenor)]
            if len(cell)<30:cell=h[x.loc[h,'option_type'].eq(side)]
            t95=pool.y.loc[cell].quantile(.95) if len(cell)>=30 else np.nan
            f.loc[sub.index,'tail95']=pool.y.loc[sub.index].gt(t95)
            thresholds.append({'month':month,'side':int(side),'tenor':int(tenor),'history_orders':len(cell),
                'q90':t90.loc[sub.index].iloc[0],'q95':t95,'source':source.loc[sub.index].iloc[0]})
    return f,pd.DataFrame(descriptors),pd.DataFrame(thresholds),info


def assign_bins(frame,dictionary):
    f=frame.copy();rows=[]
    for side,sub in f.groupby('side'):
        warm=sub.loc[sub.created.lt(START)&sub.closed.lt(START)]
        for name in dictionary.feature:
            values=warm[name].dropna()
            if name.startswith('trend_'):
                cuts=[-.5,.5];labels=['favorable','neutral','adverse']
            elif name.startswith('platform_same_'):
                cuts=[.5,2.5];labels=['none','1-2','3+']
            elif name=='risk_run':
                cuts=[.5,1.5];labels=['none','one','2+']
            else:
                cuts=sorted(set(values.quantile([1/3,2/3]).tolist())) if len(values)>=30 else []
                labels=['low','middle','high'] if len(cuts)==2 else ['low','high'] if len(cuts)==1 else ['unpartitioned']
            b=pd.cut(sub[name],[-np.inf,*cuts,np.inf],labels=labels).astype('object').fillna('missing')
            f.loc[sub.index,name+'_bin']=b
            rows.append({'side':side,'feature':name,'reference_orders':len(values),'cutpoints':json.dumps(cuts),'labels':'|'.join(labels)})
    singles=['tenor','tenor_band','utc_session','weekend']+[v+'_bin' for v in dictionary.feature]
    pairs=[('tenor_band',f'{v}_bin') for v in ['position','risk_vol_240','trend_60','risk_jump','market_trades_60','platform_same_60']]
    pairs += [(f'{a}_bin',f'{b}_bin') for a,b in [('position','risk_vol_240'),('position','trend_60'),('position','vol_ratio_60_240'),
        ('risk_vol_240','trend_60'),('risk_vol_240','vol_ratio_60_240'),('risk_vol_240','risk_jump'),
        ('risk_vol_240','market_trades_60'),('risk_vol_240','platform_same_60'),
        ('trend_60','vol_ratio_60_240'),('trend_60','platform_same_60'),('trend_60','risk_run'),('market_trades_60','platform_same_60')]]
    triples=[('tenor_band','position_bin','trend_60_bin'),('risk_vol_240_bin','trend_60_bin','vol_ratio_60_240_bin')]
    registry=[('overall',[])]+[(v,[v]) for v in singles]+[(' x '.join(p),list(p)) for p in pairs+triples]
    return f,pd.DataFrame(rows),registry


def scan(frame,registry):
    calendars={m:calendar(m) for m in MONTHS}
    records=[];linears=[];stored={}
    for side,sub in frame.groupby('side'):
        for family,columns in registry:
            labels=sub[columns].astype(str).agg('|'.join,axis=1) if columns else pd.Series('all',index=sub.index)
            for label in sorted(labels.unique()):
                group=sub.loc[labels.eq(label)]
                for month,p in group.groupby('month'):
                    row,draws,linear=group_statistics(p,*calendars[month])
                    row.update(side=side,family=family,group=label,month=month)
                    records.append(row);linears.append(linear)
                    stored[side,family,label,month]=(row,draws,linear)
    monthly=pd.DataFrame(records)
    devrows=[];devlinear=[]
    for side,family,label in monthly[['side','family','group']].drop_duplicates().itertuples(index=False,name=None):
        items=[stored.get((side,family,label,m)) for m in MONTHS[:3]]
        if any(v is None for v in items):continue
        part=pd.DataFrame([v[0] for v in items]);row=part.iloc[0].to_dict()
        row.update(month='development',orders=int(part.orders.sum()),days=int(part.days.sum()),supported=bool(part.supported.all()))
        for key in ['mean_error','day_equal_bias','mse_slope','mae_slope']:
            row[key]=part[key].mean()
        row['rmse']=np.sqrt(np.mean(part.rmse**2));row['mae']=part.mae.mean()
        for key in ['cash_bias','cash_under','cash_over']:row[key]=part[key].sum()
        for key,metric in [('bias','ci'),('mse','mse_slope_ci'),('mae','mae_slope_ci')]:
            from scripts.base_regime_statistics import interval
            mean=np.mean(np.stack([v[1][key] for v in items]),axis=0)
            lo,hi,n=interval(mean)
            row[metric+'_low'],row[metric+'_high']=lo,hi
            if key=='bias':row['valid_repeats']=n
        # Only retain explicitly meaningful macro columns; no copied monthly rates.
        keep=['side','family','group','month','orders','days','supported','mean_error','day_equal_bias','rmse','mae',
            'cash_bias','cash_under','cash_over','mse_slope','mae_slope','ci_low','ci_high','valid_repeats',
            'mse_slope_ci_low','mse_slope_ci_high','mae_slope_ci_low','mae_slope_ci_high']
        devrows.append({k:row[k] for k in keep});devlinear.append(np.mean([v[2] for v in items],axis=0))
    development=pd.DataFrame(devrows)
    devmask=monthly.month.ne('2026-07');dev_indices=np.flatnonzero(devmask)
    combined=pd.concat([monthly.loc[devmask],development],ignore_index=True)
    combined,devcritical=simultaneous(combined,[linears[i] for i in dev_indices]+devlinear)
    for col in ['sim_low','sim_high','sim_critical']:
        monthly.loc[devmask,col]=combined[col].iloc[:len(dev_indices)].to_numpy()
        development[col]=combined[col].iloc[len(dev_indices):].to_numpy()
    july_indices=np.flatnonzero(~devmask)
    july,julycritical=simultaneous(monthly.loc[~devmask].reset_index(drop=True),[linears[i] for i in july_indices])
    for col in ['sim_low','sim_high','sim_critical']:monthly.loc[~devmask,col]=july[col].to_numpy()
    stats=stability(monthly,development)
    return monthly,development,stats,{'development_critical':devcritical,'july_critical':julycritical,
        'development_eligible_comparisons':int(combined.supported.sum()),'july_eligible_comparisons':int(july.supported.sum()),
        'method':'95% max absolute studentized centered linearized-ratio block bootstrap; approximate simultaneous intervals',
        'point_intervals':'ordinary resampled ratios, >=1900 finite draws'}


def correlations(frame,dictionary):
    rows=[];pairs=[];names=dictionary.feature.tolist()
    for side,sub in frame.groupby('side'):
        for month in ['all',*MONTHS]:
            p=sub if month=='all' else sub.loc[sub.month.eq(month)]
            groups=p.month.astype(str)+'|'+p.tenor.astype(str)
            for name in names:
                rows.append({'side':side,'month':month,'feature':name,'orders':int(p[name].notna().sum()),
                    'rank_bias_correlation':rank_correlation(p[name],p.error),
                    'within_month_tenor_rank_correlation':rank_correlation(p[name],p.error,groups=groups)})
        corr=sub[names].corr(method='spearman')
        for i,a in enumerate(names):
            for b in names[i+1:]:
                if abs(corr.loc[a,b])>=.75:pairs.append({'side':side,'feature_a':a,'feature_b':b,'spearman':corr.loc[a,b]})
    return pd.DataFrame(rows),pd.DataFrame(pairs)


def analyze(output):
    if output.exists():raise ValueError('Output exists; use a fresh directory')
    started=perf_counter();guards=install_guards()
    protected={}
    for directory in ['outputs/model_studies/put_final_july_20261014','outputs/model_studies/base_multiplier_20261014','outputs/diagnostics/base_regimes_20261014']:
        for p in (PROJECT/directory).rglob('*'):
            if p.is_file():protected[str(p.relative_to(PROJECT))]=digest(p)
    protected['outputs/preprocessing/manifest.json']=digest(DEFAULT_OUTPUT/'manifest.json')
    market,market_info=read_test_market_prefix(DEFAULT_DATA/'btc_market_history.csv')
    if market_info['prefix_sha256']!='cbf5d491d167ac377beb844dc9794e4f5e9986f72859035426f7789e67924b50':raise ValueError('Old market prefix changed')
    events,event_info=old_events()
    pool=load_pool('test',{'cohort':'benchmark','use_base':True})
    frame,dictionary,tail_thresholds,feature_info=feature_frame(pool,market,events)
    frame,cuts,registry=assign_bins(frame,dictionary)
    warmup=frame.loc[frame.created.lt(START)&frame.closed.lt(START)]
    frame=frame.loc[frame.created.ge(START)&frame.created.lt(END)].copy()
    assert len(frame)==5494
    print(f'Features {len(dictionary)}, registered partitions {len(registry)}; scanning both directions',flush=True)
    monthly,dev,stable,multiple=scan(frame,registry)
    print(f'Monthly cells {len(monthly)}; checking exact-tenor structure',flush=True)
    control=exact_tenor_control(frame,warmup,registry)
    association,redundancy=correlations(frame,dictionary)
    outcome=[]
    for (month,side),p in frame.groupby(['month','side']):
        for grouping,mask in [('all',pd.Series(True,index=p.index)),('zero',p.actual.eq(0)),('positive',p.actual.gt(0)),
            ('body90',~p.tail90),('tail90',p.tail90),('tail95',p.tail95)]:
            sub=p.loc[mask]
            if len(sub):outcome.append({'month':month,'side':side,'cohort':grouping,**metrics(sub)})
    # Independently reconcile to prior Base audit, never score a new model.
    old=pd.read_csv(PROJECT/'outputs/diagnostics/base_regimes_20261014/metrics/overall_by_month.csv')
    for (month,side),p in frame.groupby(['month','side']):
        ref=old.loc[old.month.eq(month)&old.side.eq(side)].iloc[0];got=metrics(p)
        for key in ['orders','days','mean_error','rmse','mae','cash_bias','cash_under','cash_over']:
            np.testing.assert_allclose(got[key],ref[key],rtol=1e-10,atol=1e-7)
    for (side,month,family),p in monthly.groupby(['side','month','family']):
        expected=monthly.loc[monthly.side.eq(side)&monthly.month.eq(month)&monthly.family.eq('overall')].iloc[0]
        assert p.orders.sum()==expected.orders
        np.testing.assert_allclose(np.average(p.mean_error,weights=p.orders),expected.mean_error,atol=1e-9)
        np.testing.assert_allclose(p.cash_bias.sum(),expected.cash_bias,atol=1e-8)
    risk=load_pool('test',{'cohort':'risk','use_base':False});coverage=[]
    for month in MONTHS:
        for side,label in [(0,'call'),(1,'put')]:
            ids=risk.X.index[risk.X.option_type.eq(side)&risk.metadata.created_at_utc.dt.strftime('%Y-%m').eq(month)]
            present=ids.isin(frame.index)
            coverage.append({'month':month,'side':label,'risk_orders':len(ids),'base_orders':int(present.sum()),
                'base_coverage':present.mean(),'actual_mean_with_base':risk.y.loc[ids[present]].mean(),
                'actual_mean_without_base':risk.y.loc[ids[~present]].mean()})
    for directory in ['metrics','provenance','reports']:(output/directory).mkdir(parents=True)
    tables={'monthly_regimes':monthly,'development_regimes':dev,'stability':stable,'exact_tenor_control':control,
        'outcome_decomposition':pd.DataFrame(outcome),'feature_associations':association,'redundancy':redundancy,
        'coverage':pd.DataFrame(coverage)}
    for name,table in tables.items():table.to_csv(output/'metrics'/f'{name}.csv',index=False,float_format='%.15g')
    cuts.to_csv(output/'provenance/fixed_feature_cuts.csv',index=False)
    dictionary.to_csv(output/'provenance/feature_dictionary.csv',index=False)
    tail_thresholds.to_csv(output/'provenance/frozen_tail_thresholds.csv',index=False)
    source_paths=[Path(__file__),PROJECT/'scripts/base_regime_statistics.py',PROJECT/'scripts/payout_signal_analysis.py',
        PROJECT/'pp_models/directional_features.py',PROJECT/'pp_models/put_final_io.py',PROJECT/'docs/plans/2026-10-14-base-regimes-comprehensive.md']
    sources={str(p.relative_to(PROJECT)):digest(p) for p in source_paths}
    for name,value in protected.items():assert digest(PROJECT/name)==value
    assert guards=={'official_validation_open_attempts':0,'model_fit_predict_calls':0}
    provenance={'created_at_utc':datetime.now(timezone.utc).isoformat(),'orders':len(frame),'warmup_orders':len(warmup),
        'market':market_info,'events':event_info,'feature_check':feature_info,'guards':guards,'multiple_comparisons':multiple,
        'registry':registry,'source_sha256':sources,'protected_sha256':protected,'elapsed_seconds':perf_counter()-started,
        'feature_count':len(dictionary),'monthly_cells':len(monthly),'unique_regimes':len(stable),'no_model_training':True,
        'existing_artifacts_unchanged':True}
    (output/'provenance/analysis.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    manifest={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    (output/'provenance/manifest.json').write_text(json.dumps({'files':manifest},indent=2)+'\n')
    print(json.dumps({'output':str(output),'elapsed_seconds':perf_counter()-started,
        'tier_counts':stable.groupby(['side','tier']).size().to_dict().__str__()},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_RESULT)
    with threadpool_limits(limits=1):analyze(p.parse_args().output.resolve())
