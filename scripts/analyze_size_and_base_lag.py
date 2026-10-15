"""Old-data size, payout and Base lead/lag diagnostics; aggregates only, no training."""
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
from pp_models.put_final_io import read_test_market_prefix
from scripts.analyze_base_regimes import digest,install_guards
from scripts.analyze_base_regimes_comprehensive import feature_frame,assign_bins,old_events,START,END
from scripts.base_regime_statistics import MONTHS
from scripts.size_lag_statistics import size_thresholds,matched_daily,contrast_summary,lag_pairs,select_events,block_weights

RESULT=PROJECT/'outputs/diagnostics/size_base_lag_20261014'
PERIODS=[('all',START,END),('development',START,pd.Timestamp('2026-07-01',tz='UTC'))]+[
    (m,pd.Timestamp(m+'-01',tz='UTC'),min(pd.Timestamp(m+'-01',tz='UTC')+pd.offsets.MonthBegin(1),END)) for m in MONTHS]
PHASES=[('pre24_4',-24,-4),('pre4_1',-4,-1),('shock1',-1,0),('post0_4',0,4),('post4_24',4,24),('post24_48',24,48)]


def load_data():
    market,info=read_test_market_prefix(DEFAULT_DATA/'btc_market_history.csv')
    assert info['prefix_sha256']=='cbf5d491d167ac377beb844dc9794e4f5e9986f72859035426f7789e67924b50'
    pool=load_pool('test',{'cohort':'risk','use_base':False})
    common=load_pool('test',{'cohort':'benchmark','use_base':True})
    pool.X[BASE_FEATURE]=common.X[BASE_FEATURE].reindex(pool.X.index)
    events,eventinfo=old_events()
    f,dictionary,thresholds,_=feature_frame(pool,market,events)
    f,cuts,_=assign_bins(f,dictionary)
    f['spot']=pool.X.market_price_usdc_per_btc
    f['remaining']=pool.X.remaining_seconds
    f['premium']=pool.metadata.recorded_premium_usdc
    f['holding_hours']=(f.closed-f.created).dt.total_seconds()/3600
    f['base_normalized']=f.base/f.spot/np.sqrt(f.remaining/60)*10000
    f['actual_normalized']=f.actual/f.spot/np.sqrt(f.remaining/60)*10000
    f['payout_bp']=f.actual/f.spot*10000
    f['positive']=f.actual.gt(0).astype(float)
    f['cash']=f.actual*f.quantity;f['base_cash']=f.base*f.quantity
    f['notional']=f.spot*f.quantity
    f['tail_monthly']=f.tail90.astype(float)
    f['large_rolling'],sizecuts=size_thresholds(f,START,END)
    f['large_fixed']=False;fixed=[]
    for side,p in f.groupby('side'):
        warm=p.loc[p.created.lt(START)&p.closed.lt(START)]
        cut=warm.quantity.quantile(.9)
        f.loc[p.index,'large_fixed']=p.quantity.gt(cut)
        fixed.append({'side':side,'quantity_q90':cut,'history':len(warm)})
    lookup=sizecuts.set_index(['day','side','tenor']).payout_q90
    t=pd.MultiIndex.from_frame(f[['day','side','tenor']])
    cut=lookup.reindex(t).to_numpy()
    f['tail_daily']=np.where(np.isfinite(cut),f.actual.to_numpy()>cut,np.nan)
    f=f.loc[f.created.ge(START)&f.created.lt(END)].copy()
    return f,market,{'market':info,'events':eventinfo},sizecuts,pd.DataFrame(fixed),cuts,thresholds


def group_metrics(g):
    cash=g.cash.sum();q=g.quantity.sum();base=g.base.notna()
    return {'orders':len(g),'days':g.day.nunique(),'quantity_median':g.quantity.median(),'quantity_sum':q,
        'positive_rate':g.positive.mean(),'unit_mean':g.actual.mean(),'positive_unit_mean':g.loc[g.positive.eq(1),'actual'].mean(),
        'payout_bp_mean':g.payout_bp.mean(),'tail_daily_rate':g.tail_daily.mean(),'tail_monthly_rate':g.tail_monthly.mean(),
        'cash':cash,'cash_per_btc':cash/q,'cash_per_notional_bp':cash/g.notional.sum()*10000,
        'premium':g.premium.sum(),'payout_to_recorded_premium':cash/g.premium.sum() if g.premium.sum()>0 else np.nan,
        'base_orders':base.sum(),'common_cash':g.loc[base,'cash'].sum(),'base_cash':g.base_cash.sum(),
        'payout_to_base':g.loc[base,'cash'].sum()/g.base_cash.sum() if g.base_cash.sum()>0 else np.nan,
        'base_mean':g.base.mean(),'bias':g.error.mean(),'holding_hours_median':g.holding_hours.median()}


def size_analysis(f):
    raw=[];contrasts=[];linears=[];concentration=[]
    for cohort,frame in [('risk',f),('benchmark',f.loc[f.base.notna()])]:
        for side,sub in frame.groupby('side'):
            for period,start,end in PERIODS:
                p=sub.loc[sub.created.ge(start)&sub.created.lt(end)]
                for frac in [.01,.05,.1]:
                    n=int(np.ceil(len(p)*frac));top=p.nlargest(n,'cash')
                    concentration.append({'cohort':cohort,'side':side,'period':period,'top_order_fraction':frac,
                        'orders':len(p),'selected':n,'cash_share':top.cash.sum()/p.cash.sum()})
                for definition in ['large_rolling','large_fixed']:
                    for large,g in p.loc[p[definition].notna()].groupby(definition):
                        raw.append({'cohort':cohort,'side':side,'period':period,'definition':definition,'large':large,
                            **group_metrics(g),'cash_share':g.cash.sum()/p.cash.sum(),'quantity_share':g.quantity.sum()/p.quantity.sum()})
            for definition in ['large_rolling','large_fixed']:
                for control,columns in [('day_tenor',['tenor']),('day_tenor_position',['tenor','position_bin']),
                    ('day_tenor_position_vol',['tenor','position_bin','risk_vol_240_bin'])]:
                    metrics=['positive','actual','payout_bp','tail_daily','tail_monthly']+(['error'] if cohort=='benchmark' else [])
                    daily=matched_daily(sub,definition,columns,metrics)
                    if daily.empty:continue
                    for period,start,end in PERIODS+[('exclude_june24_25',START,END)]:
                        d=daily.loc[(daily.index>=start)&(daily.index<end)].copy()
                        if period=='exclude_june24_25':d=d.loc[~d.index.isin(pd.date_range('2026-06-24',periods=2,tz='UTC'))]
                        if d.empty:continue
                        for metric in metrics:
                            row,linear=contrast_summary(d,metric,start,end)
                            denom=sub.loc[sub.created.ge(start)&sub.created.lt(end)]
                            if period=='exclude_june24_25':denom=denom.loc[~denom.day.isin(pd.date_range('2026-06-24',periods=2,tz='UTC'))]
                            contrasts.append({'cohort':cohort,'side':side,'period':period,'definition':definition,
                                'control':control,'metric':metric,'coverage':d.orders.sum()/len(denom),**row})
                            linears.append(linear)
    contrasts=pd.DataFrame(contrasts)
    # An approximate simultaneous interval within each calendar period, across
    # both sides, cohorts, definitions, controls and outcomes examined here.
    for period,part in contrasts.groupby('period'):
        matrix=np.column_stack([linears[i] for i in part.index]);sd=matrix.std(axis=0,ddof=1)
        eligible=part.days.ge(10).to_numpy()&part.valid_draws.ge(1900).to_numpy()
        z=np.divide(abs(matrix),sd,out=np.zeros_like(matrix),where=sd>1e-12)
        critical=np.quantile(z[:,eligible].max(axis=1),.95) if eligible.any() else np.nan
        contrasts.loc[part.index,'sim_low']=np.where(eligible,part.difference.to_numpy()-critical*sd,np.nan)
        contrasts.loc[part.index,'sim_high']=np.where(eligible,part.difference.to_numpy()+critical*sd,np.nan)
    return pd.DataFrame(raw),contrasts,pd.DataFrame(concentration)


def time_analysis(f):
    frame=f.loc[f.base.notna()];rows=[];dailyrows=[];tests=[]
    days=pd.date_range(START,END.floor('D'),inclusive='left') # final partial UTC day excluded
    for side,p in frame.groupby('side'):
        scopes={'all':p,'tenor_0.5h':p.loc[p.tenor.eq(1800)],'tenor_4h':p.loc[p.tenor.eq(14400)],
                'near_atm':p.loc[p.position_raw.abs().le(.001)]}
        for scope,g in scopes.items():
            created=g.groupby(g.created.dt.floor('D')).agg(orders=('base','size'),base=('base','mean'),actual=('actual','mean'),
                base_cash=('base_cash','sum'),cash=('cash','sum'),base_normalized=('base_normalized','mean'),
                actual_normalized=('actual_normalized','mean')).reindex(days)
            settled=g.groupby(g.closed.dt.floor('D')).agg(orders=('base','size'),actual=('actual','mean'),cash=('cash','sum'),
                actual_normalized=('actual_normalized','mean')).reindex(days)
            for clock,d in [('created',created),('settled',settled)]:
                dailyrows.append(d.reset_index(names='day').assign(side=side,scope=scope,clock=clock))
                for period,start,end in PERIODS:
                    mask=(days>=start)&(days<end);a=d.loc[mask];b=created.loc[mask]
                    for units,xname,yname in [('unit','actual','base'),('cash','cash','base_cash'),
                        ('normalized','actual_normalized','base_normalized')]:
                        for lag in range(-3,4):
                            pairs=lag_pairs(a[xname],b[yname],lag)
                            rows.append({'side':side,'scope':scope,'clock':clock,'period':period,'units':units,'lag_days':lag,
                                'pairs':len(pairs),'pearson':pairs.x.corr(pairs.y) if len(pairs)>=10 else np.nan,
                                'spearman':pairs.x.corr(pairs.y,method='spearman') if len(pairs)>=10 else np.nan})
                        # Prespecified +1 vs 0, on identical dates. Daily blocks retain missing days.
                        joined=pd.concat([a[xname].rename('x'),b[yname].rename('b0'),
                            b[yname].shift(-1).rename('b1')],axis=1)
                        valid=joined.notna().all(axis=1);j=joined.loc[valid]
                        if len(j)<15:continue
                        weights=block_weights(len(joined));arr=joined.fillna(0).to_numpy()
                        w=weights[:,valid.to_numpy()];v=arr[valid.to_numpy()]
                        n=w.sum(axis=1);s=w@v;s2=w@(v*v)
                        cov0=w@(v[:,0]*v[:,1])-s[:,0]*s[:,1]/np.maximum(n,1)
                        cov1=w@(v[:,0]*v[:,2])-s[:,0]*s[:,2]/np.maximum(n,1)
                        var=s2-s*s/np.maximum(n[:,None],1)
                        with np.errstate(invalid='ignore',divide='ignore'):
                            draws=cov1/np.sqrt(var[:,0]*var[:,2])-cov0/np.sqrt(var[:,0]*var[:,1])
                        draws=draws[(n>=10)&np.isfinite(draws)]
                        ci=np.quantile(draws,[.025,.975]) if len(draws)>=1900 else [np.nan,np.nan]
                        tests.append({'side':side,'scope':scope,'clock':clock,'period':period,'units':units,'pairs':len(j),
                            'lag1_minus_lag0':j.x.corr(j.b1)-j.x.corr(j.b0),'ci_low':ci[0],'ci_high':ci[1],'valid_draws':len(draws)})
    return pd.DataFrame(rows),pd.concat(dailyrows,ignore_index=True),pd.DataFrame(tests)


def event_analysis(f,market):
    hourly=market.set_index('open_time_utc').close.resample('h').agg(['last','count'])
    price=hourly['last'].where(hourly['count'].eq(60));price.index=price.index+pd.Timedelta(hours=1)
    returns=np.log(price).diff();eventrows=[];phaserows=[];paired=[]
    common=f.loc[f.base.notna()]
    for side,sign in [('call',1),('put',-1)]:
        signal=(sign*returns).clip(lower=0)
        for quantile in [.99,.975]:
            threshold=signal.shift(1).rolling(720,min_periods=600).quantile(quantile)
            times=select_events(signal,threshold,START,END)
            for number,t in enumerate(times):
                key={'side':side,'quantile':quantile,'event':number,'event_time':t,'month':t.strftime('%Y-%m')}
                eventrows.append({**key,'direction_return_bp':signal.loc[t]*10000,'threshold_bp':threshold.loc[t]*10000})
                p=common.loc[common.side.eq(side)];parts={}
                for phase,lo,hi in PHASES:
                    g=p.loc[p.created.ge(t+pd.Timedelta(hours=lo))&p.created.lt(t+pd.Timedelta(hours=hi))]
                    parts[phase]=g
                    if len(g):phaserows.append({**key,'phase':phase,'base_normalized':g.base_normalized.mean(),
                        'actual_normalized':g.actual_normalized.mean(),'risk_vol_240':g.risk_vol_240.mean(),**group_metrics(g)})
                before=parts['pre24_4']
                for phase,after in parts.items():
                    if phase=='pre24_4' or len(before)<2 or len(after)<2:continue
                    for control in ['raw','tenor_position']:
                        if control=='raw':groups=[(before,after,1.)]
                        else:
                            groups=[]
                            for keys,b in before.groupby(['tenor','position_bin'],observed=True):
                                a=after.loc[after.tenor.eq(keys[0])&after.position_bin.eq(keys[1])]
                                if len(a)>=2 and len(b)>=2:groups.append((b,a,float(len(b))))
                        if not groups:continue
                        total=sum(w for _,_,w in groups)
                        row={**key,'phase':phase,'control':control,'pre_orders':sum(len(b) for b,_,_ in groups),
                            'post_orders':sum(len(a) for _,a,_ in groups),'pre_coverage':sum(len(b) for b,_,_ in groups)/len(before),
                            'post_coverage':sum(len(a) for _,a,_ in groups)/len(after)}
                        for metric in ['base','actual','error','base_normalized','actual_normalized','risk_vol_240']:
                            row[metric+'_pre']=sum(b[metric].mean()*w for b,a,w in groups)/total
                            row[metric+'_post']=sum(a[metric].mean()*w for b,a,w in groups)/total
                            row[metric+'_change']=row[metric+'_post']-row[metric+'_pre']
                        paired.append(row)
    pairs=pd.DataFrame(paired);summary=[]
    for (side,q,phase,control),g in pairs.groupby(['side','quantile','phase','control']):
        for period in ['all',*MONTHS]:
            p=g if period=='all' else g.loc[g.month.eq(period)]
            if not len(p):continue
            for metric in ['base','actual','error','base_normalized','actual_normalized','risk_vol_240']:
                v=p[metric+'_change'].to_numpy();rng=np.random.default_rng(20261014)
                boot=v[rng.integers(0,len(v),size=(2000,len(v)))].mean(axis=1)
                ci=np.quantile(boot,[.025,.975]) if len(v)>=8 else [np.nan,np.nan]
                leave=(v.sum()-v)/(len(v)-1) if len(v)>1 else np.array([np.nan])
                summary.append({'side':side,'quantile':q,'phase':phase,'control':control,'period':period,'metric':metric,
                    'events':len(p),'pre_mean':p[metric+'_pre'].mean(),'post_mean':p[metric+'_post'].mean(),
                    'change':v.mean(),'median_change':np.median(v),'positive_fraction':np.mean(v>0),
                    'ci_low':ci[0],'ci_high':ci[1],'leave_event_min':np.nanmin(leave) if len(v)>1 else np.nan,
                    'leave_event_max':np.nanmax(leave) if len(v)>1 else np.nan,
                    'pre_coverage_mean':p.pre_coverage.mean(),'post_coverage_mean':p.post_coverage.mean()})
    return pd.DataFrame(eventrows),pd.DataFrame(phaserows),pairs,pd.DataFrame(summary)


def balanced_events(f,events):
    """Same events AND common contract cells across every phase of a panel."""
    records=[];summary=[];joint=[]
    phasebounds={name:(lo,hi) for name,lo,hi in PHASES}
    panels={'shock_late':['shock1','post24_48'],
        'full_path':['pre24_4','shock1','post0_4','post24_48']}
    metrics=['base','actual','error','base_normalized']
    for ev in events.itertuples():
        p=f.loc[f.base.notna()&f.side.eq(ev.side)]
        for panel,names in panels.items():
            parts={}
            for name in names:
                lo,hi=phasebounds[name]
                parts[name]=p.loc[p.created.ge(ev.event_time+pd.Timedelta(hours=lo))&p.created.lt(ev.event_time+pd.Timedelta(hours=hi))]
            if any(len(g)<2 for g in parts.values()):continue
            for control in ['raw','tenor_position']:
                if control=='raw':groups=[(parts,1.)]
                else:
                    groups=[]
                    for key,g in parts[names[0]].groupby(['tenor','position_bin'],observed=True):
                        cells={name:s.loc[s.tenor.eq(key[0])&s.position_bin.eq(key[1])] for name,s in parts.items()}
                        if all(len(v)>=2 for v in cells.values()):groups.append((cells,float(len(g))))
                if not groups:continue
                total=sum(w for _,w in groups)
                for name in names:
                    row={'side':ev.side,'quantile':ev.quantile,'event':ev.event,'month':ev.month,
                        'panel':panel,'control':control,'phase':name,
                        'orders':sum(len(g[name]) for g,w in groups),'original_orders':len(parts[name])}
                    row.update({metric:sum(g[name][metric].mean()*w for g,w in groups)/total for metric in metrics})
                    records.append(row)
    rows=pd.DataFrame(records)
    for keys,g in rows.groupby(['side','quantile','panel','control']):
        common=dict(zip(['side','quantile','panel','control'],keys))
        wide={metric:g.pivot(index=['event','month'],columns='phase',values=metric) for metric in metrics}
        for period in ['all',*MONTHS]:
            mask=np.ones(len(wide['base']),dtype=bool) if period=='all' else wide['base'].index.get_level_values('month')==period
            if not mask.any():continue
            n=int(mask.sum())
            a={k:v.loc[mask] for k,v in wide.items()}
            for phase in a['base'].columns:
                for metric in metrics:
                    v=a[metric][phase]
                    summary.append({**common,'period':period,'events':n,'phase':phase,'metric':metric,'mean':v.mean()})
            early=a['error'].shock1;late=a['error'].post24_48
            change=a['base_normalized'].post24_48-a['base_normalized'].shock1
            rng=np.random.default_rng(20261014)
            boot=change.to_numpy()[rng.integers(0,n,size=(2000,n))].mean(axis=1)
            ci=np.quantile(boot,[.025,.975]) if n>=8 else [np.nan,np.nan]
            joint.append({**common,'period':period,'events':n,'shock_under':int((early<0).sum()),
                'late_over':int((late>0).sum()),'under_then_over':int(((early<0)&(late>0)).sum()),
                'base_raw_increase':int((a['base'].post24_48>a['base'].shock1).sum()),
                'base_normalized_increase':int((change>0).sum()),'normalized_change_mean':change.mean(),
                'normalized_change_ci_low':ci[0],'normalized_change_ci_high':ci[1]})
    return rows,pd.DataFrame(summary),pd.DataFrame(joint)


def analyze(output):
    if output.exists():raise ValueError('Choose a fresh output directory')
    start=perf_counter();guards=install_guards()
    protected={str(p.relative_to(PROJECT)):digest(p) for folder in ['outputs/preprocessing',
        'outputs/diagnostics/base_regimes_comprehensive_20261014','outputs/model_studies/base_multiplier_20261014']
        for p in (PROJECT/folder).rglob('*') if p.is_file()}
    f,market,info,sizecuts,fixedcuts,cuts,tailcuts=load_data()
    print(f'Loaded {len(f)} risk orders; {f.base.notna().sum()} with Base',flush=True)
    raw,matched,concentration=size_analysis(f);print('Size contrasts complete',flush=True)
    lag,daily,lagtests=time_analysis(f);print('Calendar lag analysis complete',flush=True)
    events,phases,pairs,eventsummary=event_analysis(f,market);print('Independent market events complete',flush=True)
    balanced,balancedsummary,joint=balanced_events(f,events)
    for folder in ['metrics','provenance','reports']:(output/folder).mkdir(parents=True,exist_ok=True)
    tables={'metrics/size_groups.csv':raw,'metrics/size_matched.csv':matched,'metrics/concentration.csv':concentration,
        'metrics/daily_time_alignment.csv':daily,'metrics/lag_correlations.csv':lag,'metrics/lag1_vs_same_day.csv':lagtests,
        'metrics/market_events.csv':events,'metrics/event_phases.csv':phases,'metrics/event_pairs.csv':pairs,
        'metrics/event_summary.csv':eventsummary,'provenance/size_daily_thresholds.csv':sizecuts,
        'metrics/balanced_event_phases.csv':balanced,'metrics/balanced_event_summary.csv':balancedsummary,
        'metrics/event_sequence.csv':joint,
        'provenance/size_fixed_thresholds.csv':fixedcuts,'provenance/feature_cuts.csv':cuts,'provenance/tail_thresholds.csv':tailcuts}
    for name,table in tables.items():table.to_csv(output/name,index=False,float_format='%.12g')
    assert guards=={'official_validation_open_attempts':0,'model_fit_predict_calls':0}
    assert all(digest(PROJECT/p)==value for p,value in protected.items())
    coverage=f.groupby(['side','month']).agg(risk_orders=('actual','size'),base_orders=('base','count'))
    coverage.to_csv(output/'metrics/coverage.csv')
    provenance={'created_at':datetime.now(timezone.utc).isoformat(),'guards':guards,'seconds':perf_counter()-start,
        'risk_orders':len(f),'benchmark_orders':int(f.base.notna().sum()),'market':info,'protected_sha256':protected,
        'source_sha256':{str(p.relative_to(PROJECT)):digest(p) for p in [Path(__file__),PROJECT/'scripts/size_lag_statistics.py',
            PROJECT/'docs/plans/2026-10-14-size-and-base-lag.md']},
        'scope':'old April-July only; no training; July already explored; official new validation sealed',
        'lag_definition':'corr(outcome on calendar t, new-order Base on t+k); k>0 Base later; final partial day excluded',
        'events':'completed hourly directional return > trailing720hour quantile with at least600; 72h spacing; no outcome selection',
        'multiple_testing':'size simultaneous intervals within calendar period; lag and event CIs exploratory only'}
    (output/'provenance/analysis.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (output/'reports/README.md').write_text('# 大单与Base滞后诊断\n\n完整解释见 `docs/reports/2026-10-14-size-and-base-lag.md`。仅保存聚合，未训练/调价。\n')
    manifest={'files':{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}}
    (output/'provenance/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:provenance[k] for k in ['risk_orders','benchmark_orders','seconds','guards']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=RESULT)
    with threadpool_limits(limits=1):analyze(parser.parse_args().output)
