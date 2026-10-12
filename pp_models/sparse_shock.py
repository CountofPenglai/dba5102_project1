"""Causal, high-threshold shock gate and event-shrunk payout correction."""
import json
import numpy as np
import pandas as pd
from pp_data.config import PROJECT

CONFIG_PATH=PROJECT/'config/sparse_shock_v1.json'


def load_settings():
    c=json.loads(CONFIG_PATH.read_text())
    if (c['version']!='sparse_shock_v1' or c['protocol']!='config/validation_protocol_v2.json'
            or c['bar_minutes']!=5 or c['return_minutes']!=15 or c['parameter_search']
            or not 0<c['off_quantile']<c['on_quantile']<c['extreme_quantile']<1
            or not 0<c['off_absolute_return']<c['on_absolute_return']<c['extreme_absolute_return']
            or any(c[k]<=0 for k in ['scale_bars','scale_floor','threshold_history_days','threshold_min_bars',
                'confirm_bars','exit_bars','lease_minutes','recovery_half_life_minutes','state_zero_cutoff',
                'contract_otm_sigma_cutoff','amplitude_min_events','amplitude_min_orders',
                'amplitude_shrink_events','amplitude_cap','fixed_probe_amplitude'])):
        raise ValueError('Invalid fixed sparse-shock settings')
    return c


def run_state_machine(signal,c):
    """One direction; zero-state starts define event IDs, including recovery tails."""
    if (not signal.index.is_unique or signal.index.tz is None
            or not signal.index.is_monotonic_increasing
            or not signal.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=5)).all()):
        raise ValueError('Shock signal must be a complete chronological 5-minute grid')
    state=0.;active=False;strong_run=weak_run=0;event=event_counter=0;last_strong=None
    rows=[];decay=2**(-c['bar_minutes']/c['recovery_half_life_minutes'])
    columns=['z','r5','r15','on','extreme','off']
    for stamp,values in zip(signal.index,signal[columns].to_numpy(float)):
        if not np.isfinite(values).all():
            state=0.;active=False;strong_run=weak_run=0;event=0;last_strong=None
            rows.append((np.nan,0,'unavailable',np.nan,np.nan));continue
        z,r5,r15,on,extreme,off=values
        strong=z>=on and r15>=c['on_absolute_return'] and r5>0
        sudden=z>=extreme and r15>=c['extreme_absolute_return'] and r5>0
        weak=z<off or r15<c['off_absolute_return'] or r5<=0
        strong_run=strong_run+1 if strong else 0
        weak_run=weak_run+1 if weak else 0
        activate=sudden or strong_run>=c['confirm_bars']
        delay=np.nan
        if (not active and activate) or (active and strong):
            if not active:
                delay=0 if sudden else c['confirm_bars']-1
                if state==0:
                    event_counter+=1;event=event_counter
            state=1. if sudden else .5+.5*np.clip((z-on)/max(extreme-on,1e-12),0,1)
            active=True;last_strong=stamp;weak_run=0;phase='strong'
        elif active:
            expired=(stamp-last_strong).total_seconds()/60>=c['lease_minutes']
            if weak_run>=c['exit_bars'] or expired:
                active=False;state*=decay;phase='recovery'
            else:phase='held'
        elif state>0:
            state*=decay;phase='recovery'
        else:phase='quiet'
        if state<c['state_zero_cutoff']:
            state=0.;event=0;phase='quiet';last_strong=None
        age=(stamp-last_strong).total_seconds()/60 if last_strong is not None else np.nan
        rows.append((state,event,phase,delay,age))
    return pd.DataFrame(rows,index=signal.index,
        columns=['state','event_id','phase','delay_bars','since_strong_minutes'])


def market_states(market,c):
    if (market.empty or market.open_time_utc.dt.tz is None or market.open_time_utc.duplicated().any()
            or not market.open_time_utc.is_monotonic_increasing
            or not market.open_time_utc.eq(market.open_time_utc.dt.floor('min')).all()
            or not market.close_time_utc.eq(market.open_time_utc+pd.Timedelta(milliseconds=59999)).all()
            or not np.isfinite(market.close).all() or market.close.le(0).any()):
        raise ValueError('Expected audited chronological complete minute candles')
    bars=market.assign(bar=market.open_time_utc.dt.floor('5min')).groupby('bar').agg(price=('close','last'),n=('close','size'))
    bars=bars.reindex(pd.date_range(bars.index.min(),bars.index.max(),freq='5min'))
    price=bars.price.where(bars.n.eq(5));r5=np.log(price).diff()
    r15=r5.rolling(3,min_periods=3).sum()
    scale=np.sqrt(r5.pow(2).shift(1).rolling(c['scale_bars'],min_periods=c['scale_bars']).mean()).clip(lower=c['scale_floor'])
    raw=pd.DataFrame({'r5':r5.to_numpy(),'r15':r15.to_numpy(),
        'z':(r15/(scale*np.sqrt(3))).to_numpy()},index=bars.index+pd.Timedelta(minutes=5))
    result=[]
    for direction,sign in [('call',1),('put',-1)]:
        f=raw*sign
        for name in ['on','extreme','off']:f[name]=np.nan
        f['threshold_bars']=0
        for day in raw.index.floor('D').unique():
            hist=f.loc[(f.index>=day-pd.Timedelta(days=c['threshold_history_days']))&(f.index<day),'z'].dropna()
            today=f.index.floor('D')==day
            f.loc[today,'threshold_bars']=len(hist)
            if len(hist)<c['threshold_min_bars']:continue
            for key,q in [('on','on_quantile'),('extreme','extreme_quantile'),('off','off_quantile')]:
                f.loc[today,key]=hist.quantile(c[q])
        state=run_state_machine(f,c)
        result.append(pd.concat([state,f[['on','extreme','off','threshold_bars']]],axis=1).add_prefix(direction+'_'))
    return pd.concat(result,axis=1)


def contract_weight(X,c):
    if (not np.isfinite(X[['option_type','log_moneyness','realized_vol_60m','remaining_seconds']]).all().all()
            or not X.option_type.isin([0,1]).all() or X.remaining_seconds.le(0).any() or X.realized_vol_60m.lt(0).any()):
        raise ValueError('Invalid contract inputs')
    uncertainty=np.maximum(X.realized_vol_60m*np.sqrt(X.remaining_seconds/3600),1e-6)
    aligned=(1-2*X.option_type)*X.log_moneyness/uncertainty
    k=c['contract_otm_sigma_cutoff']
    return ((aligned+k)/k).clip(0,1)


def order_states(pool,states,c):
    created=pool.metadata.created_at_utc
    if not created.index.equals(pool.X.index):raise ValueError('Misaligned order times')
    left=pd.DataFrame({'created':created.array,'position':np.arange(len(created))})
    right=states.rename_axis('available').reset_index()
    match=pd.merge_asof(left.sort_values('created'),right,left_on='created',right_on='available',direction='backward')
    match=match.set_index('position').reindex(range(len(created))).set_axis(created.index)
    age=(match.created-match.available).dt.total_seconds()
    if not age.ge(0).all() or not age.lt(300).all():raise ValueError('Missing latest completed shock state')
    out=pd.DataFrame(index=created.index)
    for field in ['state','event_id','phase','since_strong_minutes']:
        out[field]=np.where(pool.X.option_type.eq(0),match['call_'+field],match['put_'+field])
    if not np.isfinite(out.state.astype(float)).all():raise ValueError('Shock features unavailable; no silent order exclusion')
    out['contract_weight']=contract_weight(pool.X,c)
    out['effective_state']=out.state.astype(float)*out.contract_weight
    out['available_at_utc']=match.available
    return out


def fit_amplitude(base,state,y,event,c):
    items=[base,state,y,event]
    if (not all(isinstance(v,pd.Series) and base.index.equals(v.index) for v in items)
            or not base.index.is_unique or not all(np.isfinite(v).all() for v in items)
            or base.lt(0).any() or y.lt(0).any() or state.lt(0).any() or state.gt(1).any()
            or ((state>0)&(event<=0)).any()):raise ValueError('Invalid causal amplitude inputs')
    mask=(state>0)&(base>0)
    b,g,t,e=(v.loc[mask] for v in items)
    count=int(e.nunique());orders=int(mask.sum())
    detail={'events':count,'orders':orders,'raw_amplitude':0.,'fallback':False,'capped':False}
    if count<c['amplitude_min_events'] or orders<c['amplitude_min_orders']:
        detail['fallback']=True;return 0.,detail
    w=1/e.map(e.value_counts()).to_numpy(float);x=(b*g).to_numpy(float)
    raw=max(0.,float(np.sum(w*x*(t-b))/np.sum(w*x*x)))
    shrunk=raw*count/(count+c['amplitude_shrink_events'])
    detail.update(raw_amplitude=raw,capped=shrunk>c['amplitude_cap'])
    return min(shrunk,c['amplitude_cap']),detail


def market_diagnostics(states,protocol):
    start,end=pd.Timestamp(protocol['validation_start']),pd.Timestamp(protocol['validation_end'])
    selected=states[(states.index>=start)&(states.index<end)]
    rows=[]
    for direction in ['call','put']:
        f=selected.filter(regex='^'+direction+'_').rename(columns=lambda n:n[len(direction)+1:]).copy()
        f['month']=f.index.strftime('%Y-%m')
        for period in ['all']+protocol['report_months']:
            part=f if period=='all' else f[f.month.eq(period)]
            active=part.state.gt(0);valid=part.state.notna()
            events=part.loc[active].groupby('event_id').size()*5
            delays=part.delay_bars.dropna()*5
            rows.append({'month':period,'direction':direction,'bars':len(part),'available_bars':int(valid.sum()),
                'affected_bars':int(active.sum()),'affected_fraction':float(active.mean()),
                'strong_bars':int(part.phase.eq('strong').sum()),'recovery_bars':int(part.phase.eq('recovery').sum()),
                'events_intersecting_period':len(events),'event_minutes_median':float(events.median()) if len(events) else np.nan,
                'event_minutes_p95':float(events.quantile(.95)) if len(events) else np.nan,
                'event_minutes_max':float(events.max()) if len(events) else np.nan,
                'confirmation_delay_minutes_mean':float(delays.mean()) if len(delays) else np.nan,
                'recovery_age_minutes_p95':float(part.loc[part.phase.eq('recovery'),'since_strong_minutes'].quantile(.95)),
                'minimum_threshold_history_bars':int(part.threshold_bars.min())})
    return pd.DataFrame(rows)
