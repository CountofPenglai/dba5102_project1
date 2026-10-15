"""Fixed old-data macro event diagnostics. No model training or official holdout access."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pp_data.config import PROJECT, DEFAULT_OUTPUT
from pp_data.features import BASE_FEATURE
from pp_calendar.calendar import assert_coverage
from scripts.analyze_base_regimes import digest, install_guards, check_previous
from scripts.analyze_size_and_base_lag import load_data, START, END
from scripts.macro_event_statistics import crosses, MinuteMarket, summarize_events, matched_event

ROOT=PROJECT/'external_data/macro_events'
RESULT=PROJECT/'outputs/diagnostics/macro_events_20261014_v2'
MARKET_METRICS=['rms_bp','up_rms_bp','down_rms_bp','return_bp']
OUTCOMES=['base','actual','error','abs_error','sq_error','tail','positive','unit_shortfall',
          'future_rms_bp','future_risk_rms_bp']


def event_table(calendar, include_press=False):
    rows=[]
    for row in calendar['events']:
        t=pd.Timestamp(row['event_time_utc'])
        if START <= t < END and (include_press or row['kind']!='press_conference'):
            rows.append({**row,'time':t,'month':t.strftime('%Y-%m'),
                         'known':pd.Timestamp(row['known_by_utc'])})
    return pd.DataFrame(rows)


def pseudo_times(time, same_month=False):
    local=time.tz_convert('America/New_York')
    for days in [-14,-7,7,14]:
        t=(local+pd.DateOffset(days=days)).tz_convert('UTC')
        if START<=t<END and (not same_month or t.month==time.month):yield t


def overlaps_announcements(start,end,events):
    return bool(events.time.ge(start).mul(events.time.le(end)).any())


def market_study(engine, events):
    rows=[]
    for e in events.itertuples():
        for horizon in [15,60,240]:
            delta=pd.Timedelta(minutes=horizon)
            before=engine.window(e.time-delta,e.time);after=engine.window(e.time,e.time+delta)
            controls=[];control_times=[]
            for t in pseudo_times(e.time):
                if overlaps_announcements(t-delta,t+delta,events):continue
                w=engine.window(t,t+delta)
                if w['complete']:
                    controls.append(w);control_times.append(t.isoformat())
            row={'event_id':e.event_id,'family':e.family,'event_time':e.time,'month':e.month,
                 'horizon_minutes':horizon,'controls':len(controls),'control_times':'|'.join(control_times),
                 'complete':after['complete'],
                 'control_design':'same_NY_clock_weekday_plusminus_7_14d'}
            for metric in MARKET_METRICS:
                row[metric+'_pre']=before[metric]
                row[metric+'_post']=after[metric]
                row[metric+'_control']=np.mean([c[metric] for c in controls]) if len(controls)>=2 else np.nan
                row[metric+'_difference']=row[metric+'_post']-row[metric+'_control']
                if metric!='return_bp':
                    den=row[metric+'_control']
                    row[metric+'_ratio']=row[metric+'_post']/den if den>0 else np.nan
            rows.append(row)
    per=pd.DataFrame(rows);summaries=[]
    for family in ['all',*sorted(per.family.unique())]:
        for period in ['all','development','2026-04','2026-05','2026-06','2026-07']:
            g=per if family=='all' else per.loc[per.family.eq(family)]
            if period=='development':g=g.loc[g.month.lt('2026-07')]
            elif period!='all':g=g.loc[g.month.eq(period)]
            for h,s in g.groupby('horizon_minutes'):
                for metric in MARKET_METRICS:
                    for stat in ['difference']+(['ratio'] if metric!='return_bp' else []):
                        summaries.append({'family':family,'period':period,'horizon_minutes':h,
                                          'metric':metric,'statistic':stat,
                                          'positive_reference':1. if stat=='ratio' else 0.,
                                          **summarize_events(s[metric+'_'+stat],1. if stat=='ratio' else 0.)})
    return per,pd.DataFrame(summaries)


def annotate(frame,engine,events,all_events):
    f=frame.copy();f['cross_any']=False;f['known_cross_any']=False;f['recent_event_4h']=False
    for family in ['CPI','NFP','FOMC']:f['cross_'+family]=False
    coverage=[]
    for e in all_events.itertuples():
        mask=crosses(f.created,f.remaining,e.time)
        known=mask & f.created.ge(e.known)
        if e.kind!='press_conference':
            f['cross_any']|=mask;f['known_cross_any']|=known;f['cross_'+e.family]|=mask
            f['recent_event_4h']|=f.created.ge(e.time)&f.created.lt(e.time+pd.Timedelta(hours=4))
        coverage.append({'event_id':e.event_id,'family':e.family,'kind':e.kind,'event_time':e.time,
                         'known_by':e.known,'risk_orders':int(mask.sum()),'base_orders':int((mask&f.base.notna()).sum()),
                         'known_risk_orders':int(known.sum()),'known_base_orders':int((known&f.base.notna()).sum()),
                         'ended_before_event':int((mask&f.closed.lt(e.time)).sum()),
                         'actual_active_at_event':int((mask&f.closed.ge(e.time)).sum())})
    f['press_only']=False
    for e in all_events.loc[all_events.kind.eq('press_conference')].itertuples():
        f['press_only']|=crosses(f.created,f.remaining,e.time)&~f.cross_any
    future=[engine.window(t,t+pd.Timedelta(seconds=float(s))) for t,s in zip(f.created,f.remaining)]
    f['future_rms_bp']=[r['rms_bp'] for r in future]
    up=np.array([r['up_rms_bp'] for r in future]);down=np.array([r['down_rms_bp'] for r in future])
    f['future_risk_rms_bp']=np.where(f.side.eq('put'),down,up)
    f['future_market_complete']=[r['complete'] for r in future]
    f['abs_error']=abs(f.error);f['sq_error']=f.error**2
    f['unit_shortfall']=(-f.error).clip(lower=0);f['tail']=f.tail_monthly
    return f,pd.DataFrame(coverage)


def basic_metrics(f):
    g=f.loc[f.base.notna()]
    return {'risk_orders':len(f),'base_orders':len(g),'base_coverage':len(g)/len(f),
            'days':f.day.nunique(),'base_mean':g.base.mean(),'actual_mean_common':g.actual.mean(),
            'bias':g.error.mean(),'rmse':np.sqrt(g.sq_error.mean()),'mae':g.abs_error.mean(),
            'tail_rate_risk':f['tail'].mean(),'positive_rate_risk':f.positive.mean(),
            'quantity_median':f.quantity.median(),'remaining_hours_mean':f.remaining.mean()/3600,
            'future_complete':f.future_market_complete.sum(),'future_rms_bp':f.future_rms_bp.mean(),
            'future_risk_rms_bp':f.future_risk_rms_bp.mean(),
            'actual_cash_common':(g.quantity*g.actual).sum(),'base_cash':(g.quantity*g.base).sum(),
            'cash_bias':(g.quantity*g.error).sum(),'cash_under':(g.quantity*g.unit_shortfall).sum()}


def raw_order_study(f,events):
    rows=[]
    for period in ['all','development','2026-04','2026-05','2026-06','2026-07']:
        p=f if period=='all' else f.loc[f.month.lt('2026-07') if period=='development' else f.month.eq(period)]
        for side,g in p.groupby('side'):
            for family in ['any','CPI','NFP','FOMC','press_only']:
                flag=family if family=='press_only' else 'cross_'+family
                for value,sub in g.groupby(flag):
                    ev=events.loc[events.kind.eq('press_conference') if family=='press_only' else events.kind.ne('press_conference')]
                    if family not in ['any','press_only']:ev=ev.loc[ev.family.eq(family)]
                    n=sum(crosses(sub.created,sub.remaining,e.time).any() for e in ev.itertuples()) if value else 0
                    rows.append({'period':period,'side':side,'family':family,'crosses':bool(value),
                                 'events':n,**basic_metrics(sub)})
    return pd.DataFrame(rows)


def lead_bin(f,t):
    out=f.copy()
    lead=(t-out.created).dt.total_seconds()/3600
    out['lead_bin']=pd.cut(lead,[0,.25,1,4,24.01],labels=['<=15m','15m-1h','1-4h','4-24h']).astype(str)
    return out


def matched_order_study(f,events):
    common=f.loc[f.base.notna()];rows=[]
    for e in events.itertuples():
        treated=common.loc[crosses(common.created,common.remaining,e.time)&common.created.ge(e.known)]
        treated=lead_bin(treated,e.time)
        controls=[]
        for t in pseudo_times(e.time,same_month=True):
            if overlaps_announcements(t-pd.Timedelta(hours=24),t+pd.Timedelta(hours=4),events):continue
            mask=crosses(common.created,common.remaining,t)&~common.cross_any&~common.recent_event_4h
            c=lead_bin(common.loc[mask],t);c['pseudo_time']=t
            if len(c):controls.append(c)
        control=pd.concat(controls) if controls else common.iloc[:0].assign(lead_bin='',pseudo_time=pd.NaT)
        for side,t in treated.groupby('side'):
            c=control.loc[control.side.eq(side)]
            for level,columns in [('tenor_lead',['tenor','lead_bin']),
                                  ('tenor_lead_position_vol',['tenor','lead_bin','position_bin','risk_vol_240_bin'])]:
                result=matched_event(t,c,columns,OUTCOMES)
                if result is None:
                    result={'treated_orders':len(t),'treated_matched':0,'control_rows':0,'coverage':0.,'matched_cells':0}
                    for metric in OUTCOMES:
                        result[metric+'_support']=0
                        for suffix in ['treated','control','difference']:result[metric+'_'+suffix]=np.nan
                rows.append({'event_id':e.event_id,'family':e.family,'month':e.month,'side':side,
                             'control':level,'pseudo_dates':c.pseudo_time.nunique(),**result})
    per=pd.DataFrame(rows);summary=[]
    if per.empty:return per,pd.DataFrame()
    for (side,control),g in per.groupby(['side','control']):
        for family in ['all','CPI','NFP','FOMC']:
            for period in ['all','development','2026-04','2026-05','2026-06','2026-07']:
                p=g if family=='all' else g.loc[g.family.eq(family)]
                if period=='development':p=p.loc[p.month.lt('2026-07')]
                elif period!='all':p=p.loc[p.month.eq(period)]
                if p.empty:continue
                for metric in OUTCOMES:
                    summary.append({'side':side,'control':control,'family':family,'period':period,'metric':metric,
                                    'treated_matched':p.treated_matched.sum(),'treated_orders':p.treated_orders.sum(),
                                    'eligible_events':len(p),
                                    'outcome_treated_supported':p[metric+'_support'].sum(),
                                    'coverage':p.treated_matched.sum()/p.treated_orders.sum(),
                                    'treated_mean':p[metric+'_treated'].mean(),'control_mean':p[metric+'_control'].mean(),
                                    **summarize_events(p[metric+'_difference'])})
    return per,pd.DataFrame(summary)


def run(destination):
    if destination.exists():raise FileExistsError(destination)
    guards=install_guards();started=perf_counter()
    prior=PROJECT/'outputs/model_studies/put_final_july_20261014'
    files=json.loads((prior/'provenance/manifest.json').read_text())['files']
    for name,expected in files.items():
        if digest(prior/name)!=expected:raise ValueError('Prior result changed: '+name)
    old_manifest=digest(DEFAULT_OUTPUT/'manifest.json')
    calendar=json.loads((ROOT/'calendar.json').read_text())
    assert_coverage(calendar,START.isoformat(),END.isoformat())
    events=event_table(calendar);all_events=event_table(calendar,True)
    frame,market,info,_,_,_,_=load_data()
    if len(frame)!=6615 or frame.base.notna().sum()!=5494:raise ValueError('Old cohorts changed')
    check=frame.loc[frame.base.notna()].assign(**{BASE_FEATURE:frame.base},quantity_btc=frame.quantity,
                                             option_type=frame.side.map({'call':0,'put':1}))
    verified=check_previous(check,prior)
    engine=MinuteMarket(market)
    market_events,market_summary=market_study(engine,events)
    f,coverage=annotate(frame,engine,events,all_events)
    if not f.cross_any.equals(f.known_cross_any):raise ValueError('Historical calendar knowledge incomplete')
    raw=raw_order_study(f,all_events)
    matched,matched_summary=matched_order_study(f,events)
    event_orders=[]
    for e in all_events.itertuples():
        g=f.loc[crosses(f.created,f.remaining,e.time)]
        for side,p in g.groupby('side'):
            event_orders.append({'event_id':e.event_id,'family':e.family,'kind':e.kind,'month':e.month,
                                 'side':side,**basic_metrics(p)})
    tables={'event_coverage':coverage,'market_events':market_events,'market_summary':market_summary,
            'order_groups':raw,'order_events':pd.DataFrame(event_orders),'matched_events':matched,
            'matched_summary':matched_summary}
    for name,table in tables.items():
        path=destination/'metrics'/(name+'.csv');path.parent.mkdir(parents=True,exist_ok=True)
        table.to_csv(path,index=False)
    assert digest(DEFAULT_OUTPUT/'manifest.json')==old_manifest
    assert all(digest(prior/name)==expected for name,expected in files.items())
    provenance={'created_at_utc':datetime.now(timezone.utc).isoformat(),'calendar_as_of':calendar['as_of_utc'],
                'calendar_sha256':digest(ROOT/'calendar.json'),'source_bundle_sha256':digest(ROOT/'sources/2026-10-14/verified_schedule.json'),
                'old_data':info,'cohort':{'risk':len(f),'benchmark':int(f.base.notna().sum()),
                'cross_primary_risk':int(f.cross_any.sum()),'cross_primary_base':int((f.cross_any&f.base.notna()).sum()),
                'press_only_risk':int(f.press_only.sum()),'future_market_complete':int(f.future_market_complete.sum())},
                'main_events':len(events),'all_events_including_press':len(all_events),'previous_base_groups_verified':verified,
                'preprocessing_manifest_sha256':old_manifest,'prior_outputs_unchanged':True,'guards':guards,
                'elapsed_seconds':perf_counter()-started,'rows':{k:len(v) for k,v in tables.items()},
                'information_rule':'planned remaining horizon; all crossed main events have dated prior publication evidence; actual close diagnostic only',
                'inference':'descriptive; event bootstrap only at n>=8; no multiple-comparison correction'}
    path=destination/'provenance/run.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    manifest={str(p.relative_to(destination)):digest(p) for p in destination.rglob('*') if p.is_file()}
    code=['pp_calendar/calendar.py','scripts/update_macro_calendar.py','scripts/macro_event_statistics.py','scripts/analyze_macro_events.py']
    (destination/'provenance/manifest.json').write_text(json.dumps({'outputs':manifest,'code':{n:digest(PROJECT/n) for n in code}},indent=2)+'\n')
    print(json.dumps(provenance,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=RESULT)
    with threadpool_limits(limits=1):run(parser.parse_args().output)
