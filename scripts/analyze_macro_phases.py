"""Old-data macro phases, independent-event support and contract-controlled errors."""
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
from scripts.analyze_base_regimes import digest,install_guards,check_previous
from scripts.macro_event_statistics import MinuteMarket,crosses,matched_event,summarize_events
from scripts.macro_phase_statistics import PHASES,phase_mask,phase_overlap_minutes,phase_metrics,aggregate_cells,history_counts

ROOT=PROJECT/'external_data/macro_events'
CALENDAR=ROOT/'snapshots/20261014T123002Z.json'
RESULT=PROJECT/'outputs/diagnostics/macro_phases_20261014_v2'
START=pd.Timestamp('2026-01-01T05:00Z')
DEVELOPMENT=pd.Timestamp('2026-04-01T00:00Z')
END=pd.Timestamp('2026-07-31T17:11:48Z')
PERIODS=['warmup','development','study','2026-01','2026-02','2026-03','2026-04','2026-05','2026-06','2026-07']
METRICS=['total','call','put','return']
MATCH_OUTCOMES=['error','abs_error','sq_error','future_risk_rms_bp']


def select_period(f,period):
    if period=='warmup':return f.loc[f.month.lt('2026-04')]
    if period=='development':return f.loc[f.month.ge('2026-04')&f.month.lt('2026-07')]
    if period=='study':return f.loc[f.month.ge('2026-04')]
    return f.loc[f.month.eq(period)]


def load_inputs():
    market,info=read_test_market_prefix(DEFAULT_DATA/'btc_market_history.csv')
    if info['prefix_sha256']!='cbf5d491d167ac377beb844dc9794e4f5e9986f72859035426f7789e67924b50':
        raise ValueError('Changed historical market prefix')
    pool=load_pool('test',{'cohort':'risk','use_base':False})
    base=load_pool('test',{'cohort':'benchmark','use_base':True}).X[BASE_FEATURE]
    augmented,_=augment_pool(pool,market);x=augmented.X;meta=pool.metadata
    f=pd.DataFrame({'base':base.reindex(x.index),'actual':pool.y,'quantity':x.quantity_btc,
        'created':meta.created_at_utc,'closed':meta.closed_at_utc,'remaining':x.remaining_seconds,
        'tenor':x.product_duration_seconds.astype(int),'side':x.option_type.map({0:'call',1:'put'})})
    f['month']=f.created.dt.strftime('%Y-%m');f['day']=f.created.dt.floor('D')
    f['position']=(1-2*x.option_type)*x.log_moneyness/np.maximum(x.total_vol_240m*np.sqrt(x.remaining_seconds/60),1e-6)
    f['risk_vol']=np.where(x.option_type.eq(0),x.up_vol_240m,x.down_vol_240m)
    cuts=[]
    for side,g in f.groupby('side'):
        warm=g.loc[g.created.lt(DEVELOPMENT)&g.closed.lt(DEVELOPMENT)]
        for name in ['position','risk_vol']:
            boundaries=warm[name].quantile([1/3,2/3]).tolist()
            f.loc[g.index,name+'_bin']=pd.cut(g[name],[-np.inf,*boundaries,np.inf],labels=['low','middle','high']).astype(str)
            cuts.append({'side':side,'feature':name,'reference_orders':len(warm),'cuts':json.dumps(boundaries)})
    f['error']=f.base-f.actual;f['abs_error']=abs(f.error);f['sq_error']=f.error**2
    engine=MinuteMarket(market)
    future=[engine.window(t,t+pd.Timedelta(seconds=float(s))) for t,s in zip(f.created,f.remaining)]
    f['future_risk_rms_bp']=[r['up_rms_bp'] if side=='call' else r['down_rms_bp'] for side,r in zip(f.side,future)]
    calendar=json.loads(CALENDAR.read_text());events=[]
    for row in calendar['events']:
        time=pd.Timestamp(row['event_time_utc'])
        if START<=time<END and row['kind']!='press_conference':
            events.append({**row,'time':time,'month':time.strftime('%Y-%m'),'known_time':pd.Timestamp(row['known_by_utc'])})
    events=pd.DataFrame(events)
    if len(f)!=7055 or f.base.notna().sum()!=5934 or len(events)!=19:raise ValueError('Changed old cohort')
    return f,market,engine,events,info,pd.DataFrame(cuts)


def pseudo_times(event):
    local=event.tz_convert('America/New_York')
    for days in [-14,-7,7,14]:
        t=(local+pd.DateOffset(days=days)).tz_convert('UTC')
        if START<=t<END:yield t


def overlaps_influence(start,end,events):
    return bool(((events.time-pd.Timedelta(minutes=60)<end)&(events.time+pd.Timedelta(minutes=240)>start)).any())


def market_study(engine,events):
    rows=[]
    for e in events.itertuples():
        for phase,(lo,hi) in PHASES.items():
            start=e.time+pd.Timedelta(minutes=lo);end=e.time+pd.Timedelta(minutes=hi)
            observed=engine.window(start,end);controls=[];times=[]
            for t in pseudo_times(e.time):
                a=t+pd.Timedelta(minutes=lo);b=t+pd.Timedelta(minutes=hi)
                if overlaps_influence(a,b,events):continue
                w=engine.window(a,b)
                if w['complete']:controls.append(w);times.append(t.isoformat())
            row={'event_id':e.event_id,'family':e.family,'month':e.month,'time':e.time,'phase':phase,
                 'complete':observed['complete'],'controls':len(controls),'control_times':'|'.join(times)}
            for label,name in zip(METRICS,['rms_bp','up_rms_bp','down_rms_bp','return_bp']):
                row[label+'_observed']=observed[name]
                row[label+'_control']=np.mean([c[name] for c in controls]) if len(controls)>=2 else np.nan
                row[label+'_difference']=row[label+'_observed']-row[label+'_control']
                den=row[label+'_control']
                if label!='return':row[label+'_ratio']=row[label+'_observed']/den if den>0 else np.nan
            rows.append(row)
    per=pd.DataFrame(rows);summary=[]
    for family in ['all','CPI','NFP','FOMC']:
        for period in PERIODS:
            p=select_period(per,period)
            if family!='all':p=p.loc[p.family.eq(family)]
            for phase,g in p.groupby('phase'):
                for metric in METRICS:
                    for statistic in ['difference']+(['ratio'] if metric!='return' else []):
                        summary.append({'family':family,'period':period,'phase':phase,'metric':metric,'statistic':statistic,
                            'calendar_events':len(g),'complete_market_events':int(g.complete.sum()),
                            **summarize_events(g[metric+'_'+statistic],1. if statistic=='ratio' else 0.)})
    return per,pd.DataFrame(summary)


def label_for_event(f,event):
    out=f.copy();out['event_id']=event.event_id;out['event_time']=event.time
    out['known']=out.created.ge(event.known_time)
    out['cross_primary']=crosses(out.created,out.remaining,event.time)
    out['cross_press']=crosses(out.created,out.remaining,event.time+pd.Timedelta(minutes=30)) if event.family=='FOMC' else False
    planned=out.created+pd.to_timedelta(out.remaining,unit='s')
    out['lead_minutes']=(event.time-out.created).dt.total_seconds()/60
    out['future_post_release_minutes']=((planned-out.created.clip(lower=event.time)).dt.total_seconds()/60).clip(lower=0)
    out['cross_post_release_minutes']=((planned-event.time).dt.total_seconds()/60).where(out.cross_primary)
    return out


def quote_study(f,events):
    cells=[];overlaps=[];panels=[];cross_panels=[];cross_cells=[]
    for event in events.itertuples():
        labeled=label_for_event(f,event)
        crossing=labeled.loc[labeled.cross_primary].copy();crossing['phase']='cross_primary'
        cross_panels.append(crossing)
        for side in ['call','put']:
            g=crossing.loc[crossing.side.eq(side)]
            cross_cells.append({'event_id':event.event_id,'family':event.family,'month':event.month,
                                'side':side,'phase':'cross_primary',**phase_metrics(g)})
        for phase in PHASES:
            mask=phase_mask(labeled.created,event.time,phase)
            phase_frame=labeled.loc[mask].copy();phase_frame['phase']=phase
            if not phase_frame.known.all():raise ValueError('Missing prior schedule evidence at quote time')
            panels.append(phase_frame)
            overlap=phase_overlap_minutes(labeled.created,labeled.remaining,event.time,phase)
            for side in ['call','put']:
                g=phase_frame.loc[phase_frame.side.eq(side)]
                key={'event_id':event.event_id,'family':event.family,'month':event.month,'time':event.time,'phase':phase,'side':side}
                cells.append({**key,**phase_metrics(g)})
                g=labeled.loc[overlap.gt(0)&labeled.side.eq(side)]
                overlaps.append({**key,'risk_orders':len(g),'base_orders':int(g.base.notna().sum()),
                                 'known_orders':int(g.known.sum()),'overlap_minutes_sum':overlap.loc[g.index].sum()})
    per=pd.DataFrame(cells);overlap_per=pd.DataFrame(overlaps);panel=pd.concat(panels)
    if panel.index.has_duplicates:raise ValueError('A quote belongs to multiple primary event windows')
    summary=[];balanced=[]
    for family in ['all','CPI','NFP','FOMC']:
        for period in PERIODS:
            p=select_period(per,period)
            if family!='all':p=p.loc[p.family.eq(family)]
            for (side,phase),g in p.groupby(['side','phase']):
                summary.append({'family':family,'period':period,'side':side,'phase':phase,**aggregate_cells(g)})
            if family=='all':
                for side,g in p.groupby('side'):
                    full=g.groupby('event_id').base_orders.apply(lambda counts:len(counts)==4 and counts.gt(0).all())
                    ids=full.index[full]
                    for phase,h in g.loc[g.event_id.isin(ids)].groupby('phase'):
                        balanced.append({'period':period,'side':side,'phase':phase,**aggregate_cells(h)})
    history=[]
    history_panel=pd.concat([panel,*cross_panels])
    for origin in pd.date_range('2026-03-01',periods=5,freq='MS',tz='UTC'):
        for side in ['all','call','put']:
            for phase in [*PHASES,'cross_primary']:
                g=history_panel.loc[history_panel.phase.eq(phase)]
                if side!='all':g=g.loc[g.side.eq(side)]
                history.append({'origin':origin,'side':side,'phase':phase,**history_counts(g,origin)})
    return per,pd.DataFrame(summary),overlap_per,pd.DataFrame(balanced),pd.DataFrame(history),panel,pd.DataFrame(cross_cells)


def matched_study(f,events):
    affected=pd.Series(False,index=f.index)
    planned=f.created+pd.to_timedelta(f.remaining,unit='s')
    for e in events.itertuples():
        affected|=f.created.lt(e.time+pd.Timedelta(minutes=240))&planned.gt(e.time-pd.Timedelta(minutes=60))
    common=f.loc[f.base.notna()];safe=f.loc[f.base.notna()&~affected];rows=[]
    for e in events.itertuples():
        for phase in PHASES:
            treated=common.loc[phase_mask(common.created,e.time,phase)]
            frames=[]
            for t in pseudo_times(e.time):
                if t.month!=e.time.month:continue
                lo,hi=PHASES[phase]
                if overlaps_influence(t+pd.Timedelta(minutes=lo),t+pd.Timedelta(minutes=hi),events):continue
                c=safe.loc[phase_mask(safe.created,t,phase)].copy();c['pseudo_time']=t
                frames.append(c)
            control=pd.concat(frames) if frames else safe.iloc[:0].assign(pseudo_time=pd.NaT)
            for side in ['call','put']:
                t=treated.loc[treated.side.eq(side)];c=control.loc[control.side.eq(side)]
                levels=[('tenor',['tenor'])]
                if e.time>=DEVELOPMENT:levels.append(('tenor_position_vol',['tenor','position_bin','risk_vol_bin']))
                for level,columns in levels:
                    result=matched_event(t,c,columns,MATCH_OUTCOMES)
                    if result is None:
                        result={'treated_orders':len(t),'treated_matched':0,'control_rows':0,
                                'coverage':0. if len(t) else np.nan,'matched_cells':0}
                        for metric in MATCH_OUTCOMES:
                            result[metric+'_support']=0
                            for suffix in ['treated','control','difference']:result[metric+'_'+suffix]=np.nan
                    rows.append({'event_id':e.event_id,'family':e.family,'month':e.month,'phase':phase,
                                 'side':side,'control':level,'pseudo_dates':c.pseudo_time.nunique(),**result})
    per=pd.DataFrame(rows);summary=[]
    for period in PERIODS:
        p=select_period(per,period)
        for (side,phase,level),g in p.groupby(['side','phase','control']):
            n=g.treated_orders.sum()
            for metric in MATCH_OUTCOMES:
                summary.append({'period':period,'side':side,'phase':phase,'control':level,'metric':metric,
                    'calendar_events':len(g),'eligible_events':int(g.treated_orders.gt(0).sum()),
                    'treated_orders':n,'treated_matched':g.treated_matched.sum(),
                    'coverage':g.treated_matched.sum()/n if n else np.nan,
                    'outcome_support':g[metric+'_support'].sum(),**summarize_events(g[metric+'_difference'])})
    return per,pd.DataFrame(summary)


def run(destination):
    if destination.exists():raise FileExistsError(destination)
    guards=install_guards();started=perf_counter()
    protected={DEFAULT_OUTPUT/'manifest.json':digest(DEFAULT_OUTPUT/'manifest.json')}
    prior=PROJECT/'outputs/model_studies/put_final_july_20261014'
    for name,sha in json.loads((prior/'provenance/manifest.json').read_text())['files'].items():
        if digest(prior/name)!=sha:raise ValueError('Changed previous result')
        protected[prior/name]=sha
    previous=PROJECT/'outputs/diagnostics/macro_events_20261014_v2'
    for name,sha in json.loads((previous/'provenance/manifest.json').read_text())['outputs'].items():
        if digest(previous/name)!=sha:raise ValueError('Changed previous macro result')
        protected[previous/name]=sha
    f,market,engine,events,info,cuts=load_inputs()
    scored=f.loc[f.created.ge(DEVELOPMENT)&f.base.notna()].assign(**{BASE_FEATURE:f.base},quantity_btc=f.quantity,option_type=f.side.map({'call':0,'put':1}))
    verified=check_previous(scored,prior)
    me,ms=market_study(engine,events)
    oe,os,exposure,balanced,history,panel,cross_support=quote_study(f,events)
    matched,matched_summary=matched_study(f,events)
    tables={'market_phases':me,'market_summary':ms,'order_phases':oe,'order_summary':os,
            'exposure_support':exposure,'balanced_order_summary':balanced,'historical_support':history,
            'matched_phases':matched,'matched_summary':matched_summary,'cross_event_support':cross_support}
    for name,table in tables.items():
        path=destination/'metrics'/(name+'.csv');path.parent.mkdir(parents=True,exist_ok=True);table.to_csv(path,index=False)
    prov=destination/'provenance';prov.mkdir(parents=True,exist_ok=True);cuts.to_csv(prov/'fixed_bins.csv',index=False)
    for path,sha in protected.items():assert digest(path)==sha,str(path)
    result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'calendar_path':str(CALENDAR.relative_to(PROJECT)),
        'calendar_sha256':digest(CALENDAR),'old_market':info,'risk_orders':len(f),'base_orders':int(f.base.notna().sum()),
        'warmup_orders':int(f.created.lt(DEVELOPMENT).sum()),'first_order':f.created.min().isoformat(),
        'main_events':len(events),'phase_orders':len(panel),'phase_base_orders':int(panel.base.notna().sum()),
        'all_phase_orders_have_prior_evidence':bool(panel.known.all()),'previous_base_groups_verified':verified,
        'phase_interpretation':'creation-time phase; whole-contract outcome; planned overlap support separate',
        'guards':guards,'protected_hashes':{str(k.relative_to(PROJECT)):v for k,v in protected.items()},
        'elapsed_seconds':perf_counter()-started,'rows':{k:len(v) for k,v in tables.items()}}
    (prov/'run.json').write_text(json.dumps(result,indent=2)+'\n')
    names=['scripts/analyze_macro_phases.py','scripts/macro_phase_statistics.py','scripts/macro_event_statistics.py','pp_calendar/calendar.py']
    manifest={'outputs':{str(p.relative_to(destination)):digest(p) for p in destination.rglob('*') if p.is_file()},
              'code':{name:digest(PROJECT/name) for name in names}}
    (prov/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='protected_hashes'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=RESULT)
    with threadpool_limits(limits=1):run(p.parse_args().output)
