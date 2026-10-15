"""Independent direct-slice arithmetic and coverage checks; no model calls."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pp_data.config import PROJECT
from scripts.analyze_base_regimes import digest, install_guards
from scripts.analyze_size_and_base_lag import load_data
from scripts.analyze_macro_events import RESULT


def verify(root):
    guards=install_guards()
    manifest=json.loads((root/'provenance/manifest.json').read_text())
    for name,sha in manifest['outputs'].items():assert digest(root/name)==sha,name
    for name,sha in manifest['code'].items():assert digest(PROJECT/name)==sha,name
    run=json.loads((root/'provenance/run.json').read_text())
    calendar=json.loads((PROJECT/'external_data/macro_events/calendar.json').read_text())
    assert digest(PROJECT/'external_data/macro_events/calendar.json')==run['calendar_sha256']
    for event in calendar['events']:
        expected=pd.Timestamp(event['local_date']+'T'+event['local_time'],tz='America/New_York').tz_convert('UTC')
        assert expected==pd.Timestamp(event['event_time_utc'])
    frame,market,*_=load_data()
    prices=market.set_index('open_time_utc').close
    prices=prices.reindex(pd.date_range(prices.index.min(),prices.index.max(),freq='min'))
    returns=np.log(prices/prices.shift(1))

    def direct_window(start,end):
        index=pd.date_range(pd.Timestamp(start).ceil('min'),pd.Timestamp(end).floor('min'),freq='min',inclusive='left')
        r=returns.reindex(index).to_numpy()
        if not len(r) or not np.isfinite(r).all():return np.full(4,np.nan)
        return np.array([np.sqrt(np.mean(r*r)),np.sqrt(np.mean(np.maximum(r,0)**2)),
                         np.sqrt(np.mean(np.minimum(r,0)**2)),r.sum()])*10000

    market_rows=pd.read_csv(root/'metrics/market_events.csv')
    names=['rms_bp','up_rms_bp','down_rms_bp','return_bp']
    for row in market_rows.itertuples():
        time=pd.Timestamp(row.event_time);delta=pd.Timedelta(minutes=row.horizon_minutes)
        np.testing.assert_allclose(direct_window(time-delta,time),[getattr(row,n+'_pre') for n in names],rtol=1e-8,atol=1e-7)
        np.testing.assert_allclose(direct_window(time,time+delta),[getattr(row,n+'_post') for n in names],rtol=1e-8,atol=1e-7)
        controls=[] if pd.isna(row.control_times) else [pd.Timestamp(t) for t in row.control_times.split('|')]
        assert len(controls)==row.controls
        for t in controls:
            local=t.tz_convert('America/New_York');event_local=time.tz_convert('America/New_York')
            assert local.weekday()==event_local.weekday() and local.time()==event_local.time()
            assert abs((local.date()-event_local.date()).days) in [7,14]
        expected=np.mean([direct_window(t,t+delta) for t in controls],axis=0) if len(controls)>=2 else np.full(4,np.nan)
        np.testing.assert_allclose(expected,[getattr(row,n+'_control') for n in names],rtol=1e-8,atol=1e-7)
    events=[e for e in calendar['events'] if '2026-04-01'<=e['event_time_utc']<'2026-08-01']
    masks={}
    planned_end=frame.created+pd.to_timedelta(frame.remaining,unit='s')
    for e in events:
        time=pd.Timestamp(e['event_time_utc'])
        masks[e['event_id']]=(frame.created<time)&(planned_end>=time)
    coverage=pd.read_csv(root/'metrics/event_coverage.csv')
    for row in coverage.itertuples():
        mask=masks[row.event_id]
        assert mask.sum()==row.risk_orders
        assert (mask&frame.base.notna()).sum()==row.base_orders
        assert (mask&(frame.closed<pd.Timestamp(row.event_time))).sum()==row.ended_before_event
    primary=[e for e in events if e['kind']!='press_conference']
    cross=pd.concat([masks[e['event_id']] for e in primary],axis=1).any(axis=1)
    groups=pd.read_csv(root/'metrics/order_groups.csv')
    selected=groups.loc[groups.family.eq('any')]
    for row in selected.itertuples():
        period=pd.Series(True,index=frame.index)
        if row.period=='development':period=frame.month.lt('2026-07')
        elif row.period!='all':period=frame.month.eq(row.period)
        mask=period&frame.side.eq(row.side)&cross.eq(row.crosses)
        f=frame.loc[mask];g=f.loc[f.base.notna()]
        error=(g.base-g.actual).to_numpy()
        np.testing.assert_allclose([len(f),len(g),error.mean(),np.sqrt(np.mean(error**2)),np.mean(abs(error))],
                                   [row.risk_orders,row.base_orders,row.bias,row.rmse,row.mae],rtol=1e-9,atol=1e-8)
        assert row.events==(sum((masks[e['event_id']]&mask).any() for e in primary) if row.crosses else 0)
    matched=pd.read_csv(root/'metrics/matched_events.csv')
    summary=pd.read_csv(root/'metrics/matched_summary.csv')
    for (side,control),g in matched.groupby(['side','control']):
        assert g.treated_orders.sum()==(cross&frame.side.eq(side)&frame.base.notna()).sum()
        for row in summary.loc[summary.side.eq(side)&summary.control.eq(control)&summary.family.eq('all')&summary.period.eq('all')].itertuples():
            value=g[row.metric+'_difference'].dropna()
            assert row.events==len(value) and row.eligible_events==len(g)
            np.testing.assert_allclose([g.treated_matched.sum()/g.treated_orders.sum(),value.mean()],
                                       [row.coverage,row.mean],rtol=1e-9,atol=1e-8)
    assert guards=={'official_validation_open_attempts':0,'model_fit_predict_calls':0}
    result={'calendar_timestamps_recomputed':len(calendar['events']),'market_windows_and_controls_recomputed':len(market_rows),
            'event_coverage_rows_recomputed':len(coverage),'raw_any_groups_recomputed':len(selected),
            'matched_coverage_denominators':{'call':int((cross&frame.side.eq('call')&frame.base.notna()).sum()),
                                            'put':int((cross&frame.side.eq('put')&frame.base.notna()).sum())},
            'output_code_calendar_checksums':True,'guards':guards,'verifier_sha256':digest(Path(__file__))}
    path=root/'provenance/verification.json';path.write_text(json.dumps(result,indent=2)+'\n')
    manifest['outputs']['provenance/verification.json']=digest(path)
    (root/'provenance/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=RESULT)
    with threadpool_limits(limits=1):verify(p.parse_args().output)
