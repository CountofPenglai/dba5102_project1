"""Compact, paired evaluation of both directions and whole announcement clusters."""
import hashlib
import numpy as np
import pandas as pd
from .base_multiplier_statistics import tables as core_tables
from .put_final_statistics import draw_weights, paired_daily, differences, scope_frames
from .two_stage_robustness import interval_summary
from .two_stage_study import metric_values


def tables(frame, start, end, candidate='announcement', development_eligible=True):
    result = core_tables(frame, start, end, [candidate], False, {candidate:development_eligible})
    assessment = result['assessment']
    assessment['adoption_point_pass'] &= assessment.accuracy_vs_official_raw_pass
    days = pd.date_range(pd.Timestamp(start).floor('D'), pd.Timestamp(end).ceil('D'), inclusive='left')
    draws = {block:draw_weights(days, block) for block in (3, 7, 14)}
    intervals, removals = [], []
    for scope, part in scope_frames(frame):
        if part.empty: continue
        for reference in ('official_raw', 'h0'):
            stats = paired_daily(part, candidate, reference, days)
            point, point_valid = differences(stats, np.ones((1, len(days))), days)
            for block, w in draws.items():
                deltas, valid = differences(stats, w, days)
                for metric, values in deltas.items():
                    intervals.append({'scope':scope,'candidate':candidate,'reference':reference,'block_days':block,
                        'metric':metric,'point_delta':point[metric][0] if point_valid[0] else np.nan,
                        'draws_sha256':hashlib.sha256(w.astype('<i8').tobytes()).hexdigest(),
                        **interval_summary(values[valid])})
            if scope not in ('all','call','put'): continue
            for event_id in sorted(set(part.event_id)-{''}):
                reduced = part.loc[part.event_id.ne(event_id)]
                if reduced.empty: continue
                d, valid = differences(paired_daily(reduced,candidate,reference,days),np.ones((1,len(days))),days)
                removals.append({'scope':scope,'reference':reference,'removed_event':event_id,
                    'removed_orders':int((part.model.eq(candidate)&part.event_id.eq(event_id)).sum()),
                    'valid':bool(valid[0]),**{m:float(v[0]) for m,v in d.items()}})
    result['paired_intervals'] = pd.DataFrame(intervals)
    result['event_deletion'] = pd.DataFrame(removals)
    # Event and phase metrics are descriptive, not extra candidates or selection rules.
    rows=[]
    for grouping in ('event_id','phase'):
        for (month,model,side,value),g in frame.groupby(['month','model','side',grouping]):
            rows.append({'group':grouping,'value':value,'month':month,'model':model,'side':side,
                         'events':g.loc[g.event_id.ne(''),'event_id'].nunique(),**metric_values(g)})
    result['event_phase'] = pd.DataFrame(rows)
    p=frame.loc[frame.model.eq(candidate)]; rows=[]
    for (month,side),g in p.groupby(['month','side']):
        m=g.multiplier
        rows.append({'month':month,'side':side,'orders':len(g),'known_event_orders':int(g.calendar_status.eq('known_event').sum()),
            'known_events':g.loc[g.calendar_status.eq('known_event'),'event_id'].nunique(),
            'supported_orders':int(g.supported.sum()),'raised_orders':int(m.gt(1+1e-10).sum()),
            'lowered_orders':int(m.lt(1-1e-10).sum()),'mean_multiplier':m.mean(),
            'min_multiplier':m.min(),'max_multiplier':m.max(),'max_direction_adjustment':g.direction_adjustment.abs().max()})
    result['multipliers']=pd.DataFrame(rows)
    result['fallbacks']=p.groupby(['month','side','fallback_reason']).size().rename('orders').reset_index()
    return result
