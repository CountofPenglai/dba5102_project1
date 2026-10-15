"""Non-overlapping quote phases and event support; descriptive, no fitting."""
import numpy as np
import pandas as pd
from scripts.macro_event_statistics import summarize_events

PHASES={'pre60':(-60,0),'post0_15':(0,15),'post15_60':(15,60),'post60_240':(60,240)}


def phase_mask(created,event,phase):
    lo,hi=PHASES[phase]
    return created.ge(event+pd.Timedelta(minutes=lo)) & created.lt(event+pd.Timedelta(minutes=hi))


def phase_overlap_minutes(created,remaining,event,phase):
    lo,hi=PHASES[phase]
    starts=created.clip(lower=event+pd.Timedelta(minutes=lo))
    ends=(created+pd.to_timedelta(remaining,unit='s')).clip(upper=event+pd.Timedelta(minutes=hi))
    return ((ends-starts).dt.total_seconds()/60).clip(lower=0)


def phase_metrics(f):
    g=f.loc[f.base.notna()];n=len(g);e=g.base-g.actual
    row={'risk_orders':len(f),'base_orders':n,'known_orders':int(f.known.sum()),
         'cross_primary_orders':int(f.cross_primary.sum()),'cross_press_orders':int(f.cross_press.sum()),
         'base_coverage':n/len(f) if len(f) else np.nan,
         'actual_mean_risk':f.actual.mean(),'actual_mean_common':g.actual.mean(),'base_mean':g.base.mean(),
         'bias':e.mean(),'mse':(e**2).mean(),'mae':abs(e).mean(),
         'remaining_hours_mean':f.remaining.mean()/3600,
         'future_risk_rms_bp':f.future_risk_rms_bp.mean(),
         'future_complete':int(f.future_risk_rms_bp.notna().sum()),
         'cash_under':float((g.quantity*(-e).clip(lower=0)).sum()) if n else np.nan,
         'cash_over':float((g.quantity*e.clip(lower=0)).sum()) if n else np.nan,
         'cash_bias':float((g.quantity*e).sum()) if n else np.nan}
    row['rmse']=np.sqrt(row['mse'])
    for name in ['lead_minutes','future_post_release_minutes','cross_post_release_minutes']:
        row[name+'_mean']=f[name].mean() if name in f else np.nan
    return row


def aggregate_cells(cells):
    risk=cells.loc[cells.risk_orders.gt(0)];base=cells.loc[cells.base_orders.gt(0)]
    n=int(base.base_orders.sum())
    row={'calendar_events':cells.event_id.nunique(),'risk_events':risk.event_id.nunique(),
         'base_events':base.event_id.nunique(),'events_ge5_base':int(cells.base_orders.ge(5).sum()),
         'risk_orders':int(cells.risk_orders.sum()),'base_orders':n,
         'base_orders_median_active_event':base.base_orders.median(),
         'largest_event_base_share':base.base_orders.max()/n if n else np.nan}
    for metric in ['bias','mse','mae']:
        row[metric+'_order_equal']=(base[metric]*base.base_orders).sum()/n if n else np.nan
        row[metric+'_event_equal']=base[metric].mean()
    row['rmse_order_equal']=np.sqrt(row['mse_order_equal'])
    row['rmse_event_equal']=np.sqrt(row['mse_event_equal'])
    uncertainty=summarize_events(base.bias)
    for k in ['ci_low','ci_high','leave_one_min','leave_one_max','positive_fraction']:
        row['bias_event_'+k]=uncertainty[k]
    for metric in ['cash_under','cash_over','cash_bias']:
        if metric in cells:row[metric]=base[metric].sum() if n else np.nan
    if 'future_risk_rms_bp' in cells:
        row['future_risk_rms_event_equal']=risk.future_risk_rms_bp.mean()
        row['future_complete']=int(cells.future_complete.sum())
    return row


def history_counts(f,origin):
    h=f.loc[f.created.lt(origin)&f.closed.lt(origin)&f.event_time.lt(origin)&f.known]
    b=h.loc[h.base.notna()]
    return {'risk_orders':len(h),'base_orders':len(b),'risk_events':h.event_id.nunique(),'base_events':b.event_id.nunique()}
