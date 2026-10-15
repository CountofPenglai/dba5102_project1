"""RMSE-first evidence, separate MAE cost and direction/cash risk checks."""
import hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .two_stage_study import aggregate, metric_values
from .put_final_statistics import draw_weights, paired_daily, differences, scope_frames

CANDIDATES=('contract30','g1a','static')
PAIRS=[(c,r) for c in CANDIDATES for r in ('official_raw','h0')]


def comparisons(macro):
    lookup=macro.set_index('model');rows=[]
    for c,r in PAIRS:
        a,b=lookup.loc[c],lookup.loc[r]
        rows.append(dict(candidate=c,reference=r,
            rmse_delta=a.macro_rmse_per_btc-b.macro_rmse_per_btc,
            rmse_change=a.macro_rmse_per_btc/b.macro_rmse_per_btc-1,
            mae_delta=a.macro_mae_per_btc-b.macro_mae_per_btc,
            mae_change=a.macro_mae_per_btc/b.macro_mae_per_btc-1,
            rmse_improved=bool(a.macro_rmse_per_btc<b.macro_rmse_per_btc),
            mae_role='cost_monitor_only',selection_performed=False))
    return pd.DataFrame(rows)


def ratio(a,b):
    return float(a/b) if b else (1. if a==0 else np.inf)


def risks(frame):
    rows=[]
    for c,r in PAIRS:
        total={m:metric_values(frame.loc[frame.model.eq(m)]) for m in (c,r)}
        for side in (0,1):
            parts={m:frame.loc[frame.model.eq(m)&frame.side.eq(side)] for m in (c,r)}
            tail={m:p.loc[p.tail_group.eq('tail')] for m,p in parts.items()}
            supported=all(len(p)>=30 and p.day.nunique()>=5 for p in tail.values())
            row=dict(candidate=c,reference=r,side=side,tail_orders=len(tail[c]),tail_days=tail[c].day.nunique(),
                     support_sufficient=supported)
            if any(p.empty for p in [*parts.values(),*tail.values()]):
                row.update(direction_risk_pass=False,unavailable=True)
            else:
                mv={m:metric_values(parts[m]) for m in (c,r)};tv={m:metric_values(tail[m]) for m in (c,r)}
                row.update(unavailable=False,
                    tail_rmse_ratio=ratio(tv[c]['rmse_per_btc'],tv[r]['rmse_per_btc']),
                    gross_under_ratio=ratio(mv[c]['gross_underprediction_usdc'],mv[r]['gross_underprediction_usdc']),
                    tail_gross_under_ratio=ratio(tv[c]['gross_underprediction_usdc'],tv[r]['gross_underprediction_usdc']),
                    absolute_bias_ratio=ratio(mv[c]['absolute_total_bias_usdc'],mv[r]['absolute_total_bias_usdc']),
                    overall_absolute_bias_ratio=ratio(total[c]['absolute_total_bias_usdc'],total[r]['absolute_total_bias_usdc']))
                row['direction_risk_pass']=bool(supported and row['tail_rmse_ratio']<=1.02
                    and row['gross_under_ratio']<=1.02 and row['tail_gross_under_ratio']<=1.02
                    and row['absolute_bias_ratio']<=1 and row['overall_absolute_bias_ratio']<=1)
            rows.append(row)
    return pd.DataFrame(rows)


def scoped_macro(frame,months):
    rows=[]
    for scope,part in scope_frames(frame):
        for model,p in part.groupby('model'):
            monthly=[metric_values(p.loc[p.month.eq(m)]) for m in months if p.month.eq(m).any()]
            complete=len(monthly)==len(months)
            rows.append(dict(scope=scope,model=model,months=len(monthly),expected_months=len(months),
                complete=complete,macro_rmse_per_btc=np.sqrt(np.mean([v['rmse_per_btc']**2 for v in monthly])) if complete else np.nan,
                macro_mae_per_btc=np.mean([v['mae_per_btc'] for v in monthly]) if complete else np.nan))
    return pd.DataFrame(rows)


def tables(frame,start,end,uncertainty=True):
    days=pd.date_range(pd.Timestamp(start).floor('D'),pd.Timestamp(end).ceil('D'),inclusive='left')
    months=sorted(set(days.strftime('%Y-%m')))
    summary,daily,macro=aggregate(frame)
    if not macro.months.eq(len(months)).all():raise ValueError('Incomplete overall scoring months')
    result={'summary':summary,'daily':daily,'macro_monthly':macro,'comparisons':comparisons(macro),
            'risk_constraints':risks(frame),'scoped_macro':scoped_macro(frame,months)}
    cells=[];multipliers=[]
    for (month,model,side,tenor,tail),p in frame.groupby(['month','model','side','tenor','tail_group']):
        cells.append(dict(month=month,model=model,side=side,tenor=tenor,tail_group=tail,
                          days=p.day.nunique(),**metric_values(p)))
    for (month,model,side),p in frame.groupby(['month','model','side']):
        m=p.multiplier
        multipliers.append(dict(month=month,model=model,side=side,orders=len(p),mean=m.mean(),min=m.min(),
            p10=m.quantile(.1),median=m.median(),p90=m.quantile(.9),max=m.max(),
            above_base_orders=int(m.gt(1+1e-10).sum()),below_base_orders=int(m.lt(1-1e-10).sum())))
    result.update(direction_tenor=pd.DataFrame(cells),multipliers=pd.DataFrame(multipliers))
    if not uncertainty:return result
    intervals,influence,sufficient=[],[],[]
    draws={b:draw_weights(days,b) for b in (3,7,14)}
    with threadpool_limits(limits=1):
        for scope,part in scope_frames(frame):
            if part.empty:continue
            for c,r in PAIRS:
                stats=paired_daily(part,c,r,days)
                sufficient.append(stats.reset_index(names='day').assign(scope=scope,candidate=c,reference=r))
                point,pv=differences(stats,np.ones((1,len(days))),days)
                for block,weights in draws.items():
                    delta,valid=differences(stats,weights,days)
                    digest=hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest()
                    for metric,values in delta.items():
                        usable=valid&np.isfinite(values);n=int(usable.sum());supported=n>=1900 and bool(pv[0])
                        low,high=np.quantile(values[usable],[.025,.975]) if supported else (np.nan,np.nan)
                        # Six prespecified candidate/reference overall RMSE comparisons per phase.
                        family=scope=='all' and metric=='rmse_delta'
                        fl,fh=np.quantile(values[usable],[.05/12,1-.05/12]) if supported and family else (np.nan,np.nan)
                        intervals.append(dict(scope=scope,candidate=c,reference=r,block_days=block,metric=metric,
                            point_delta=float(point[metric][0]) if pv[0] else np.nan,valid_repeats=n,
                            supported=supported,lower_95=low,upper_95=high,simultaneous_lower=fl,simultaneous_upper=fh,
                            calendar_draws_sha256=digest))
                if scope not in ('all','call','put'):continue
                for width in (1,7):
                    n=len(days)-width+1
                    if n<=0:continue
                    weights=np.ones((n,len(days)))
                    for j in range(n):weights[j,j:j+width]=0
                    delta,valid=differences(stats,weights,days)
                    for metric in ('rmse_delta','mae_delta','gross_underprediction_delta_usdc','absolute_total_bias_delta_usdc'):
                        vals=delta[metric];usable=valid&np.isfinite(vals);pos=np.flatnonzero(usable)
                        influence.append(dict(scope=scope,candidate=c,reference=r,deleted_days=width,metric=metric,
                            valid_windows=len(pos),total_windows=n,improvement_windows=int((vals[usable]<0).sum()),
                            min_delta=float(vals[usable].min()) if len(pos) else np.nan,
                            max_delta=float(vals[usable].max()) if len(pos) else np.nan,
                            worst_window_start=days[pos[np.argmax(vals[usable])]].isoformat() if len(pos) else None))
    result.update(intervals=pd.DataFrame(intervals),influence=pd.DataFrame(influence),
                  paired_daily=pd.concat(sufficient,ignore_index=True))
    return result
