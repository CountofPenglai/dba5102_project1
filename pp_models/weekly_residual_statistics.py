"""Fixed comparisons, tail/cash constraints and forward signal diagnostics."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .two_stage_study import aggregate,metric_values
from .frozen_candidate_statistics import scoped_macro,ratio
from .put_final_statistics import scope_frames,paired_daily,differences,draw_weights

CANDIDATES=('level_add','level_scale','jump','streak05','streak10')
PAIRS=[('contract_weekly',r) for r in ('official_raw','h0','contract_monthly')]+[(c,'contract_weekly') for c in CANDIDATES]+[
    (c,'uniform_'+c) for c in ('jump','streak05','streak10')]


def risk_table(frame,scoped):
    sm=scoped.set_index(['scope','model']);rows=[]
    for c in ('contract_weekly',)+CANDIDATES:
        for r in ('official_raw','h0','contract_weekly'):
            if c==r:continue
            overall={m:metric_values(frame.loc[frame.model.eq(m)]) for m in (c,r)}
            for side,label in ((0,'call'),(1,'put')):
                p={m:frame.loc[frame.model.eq(m)&frame.side.eq(side)] for m in (c,r)}
                t={m:v.loc[v.tail_group.eq('tail')] for m,v in p.items()}
                complete=all((label+'_tail',m) in sm.index and bool(sm.loc[(label+'_tail',m),'complete']) for m in (c,r))
                enough=complete and all(len(v)>=30 and v.day.nunique()>=5 for v in t.values())
                row=dict(candidate=c,reference=r,side=side,tail_orders=len(t[c]),tail_days=t[c].day.nunique(),support_sufficient=enough,direction_risk_pass=False)
                if all(not v.empty for v in [*p.values(),*t.values()]):
                    mv={m:metric_values(v) for m,v in p.items()};tv={m:metric_values(v) for m,v in t.items()}
                    row.update(tail_rmse_ratio=ratio(tv[c]['rmse_per_btc'],tv[r]['rmse_per_btc']),
                        gross_under_ratio=ratio(mv[c]['gross_underprediction_usdc'],mv[r]['gross_underprediction_usdc']),
                        tail_gross_under_ratio=ratio(tv[c]['gross_underprediction_usdc'],tv[r]['gross_underprediction_usdc']),
                        absolute_bias_ratio=ratio(mv[c]['absolute_total_bias_usdc'],mv[r]['absolute_total_bias_usdc']),
                        overall_absolute_bias_ratio=ratio(overall[c]['absolute_total_bias_usdc'],overall[r]['absolute_total_bias_usdc']))
                    row['direction_risk_pass']=bool(enough and row['tail_rmse_ratio']<=1.02 and row['gross_under_ratio']<=1.02
                        and row['tail_gross_under_ratio']<=1.02 and row['absolute_bias_ratio']<=1+1e-10 and row['overall_absolute_bias_ratio']<=1+1e-10)
                rows.append(row)
    return pd.DataFrame(rows)


def tables(frame,start,end,uncertainty=True):
    days=pd.date_range(pd.Timestamp(start).floor('D'),pd.Timestamp(end).ceil('D'),inclusive='left');months=sorted(set(days.strftime('%Y-%m')))
    summary,daily,macro=aggregate(frame);scoped=scoped_macro(frame,months);risk=risk_table(frame,scoped)
    if not macro.months.eq(len(months)).all():raise ValueError('Missing scoring month')
    lookup=macro.set_index('model');compare=[]
    for c,r in PAIRS:
        a,b=lookup.loc[c],lookup.loc[r]
        compare.append(dict(candidate=c,reference=r,rmse_delta=a.macro_rmse_per_btc-b.macro_rmse_per_btc,
            rmse_change=ratio(a.macro_rmse_per_btc,b.macro_rmse_per_btc)-1,mae_delta=a.macro_mae_per_btc-b.macro_mae_per_btc))
    cells=[];changes=[];alarms=[]
    for (month,model,side,tenor,tail),p in frame.groupby(['month','model','side','tenor','tail_group']):
        cells.append(dict(month=month,model=model,side=side,tenor=tenor,tail_group=tail,days=p.day.nunique(),**metric_values(p)))
    for (month,model,side),p in frame.groupby(['month','model','side']):
        raised=p.prediction>p.h+1e-10;tail=p.tail_group.eq('tail')
        changes.append(dict(month=month,model=model,side=side,orders=len(p),supported=int(p.supported.sum()),
            raised_vs_h=int(raised.sum()),lowered_vs_h=int((p.prediction<p.h-1e-10).sum()),
            raised_tail_orders=int((raised&tail).sum()),raised_body_orders=int((raised&~tail).sum()),
            tail_orders=int(tail.sum()),raised_days=p.loc[raised,'day'].nunique(),
            extra_predicted_usdc=float(((p.prediction-p.h)*p.quantity).sum()),
            mean_multiplier=p.multiplier.mean(),min_multiplier=p.multiplier.min(),max_multiplier=p.multiplier.max()))
    base=frame.loc[frame.model.eq('contract_weekly')]
    for signal in ('jump','streak'):
        for (month,side),p in base.groupby(['month','side']):
            active=p[signal]>0;tail=p.tail_group.eq('tail')
            for label,use in [('active',active),('inactive',~active)]:
                v=p.loc[use]
                alarms.append(dict(signal=signal,month=month,side=side,status=label,orders=len(v),days=v.day.nunique(),
                    tail_orders=int(v.tail_group.eq('tail').sum()),tail_rate=float(v.tail_group.eq('tail').mean()) if len(v) else np.nan,
                    tail_coverage=float((use&tail).sum()/tail.sum()) if tail.sum() else np.nan,
                    mean_residual=float((v.actual-v.prediction).mean()) if len(v) else np.nan,
                    positive_residual_usdc=float((np.maximum(v.actual-v.prediction,0)*v.quantity).sum())))
    result=dict(summary=summary,daily=daily,macro_monthly=macro,scoped_macro=scoped,risk_constraints=risk,
        comparisons=pd.DataFrame(compare),direction_tenor=pd.DataFrame(cells),multipliers=pd.DataFrame(changes),signal_diagnostics=pd.DataFrame(alarms))
    if not uncertainty:return result
    intervals=[];influence=[];sufficient=[];draws={b:draw_weights(days,b) for b in (3,7,14)}
    with threadpool_limits(limits=1):
        for scope,part in scope_frames(frame):
            if part.empty:continue
            for c,r in PAIRS:
                stats=paired_daily(part,c,r,days);point,pv=differences(stats,np.ones((1,len(days))),days)
                sufficient.append(stats.reset_index(names='day').assign(scope=scope,candidate=c,reference=r))
                for block,w in draws.items():
                    delta,valid=differences(stats,w,days)
                    for metric,values in delta.items():
                        use=valid&np.isfinite(values);n=int(use.sum());supported=bool(pv[0] and n>=1900)
                        lo,hi=np.quantile(values[use],[.025,.975]) if supported else (np.nan,np.nan)
                        fl,fh=np.quantile(values[use],[.05/(2*len(PAIRS)),1-.05/(2*len(PAIRS))]) if supported and scope=='all' and metric=='rmse_delta' else (np.nan,np.nan)
                        intervals.append(dict(scope=scope,candidate=c,reference=r,block_days=block,metric=metric,
                            point_delta=float(point[metric][0]) if pv[0] else np.nan,valid_repeats=n,supported=supported,
                            lower_95=lo,upper_95=hi,simultaneous_lower=fl,simultaneous_upper=fh))
                if scope not in ('all','call','put'):continue
                for width in (1,7):
                    n=len(days)-width+1;w=np.ones((n,len(days)))
                    for j in range(n):w[j,j:j+width]=0
                    delta,valid=differences(stats,w,days)
                    for metric in ('rmse_delta','mae_delta','gross_underprediction_delta_usdc','absolute_total_bias_delta_usdc'):
                        values=delta[metric];use=valid&np.isfinite(values)
                        influence.append(dict(scope=scope,candidate=c,reference=r,deleted_days=width,metric=metric,
                            valid_windows=int(use.sum()),total_windows=n,improvement_windows=int((values[use]<0).sum()),
                            min_delta=float(values[use].min()) if use.any() else np.nan,max_delta=float(values[use].max()) if use.any() else np.nan))
    result.update(intervals=pd.DataFrame(intervals),influence=pd.DataFrame(influence),paired_daily=pd.concat(sufficient,ignore_index=True))
    return result
