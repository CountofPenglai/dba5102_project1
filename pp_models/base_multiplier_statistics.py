"""Aggregate-only matched evaluation for frozen multiplier experiments."""
import hashlib
import numpy as np
import pandas as pd
from .put_final_statistics import draw_weights, paired_daily, differences, scope_frames
from .two_stage_robustness import interval_summary
from .two_stage_study import aggregate


def risk_table(summary, candidates):
    all_rows = summary.loc[summary.month.eq('all')]
    rows=[]
    def get(model,group,value):
        part=all_rows.loc[all_rows.model.eq(model)&all_rows.group.eq(group)&all_rows.value.eq(value)]
        return part.iloc[0] if len(part)==1 else None
    for model in ['h0',*candidates]:
        for side in (0,1):
            own,base=(get(n,'direction',str(side)) for n in (model,'official_raw'))
            tail,btail=(get(n,'direction_tail',f'{side}|tail') for n in (model,'official_raw'))
            body,bbody=(get(n,'direction_tail',f'{side}|body') for n in (model,'official_raw'))
            support=tail is not None and tail.orders>=30 and tail.days>=5
            row={'model':model,'side':side,'tail_support_pass':bool(support)}
            if any(x is None for x in (own,base,tail,btail,body,bbody)):
                rows.append(row|{'direction_pass':False});continue
            pairs={'tail_rmse':(tail.rmse_per_btc,btail.rmse_per_btc,1.02),
                   'gross_under':(own.gross_underprediction_usdc,base.gross_underprediction_usdc,1.02),
                   'tail_gross_under':(tail.gross_underprediction_usdc,btail.gross_underprediction_usdc,1.02),
                   'absolute_bias':(own.absolute_total_bias_usdc,base.absolute_total_bias_usdc,1.),
                   'body_mae':(body.mae_per_btc,bbody.mae_per_btc,1.)}
            for metric,(a,b,limit) in pairs.items():
                row.update({metric:a,metric+'_base':b,metric+'_pass':bool(a<=limit*b+1e-10)})
            row['direction_pass']=bool(support and all(row[k+'_pass'] for k in pairs))
            rows.append(row)
    return pd.DataFrame(rows)


def tables(frame,start,end,candidates,uncertainty=True,development_eligibility=None):
    summary,daily,macro=aggregate(frame)
    risk=risk_table(summary,candidates)
    lookup=macro.set_index('model')
    biases=summary.query("month == 'all' and group == 'overall'").set_index('model').absolute_total_bias_usdc
    decisions=[]
    for model in candidates:
        row={'model':model}
        for ref in ('h0','official_raw'):
            dr=lookup.loc[model,'macro_rmse_per_btc']-lookup.loc[ref,'macro_rmse_per_btc']
            da=lookup.loc[model,'macro_mae_per_btc']-lookup.loc[ref,'macro_mae_per_btc']
            row.update({f'rmse_delta_vs_{ref}':dr,f'mae_delta_vs_{ref}':da,
                        f'accuracy_vs_{ref}_pass':bool(dr<0 and da<=0)})
        row['overall_bias_pass']=bool(biases[model]<=biases['official_raw']+1e-10)
        row['mae_vs_base_pass']=bool(lookup.loc[model,'macro_mae_per_btc']<=lookup.loc['official_raw','macro_mae_per_btc'])
        row['risk_pass']=bool(row['overall_bias_pass'] and risk.loc[risk.model.eq(model)].direction_pass.all())
        row['locked_development_eligible']=bool((development_eligibility or {}).get(model,True))
        row['adoption_point_pass']=bool(row['risk_pass'] and row['accuracy_vs_h0_pass']
            and row['mae_vs_base_pass'] and row['locked_development_eligible'])
        decisions.append(row)
    result={'summary':summary,'macro_monthly':macro,'risk_constraints':risk,'assessment':pd.DataFrame(decisions)}
    # Keep compact daily losses for reproducibility, no order-level outputs.
    result['daily']=daily
    tenor_rows=[]
    for (month,model,side,tenor),part in frame.groupby(['month','model','side','tenor']):
        for cohort,sub in [('all',part),*part.groupby('tail_group')]:
            e=sub.prediction-sub.actual
            tenor_rows.append({'month':month,'model':model,'side':side,'tenor':tenor,'cohort':cohort,'orders':len(sub),
                'rmse':np.sqrt(np.mean(e**2)),'mae':np.mean(abs(e)),
                'cash_bias':np.sum(e*sub.quantity),'cash_under':np.sum((-e).clip(lower=0)*sub.quantity)})
    result['direction_tenor']=pd.DataFrame(tenor_rows)
    if not uncertainty:return result
    days=pd.date_range(pd.Timestamp(start).floor('D'),pd.Timestamp(end).ceil('D'),inclusive='left')
    pairs=[(c,r) for c in candidates for r in ('official_raw','h0')]
    if set(('constant','contract','market')).issubset(candidates):
        pairs += [('contract','constant'),('market','contract')]
    draws={b:draw_weights(days,b) for b in (3,7,14)}
    intervals,influence,tenor_influence=[],[],[]
    for scope,part in scope_frames(frame):
        if scope.startswith('call') or part.empty:continue
        for c,r in pairs:
            stats=paired_daily(part,c,r,days)
            point,point_valid=differences(stats,np.ones((1,len(days))),days)
            for block,weights in draws.items():
                deltas,valid=differences(stats,weights,days)
                for metric,values in deltas.items():
                    intervals.append({'scope':scope,'candidate':c,'reference':r,'block_days':block,
                        'metric':metric,'point_delta':point[metric][0] if point_valid[0] else np.nan,
                        'draws_sha256':hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest(),
                        **interval_summary(values[valid])})
            if scope not in ('all','put'):continue
            for width in (1,7):
                weights=np.ones((len(days)-width+1,len(days)))
                for i in range(len(weights)):weights[i,i:i+width]=0
                deltas,valid=differences(stats,weights,days)
                for metric in ('rmse_delta','mae_delta'):
                    v=deltas[metric][valid]
                    influence.append({'scope':scope,'candidate':c,'reference':r,'excluded_days':width,
                        'metric':metric,'point_delta':point[metric][0],'valid_windows':int(valid.sum()),
                        'min_delta':v.min() if len(v) else np.nan,'max_delta':v.max() if len(v) else np.nan,
                        'sign_flip_windows':int((np.sign(v)!=np.sign(point[metric][0])).sum())})
            for tenor in sorted(part.tenor.unique()):
                reduced=part.loc[part.tenor.ne(tenor)]
                if reduced.empty:continue
                td=paired_daily(reduced,c,r,days)
                delta,valid=differences(td,np.ones((1,len(days))),days)
                tenor_influence.append({'scope':scope,'candidate':c,'reference':r,'excluded_tenor':tenor,
                    'valid':bool(valid[0]),'rmse_delta':delta['rmse_delta'][0], 'mae_delta':delta['mae_delta'][0]})
    result.update(paired_intervals=pd.DataFrame(intervals),date_influence=pd.DataFrame(influence),
                  tenor_influence=pd.DataFrame(tenor_influence))
    return result
