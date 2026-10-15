"""Descriptive Base error statistics; no candidate fitting or pricing rules."""
import numpy as np
import pandas as pd

MONTHS=['2026-04','2026-05','2026-06','2026-07']


def calendar(month):
    start=pd.Timestamp(month+'-01',tz='UTC')
    days=pd.date_range(start,start+pd.offsets.MonthBegin(1),inclusive='left')
    rng=np.random.default_rng(20261014+int(month[-2:]))
    starts=rng.integers(0,len(days),size=(2000,int(np.ceil(len(days)/7))))
    idx=((starts[:,:,None]+np.arange(7))%len(days)).reshape(2000,-1)[:,:len(days)]
    w=np.zeros((2000,len(days)),dtype=int)
    np.add.at(w,(np.repeat(np.arange(2000),len(days)),idx.ravel()),1)
    return days,w


def metrics(f):
    e=f.base-f.actual;q=f.quantity;under=(-e).clip(lower=0)
    tail=f.tail90.astype(bool)
    return {'orders':len(f),'days':int(f.day.nunique()),'supported':bool(len(f)>=30 and f.day.nunique()>=5),
        'mean_error':e.mean(),'day_equal_bias':e.groupby(f.day).mean().mean(),
        'rmse':np.sqrt(np.mean(e**2)),'mae':abs(e).mean(),'base_mean':f.base.mean(),'actual_mean':f.actual.mean(),
        'zero_rate':f.actual.eq(0).mean(),'under_order_rate':e.lt(0).mean(),
        'cash_bias':np.sum(e*q),'cash_under':np.sum(under*q),'cash_over':np.sum(e.clip(lower=0)*q),
        'tail_rate':tail.mean(),'tail_error_share':np.sum(under[tail])/np.sum(under) if under.sum()>0 else np.nan,
        'mse_slope':np.mean(2*f.base*e),'mae_slope':np.mean(f.base*np.sign(e)),
        'body_mean_error':e[~tail].mean(),'tail_mean_error':e[tail].mean()}


def interval(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    if len(values)<1900:return np.nan,np.nan,len(values)
    return *np.quantile(values,[.025,.975]),len(values)


def group_statistics(f,days,weights):
    row=metrics(f);e=f.base-f.actual
    daily=pd.DataFrame({'day':f.day,'n':1.,'bias':e,'mse':2*f.base*e,
                        'mae':f.base*np.sign(e)}).groupby('day').sum().reindex(days,fill_value=0.)
    draws=weights@daily.to_numpy(float);n=draws[:,0]
    ratios=np.divide(draws[:,1:],n[:,None],out=np.full_like(draws[:,1:],np.nan),where=n[:,None]>0)
    values={k:ratios[:,i] for i,k in enumerate(['bias','mse','mae'])}
    row['ci_low'],row['ci_high'],row['valid_repeats']=interval(values['bias'])
    for metric in ('mse','mae'):
        row[metric+'_slope_ci_low'],row[metric+'_slope_ci_high'],_=interval(values[metric])
    deletion=[]
    for j in range(len(days)-6):
        den=len(f)-daily.n.iloc[j:j+7].sum()
        if den>=10:deletion.append((e.sum()-daily.bias.iloc[j:j+7].sum())/den)
    row.update(delete7_valid=len(deletion),delete7_expected=len(days)-6,delete7_bias_min=min(deletion) if deletion else np.nan,
               delete7_bias_max=max(deletion) if deletion else np.nan)
    # Linearized ratio estimator avoids undefined empty-group bootstrap ratios in
    # the simultaneous max statistic. Ordinary point intervals retain full ratios.
    influence=daily.bias-row['mean_error']*daily.n
    linear=weights@influence.to_numpy()/len(f)
    return row,values,linear


def simultaneous(rows,linears):
    rows=rows.copy();matrix=np.column_stack(linears)
    eligible=rows.supported.to_numpy(bool)
    sd=np.std(matrix,axis=0,ddof=1)
    z=np.divide(np.abs(matrix),sd[None,:],out=np.zeros_like(matrix),where=sd[None,:]>1e-12)
    critical=float(np.quantile(z[:,eligible].max(axis=1),.95)) if eligible.any() else np.nan
    rows['sim_low']=np.where(eligible,rows.mean_error-critical*sd,np.nan)
    rows['sim_high']=np.where(eligible,rows.mean_error+critical*sd,np.nan)
    rows['sim_critical']=critical
    return rows,critical


def stability(monthly,development):
    rows=[]
    for (side,family,group),p in monthly.groupby(['side','family','group'],sort=True):
        dev=development.loc[development.side.eq(side)&development.family.eq(family)&development.group.eq(group)]
        four=len(p)==4 and set(p.month)==set(MONTHS)
        enough=bool(four and p.supported.all())
        same=bool(four and ((p.mean_error>0).all() or (p.mean_error<0).all()))
        sign=1 if same and p.mean_error.iloc[0]>0 else -1 if same else 0
        day=bool(same and (p.day_equal_bias*sign>0).all())
        complete=bool('delete7_valid' in p and 'delete7_expected' in p
                      and p.delete7_valid.eq(p.delete7_expected).all())
        delete=bool(same and complete and ((p.delete7_bias_min>0).all() if sign==1 else (p.delete7_bias_max<0).all()))
        all_ci=bool(same and ((p.ci_low>0).all() if sign==1 else (p.ci_high<0).all()))
        july=p.loc[p.month.eq('2026-07')]
        point=sim=False
        if same and len(dev)==1 and len(july)==1:
            d,j=dev.iloc[0],july.iloc[0]
            point=bool(d.ci_low>0 and j.ci_low>0) if sign==1 else bool(d.ci_high<0 and j.ci_high<0)
            sim=bool(d.sim_low>0 and j.sim_low>0) if sign==1 else bool(d.sim_high<0 and j.sim_high<0)
        tier=0
        if enough and same:
            tier=1
            if day and delete:
                tier=2
                if point:
                    tier=3
                    if sim:tier=4
        rows.append({'side':side,'family':family,'group':group,'months_present':len(p),
            'minimum_month_orders':int(p.orders.min()),'minimum_month_days':int(p.days.min()),
            'four_month_supported':enough,'same_bias_sign':same,'bias_sign':sign,
            'day_equal_same_sign':day,'delete7_complete':complete,'delete7_same_sign':delete,'all_four_point_intervals':all_ci,
            'development_july_point_intervals':point,'development_july_simultaneous_intervals':sim,
            'all_mse_slopes_same_as_bias':bool(same and (p.mse_slope*sign>0).all()),
            'all_mae_slopes_same_as_bias':bool(same and (p.mae_slope*sign>0).all()),
            'min_mean_error':p.mean_error.min(),'max_mean_error':p.mean_error.max(),'tier':tier})
    return pd.DataFrame(rows)


def exact_tenor_control(frame,warmup,registry):
    """Common exact-tenor support in group AND complement in all four months."""
    output=[];calendars={m:calendar(m) for m in MONTHS}
    for side,side_frame in frame.groupby('side'):
        ref=warmup.loc[warmup.side.eq(side)]
        weights=ref.tenor.value_counts(normalize=True)
        for family,columns in registry:
            if family=='overall' or len(columns)>2 or 'tenor' in columns or 'tenor_band' in columns:continue
            labels=side_frame[columns].astype(str).agg('|'.join,axis=1)
            for label in sorted(labels.unique()):
                target=labels.eq(label);common=[]
                for tenor in weights.index:
                    adequate=True
                    for month in MONTHS:
                        scope=side_frame.month.eq(month)&side_frame.tenor.eq(tenor)
                        for membership in (target,~target):
                            sub=side_frame.loc[scope&membership]
                            if len(sub)<10 or sub.day.nunique()<3:adequate=False;break
                        if not adequate:break
                    if adequate:common.append(tenor)
                for month in MONTHS:
                    group=side_frame.loc[target&side_frame.month.eq(month)]
                    if group.empty:continue
                    control=side_frame.loc[~target&side_frame.month.eq(month)]
                    own=other=np.nan
                    ci={k:np.nan for k in ['standardized_ci_low','standardized_ci_high','contrast_ci_low','contrast_ci_high']}
                    ci.update(standardized_valid=0,contrast_valid=0)
                    if common:
                        w=weights.loc[common]/weights.loc[common].sum()
                        own=sum(w[t]*group.loc[group.tenor.eq(t),'error'].mean() for t in common)
                        other=sum(w[t]*control.loc[control.tenor.eq(t),'error'].mean() for t in common)
                        days,draws=calendars[month]
                        def samples(part):
                            values=np.zeros(2000)
                            for tenor in common:
                                sub=part.loc[part.tenor.eq(tenor)]
                                daily=sub.groupby('day').error.agg(['sum','size']).reindex(days,fill_value=0)
                                sampled=draws@daily.to_numpy(float)
                                ratios=np.divide(sampled[:,0],sampled[:,1],out=np.full(2000,np.nan),where=sampled[:,1]>0)
                                values+=w[tenor]*ratios
                            return values
                        own_samples,other_samples=samples(group),samples(control)
                        ci['standardized_ci_low'],ci['standardized_ci_high'],ci['standardized_valid']=interval(own_samples)
                        ci['contrast_ci_low'],ci['contrast_ci_high'],ci['contrast_valid']=interval(own_samples-other_samples)
                    output.append({'side':side,'family':family,'group':label,'month':month,
                        'common_tenors':'|'.join(str(int(t)) for t in common),'orders':len(group),
                        'covered_orders':int(group.tenor.isin(common).sum()),
                        'coverage':group.tenor.isin(common).mean(),'raw_bias':group.error.mean(),
                        'standardized_bias':own,'complement_standardized_bias':other,
                        'conditional_bias_difference':own-other,**ci})
    return pd.DataFrame(output)
