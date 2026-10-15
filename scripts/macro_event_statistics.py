"""Descriptive event-study primitives; no learned models or price recommendations."""
import numpy as np
import pandas as pd


def crosses(created, remaining, event):
    return created.lt(event) & (created+pd.to_timedelta(remaining,unit='s')).ge(event)


class MinuteMarket:
    def __init__(self, market):
        prices = market.set_index('open_time_utc').close.sort_index()
        if prices.index.has_duplicates or (prices <= 0).any():
            raise ValueError('Invalid minute market')
        self.index = pd.date_range(prices.index.min(), prices.index.max(), freq='min')
        prices = prices.reindex(self.index)
        r = np.log(prices).diff().to_numpy()
        self.valid = np.isfinite(r)
        r = np.nan_to_num(r, nan=0.)
        values = np.column_stack([r*r, np.maximum(r,0)**2, np.minimum(r,0)**2,r,self.valid])
        self.sums = np.vstack([np.zeros(5), values.cumsum(axis=0)])

    def window(self, start, end):
        # A candle covers [open,open+1m]; drop both partial endpoint candles.
        start, end = pd.Timestamp(start).ceil('min'), pd.Timestamp(end).floor('min')
        n = int((end-start).total_seconds()/60)
        lo, hi = self.index.searchsorted(start), self.index.searchsorted(end)
        sums = self.sums[hi]-self.sums[lo]
        complete = n > 0 and hi-lo == n and sums[4] == n
        result = {'minutes': n, 'observed_minutes': int(sums[4]), 'complete': bool(complete)}
        for j,name in enumerate(['rms_bp','up_rms_bp','down_rms_bp']):
            result[name] = float(np.sqrt(sums[j]/n)*10000) if complete else np.nan
        result['return_bp'] = float(sums[3]*10000) if complete else np.nan
        return result


def summarize_events(values, reference=0.):
    v = np.asarray(values,dtype=float); v = v[np.isfinite(v)]
    n = len(v)
    if not n:
        return {'events': 0, 'mean': np.nan, 'median': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                'positive_fraction': np.nan, 'leave_one_min': np.nan, 'leave_one_max': np.nan}
    ci = [np.nan, np.nan]
    if n >= 8:
        rng = np.random.default_rng(20261014)
        ci = np.quantile(v[rng.integers(0,n,size=(2000,n))].mean(axis=1),[.025,.975])
    leave = (v.sum()-v)/(n-1) if n>1 else np.array([np.nan])
    return {'events': n, 'mean': v.mean(), 'median': np.median(v), 'ci_low': ci[0], 'ci_high': ci[1],
            'positive_fraction': (v>reference).mean(), 'leave_one_min': min(leave), 'leave_one_max': max(leave)}


def matched_event(treated, control, columns, metrics):
    matched=[]
    for key,g in treated.groupby(columns,observed=True):
        if len(columns)==1: key=(key[0],) if isinstance(key,tuple) else (key,)
        c=control
        for col,value in zip(columns,key): c=c.loc[c[col].eq(value)]
        if len(c): matched.append((g,c,len(g)))
    if not matched:return None
    total=sum(n for _,_,n in matched)
    row={'treated_orders':len(treated),'treated_matched':total,'control_rows':sum(len(c) for _,c,_ in matched),
         'coverage':total/len(treated),'matched_cells':len(matched)}
    for metric in metrics:
        # Missing market values must not become a healthy zero. Use the same complete
        # cells on both sides; report separate support for each outcome.
        cells=[(g,c,n) for g,c,n in matched if g[metric].notna().all() and c[metric].notna().all()]
        support=sum(n for _,_,n in cells)
        row[metric+'_support']=support
        for name,pos in [('treated',0),('control',1)]:
            row[metric+'_'+name]=sum(pair[pos][metric].mean()*pair[2] for pair in cells)/support if support else np.nan
        row[metric+'_difference']=row[metric+'_treated']-row[metric+'_control']
    return row
