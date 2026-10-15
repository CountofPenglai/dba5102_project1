"""Paired calendar-block uncertainty; preserves the same fixed 2000 draws."""
import numpy as np
import pandas as pd
LOSS_COLUMNS = ["orders", "csse", "rsse", "csae", "rsae", "cbias", "rbias", "cunder", "runder"]

def scope_frames(frame):
    yield 'all', frame
    for side, label in [(0, 'call'), (1, 'put')]:
        part = frame.loc[frame.side.eq(side)]
        yield label, part
        for cohort in ('body', 'tail'):
            yield label+'_'+cohort, part.loc[part.tail_group.eq(cohort)]


def draw_weights(days, block):
    weights = np.zeros((2000, len(days)), dtype=np.int64)
    rng = np.random.default_rng(20261012 + block*1000)
    month_labels = days.strftime('%Y-%m')
    groups = [np.flatnonzero(month_labels == month) for month in sorted(set(month_labels))]
    for repeat in range(2000):
        for indices in groups:
            n = len(indices)
            starts = rng.integers(0, n, size=int(np.ceil(n/block)))
            chosen = ((starts[:, None]+np.arange(block)) % n).ravel()[:n]
            weights[repeat, indices] = np.bincount(chosen, minlength=n)
    return weights


def paired_daily(frame, candidate, reference, days):
    c, r = (frame.loc[frame.model.eq(name)] for name in (candidate, reference))
    if (c.empty or not c.index.is_unique or not c.index.equals(r.index)
            or not c[['actual','day','side','quantity','tail_group']].equals(r[['actual','day','side','quantity','tail_group']])):
        raise ValueError('Invalid final-study paired rows')
    if not c.day.isin(days).all(): raise ValueError('Scoring rows outside evaluation calendar')
    ce, re = c.prediction-c.actual, r.prediction-r.actual
    daily = pd.DataFrame({'day': c.day, 'orders': 1, 'csse': ce**2, 'rsse': re**2,
        'csae': ce.abs(), 'rsae': re.abs(), 'cbias': ce*c.quantity, 'rbias': re*c.quantity,
        'cunder': (-ce).clip(lower=0)*c.quantity, 'runder': (-re).clip(lower=0)*r.quantity})
    return daily.groupby('day').sum().reindex(days, fill_value=0)[LOSS_COLUMNS]


def differences(daily, weights, days):
    """Monthly equal unit errors; cash sums over the complete evaluation period."""
    losses, valid = [], np.ones(len(weights), dtype=bool)
    month_labels = days.strftime('%Y-%m')
    for month in sorted(set(month_labels)):
        mask = month_labels == month
        totals = weights[:, mask] @ daily.loc[mask, LOSS_COLUMNS[:5]].to_numpy(float)
        counts = totals[:, 0]
        valid &= counts > 0
        rates = np.full((len(weights), 4), np.nan)
        np.divide(totals[:, 1:], counts[:, None], out=rates, where=counts[:, None] > 0)
        losses.append(rates)
    mean = np.mean(losses, axis=0)
    cash = weights @ daily[LOSS_COLUMNS[5:]].to_numpy(float)
    result = {'rmse_delta': np.sqrt(mean[:, 0])-np.sqrt(mean[:, 1]),
        'mae_delta': mean[:, 2]-mean[:, 3], 'total_bias_delta_usdc': cash[:, 0]-cash[:, 1],
        'absolute_total_bias_delta_usdc': np.abs(cash[:, 0])-np.abs(cash[:, 1]),
        'gross_underprediction_delta_usdc': cash[:, 2]-cash[:, 3]}
    return result, valid
