"""Final-study paired calendar statistics; July is evaluated on its own calendar."""
import hashlib

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .two_stage_component_study import risk_constraints
from .two_stage_robustness import interval_summary
from .two_stage_study import aggregate

PAIRS = [('put_combined', 'h0'), ('put_combined', 'official_raw'), ('h0', 'official_raw')]
LOSS_COLUMNS = ['orders', 'csse', 'rsse', 'csae', 'rsae', 'cbias', 'rbias', 'cunder', 'runder']


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


def build_tables(frame, start, end, uncertainty=True):
    summary, daily, macro = aggregate(frame)
    risk = risk_constraints(frame, candidates=('put_combined',))
    lookup = macro.set_index('model')
    all_rows = summary.query("month == 'all' and group == 'overall'").set_index('model')
    row = lookup.loc['put_combined']; reference = lookup.loc['h0']
    risk_pass = bool(risk.query("model == 'put_combined'").direction_pass.all()
                     and all_rows.loc['put_combined','absolute_total_bias_usdc'] <= all_rows.loc['official_raw','absolute_total_bias_usdc'])
    assessment = pd.DataFrame([{'model': 'put_combined', 'rmse_delta_vs_h0': row.macro_rmse_per_btc-reference.macro_rmse_per_btc,
        'mae_delta_vs_h0': row.macro_mae_per_btc-reference.macro_mae_per_btc,
        'accuracy_vs_h0_pass': bool(row.macro_rmse_per_btc < reference.macro_rmse_per_btc and row.macro_mae_per_btc <= reference.macro_mae_per_btc),
        'risk_working_proposal_pass': risk_pass, 'selection_performed': False}])
    tables = {'summary': summary, 'daily': daily, 'macro_monthly': macro, 'risk_constraints': risk, 'assessment': assessment}
    if not uncertainty: return tables
    days = pd.date_range(pd.Timestamp(start).floor('D'), pd.Timestamp(end).ceil('D'), inclusive='left')
    draws = {block: draw_weights(days, block) for block in (3,7,14)}
    intervals, influence = [], []
    with threadpool_limits(limits=1):
        for scope, part in scope_frames(frame):
            if part.empty: continue
            for candidate, reference in PAIRS:
                stats = paired_daily(part, candidate, reference, days)
                point, point_valid = differences(stats, np.ones((1,len(days))), days)
                for block, weights in draws.items():
                    deltas, valid = differences(stats, weights, days)
                    for metric, values in deltas.items():
                        intervals.append({'scope': scope, 'candidate': candidate, 'reference': reference,
                            'block_days': block, 'metric': metric, 'point_delta': float(point[metric][0]) if point_valid[0] else np.nan,
                            'calendar_draws_sha256': hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest(),
                            **interval_summary(values[valid])})
                if scope not in ('all','call','put'): continue
                for width, label in [(1,'single_day'), (7,'seven_day')]:
                    windows = len(days)-width+1
                    weights = np.ones((windows,len(days)))
                    for i in range(windows): weights[i,i:i+width]=0
                    deltas, valid = differences(stats, weights, days)
                    for metric in ('rmse_delta','mae_delta'):
                        values = deltas[metric][valid]; positions = np.flatnonzero(valid)
                        influence.append({'scope': scope, 'candidate': candidate, 'reference': reference,
                            'window_type': label, 'metric': metric, 'full_delta': point[metric][0],
                            'minimum_delta': float(values.min()) if len(values) else np.nan,
                            'maximum_delta': float(values.max()) if len(values) else np.nan,
                            'sign_flip_windows': int((np.sign(values)!=np.sign(point[metric][0])).sum()),
                            'valid_windows': int(valid.sum()), 'total_windows': windows,
                            'maximum_start': days[positions[values.argmax()]].isoformat() if len(values) else None})
    tables.update(paired_intervals=pd.DataFrame(intervals), date_influence=pd.DataFrame(influence))
    return tables
