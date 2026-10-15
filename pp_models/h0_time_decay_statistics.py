"""Fixed paired comparisons and stage diagnostics for the time-decay experiment."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .two_stage_study import aggregate, metric_values
from .frozen_candidate_statistics import scoped_macro, ratio
from .put_final_statistics import scope_frames, paired_daily, differences, draw_weights

PAIRS = [('h0', 'official_raw')] + [(c, r) for c in ('h0_decay7', 'h0_decay14') for r in ('h0', 'official_raw')]


def risks(frame, scoped):
    rows = []; sm = scoped.set_index(['scope', 'model'])
    for c, r in PAIRS:
        total = {m: metric_values(frame.loc[frame.model.eq(m)]) for m in (c, r)}
        for side, label in ((0, 'call'), (1, 'put')):
            parts = {m: frame.loc[frame.model.eq(m) & frame.side.eq(side)] for m in (c, r)}
            tail = {m: p.loc[p.tail_group.eq('tail')] for m, p in parts.items()}
            complete = all((label+'_tail', m) in sm.index and bool(sm.loc[(label+'_tail', m), 'complete']) for m in (c, r))
            enough = complete and all(len(p) >= 30 and p.day.nunique() >= 5 for p in tail.values())
            row = dict(candidate=c, reference=r, side=side, tail_orders=len(tail[c]), tail_days=tail[c].day.nunique(),
                support_sufficient=enough, direction_risk_pass=False)
            if all(not p.empty for p in [*parts.values(), *tail.values()]):
                mv = {m: metric_values(p) for m, p in parts.items()}; tv = {m: metric_values(p) for m, p in tail.items()}
                row.update(tail_rmse_ratio=ratio(tv[c]['rmse_per_btc'], tv[r]['rmse_per_btc']),
                    gross_under_ratio=ratio(mv[c]['gross_underprediction_usdc'], mv[r]['gross_underprediction_usdc']),
                    tail_gross_under_ratio=ratio(tv[c]['gross_underprediction_usdc'], tv[r]['gross_underprediction_usdc']),
                    absolute_bias_ratio=ratio(mv[c]['absolute_total_bias_usdc'], mv[r]['absolute_total_bias_usdc']),
                    overall_absolute_bias_ratio=ratio(total[c]['absolute_total_bias_usdc'], total[r]['absolute_total_bias_usdc']))
                row['direction_risk_pass'] = bool(enough and row['tail_rmse_ratio'] <= 1.02 and row['gross_under_ratio'] <= 1.02
                    and row['tail_gross_under_ratio'] <= 1.02 and row['absolute_bias_ratio'] <= 1+1e-10
                    and row['overall_absolute_bias_ratio'] <= 1+1e-10)
            rows.append(row)
    return pd.DataFrame(rows)


def stage_diagnostics(frame):
    rows = []
    for (month, model, side), part in frame.loc[~frame.model.eq('official_raw')].groupby(['month', 'model', 'side']):
        positive = part.actual.gt(0); pos = part.loc[positive]
        rows.append(dict(month=month, model=model, side=side, orders=len(part), positive_orders=len(pos),
            brier=float(np.mean((part.probability-positive.astype(int))**2)),
            mean_probability=part.probability.mean(), payout_frequency=positive.mean(),
            positive_severity_rmse=float(np.sqrt(np.mean((pos.severity-pos.actual)**2))) if len(pos) else np.nan,
            h0_rmse=float(np.sqrt(np.mean((part.h0_probability*part.h0_severity-part.actual)**2))),
            only_probability_rmse=float(np.sqrt(np.mean((part.probability*part.h0_severity-part.actual)**2))),
            only_amount_rmse=float(np.sqrt(np.mean((part.h0_probability*part.severity-part.actual)**2))),
            both_rmse=float(np.sqrt(np.mean((part.prediction-part.actual)**2)))))
    return pd.DataFrame(rows)


def tables(frame, start, end, uncertainty=True):
    days = pd.date_range(pd.Timestamp(start).floor('D'), pd.Timestamp(end).ceil('D'), inclusive='left')
    months = sorted(set(days.strftime('%Y-%m')))
    summary, daily, macro = aggregate(frame); scoped = scoped_macro(frame, months)
    if not macro.months.eq(len(months)).all(): raise ValueError('Missing validation month')
    lookup = macro.set_index('model'); compare = []; cells = []
    for c, r in PAIRS:
        a, b = lookup.loc[c], lookup.loc[r]
        compare.append(dict(candidate=c, reference=r, rmse_delta=a.macro_rmse_per_btc-b.macro_rmse_per_btc,
            rmse_change=ratio(a.macro_rmse_per_btc, b.macro_rmse_per_btc)-1,
            mae_delta=a.macro_mae_per_btc-b.macro_mae_per_btc, mae_role='cost_monitor', selection_performed=False))
    for (month, model, side, tenor, tail), p in frame.groupby(['month', 'model', 'side', 'tenor', 'tail_group']):
        cells.append(dict(month=month, model=model, side=side, tenor=tenor, tail_group=tail,
            days=p.day.nunique(), **metric_values(p)))
    result = dict(summary=summary, daily=daily, macro_monthly=macro, scoped_macro=scoped,
        comparisons=pd.DataFrame(compare), risk_constraints=risks(frame, scoped),
        direction_tenor=pd.DataFrame(cells), stage_diagnostics=stage_diagnostics(frame))
    if not uncertainty: return result
    intervals = []; influence = []; sufficient = []; draws = {b: draw_weights(days, b) for b in (3, 7, 14)}
    with threadpool_limits(limits=1):
        for scope, part in scope_frames(frame):
            if part.empty: continue
            for c, r in PAIRS:
                stats = paired_daily(part, c, r, days); point, pv = differences(stats, np.ones((1, len(days))), days)
                sufficient.append(stats.reset_index(names='day').assign(scope=scope, candidate=c, reference=r))
                for block, w in draws.items():
                    delta, valid = differences(stats, w, days)
                    for metric, values in delta.items():
                        use = valid & np.isfinite(values); n = int(use.sum()); supported = bool(pv[0] and n >= 1900)
                        lo, hi = np.quantile(values[use], [.025, .975]) if supported else (np.nan, np.nan)
                        fl, fh = (np.quantile(values[use], [.05/(2*len(PAIRS)), 1-.05/(2*len(PAIRS))])
                            if supported and scope == 'all' and metric == 'rmse_delta' else (np.nan, np.nan))
                        intervals.append(dict(scope=scope, candidate=c, reference=r, block_days=block, metric=metric,
                            point_delta=float(point[metric][0]) if pv[0] else np.nan, valid_repeats=n, supported=supported,
                            lower_95=lo, upper_95=hi, simultaneous_lower=fl, simultaneous_upper=fh))
                if scope not in ('all', 'call', 'put'): continue
                for width in (1, 7):
                    n = len(days)-width+1; w = np.ones((n, len(days)))
                    for j in range(n): w[j, j:j+width] = 0
                    delta, valid = differences(stats, w, days)
                    for metric in ('rmse_delta', 'mae_delta', 'gross_underprediction_delta_usdc', 'absolute_total_bias_delta_usdc'):
                        values = delta[metric]; use = valid & np.isfinite(values)
                        influence.append(dict(scope=scope, candidate=c, reference=r, deleted_days=width, metric=metric,
                            valid_windows=int(use.sum()), total_windows=n, improvement_windows=int((values[use] < 0).sum()),
                            min_delta=float(values[use].min()) if use.any() else np.nan,
                            max_delta=float(values[use].max()) if use.any() else np.nan))
    result.update(intervals=pd.DataFrame(intervals), influence=pd.DataFrame(influence), paired_daily=pd.concat(sufficient, ignore_index=True))
    return result
