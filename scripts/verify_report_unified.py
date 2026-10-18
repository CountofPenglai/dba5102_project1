"""Read-only replay of common cohorts, fixed blends and announcement parameters.

No estimator is fitted. Independently aggregate daily moments for every recipe;
replay fixed/announcement row predictions without the production predictors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path: sys.path.insert(0, str(PROJECT))
from pp_data.features import BASE_FEATURE
from pp_models.runtime import AccessGuard
from pp_data.io import load_dataset
from pp_data.io import canonical_identity
from pp_models.runtime import atomic_json, file_digest
from pp_models.verification import metrics
from pp_models.verification import h0_values
from pp_models.verification import quadratic_check


def announcement_values(created, calendar, rule):
    """Independent vectorized replay: available anchors, event maxima, capped sum."""
    at = pd.DatetimeIndex(created)
    per_event = {}
    for row in calendar['events']:
        if row.get('known_by_utc') is None: continue
        release, known = pd.Timestamp(row['event_time_utc']), pd.Timestamp(row['known_by_utc'])
        if known >= release: continue
        key = ('US_FOMC_'+release.tz_convert('America/New_York').strftime('%Y-%m-%d')
               if row['family'] == 'FOMC' else row['event_id'])
        seconds = (at-release).total_seconds().to_numpy()
        value = np.zeros(len(at)); before = seconds < 0
        value[before] = 2.**(seconds[before]/(3600*rule['pre_half_life_hours']))
        value[~before] = np.maximum(0., 1-seconds[~before]/(60*rule['post_minutes']))
        value[at < known] = 0.
        per_event[key] = np.maximum(per_event.get(key, np.zeros(len(at))), value)
    adjustment = np.minimum(rule['total_cap'], sum(per_event.values(), np.zeros(len(at)))*rule['per_event_cap'])
    for coverage in calendar['coverage'].values():
        lower = pd.Timestamp(coverage['start'], tz='America/New_York')
        upper = pd.Timestamp(coverage['end_exclusive'], tz='America/New_York')
        outside = (at-pd.Timedelta(minutes=rule['post_minutes']) < lower) | (at+pd.Timedelta(microseconds=1) > upper)
        adjustment[outside] = 0.
    return pd.Series(1+adjustment, index=created.index)


def check_daily(summary, daily, macro):
    """Check all models' monthly/pooled metrics from independently summed moments."""
    rows = 0
    for row in summary.loc[summary.group.isin(['overall', 'direction'])].itertuples():
        part = daily.loc[daily.model.eq(row.model) & daily.group.eq(row.group) & daily.value.eq(row.value)]
        if row.month != 'all': part = part.loc[part.month.eq(row.month)]
        n = part.orders.sum()
        assert n == row.orders and n > 0
        assert pd.to_datetime(part.day).nunique() == row.days
        np.testing.assert_allclose([np.sqrt((part.orders*part.rmse_per_btc**2).sum()/n),
            (part.orders*part.mae_per_btc).sum()/n], [row.rmse_per_btc, row.mae_per_btc], atol=1e-8)
        for col in ['actual_total_usdc', 'predicted_total_usdc', 'total_bias_usdc',
                    'gross_overprediction_usdc', 'gross_underprediction_usdc']:
            np.testing.assert_allclose(part[col].sum(), getattr(row, col), rtol=1e-10, atol=1e-7)
        rows += 1
    for row in macro.itertuples():
        m = summary.loc[summary.model.eq(row.model) & summary.group.eq('overall') & summary.month.ne('all')]
        assert len(m) == 3 and m.orders.sum() == row.orders
        np.testing.assert_allclose([np.sqrt(np.mean(m.rmse_per_btc**2)), m.mae_per_btc.mean()],
                                  [row.macro_rmse_per_btc, row.macro_mae_per_btc], atol=1e-8)
    return rows


def verify(root, data_dir=None):
    root = Path(root)
    read = lambda name: json.loads((root/'provenance'/name).read_text())
    manifest, pre, protocol, states = (read(name) for name in
        ['manifest.json', 'preflight.json', 'protocol.json', 'parameters.json'])
    assert manifest['producer'] == 'report_unified_v3', 'Use the archived source to verify an earlier producer'
    assert manifest['status'] == 'complete' and manifest['official_validation_open_attempts'] == 0
    for name, digest in manifest['artifacts'].items(): assert file_digest(root/name) == digest, name
    with ZipFile(root/'provenance/source.zip') as archive:
        for name, digest in pre['sources'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    guard = AccessGuard(); sys.addaudithook(guard.check)
    # A checkout can move; the dataset identity, not its original absolute path,
    # must still match the immutable run provenance.
    data_dir = Path(pre['data_dir'] if data_dir is None else data_dir)
    if canonical_identity(data_dir) != pre['canonical']:
        raise ValueError('Replay dataset differs from the recorded report data')
    pool = load_dataset(data_dir, cohort='benchmark', use_base=True)
    created, closed = pool.metadata.created_at_utc, pool.metadata.closed_at_utc
    start, end, first = (pd.Timestamp(protocol[k]) for k in ['validation_start', 'validation_end', 'training_start'])
    folder = root/'metrics/validation'
    summary = pd.read_csv(folder/'summary.csv', dtype={'value': str})
    daily = pd.read_csv(folder/'daily.csv', dtype={'value': str})
    macro = pd.read_csv(folder/'macro_monthly.csv')
    count = check_daily(summary, daily, macro)
    risk_rows = 0
    lookup = summary.set_index(['month', 'model', 'group', 'value'])
    for filename in ['risk_constraints.csv', 'monthly_risk_constraints.csv']:
        for row in pd.read_csv(folder/filename).itertuples():
            month = getattr(row, 'month', 'all'); side = str(row.side)
            a = lookup.loc[(month, row.candidate, 'direction', side)]
            b = lookup.loc[(month, row.reference, 'direction', side)]
            at = lookup.loc[(month, row.candidate, 'direction_tail', side+'|tail')]
            bt = lookup.loc[(month, row.reference, 'direction_tail', side+'|tail')]
            aa = lookup.loc[(month, row.candidate, 'overall', 'all')]
            bb = lookup.loc[(month, row.reference, 'overall', 'all')]
            pairs = [(at.rmse_per_btc, bt.rmse_per_btc),
                (a.gross_underprediction_usdc, b.gross_underprediction_usdc),
                (at.gross_underprediction_usdc, bt.gross_underprediction_usdc),
                (abs(a.total_bias_usdc), abs(b.total_bias_usdc)), (abs(aa.total_bias_usdc), abs(bb.total_bias_usdc))]
            ratios = [v/w if w else 1. if not v else np.inf for v, w in pairs]
            np.testing.assert_allclose(ratios, [row.tail_rmse_ratio, row.gross_under_ratio,
                row.tail_gross_under_ratio, row.absolute_bias_ratio, row.overall_absolute_bias_ratio], atol=1e-9)
            expected = row.support_sufficient and all(v <= 1.02 for v in ratios[:3]) and all(v <= 1+1e-10 for v in ratios[3:])
            assert bool(row.direction_risk_pass) == bool(expected)
            risk_rows += 1
    grid = pd.read_csv(folder/'fixed_grid.csv')
    fixed = {r.model: (r.call_h0_weight, r.put_h0_weight) for r in grid.itertuples()}
    rule = states['announcement']
    assert rule['model_fit_calls'] == manifest['fits']['announcement'] == 0
    with ZipFile(root/'provenance/source.zip') as archive:
        calendar_bytes = archive.read(rule['calendar'])
    assert hashlib.sha256(calendar_bytes).hexdigest() == rule['calendar_sha256']
    calendar = json.loads(calendar_bytes)
    multipliers = announcement_values(created, calendar, rule)
    cohorts = pd.read_csv(folder/'training_cohorts.csv').set_index('origin')
    frames = []; origins = []; announcement_checks = 0
    for origin in pd.date_range(start, end, freq='7D', inclusive='left'):
        stop = min(end, origin+pd.Timedelta(days=7)); key = origin.isoformat()
        queries = pool.X.index[created.ge(origin) & created.lt(stop)]
        if not len(queries): continue
        origins.append(key)
        h = pool.X.index[created.ge(max(first, origin-pd.Timedelta(days=30))) & created.lt(origin) & closed.lt(origin)]
        c = cohorts.loc[key]
        assert c.history_orders == len(h) and c.query_orders == len(queries)
        assert c.history_ids_sha256 == hashlib.sha256('\n'.join(h.astype(str)).encode()).hexdigest()
        assert c.earliest_training_create == created.loc[h].min().isoformat()
        assert c.latest_training_close == closed.loc[h].max().isoformat()
        h0 = states['coarse'][key]['h0']
        assert h0 == states['state_fallback'][key]['h0'] == states['hourly'][key]['h0']
        assert h0 == states['time_decay'][key]['models']['h0']['parameters']
        for family in ['coarse', 'state_fallback', 'hourly', 'time_decay']:
            st = states[family][key]
            assert st['history_orders'] == len(h) and st['query_orders'] == len(queries)
            assert pd.Timestamp(st['stop']) == stop
        q = queries[closed.loc[queries].lt(end)]; x = pool.X.loc[q]
        p, s = h0_values(x, h0); bp = x[BASE_FEATURE].to_numpy()
        predicted = {name: (np.where(x.option_type.eq(0), ca, pa)*(p*s-bp)+bp)
                     for name, (ca, pa) in fixed.items()}
        assert multipliers.loc[q].between(1., 1+rule['total_cap']).all()
        predicted[rule['model']] = bp*multipliers.loc[q].to_numpy()
        announcement_checks += len(q)
        threshold = pd.Series(index=q, dtype=float)
        for month in created.loc[q].dt.strftime('%Y-%m').unique():
            mo = pd.Timestamp(month+'-01', tz='UTC'); mq = q[created.loc[q].dt.strftime('%Y-%m').eq(month)]
            past = pool.X.index[created.ge(first) & created.lt(mo) & closed.lt(mo)]
            for (side, tenor), part in pool.X.loc[mq].groupby(['option_type', 'product_duration_seconds']):
                ids = past[pool.X.loc[past, 'option_type'].eq(side)]
                cell = ids[pool.X.loc[ids, 'product_duration_seconds'].eq(tenor)]
                ids = cell if len(cell) >= 30 else ids
                threshold.loc[part.index] = pool.y.loc[ids].quantile(.9) if len(ids) >= 30 else np.nan
        for name, prediction in predicted.items():
            frames.append(pd.DataFrame(dict(model=name, month=created.loc[q].dt.strftime('%Y-%m'),
                day=created.loc[q].dt.floor('D'), side=x.option_type, tenor=x.product_duration_seconds,
                quantity=x.quantity_btc, actual=pool.y.loc[q], prediction=prediction,
                tail_group=np.where(threshold.isna(), 'unavailable', np.where(pool.y.loc[q] > threshold, 'tail', 'body'))), index=q))
    assert sorted(states['hourly']) == sorted(states['time_decay']) == sorted(cohorts.index) == origins
    frame = pd.concat(frames); replay_names = set(frame.model); replay_rows = 0
    for row in summary.loc[summary.model.isin(replay_names)].itertuples():
        part = frame.loc[frame.model.eq(row.model)]
        if row.month != 'all': part = part.loc[part.month.eq(row.month)]
        if row.group == 'direction': part = part.loc[part.side.eq(int(row.value))]
        elif row.group == 'tenor': part = part.loc[part.tenor.eq(int(row.value))]
        elif row.group == 'direction_tail':
            side, tail = row.value.split('|'); part = part.loc[part.side.eq(int(side)) & part.tail_group.eq(tail)]
        for name, value in metrics(part).items():
            np.testing.assert_allclose(value, getattr(row, name), rtol=1e-8, atol=1e-6)
        replay_rows += 1
    pooled = summary.loc[summary.month.eq('all') & summary.group.eq('overall')].set_index('model')
    quadratic = quadratic_check(root, fixed, macro.set_index('model'), pooled)
    refs = pd.read_csv(folder/'reference_checks.csv')
    assert refs.reproduced.all() and len(refs) == 1 and refs.models.eq(len(macro)-1).all()
    out = dict(status='passed', weekly_cohorts=len(origins), models_daily_checked=len(macro),
        daily_aggregate_rows_checked=count, independently_replayed_models=len(replay_names),
        independently_replayed_summary_rows=replay_rows, quadratic_rmse_models_checked=quadratic,
        announcement_orders_independently_replayed=announcement_checks, risk_rows_checked=risk_rows, model_fit_calls=0,
        official_validation_open_attempts=guard.official_attempts,
        scope='hashes, causal cohorts, H0 parameter identity, all daily moments, fixed and announcement replay; uncertainty uses existing tested statistics implementation')
    atomic_json(root/'provenance/independent_verification.json', out)
    print(json.dumps(out)); return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root')
    p.add_argument('--data-dir', type=Path, help='Relocated dataset; its identity must match the recorded run')
    args = p.parse_args()
    verify(args.root, args.data_dir)
