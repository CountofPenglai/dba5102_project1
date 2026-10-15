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
from pp_models.base_multiplier_study import AccessGuard
from pp_models.data import load_pool
from pp_models.frozen_candidate_study import canonical_identity
from pp_models.macro_multiplier_features import BASIS
from pp_models.macro_multiplier_study import pool_features
from pp_models.scoped_data import development_snapshot
from pp_models.study_storage import atomic_json, file_digest
from scripts.verify_frozen_candidates import metrics
from scripts.verify_h0_hourly_amount import h0_values
from scripts.verify_h0_blend_fine import quadratic_check


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


def verify(root):
    root = Path(root)
    read = lambda name: json.loads((root/'provenance'/name).read_text())
    manifest, pre, protocol, states = (read(name) for name in
        ['manifest.json', 'preflight.json', 'protocol.json', 'parameters.json'])
    assert manifest['status'] == 'complete' and manifest['official_validation_open_attempts'] == 0
    for name, digest in manifest['artifacts'].items(): assert file_digest(root/name) == digest, name
    with ZipFile(root/'provenance/source.zip') as archive:
        for name, digest in pre['sources'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    guard = AccessGuard(); guard.allow_pressure = True; sys.addaudithook(guard.check)
    data_dir = Path(pre['data_dir'])
    assert canonical_identity(data_dir, include_test=True) == pre['canonical']
    assert development_snapshot(data_dir)[0] == pre['development']
    pool = load_pool('test', {'cohort': 'benchmark', 'use_base': True}, data_dir)
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
    calendar = json.loads((PROJECT/'external_data/macro_events/snapshots/20261014T123002Z.json').read_text())
    features = pool_features(pool, calendar)
    cohorts = pd.read_csv(folder/'training_cohorts.csv').set_index('origin')
    frames = []; origins = []; support_checks = 0
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
        for fit in [f for f in states['announcement'] if f['origin'] == key]:
            f = features.loc[h]
            f = f.loc[f.calendar_status.eq('known_event') & f.event_time.lt(origin) & f[BASIS].gt(0).any(axis=1)]
            assert fit['training_orders'] == len(f) and fit['training_events'] == f.event_id.nunique()
            for scope, minimum in [('shared', (30, 5, 2)), ('call', (60, 8, 3)), ('put', (60, 8, 3))]:
                g = f if scope == 'shared' else f.loc[f.option_type.eq(0 if scope == 'call' else 1)]
                for i, b in enumerate(BASIS):
                    z = g.loc[g[b].gt(0)]; counts = [len(z), z.event_id.nunique(), z.event_month.nunique()]
                    expected = dict(basis=b, orders=counts[0], events=counts[1], months=counts[2],
                                    enabled=all(a >= c for a, c in zip(counts, minimum)))
                    assert fit['support'][scope][i] == expected
                    enabled = expected['enabled'] and (scope == 'shared' or fit['shared_enabled'][i])
                    actual = fit['shared_enabled'][i] if scope == 'shared' else fit['direction_enabled'][0 if scope == 'call' else 1][i]
                    assert enabled == actual
                    if not enabled:
                        assert fit['coefficients'][i+(0 if scope == 'shared' else 3 if scope == 'call' else 6)] == 0
                    support_checks += 1
            fq = features.loc[q]; v = fq[BASIS].to_numpy()
            design = np.column_stack([v*fit['shared_enabled'],
                *[v*fit['direction_enabled'][side]*fq.option_type.eq(side).to_numpy()[:, None] for side in (0, 1)]])
            delta = design@np.array(fit['coefficients'])
            assert np.abs(delta).max() <= .05+1e-10
            predicted[fit['model']] = bp*(1+np.clip(delta, -.05, .05))
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
    refs = pd.read_csv(folder/'reference_checks.csv'); assert refs.reproduced.all() and len(refs) == 5
    out = dict(status='passed', weekly_cohorts=len(origins), models_daily_checked=len(macro),
        daily_aggregate_rows_checked=count, independently_replayed_models=len(replay_names),
        independently_replayed_summary_rows=replay_rows, quadratic_rmse_models_checked=quadratic,
        announcement_support_cells_checked=support_checks, risk_rows_checked=risk_rows, model_fit_calls=0,
        official_validation_open_attempts=guard.official_attempts,
        scope='hashes, causal cohorts, H0 parameter identity, all daily moments, fixed and announcement replay; uncertainty uses existing tested statistics implementation')
    atomic_json(root/'provenance/independent_verification.json', out)
    print(json.dumps(out)); return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root'); verify(p.parse_args().root)
