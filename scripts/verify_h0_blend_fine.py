"""Independent fixed-grid replay and RMSE reconstruction from old daily anchors."""
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
from pp_models.scoped_data import development_snapshot
from pp_models.study_storage import atomic_json, file_digest
from scripts.verify_frozen_candidates import metrics
from scripts.verify_h0_hourly_amount import h0_values


def quadratic_check(reference, fixed, macro, pooled):
    daily = pd.read_csv(reference/'metrics/validation/daily.csv', dtype={'value': str})
    daily = daily.loc[daily.group.eq('direction') & daily.model.isin(['official_raw', 'h0', 'half_blend'])].copy()
    daily['sse'] = daily.orders*daily.rmse_per_btc**2
    piv = daily.pivot(index=['day', 'month', 'value'], columns='model', values='sse')
    n = daily.loc[daily.model.eq('official_raw')].set_index(['day', 'month', 'value']).orders.reindex(piv.index)
    d2 = 2*(piv.h0+piv.official_raw)-4*piv.half_blend
    assert (d2 >= -1e-5).all()
    checks = 0
    for name, (c, p) in fixed.items():
        a = np.where(piv.index.get_level_values('value') == '0', c, p)
        sse = (1-a)*piv.official_raw+a*piv.h0-a*(1-a)*d2
        rates = sse.groupby(level='month').sum()/n.groupby(level='month').sum()
        expected_macro = np.sqrt(rates.mean()); expected_pooled = np.sqrt(sse.sum()/n.sum())
        np.testing.assert_allclose(macro.loc[name, 'macro_rmse_per_btc'], expected_macro, rtol=1e-10, atol=1e-7)
        np.testing.assert_allclose(pooled.loc[name, 'rmse_per_btc'], expected_pooled, rtol=1e-10, atol=1e-7)
        checks += 1
    return checks


def verify(root):
    root = Path(root); manifest = json.loads((root/'provenance/manifest.json').read_text())
    assert manifest['status'] == 'complete'
    pre = json.loads((root/'provenance/preflight.json').read_text())
    config = json.loads((root/'provenance/config.json').read_text())
    protocol = json.loads((root/'provenance/protocol.json').read_text())
    for name, digest in manifest['artifacts'].items(): assert file_digest(root/name) == digest, name
    with ZipFile(root/'provenance/source.zip') as archive:
        for name, digest in pre['sources'].items(): assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    for name, digest in {**pre['sources'], **pre['protected']}.items(): assert file_digest(PROJECT/name) == digest, name
    guard = AccessGuard(); guard.allow_pressure = True; sys.addaudithook(guard.check)
    assert canonical_identity(include_test=True) == pre['canonical']
    assert development_snapshot()[0] == pre['development_data']
    pool = load_pool('test', {'cohort': 'benchmark', 'use_base': True})
    created, closed = pool.metadata.created_at_utc, pool.metadata.closed_at_utc
    start, end, first = (pd.Timestamp(protocol[k]) for k in ('validation_start', 'validation_end', 'training_start'))
    states = json.loads((root/'provenance/parameters.json').read_text())
    reference = PROJECT/config['reference_run']; refs = json.loads((reference/'provenance/parameters.json').read_text())
    expected = [o.isoformat() for o in pd.date_range(start, end, freq='7D', inclusive='left')
        if (created.ge(o) & created.lt(min(end, o+pd.Timedelta(days=7)))).any()]
    assert sorted(states) == expected
    assert len(config['fixed_ratios']) == 67 and len(config['fine_models']) == 63
    assert sorted(tuple(config['fixed_ratios'][m]) for m in config['fine_models']) == [
        (c/100, p/100) for c in range(25, 66, 5) for p in range(10, 41, 5)]
    assert len(config['pairs']) == 191 and len(set(map(tuple, config['pairs']))) == 191
    frames = []
    for origin_text, state in states.items():
        origin = pd.Timestamp(origin_text); stop = min(end, origin+pd.Timedelta(days=7))
        assert state == {k: v for k, v in refs[origin_text].items() if k != 'weights'}
        assert pd.Timestamp(state['stop']) == stop and state['is_validation'] and state['h0_supported']
        h = pool.X.index[created.ge(max(first, origin-pd.Timedelta(days=30))) & created.lt(origin) & closed.lt(origin)]
        queries = pool.X.index[created.ge(origin) & created.lt(stop)]
        assert len(h) == state['history_orders'] and len(queries) == state['query_orders'] and len(h) >= 100
        assert created.loc[h].min().isoformat() == state['earliest_training_create']
        assert closed.loc[h].max().isoformat() == state['latest_training_close']
        q = queries[closed.loc[queries].lt(end)]; x = pool.X.loc[q]
        p, s = h0_values(x, state['h0']); hp = p*s; bp = x[BASE_FEATURE].to_numpy()
        threshold = pd.Series(index=q, dtype=float)
        for month in sorted(created.loc[q].dt.strftime('%Y-%m').unique()):
            mo = pd.Timestamp(month+'-01', tz='UTC'); mq = q[created.loc[q].dt.strftime('%Y-%m').eq(month)]
            past = pool.X.index[created.ge(first) & created.lt(mo) & closed.lt(mo)]
            for (side, tenor), part in pool.X.loc[mq].groupby(['option_type', 'product_duration_seconds']):
                ids = past[pool.X.loc[past, 'option_type'].eq(side)]
                cell = ids[pool.X.loc[ids, 'product_duration_seconds'].eq(tenor)]; ids = cell if len(cell) >= 30 else ids
                threshold.loc[part.index] = pool.y.loc[ids].quantile(.9) if len(ids) >= 30 else np.nan
        for name, pair in config['fixed_ratios'].items():
            a = np.where(x.option_type.eq(0), pair[0], pair[1]); pred = a*hp+(1-a)*bp
            frames.append(pd.DataFrame(dict(model=name, day=created.loc[q].dt.floor('D'), month=created.loc[q].dt.strftime('%Y-%m'),
                side=x.option_type, tenor=x.product_duration_seconds, quantity=x.quantity_btc, actual=pool.y.loc[q], prediction=pred,
                tail_group=np.where(threshold.isna(), 'unavailable', np.where(pool.y.loc[q] > threshold, 'tail', 'body'))), index=q))
    frame = pd.concat(frames); saved = pd.read_csv(root/'metrics/validation/summary.csv', dtype={'value': str}); largest = 0.
    for row in saved.itertuples():
        part = frame.loc[frame.model.eq(row.model)]
        if row.month != 'all': part = part.loc[part.month.eq(row.month)]
        if row.group == 'direction': part = part.loc[part.side.eq(int(row.value))]
        elif row.group == 'tenor': part = part.loc[part.tenor.eq(int(row.value))]
        elif row.group == 'direction_tail':
            side, tail = row.value.split('|'); part = part.loc[part.side.eq(int(side)) & part.tail_group.eq(tail)]
        for name, value in metrics(part).items():
            largest = max(largest, abs(value-getattr(row, name)))
            np.testing.assert_allclose(value, getattr(row, name), rtol=1e-8, atol=1e-6)
    macro = pd.read_csv(root/'metrics/validation/macro_monthly.csv').set_index('model')
    for model, part in frame.groupby('model'):
        mses = []; maes = []
        for _, month in part.groupby('month'):
            e = month.prediction-month.actual; mses.append(float(np.mean(e**2))); maes.append(float(np.mean(abs(e))))
        np.testing.assert_allclose(macro.loc[model, ['macro_rmse_per_btc', 'macro_mae_per_btc']].to_numpy(float),
                                   [np.sqrt(np.mean(mses)), np.mean(maes)], rtol=1e-10, atol=1e-8)
    prior = pd.read_csv(reference/'metrics/validation/summary.csv', dtype={'value': str})
    for name in manifest['overlapping_reference_models']:
        pd.testing.assert_frame_equal(saved.loc[saved.model.eq(name)].reset_index(drop=True),
            prior.loc[prior.model.eq(name)].reset_index(drop=True), rtol=1e-10, atol=1e-7)
    pooled = saved.loc[saved.month.eq('all') & saved.group.eq('overall')].set_index('model')
    quadratic = quadratic_check(reference, config['fixed_ratios'], macro, pooled)
    grid = pd.read_csv(root/'metrics/validation/fixed_grid.csv').set_index('model')
    for name in config['fixed_ratios']:
        np.testing.assert_allclose(grid.loc[name, ['macro_rmse_per_btc', 'macro_mae_per_btc']].to_numpy(float),
                                   macro.loc[name, ['macro_rmse_per_btc', 'macro_mae_per_btc']].to_numpy(float))
        np.testing.assert_allclose(grid.loc[name, ['pooled_rmse_per_btc', 'pooled_mae_per_btc']].to_numpy(float),
                                   pooled.loc[name, ['rmse_per_btc', 'mae_per_btc']].to_numpy(float))
    assert len(frame.loc[frame.model.eq('h0')]) == manifest['coverage']['scored_orders']
    out = dict(status='passed', orders=manifest['coverage']['scored_orders'], models_checked=len(macro),
        summary_rows_checked=len(saved), prior_h0_states_checked=len(states), quadratic_rmse_models_checked=quadratic,
        largest_csv_rounding_error=largest, references_match=True, model_fit_calls=0,
        official_validation_open_attempts=guard.official_attempts,
        scope='independent H0 replay, cohorts, summary/macro errors and quadratic RMSE from old daily anchors; interval/risk cells reviewed separately')
    atomic_json(root/'provenance/independent_verification.json', out); print(json.dumps(out)); return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); verify(parser.parse_args().root)
