"""Independent minute-window and saved-parameter replay; no estimator fit calls."""
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
from pp_data.config import DEFAULT_DATA
from pp_data.features import BASE_FEATURE
from pp_models.base_multiplier_study import AccessGuard
from pp_models.data import load_pool
from pp_models.put_final_io import read_test_market_prefix
from pp_models.frozen_candidate_study import canonical_identity
from pp_models.scoped_data import development_snapshot
from pp_models.study_storage import atomic_json, file_digest
from scripts.verify_frozen_candidates import metrics


def independent_market(created, market):
    closes = pd.DatetimeIndex(market.close_time_utc).asi8; opens = pd.DatetimeIndex(market.open_time_utc).asi8
    logs = np.log(market.close.to_numpy()); positions = np.searchsorted(closes, pd.DatetimeIndex(created).asi8, side='right')-1
    rows = []
    for stamp, j in zip(pd.DatetimeIndex(created).asi8, positions):
        row = {}
        for w in (60, 240):
            good = j >= w and stamp-closes[j] <= 60_000_000_000 and np.all(np.diff(opens[j-w:j+1]) == 60_000_000_000)
            r = np.diff(logs[j-w:j+1]) if good else np.full(w, np.nan)
            row[f'up_vol_{w}m'] = float(np.sqrt(np.mean(np.maximum(r, 0)**2)))
            row[f'down_vol_{w}m'] = float(np.sqrt(np.mean(np.minimum(r, 0)**2)))
        row['available'] = all(np.isfinite(v) for v in row.values()); rows.append(row)
    return pd.DataFrame(rows, index=created.index)


def feature_values(x, f, anchor):
    v = np.where(x.option_type.eq(0), f.up_vol_60m, f.down_vol_60m)
    background = np.where(x.option_type.eq(0), f.up_vol_240m, f.down_vol_240m)
    return pd.DataFrame({'level': np.log1p(x.market_price_usdc_per_btc.to_numpy()*v*np.sqrt(x.remaining_seconds.to_numpy()/60)/np.maximum(anchor, 1)),
        'speed': np.log((v+1e-8)/(background+1e-8)), 'available': f.available}, index=x.index)


def h0_values(x, state):
    probability = np.empty(len(x)); severity = np.empty(len(x))
    for side in (0, 1):
        mask = x.option_type.eq(side); xx = x.loc[mask]; p = state[str(side)]
        raw = np.c_[np.log(xx[BASE_FEATURE]), np.log(xx.remaining_seconds),
            np.clip((1-2*side)*xx.log_moneyness / np.maximum(xx.realized_vol_60m*np.sqrt(xx.remaining_seconds/3600), 1e-6), -5, 5)]
        probability[mask] = 1/(1+np.exp(-(np.clip((raw-p['mean'])/p['scale'], -3, 3)@p['coef']+p['intercept']))) if 'coef' in p else p['probability']
        severity[mask] = xx[BASE_FEATURE]*p['severity']
    return probability, severity


def basis(f, par, name):
    if name == 'amount_gate':
        return (np.clip((f.level-par['q75'])/(par['q95']-par['q75']), 0, 1).to_numpy()*
            np.clip((np.exp(np.clip(f.speed.to_numpy(), -30, 30))-1.25)/.75, 0, 1))[:, None]
    return np.clip((f[par['columns']].to_numpy()-par['mean'])/par['scale'], -3, 3)-par['center']


def verify(root):
    root = Path(root); manifest = json.loads((root/'provenance/manifest.json').read_text())
    assert manifest['status'] == 'complete'
    pre = json.loads((root/'provenance/preflight.json').read_text()); config = json.loads((root/'provenance/config.json').read_text())
    protocol = json.loads((root/'provenance/protocol.json').read_text())
    for name, digest in manifest['artifacts'].items(): assert file_digest(root/name) == digest, name
    with ZipFile(root/'provenance/source.zip') as archive:
        for name, digest in pre['sources'].items(): assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    for name, digest in pre['protected'].items(): assert file_digest(PROJECT/name) == digest, name
    guard = AccessGuard(); guard.allow_pressure = True; sys.addaudithook(guard.check)
    assert canonical_identity(include_test=True) == pre['canonical']; assert development_snapshot()[0] == pre['development_data']
    pool = load_pool('test', {'cohort': 'benchmark', 'use_base': True})
    market, market_info = read_test_market_prefix(DEFAULT_DATA/'btc_market_history.csv')
    assert market_info['prefix_sha256'] == config['market_prefix_sha256']
    f = independent_market(pool.metadata.created_at_utc, market)
    for w in (60, 240):
        ok = f.available
        np.testing.assert_allclose(np.sqrt(w*(f.loc[ok, f'up_vol_{w}m']**2+f.loc[ok, f'down_vol_{w}m']**2)),
            pool.X.loc[ok, f'realized_vol_{w}m'], rtol=1e-9, atol=1e-12)
    created, closed = pool.metadata.created_at_utc, pool.metadata.closed_at_utc
    start, end = pd.Timestamp(protocol['validation_start']), pd.Timestamp(protocol['validation_end']); first = pd.Timestamp(protocol['training_start'])
    states = json.loads((root/'provenance/parameters.json').read_text())
    refs = json.loads((PROJECT/config['reference_run']/'provenance/parameters.json').read_text())
    expected = [o.isoformat() for o in pd.date_range(start, end, freq='7D', inclusive='left')
        if (created.ge(o) & created.lt(min(end, o+pd.Timedelta(days=7)))).any()]
    assert sorted(states) == expected
    frames = []; fit_checks = 0; all_states = 0; largest_score = 0.
    for origin_text, state in states.items():
        origin = pd.Timestamp(origin_text); stop = min(end, origin+pd.Timedelta(days=7))
        assert pd.Timestamp(state['stop']) == stop
        h = pool.X.index[created.ge(max(first, origin-pd.Timedelta(days=30))) & created.lt(origin) & closed.lt(origin)]
        assert len(h) == state['history_orders']; assert closed.loc[h].max().isoformat() == state['latest_training_close']
        assert created.loc[h].min().isoformat() == state['earliest_training_create']
        assert state['h0'] == refs[origin_text]['models']['h0']['parameters']
        queries = pool.X.index[created.ge(origin) & created.lt(stop)]; assert len(queries) == state['query_orders']
        q = queries[closed.loc[queries].lt(end)]; x = pool.X.loc[q]; hx = pool.X.loc[h]
        p0, s0 = h0_values(x, state['h0']); _, sh = h0_values(hx, state['h0'])
        fh = feature_values(hx, f.loc[h], sh); fq = feature_values(x, f.loc[q], s0)
        predictions = dict(official_raw=x[BASE_FEATURE].to_numpy(), h0=p0*s0)
        for name, parameters in state['models'].items():
            amount = s0.copy()
            for side in (0, 1):
                par = parameters[str(side)]; all_states += 1
                mask = hx.option_type.eq(side) & fh.available & pool.y.loc[h].gt(0) & (sh > 0)
                n = int(mask.sum()); days = int(created.loc[h[mask]].dt.floor('D').nunique())
                assert n == par['positive_orders'] and days == par['positive_days']
                expected_support = n >= 40 and days >= 5
                hfeatures = fh.loc[mask]; anchor = sh[mask]; actual = pool.y.loc[h[mask]].to_numpy()
                if expected_support:
                    if name == 'amount_gate':
                        quantiles = fh.loc[hx.option_type.eq(side) & fh.available, 'level'].quantile([.75, .95]).to_numpy()
                        np.testing.assert_allclose([par['q75'], par['q95']], quantiles, rtol=1e-10, atol=1e-12)
                        expected_support = quantiles[1]-quantiles[0] >= 1e-6
                        if expected_support:
                            active = basis(hfeatures, par, name)[:, 0] > 0
                            adays = int(created.loc[h[mask]].loc[active].dt.floor('D').nunique())
                            assert par['active_positive_orders'] == int(active.sum()) and par['active_positive_days'] == adays
                            expected_support = active.sum() >= 15 and adays >= 3
                    else:
                        raw = hfeatures[par['columns']].to_numpy(); weight = anchor**2
                        mean = np.average(raw, axis=0, weights=weight)
                        scale = np.sqrt(np.average((raw-mean)**2, axis=0, weights=weight)); scale[scale < 1e-8] = 1
                        center = np.average(np.clip((raw-mean)/scale, -3, 3), axis=0, weights=weight)
                        for key, value in [('mean', mean), ('scale', scale), ('center', center)]:
                            np.testing.assert_allclose(par[key], value, atol=1e-10, rtol=1e-9)
                assert bool(par['supported']) == bool(expected_support)
                if not par['supported']: continue
                scale = float(np.sqrt(np.mean(anchor**2))); np.testing.assert_allclose(par['fit_scale'], scale)
                design = basis(hfeatures, par, name)*(anchor/scale)[:, None]
                rhs = design.T@((actual-anchor)/scale); coef = np.array(par['beta'])
                grad = (design.T@design+100*np.eye(len(coef)))@coef-rhs
                if name == 'amount_gate' and coef[0] == 0:
                    assert rhs[0] <= 1e-8
                else:
                    score = np.max(np.abs(grad))/max(1., np.max(np.abs(rhs)))
                    largest_score = max(largest_score, float(score)); assert score < 1e-8
                fit_checks += 1
                use = x.option_type.eq(side) & fq.available & (s0 > 0)
                delta = np.clip(basis(fq.loc[use], par, name)@coef, 0 if name == 'amount_gate' else -.1, .1)
                amount[use] *= 1+delta
            predictions[name] = p0*amount
        threshold = pd.Series(index=q, dtype=float)
        for month in sorted(created.loc[q].dt.strftime('%Y-%m').unique()):
            mo = pd.Timestamp(month+'-01', tz='UTC'); mq = q[created.loc[q].dt.strftime('%Y-%m').eq(month)]
            past = pool.X.index[created.ge(first) & created.lt(mo) & closed.lt(mo)]
            for (side, tenor), part in pool.X.loc[mq].groupby(['option_type', 'product_duration_seconds']):
                ids = past[pool.X.loc[past, 'option_type'].eq(side)]; cell = ids[pool.X.loc[ids, 'product_duration_seconds'].eq(tenor)]
                ids = cell if len(cell) >= 30 else ids
                threshold.loc[part.index] = pool.y.loc[ids].quantile(.9) if len(ids) >= 30 else np.nan
        for name, pred in predictions.items():
            frames.append(pd.DataFrame(dict(model=name, month=created.loc[q].dt.strftime('%Y-%m'), side=x.option_type,
                tenor=x.product_duration_seconds, quantity=x.quantity_btc, actual=pool.y.loc[q], prediction=pred,
                tail_group=np.where(threshold.isna(), 'unavailable', np.where(pool.y.loc[q] > threshold, 'tail', 'body'))), index=q))
    frame = pd.concat(frames); saved = pd.read_csv(root/'metrics/validation/summary.csv', dtype={'value': str}); largest = 0.
    for row in saved.itertuples():
        p = frame.loc[frame.model.eq(row.model)]
        if row.month != 'all': p = p.loc[p.month.eq(row.month)]
        if row.group == 'direction': p = p.loc[p.side.eq(int(row.value))]
        elif row.group == 'tenor': p = p.loc[p.tenor.eq(int(row.value))]
        elif row.group == 'direction_tail':
            s, tail = row.value.split('|'); p = p.loc[p.side.eq(int(s)) & p.tail_group.eq(tail)]
        for key, value in metrics(p).items():
            largest = max(largest, abs(value-getattr(row, key)))
            np.testing.assert_allclose(value, getattr(row, key), rtol=1e-9, atol=1e-7)
    prior = pd.read_csv(PROJECT/config['reference_run']/'metrics/validation/summary.csv', dtype={'value': str})
    for name in ('h0', 'official_raw'):
        pd.testing.assert_frame_equal(saved.loc[saved.model.eq(name)].reset_index(drop=True), prior.loc[prior.model.eq(name)].reset_index(drop=True), rtol=1e-10, atol=1e-7)
    assert len(frame.loc[frame.model.eq('h0')]) == manifest['coverage']['scored_orders']
    out = dict(status='passed', orders=manifest['coverage']['scored_orders'], summary_rows_checked=len(saved),
        candidate_direction_states=all_states, supported_direction_fits_checked=fit_checks,
        largest_relative_normal_equation_error=largest_score, largest_csv_rounding_error=largest,
        independent_minute_windows=True, original_probabilities_unchanged=True, references_match=True,
        model_fit_calls=0, official_validation_open_attempts=guard.official_attempts,
        scope='independent minute windows, saved H0 parameters, amount features and normal equations, support, predictions and summary; not every risk/interval cell')
    atomic_json(root/'provenance/independent_verification.json', out); print(json.dumps(out)); return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); verify(parser.parse_args().root)
