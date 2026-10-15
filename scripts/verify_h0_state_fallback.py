"""Independent completed-hour state and prequential mixture replay; no estimator fits."""
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
from scripts.verify_h0_hourly_amount import h0_values


def independent_states(market):
    opened = pd.DatetimeIndex(market.open_time_utc); times = opened.asi8
    logs = np.log(market.close.to_numpy()); rows = []
    hourly = pd.date_range((opened[0]+pd.Timedelta(minutes=1)).ceil('h'),
        (opened[-1]+pd.Timedelta(minutes=1)).floor('h'), freq='h')
    for stamp in hourly:
        target = stamp.value-60_000_000_000; j = np.searchsorted(times, target, side='right')-1
        if j < 1440 or times[j] != target or times[j]-times[j-1440] != 1440*60_000_000_000:
            rows.append([np.nan]*4); continue
        r = np.diff(logs[j-1440:j+1]); var = float(r@r)
        rows.append([np.log(max(np.sqrt(np.sum(r[-60:]**2)), 1e-6)), np.log(max(np.sqrt(var), 1e-6)),
            (np.sum(np.maximum(r, 0)**2)-np.sum(np.minimum(r, 0)**2))/(var+1e-12),
            np.clip((logs[j]-logs[j-1440])/max(np.sqrt(var), 1e-6), -5, 5)])
    return pd.DataFrame(rows, index=hourly, columns=['logv60', 'logv1440', 'imbalance', 'trend']).rolling(6, min_periods=6).mean()


def verify(root):
    root = Path(root); manifest = json.loads((root/'provenance/manifest.json').read_text()); assert manifest['status'] == 'complete'
    pre = json.loads((root/'provenance/preflight.json').read_text()); config = json.loads((root/'provenance/config.json').read_text())
    protocol = json.loads((root/'provenance/protocol.json').read_text())
    for name, digest in manifest['artifacts'].items(): assert file_digest(root/name) == digest, name
    with ZipFile(root/'provenance/source.zip') as archive:
        for name, digest in pre['sources'].items(): assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    for name, digest in pre['protected'].items(): assert file_digest(PROJECT/name) == digest, name
    guard = AccessGuard(); guard.allow_pressure = True; sys.addaudithook(guard.check)
    assert canonical_identity(include_test=True) == pre['canonical']; assert development_snapshot()[0] == pre['development_data']
    pool = load_pool('test', {'cohort': 'benchmark', 'use_base': True}); raw, info = read_test_market_prefix(DEFAULT_DATA/'btc_market_history.csv')
    assert info['prefix_sha256'] == config['market_prefix_sha256']
    market = independent_states(raw); created, closed = pool.metadata.created_at_utc, pool.metadata.closed_at_utc
    at_quote = market.reindex(pd.DatetimeIndex(created).floor('h')).set_axis(pool.X.index)
    start, end, first = (pd.Timestamp(protocol[k]) for k in ('validation_start', 'validation_end', 'training_start'))
    states = json.loads((root/'provenance/parameters.json').read_text())
    refs = json.loads((PROJECT/config['reference_run']/'provenance/parameters.json').read_text())
    expected = [o.isoformat() for o in pd.date_range(start-pd.Timedelta(days=28), end, freq='7D', inclusive='left')
        if (created.ge(o) & created.lt(min(end, o+pd.Timedelta(days=7)))).any()]
    assert sorted(states) == expected
    cache = []; frames = []; weight_checks = 0; state_checks = 0; warmup_checks = 0
    for origin_text, state in states.items():
        origin = pd.Timestamp(origin_text); stop = min(end, origin+pd.Timedelta(days=7))
        assert pd.Timestamp(state['stop']) == stop and state['is_validation'] == (origin >= start)
        h = pool.X.index[created.ge(max(first, origin-pd.Timedelta(days=30))) & created.lt(origin) & closed.lt(origin)]
        queries = pool.X.index[created.ge(origin) & created.lt(stop)]
        assert len(h) == state['history_orders'] and len(queries) == state['query_orders']
        assert state['h0_supported'] == (len(h) >= 100)
        if len(h):
            assert created.loc[h].min().isoformat() == state['earliest_training_create']
            assert closed.loc[h].max().isoformat() == state['latest_training_close']
        if not state['h0_supported']: assert origin < start; continue
        q = queries[closed.loc[queries].lt(end)]; x = pool.X.loc[q]
        all_p, all_s = h0_values(pool.X.loc[queries], state['h0']); all_h0 = pd.Series(all_p*all_s, index=queries)
        if origin < start:
            # Verify warmup models against only their own eligible training labels.
            for side in (0, 1):
                ids = h[pool.X.loc[h, 'option_type'].eq(side)]; xx = pool.X.loc[ids]; yy = pool.y.loc[ids].to_numpy()
                par = state['h0'][str(side)]; pos = yy > 0; base = xx[BASE_FEATURE].to_numpy()
                severity = base[pos]@yy[pos]/(base[pos]@base[pos]) if pos.any() else 0.
                np.testing.assert_allclose(par['severity'], severity, rtol=1e-12, atol=1e-12)
                if 'coef' in par:
                    rd = np.c_[np.log(base), np.log(xx.remaining_seconds),
                        np.clip((1-2*side)*xx.log_moneyness / np.maximum(xx.realized_vol_60m*np.sqrt(xx.remaining_seconds/3600), 1e-6), -5, 5)]
                    z = np.clip((rd-par['mean'])/par['scale'], -3, 3)
                    p = 1/(1+np.exp(-(z@par['coef']+par['intercept'])))
                    gradient = np.r_[z.T@(p-pos)+np.array(par['coef'])/.1, np.sum(p-pos)]/len(ids)
                    assert np.max(np.abs(gradient)) < 1e-6
                warmup_checks += 1
        else:
            assert state['h0'] == refs[origin_text]['models']['h0']['parameters']
            hp = all_h0.loc[q].to_numpy(); bp = x[BASE_FEATURE].to_numpy()
            previous = pd.concat(cache); use = previous.created.ge(origin-pd.Timedelta(days=30)) & closed.reindex(previous.index).lt(origin)
            past = previous.loc[use]; alpha = np.empty(len(q))
            for side in (0, 1):
                pp = past.loc[past.side.eq(side)]; par = state['weights'][str(side)]; n = len(pp); days = pp.created.dt.floor('D').nunique()
                assert par['orders'] == n and par['days'] == days
                if n:
                    assert par['latest_close'] == closed.loc[pp.index].max().isoformat()
                    assert (pp.forecast_origin <= pp.created).all() and (pp.created < origin).all()
                a = 0.; supported = False
                if n >= 100 and days >= 10:
                    d = pp.h0.to_numpy()-pp.base.to_numpy(); delta = pool.y.loc[pp.index].to_numpy()-pp.base.to_numpy()
                    d2 = float(d@d)
                    if d2 > 1e-12:
                        penalty = 100*d2/n; a = float(np.clip((d@delta+.5*penalty)/(d2+penalty), 0, 1)); supported = True
                        np.testing.assert_allclose([par['difference_ss'], par['cross'], par['penalty']], [d2, d@delta, penalty])
                assert par['supported'] == supported; np.testing.assert_allclose(par['alpha'], a, atol=1e-11)
                alpha[x.option_type.eq(side)] = a; weight_checks += 1
            past_market = market.loc[(market.index >= max(first, origin-pd.Timedelta(days=30))) & (market.index < origin)].dropna()
            par = state['market']; assert par['hours'] == len(past_market)
            if len(past_market): assert par['latest_hour'] == past_market.index.max().isoformat()
            g = np.ones(len(q)); supported = len(past_market) >= 168
            if supported:
                vals = past_market.to_numpy(); center = np.median(vals, axis=0)
                scale = (np.quantile(vals, .75, axis=0)-np.quantile(vals, .25, axis=0))/1.349; scale[scale < 1e-6] = 1
                distances = np.sqrt(np.mean(((vals-center)/scale)**2, axis=1)); lo, hi = np.quantile(distances, [.9, .99])
                supported = hi-lo >= 1e-6
                if supported:
                    for key, value in [('center', center), ('scale', scale), ('q90', lo), ('q99', hi)]:
                        np.testing.assert_allclose(par[key], value, atol=1e-8, rtol=1e-8)
                    good = np.isfinite(at_quote.loc[q]).all(axis=1)
                    d = np.sqrt(np.mean(((at_quote.loc[q[good]].to_numpy()-center)/scale)**2, axis=1))
                    g[good] = np.clip((d-lo)/(hi-lo), 0, 1)
            assert par['supported'] == supported; state_checks += 1
            alphas = dict(half_blend=np.full(len(q), .5), state_fallback=1-g, history_blend=alpha, state_history_blend=alpha*(1-g))
            predictions = dict(h0=hp, official_raw=bp, **{m: a*hp+(1-a)*bp for m, a in alphas.items()})
            threshold = pd.Series(index=q, dtype=float)
            for month in sorted(created.loc[q].dt.strftime('%Y-%m').unique()):
                mo = pd.Timestamp(month+'-01', tz='UTC'); mq = q[created.loc[q].dt.strftime('%Y-%m').eq(month)]
                past_ids = pool.X.index[created.ge(first) & created.lt(mo) & closed.lt(mo)]
                for (side, tenor), part in pool.X.loc[mq].groupby(['option_type', 'product_duration_seconds']):
                    ids = past_ids[pool.X.loc[past_ids, 'option_type'].eq(side)]; cell = ids[pool.X.loc[ids, 'product_duration_seconds'].eq(tenor)]
                    ids = cell if len(cell) >= 30 else ids
                    threshold.loc[part.index] = pool.y.loc[ids].quantile(.9) if len(ids) >= 30 else np.nan
            for name, pred in predictions.items():
                frames.append(pd.DataFrame(dict(model=name, month=created.loc[q].dt.strftime('%Y-%m'), side=x.option_type,
                    tenor=x.product_duration_seconds, quantity=x.quantity_btc, actual=pool.y.loc[q], prediction=pred,
                    tail_group=np.where(threshold.isna(), 'unavailable', np.where(pool.y.loc[q] > threshold, 'tail', 'body'))), index=q))
        cache.append(pd.DataFrame(dict(h0=all_h0, base=pool.X.loc[queries, BASE_FEATURE], side=pool.X.loc[queries, 'option_type'],
            created=created.loc[queries], forecast_origin=origin), index=queries))
    frame = pd.concat(frames); saved = pd.read_csv(root/'metrics/validation/summary.csv', dtype={'value': str}); largest = 0.
    for row in saved.itertuples():
        p = frame.loc[frame.model.eq(row.model)]
        if row.month != 'all': p = p.loc[p.month.eq(row.month)]
        if row.group == 'direction': p = p.loc[p.side.eq(int(row.value))]
        elif row.group == 'tenor': p = p.loc[p.tenor.eq(int(row.value))]
        elif row.group == 'direction_tail':
            side, tail = row.value.split('|'); p = p.loc[p.side.eq(int(side)) & p.tail_group.eq(tail)]
        for name, value in metrics(p).items():
            largest = max(largest, abs(value-getattr(row, name)))
            np.testing.assert_allclose(value, getattr(row, name), rtol=1e-8, atol=1e-6)
    prior = pd.read_csv(PROJECT/config['reference_run']/'metrics/validation/summary.csv', dtype={'value': str})
    for name in ('h0', 'official_raw'):
        pd.testing.assert_frame_equal(saved.loc[saved.model.eq(name)].reset_index(drop=True), prior.loc[prior.model.eq(name)].reset_index(drop=True), rtol=1e-10, atol=1e-7)
    assert len(frame.loc[frame.model.eq('h0')]) == manifest['coverage']['scored_orders']
    out = dict(status='passed', orders=manifest['coverage']['scored_orders'], summary_rows_checked=len(saved),
        warmup_direction_fits_checked=warmup_checks, history_direction_weights_checked=weight_checks,
        hourly_backgrounds_checked=state_checks, largest_csv_rounding_error=largest,
        independent_hourly_states=True, strict_prior_forecast_weights=True, references_match=True,
        model_fit_calls=0, official_validation_open_attempts=guard.official_attempts,
        scope='independent hourly states, warmup equations, saved weekly H0, causal weight histories, convex predictions and summary; not every interval/risk cell')
    atomic_json(root/'provenance/independent_verification.json', out); print(json.dumps(out)); return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); verify(parser.parse_args().root)
