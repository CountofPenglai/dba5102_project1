"""Independent grid-loss and saved-H0 replay; never fit any estimator."""
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
    refs = json.loads((PROJECT/config['reference_run']/'provenance/parameters.json').read_text())
    expected = [o.isoformat() for o in pd.date_range(start-pd.Timedelta(days=28), end, freq='7D', inclusive='left')
        if (created.ge(o) & created.lt(min(end, o+pd.Timedelta(days=7)))).any()]
    assert sorted(states) == expected
    assert len(config['fixed_ratios']) == 25 and len(config['models']) == 27
    assert sorted(map(tuple, config['fixed_ratios'].values())) == [(a, b) for a in [0, .25, .5, .75, 1] for b in [0, .25, .5, .75, 1]]
    cache = []; frames = []; weight_checks = 0; h0_checks = 0
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
        assert state.get('h0') == refs[origin_text].get('h0')
        if not state['h0_supported']: assert origin < start; continue
        q = queries[closed.loc[queries].lt(end)]; x = pool.X.loc[q]
        p, s = h0_values(pool.X.loc[queries], state['h0']); all_h0 = pd.Series(p*s, index=queries); h0_checks += 1
        if origin >= start:
            hp = all_h0.loc[q].to_numpy(); bp = x[BASE_FEATURE].to_numpy()
            previous = pd.concat(cache)
            alphas = {name: np.where(x.option_type.eq(0), pair[0], pair[1]) for name, pair in config['fixed_ratios'].items()}
            for name, days in config['dynamic_windows'].items():
                use = previous.created.ge(origin-pd.Timedelta(days=days)) & previous.created.lt(origin) & closed.reindex(previous.index).lt(origin)
                past = previous.loc[use]; a = np.empty(len(q))
                for side in (0, 1):
                    pp = past.loc[past.side.eq(side)]; par = state['weights'][name][str(side)]
                    n = len(pp); day_count = pp.created.dt.floor('D').nunique()
                    assert par['orders'] == n and par['days'] == day_count and par['history_days'] == days
                    if n:
                        assert par['latest_close'] == closed.loc[pp.index].max().isoformat()
                        assert par['latest_forecast_origin'] == pp.forecast_origin.max().isoformat()
                        assert (pp.forecast_origin <= pp.created).all() and (pp.created < origin).all()
                    chosen = 0.; supported = False
                    if n >= 100 and day_count >= 10:
                        d = pp.h0.to_numpy()-pp.base.to_numpy(); d2 = float(d@d)
                        if d2 > 1e-12:
                            yy = pool.y.loc[pp.index].to_numpy(); penalty = 100*d2/n
                            raw = [float(np.mean((a0*pp.h0.to_numpy()+(1-a0)*pp.base.to_numpy()-yy)**2)) for a0 in config['fixed_grid']]
                            losses = [err+penalty/n*(a0-.5)**2 for a0, err in zip(config['fixed_grid'], raw)]
                            chosen = config['fixed_grid'][int(np.argmin(losses))]; supported = True
                            np.testing.assert_allclose(par['grid_mse'], raw, atol=1e-7, rtol=1e-11)
                            np.testing.assert_allclose(par['grid_penalized_mse'], losses, atol=1e-7, rtol=1e-11)
                            continuous = np.clip((d@(yy-pp.base.to_numpy())+.5*penalty)/(d2+penalty), 0, 1)
                            np.testing.assert_allclose(par['continuous_alpha'], continuous, atol=1e-11)
                    assert par['supported'] == supported and par['alpha'] == chosen
                    a[x.option_type.eq(side)] = chosen; weight_checks += 1
                alphas[name] = a
            threshold = pd.Series(index=q, dtype=float)
            for month in sorted(created.loc[q].dt.strftime('%Y-%m').unique()):
                mo = pd.Timestamp(month+'-01', tz='UTC'); mq = q[created.loc[q].dt.strftime('%Y-%m').eq(month)]
                past_ids = pool.X.index[created.ge(first) & created.lt(mo) & closed.lt(mo)]
                for (side, tenor), part in pool.X.loc[mq].groupby(['option_type', 'product_duration_seconds']):
                    ids = past_ids[pool.X.loc[past_ids, 'option_type'].eq(side)]
                    cell = ids[pool.X.loc[ids, 'product_duration_seconds'].eq(tenor)]
                    ids = cell if len(cell) >= 30 else ids
                    threshold.loc[part.index] = pool.y.loc[ids].quantile(.9) if len(ids) >= 30 else np.nan
            for name, alpha in alphas.items():
                pred = alpha*hp+(1-alpha)*bp
                frames.append(pd.DataFrame(dict(model=name, day=created.loc[q].dt.floor('D'), month=created.loc[q].dt.strftime('%Y-%m'),
                    side=x.option_type, tenor=x.product_duration_seconds, quantity=x.quantity_btc, actual=pool.y.loc[q], prediction=pred,
                    tail_group=np.where(threshold.isna(), 'unavailable', np.where(pool.y.loc[q] > threshold, 'tail', 'body'))), index=q))
        cache.append(pd.DataFrame(dict(h0=all_h0, base=pool.X.loc[queries, BASE_FEATURE], side=pool.X.loc[queries, 'option_type'],
            created=created.loc[queries], forecast_origin=origin), index=queries))
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
    prior = pd.read_csv(PROJECT/config['reference_run']/'metrics/validation/summary.csv', dtype={'value': str})
    for name in ('h0', 'official_raw', 'half_blend'):
        pd.testing.assert_frame_equal(saved.loc[saved.model.eq(name)].reset_index(drop=True),
            prior.loc[prior.model.eq(name)].reset_index(drop=True), rtol=1e-10, atol=1e-7)
    assert len(frame.loc[frame.model.eq('h0')]) == manifest['coverage']['scored_orders']
    out = dict(status='passed', orders=manifest['coverage']['scored_orders'], models_checked=len(macro),
        summary_rows_checked=len(saved), prior_h0_states_checked=h0_checks, grid_direction_weights_checked=weight_checks,
        largest_csv_rounding_error=largest, strict_prior_forecast_weights=True, references_match=True,
        model_fit_calls=0, official_validation_open_attempts=guard.official_attempts,
        scope='independent grid objectives, historical timing, H0 parameter replay, cohorts, summary and macro errors; interval/risk cells reviewed separately')
    atomic_json(root/'provenance/independent_verification.json', out); print(json.dumps(out)); return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); verify(parser.parse_args().root)
