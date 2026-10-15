"""Independent arithmetic replay, fit-boundary checks and weighted score equations; no fits."""
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


def design(x):
    return np.c_[np.log(x[BASE_FEATURE]), np.log(x.remaining_seconds),
        np.clip((1-2*x.option_type)*x.log_moneyness /
            np.maximum(x.realized_vol_60m*np.sqrt(x.remaining_seconds/3600), 1e-6), -5, 5)]


def probability(x, par):
    if 'coef' not in par: return np.full(len(x), par['probability'])
    z = np.clip((design(x)-par['mean'])/par['scale'], -3, 3)
    return 1/(1+np.exp(-(z@par['coef']+par['intercept'])))


def verify(root):
    root = Path(root); manifest = json.loads((root/'provenance/manifest.json').read_text())
    assert manifest['status'] == 'complete'
    config = json.loads((root/'provenance/config.json').read_text())
    protocol = json.loads((root/'provenance/protocol.json').read_text())
    pre = json.loads((root/'provenance/preflight.json').read_text())
    for name, digest in manifest['artifacts'].items(): assert file_digest(root/name) == digest, name
    with ZipFile(root/'provenance/source.zip') as archive:
        for name, digest in pre['sources'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    guard = AccessGuard(); guard.allow_pressure = True; sys.addaudithook(guard.check)
    assert canonical_identity(include_test=True) == pre['canonical']
    assert development_snapshot()[0] == pre['development_data']
    pool = load_pool('test', {'cohort': 'benchmark', 'use_base': True})
    created = pool.metadata.created_at_utc; closed = pool.metadata.closed_at_utc
    start = pd.Timestamp(protocol['validation_start']); end = pd.Timestamp(protocol['validation_end'])
    training_start = pd.Timestamp(protocol['training_start'])
    states = json.loads((root/'provenance/parameters.json').read_text()); frames = []
    expected = [o.isoformat() for o in pd.date_range(start, end, freq='7D', inclusive='left')
        if ((created >= o) & (created < min(end, o+pd.Timedelta(days=7)))).any()]
    assert sorted(states) == expected
    score_gradient = 0.; coefficient_checks = 0
    for origin_string, state in states.items():
        origin = pd.Timestamp(origin_string); stop = min(end, origin+pd.Timedelta(days=7))
        assert pd.Timestamp(state['stop']) == stop
        history = pool.X.index[created.ge(max(training_start, origin-pd.Timedelta(days=30))) & created.lt(origin) & closed.lt(origin)]
        assert len(history) == state['history_orders']
        assert created.loc[history].min().isoformat() == state['earliest_training_create']
        assert closed.loc[history].max().isoformat() == state['latest_training_close']
        queries = pool.X.index[created.ge(origin) & created.lt(stop)]
        assert len(queries) == state['query_orders']
        q = queries[closed.loc[queries].lt(end)]; x = pool.X.loc[q]; b = x[BASE_FEATURE].to_numpy()
        preds = {'official_raw': b}; probs = {}; severities = {}
        for model, half in config['half_life_days'].items():
            ps = np.zeros(len(q)); ss = np.zeros(len(q))
            for side in (0, 1):
                par = state['models'][model]['parameters'][str(side)]
                ids = history[pool.X.loc[history, 'option_type'].eq(side)]; h = pool.X.loc[ids]; y = pool.y.loc[ids].to_numpy()
                hb = h[BASE_FEATURE].to_numpy(); positive = y > 0; age = (origin-created.loc[ids]).dt.total_seconds().to_numpy()/86400
                raw = np.ones(len(ids)) if half is None else 2.**(-age/half)
                weights = raw/raw.mean() if len(raw) else raw
                diag = state['models'][model]['weights'][str(side)]
                assert diag['orders'] == len(ids) and diag['positive_orders'] == int(positive.sum())
                np.testing.assert_allclose(diag['normalized_weight_sum'], len(ids), atol=1e-10)
                if len(ids):
                    coef = np.sum(raw[positive]*hb[positive]*y[positive])/np.sum(raw[positive]*hb[positive]**2) if positive.any() else 0.
                    np.testing.assert_allclose(par['severity'], coef, rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(par['probability'], np.average(positive, weights=raw), atol=1e-12)
                    if 'coef' in par:
                        raw_design = design(h); mean = raw_design.mean(axis=0); scale = raw_design.std(axis=0); scale[scale < 1e-12] = 1.
                        np.testing.assert_allclose(par['mean'], mean, atol=1e-9)
                        np.testing.assert_allclose(par['scale'], scale, atol=1e-9)
                        z = np.clip((raw_design-par['mean'])/par['scale'], -3, 3)
                        error = weights*(probability(h, par)-positive)
                        gradient = np.r_[z.T@error+np.asarray(par['coef'])/.1, error.sum()]/len(ids)
                        score_gradient = max(score_gradient, float(np.max(np.abs(gradient))))
                        assert np.max(np.abs(gradient)) < 1e-6, (origin_string, model, side, gradient)
                else: assert par['probability'] == par['severity'] == 1.
                use = x.option_type.eq(side).to_numpy(); ps[use] = probability(x.loc[use], par); ss[use] = b[use]*par['severity']
                coefficient_checks += 1
            preds[model] = ps*ss; probs[model] = ps; severities[model] = ss
        threshold = pd.Series(index=q, dtype=float)
        for month in sorted(created.loc[q].dt.strftime('%Y-%m').unique()):
            mo = pd.Timestamp(month+'-01', tz='UTC'); mq = q[created.loc[q].dt.strftime('%Y-%m').eq(month)]
            past = pool.X.index[created.ge(training_start) & created.lt(mo) & closed.lt(mo)]
            for (side, tenor), part in pool.X.loc[mq].groupby(['option_type', 'product_duration_seconds']):
                s = past[pool.X.loc[past, 'option_type'].eq(side)]; cell = s[pool.X.loc[s, 'product_duration_seconds'].eq(tenor)]
                ids = cell if len(cell) >= 30 else s
                threshold.loc[part.index] = pool.y.loc[ids].quantile(.9) if len(ids) >= 30 else np.nan
        for model, pred in preds.items():
            frames.append(pd.DataFrame(dict(model=model, month=created.loc[q].dt.strftime('%Y-%m'), side=x.option_type,
                tenor=x.product_duration_seconds, quantity=x.quantity_btc, actual=pool.y.loc[q], prediction=pred,
                tail_group=np.where(threshold.isna(), 'unavailable', np.where(pool.y.loc[q] > threshold, 'tail', 'body'))), index=q))
    frame = pd.concat(frames); saved = pd.read_csv(root/'metrics/validation/summary.csv', dtype={'value': str})
    largest = 0.
    for row in saved.itertuples():
        part = frame.loc[frame.model.eq(row.model)]
        if row.month != 'all': part = part.loc[part.month.eq(row.month)]
        if row.group == 'direction': part = part.loc[part.side.eq(int(row.value))]
        elif row.group == 'tenor': part = part.loc[part.tenor.eq(int(row.value))]
        elif row.group == 'direction_tail':
            side, tail = row.value.split('|'); part = part.loc[part.side.eq(int(side)) & part.tail_group.eq(tail)]
        for name, value in metrics(part).items():
            largest = max(largest, abs(value-getattr(row, name)))
            np.testing.assert_allclose(value, getattr(row, name), rtol=1e-10, atol=1e-7)
    macro = pd.read_csv(root/'metrics/validation/macro_monthly.csv').set_index('model')
    for name, part in frame.groupby('model'):
        monthly = [metrics(p) for _, p in part.groupby('month')]
        np.testing.assert_allclose(macro.loc[name, 'macro_rmse_per_btc'], np.sqrt(np.mean([v['rmse_per_btc']**2 for v in monthly])), atol=1e-9)
        np.testing.assert_allclose(macro.loc[name, 'macro_mae_per_btc'], np.mean([v['mae_per_btc'] for v in monthly]), atol=1e-9)
    assert len(frame.loc[frame.model.eq('official_raw')]) == manifest['coverage']['scored_orders']
    assert frame.month.isin(protocol['report_months']).all()
    out = dict(status='passed', orders=manifest['coverage']['scored_orders'], summary_rows_checked=len(saved),
        saved_direction_fits_checked=coefficient_checks, largest_normalized_logistic_gradient=score_gradient,
        largest_csv_rounding_error=largest, month_equal_metrics_checked=True, model_fit_calls=0,
        official_validation_open_attempts=guard.official_attempts,
        scope='saved-coefficient predictions, closed-form severity, logistic score equations, training boundaries, summary and macro metrics; not all interval/risk rows')
    atomic_json(root/'provenance/independent_verification.json', out); print(json.dumps(out)); return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); verify(parser.parse_args().root)
