"""Independent aggregate/as-of/formula checks for the fixed Apr--Jun study.

Run as a module; retains only a compact verification record, never row outputs.
This verifier does not call the study's batching, model, or metric functions.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from pp_data.config import DEFAULT_OUTPUT, PROJECT
from pp_data.features import BASE_FEATURE
from pp_models.directional_study import install_development_guard
from pp_models.runs import fingerprint
from pp_models.scoped_data import development_snapshot
from pp_models.study_storage import atomic_json, file_digest


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-8, equal_nan=True)


def manual_metrics(y, prediction, quantity):
    error = prediction - y
    actual, predicted = np.dot(y, quantity), np.dot(prediction, quantity)
    return dict(orders=len(y), rmse_per_btc=np.sqrt(np.mean(error**2)),
                mae_per_btc=np.mean(abs(error)), actual_total_usdc=actual,
                predicted_total_usdc=predicted, total_bias_usdc=predicted-actual,
                gross_overprediction_usdc=np.dot(np.maximum(error, 0), quantity),
                gross_underprediction_usdc=np.dot(np.maximum(-error, 0), quantity),
                predicted_actual_ratio=predicted/actual if actual else np.nan)


def design(frame):
    uncertainty = np.maximum(frame.realized_vol_60m * np.sqrt(frame.remaining_seconds/3600), 1e-6)
    m = (1-2*frame.option_type) * frame.log_moneyness / uncertainty
    return np.column_stack([np.log(frame[BASE_FEATURE]), np.log(frame.remaining_seconds), np.clip(m, -5, 5)])


def verify(root):
    guard = install_development_guard(DEFAULT_OUTPUT)
    manifest = json.loads((root/'provenance/manifest.json').read_text())
    assert manifest['status'] == 'complete'
    for name, digest in manifest['files'].items():
        assert file_digest(root/name) == digest, name
    source = json.loads((root/'provenance/source.json').read_text())
    for name, digest in source['source']['code_sha256'].items():
        assert file_digest(PROJECT/name) == digest, name
    for name, digest in source['config_sha256'].items():
        assert file_digest(PROJECT/name) == digest, name
    identity, captured = development_snapshot(DEFAULT_OUTPUT)
    assert fingerprint(identity) == source['source']['dataset_sha256']
    tables = {}
    for kind in ['features', 'targets', 'metadata']:
        tables[kind] = pd.concat([pd.read_csv(BytesIO(captured[f'datasets/{s}/{kind}.csv']),
                                            float_precision='round_trip').set_index('order_id')
                                 for s in ['train', 'validation']])
    X, target, meta = (tables[k] for k in ['features', 'targets', 'metadata'])
    assert X.index.equals(target.index) and X.index.equals(meta.index)
    for column in ['created_at_utc', 'closed_at_utc']:
        meta[column] = pd.to_datetime(meta[column], utc=True)
    ids = meta[meta.benchmark_eligible].sort_values('created_at_utc', kind='stable').index
    X, y, meta = X.loc[ids], target.loc[ids, 'payout_per_btc_usdc'], meta.loc[ids]
    read = lambda name: pd.read_csv(root/f'metrics/{name}.csv', float_precision='round_trip', dtype={'value': str})
    daily, summary, cohorts, macro = (read(n) for n in ['daily', 'summary', 'daily_cohorts', 'macro_monthly'])
    additive = ['orders', 'actual_total_usdc', 'predicted_total_usdc', 'total_bias_usdc',
                'gross_overprediction_usdc', 'gross_underprediction_usdc']
    reconciled = 0
    for row in summary[summary.group.isin(['overall', 'direction'])].itertuples():
        part = daily[(daily.model == row.model) & (daily.group == row.group) & (daily.value == row.value)]
        if row.month != 'all':
            part = part[part.month == row.month]
        assert part.day.nunique() == row.days
        for column in additive:
            close(part[column].sum(), getattr(row, column))
        close(np.sqrt(np.average(part.rmse_per_btc**2, weights=part.orders)), row.rmse_per_btc)
        close(np.average(part.mae_per_btc, weights=part.orders), row.mae_per_btc)
        reconciled += 1
    for row in macro.itertuples():
        part = summary[(summary.model == row.model) & (summary.group == 'overall') & (summary.month != 'all')]
        assert len(part) == row.months == 3
        close(np.sqrt(np.mean(part.rmse_per_btc**2)), row.macro_rmse_per_btc)
        close(part.mae_per_btc.mean(), row.macro_mae_per_btc)
    # Independent calendar masks and six closed-form/small Logistic comparators.
    checked_days, checked_rows, all_scored, probabilities = 0, 0, [], []
    for day in pd.date_range('2026-04-01', '2026-06-30', tz='UTC'):
        score = meta.created_at_utc.ge(day) & meta.created_at_utc.lt(day+pd.Timedelta(days=1)) & meta.closed_at_utc.lt('2026-07-01')
        if not score.any():
            continue
        train = meta.created_at_utc.ge(day-pd.Timedelta(days=30)) & meta.created_at_utc.lt(day) & meta.closed_at_utc.lt(day)
        h, q = X.index[train], X.index[score]
        row = cohorts[cohorts.day == day.isoformat()].iloc[0]
        assert (len(h), len(q)) == (row.training_orders, row.scoring_orders)
        assert fingerprint(list(map(str, h))) == row.train_ids_sha256
        assert fingerprint(list(map(str, q))) == row.score_ids_sha256
        assert pd.Timestamp(row.latest_training_close) == meta.loc[h, 'closed_at_utc'].max() < day
        all_scored.extend(map(str, q)); checked_days += 1
        hx, qx, hy = X.loc[h], X.loc[q], y.loc[h]
        predictions = {'official_raw': qx[BASE_FEATURE].to_numpy().copy()}
        g = hx[['option_type', 'product_duration_seconds']].assign(y=hy).groupby(['option_type', 'product_duration_seconds']).y.agg(['sum', 'size'])
        gm = (g['sum']+30*hy.mean())/(g['size']+30)
        predictions['grouped_mean'] = np.array([gm.get(tuple(v), hy.mean()) for v in qx[['option_type', 'product_duration_seconds']].to_numpy()])
        for name in ['direction_scaled', 'direction_shrunk', 'day_balanced_shrunk', 'hurdle_base']:
            predictions[name] = qx[BASE_FEATURE].to_numpy().copy()
        for side in [0, 1]:
            hs, qs = hx.option_type.eq(side), qx.option_type.eq(side)
            if not qs.any():
                continue
            b, t = hx.loc[hs, BASE_FEATURE].to_numpy(), hy.loc[hs].to_numpy()
            qb = qx.loc[qs, BASE_FEATURE].to_numpy()
            assert len(t) > 0
            c = np.dot(b, t)/np.dot(b, b)
            predictions['direction_scaled'][qs] = qb*c
            predictions['direction_shrunk'][qs] = qb*(c+1)/2
            dates = meta.loc[h[hs], 'created_at_utc'].dt.floor('D')
            w = np.array([1/(dates == d).sum() for d in dates])
            predictions['day_balanced_shrunk'][qs] = qb*(np.sum(w*b*t)/np.sum(w*b*b)+1)/2
            positive = t > 0
            p = np.full(qs.sum(), positive.mean())
            if 0 < positive.sum() < len(positive):
                ht, qt = design(hx.loc[hs]), design(qx.loc[qs])
                center, scale = ht.mean(axis=0), ht.std(axis=0)
                scale[scale == 0] = 1
                classifier = LogisticRegression(C=.1, max_iter=1000, tol=1e-9, solver='lbfgs')
                classifier.fit(np.clip((ht-center)/scale, -3, 3), positive.astype(int))
                p = classifier.predict_proba(np.clip((qt-center)/scale, -3, 3))[:, 1]
            severity = np.dot(b[positive], t[positive])/np.dot(b[positive], b[positive]) if positive.any() else 0
            predictions['hurdle_base'][qs] = qb*p*severity
            probabilities.append(pd.DataFrame({'month': day.strftime('%Y-%m'), 'side': side,
                'positive': y.loc[q[qs]].gt(0).to_numpy(), 'hurdle_probability': p,
                'historical_probability': (positive.sum()+.5)/(len(t)+1)}))
        for name, prediction in predictions.items():
            for group, value, mask in [('overall', 'all', np.ones(len(q), bool))] + [('direction', str(side), qx.option_type.eq(side).to_numpy()) for side in [0, 1]]:
                if not mask.any():
                    continue
                observed = daily[(daily.day == day.isoformat()) & (daily.model == name) & (daily.group == group) & (daily.value == value)]
                assert len(observed) == 1
                for column, expected in manual_metrics(y.loc[q].to_numpy()[mask], prediction[mask], qx.quantity_btc.to_numpy()[mask]).items():
                    close(observed[column].iloc[0], expected)
                checked_rows += 1
    assert checked_days == len(cohorts) == source['execution']['scoring_days']
    assert len(all_scored) == len(set(all_scored)) == source['execution']['unique_scoring_orders']
    assert fingerprint(all_scored) == source['execution']['scoring_ids_sha256']
    probability = pd.concat(probabilities)
    probability_checks = 0
    for row in read('probability').query("scope == 'all_orders'").itertuples():
        part = probability[probability.side == row.direction]
        if row.month != 'all':
            part = part[part.month == row.month]
        close(len(part), row.orders)
        close(np.mean((part[row.method]-part.positive)**2), row.brier)
        probability_checks += 1
    legacy = read('legacy_june'); legacy_checks = 0
    references = [('official_raw', 'anchor_comparison_20261008', 'official_raw'),
                  ('grouped_mean', 'anchor_comparison_20261008', 'grouped_raw'),
                  ('direction_scaled', 'constrained_pricing_20261009_verified', 'constrained_scaled'),
                  ('terminal_static', 'terminal_distribution_20261009', 'official_terminal_static')]
    reference_hashes = {}
    for model, run, old_model in references:
        path = root.parent/run/'metrics/followup.csv'
        reference_hashes[str(path.relative_to(PROJECT))] = file_digest(path)
        old = pd.read_csv(path, dtype={'value': str}, float_precision='round_trip')
        for group, old_group, value in [('overall', 'overall', 'all'), ('direction', 'option_type', '0'), ('direction', 'option_type', '1')]:
            a = legacy[(legacy.month == 'late') & (legacy.model == model) & (legacy.group == group) & (legacy.value == value)]
            b = old[(old.model == old_model) & (old.group == old_group) & (old.value == value)]
            assert len(a) == len(b) == 1
            for column in additive+['rmse_per_btc', 'mae_per_btc', 'predicted_actual_ratio']:
                close(a[column].iloc[0], b[column].iloc[0])
            legacy_checks += 1
    assert guard['blocked_test_opens'] == 0
    result = {'status': 'passed', 'scoring_days': checked_days, 'unique_scoring_orders': len(all_scored),
              'independent_daily_metric_rows': checked_rows, 'daily_to_summary_rows': reconciled,
              'probability_brier_checks': probability_checks, 'legacy_reference_rows': legacy_checks,
              'test_csv_read': False, 'market_csv_read': False,
              'original_manifest_sha256': file_digest(root/'provenance/manifest.json'),
              'verifier_sha256': file_digest(Path(__file__)), 'reference_sha256': reference_hashes}
    atomic_json(root/'provenance/verification.json', result)
    (root/'provenance/verify_rolling_study.py').write_bytes(Path(__file__).read_bytes())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(verify(args.run_dir.resolve()), indent=2))
