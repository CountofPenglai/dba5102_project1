"""Prepare and audit peak-payout classification data; never train a classifier."""
import argparse
import json
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from pp_data.config import DEFAULT_DATA, DEFAULT_OUTPUT, PROJECT
from .directional_features import read_market_prefix
from .directional_study import install_development_guard
from .high_risk_features import FEATURE_NAMES, build_high_risk_features
from .high_risk_labels import TARGET_COLUMNS, fit_peak_thresholds, label_orders, load_settings
from .market_activity_features import read_order_prefix
from .rolling_protocol import load_protocol, load_rolling_pool, rolling_batches
from .runs import fingerprint, run_destination
from .scoped_data import development_snapshot
from .study_storage import atomic_json, file_digest


def prepare_day(pool, features, day, settings):
    """Single train/score interface; X never contains outcomes or label thresholds."""
    if (not features.index.equals(pool.X.index) or not features.columns.is_unique
            or features.shape[1] == 0 or not set(features).issubset(FEATURE_NAMES)
            or np.isinf(features.to_numpy(float)).any()):
        raise ValueError('High-risk feature IDs or whitelist mismatch')
    fitted = fit_peak_thresholds(pool, day, settings)
    day = fitted['day']; end = pd.Timestamp(load_protocol()['validation_end'])
    meta = pool.metadata
    ids = pool.X.index[meta.created_at_utc.ge(day) & meta.created_at_utc.lt(day+pd.Timedelta(days=1)) & meta.closed_at_utc.lt(end)]
    history = fitted['history_ids']
    return {'day': day, 'train_X': features.loc[history].copy(), 'score_X': features.loc[ids].copy(),
            'train_labels': label_orders(pool, history, fitted), 'score_labels': label_orders(pool, ids, fitted),
            'thresholds': fitted['thresholds'].copy()}


def label_statistics(labels, pool, target):
    ids = labels.index; available = labels[target].notna(); positive = labels[target].eq(1).fillna(False)
    days = pool.metadata.loc[ids, 'created_at_utc'].dt.floor('D')
    counts = days[positive].value_counts()
    cash = pool.y.loc[ids] * pool.X.loc[ids, 'quantity_btc']
    n_positive = int(positive.sum()); total = float(cash.sum())
    return {'orders': len(ids), 'available_labels': int(available.sum()), 'positive_labels': n_positive,
            'positive_fraction': n_positive/int(available.sum()) if available.any() else np.nan,
            'positive_creation_days': int(counts.size),
            'largest_day_positive_share': float(counts.max()/n_positive) if n_positive else np.nan,
            'total_payout_usdc': total, 'positive_payout_usdc': float(cash[positive].sum()),
            'positive_cash_share': float(cash[positive].sum()/total) if total else np.nan}


def collect_audit(pool, features, dictionary, settings):
    protocol = load_protocol(); training = []; scoring = []; threshold_rows = []; queues = []
    for day, history, today in rolling_batches(pool, protocol):
        batch = prepare_day(pool, features, day, settings)
        if not batch['train_X'].index.equals(history) or not batch['score_X'].index.equals(today):
            raise ValueError('High-risk daily queue differs from the canonical rolling protocol')
        queues.append({'day_utc': day.isoformat(), 'history_orders': len(history), 'scoring_orders': len(today),
                       'history_ids_sha256': fingerprint(list(map(str, history))),
                       'scoring_ids_sha256': fingerprint(list(map(str, today)))})
        train, score = batch['train_labels'], batch['score_labels']
        for labels in [train, score]:
            for kind in ['unit', 'cash']:
                a, b = labels[f'high_risk_{kind}_q975'], labels[f'high_risk_{kind}_q950']
                if (a.eq(1) & b.notna() & b.ne(1)).fillna(False).any():
                    raise ValueError('Peak labels violate nested quantile thresholds')
        for side, label in [(0, 'Call'), (1, 'Put')]:
            part = train.loc[pool.X.loc[train.index, 'option_type'].eq(side)]
            for target in TARGET_COLUMNS:
                training.append({'day_utc': day.isoformat(), 'direction': label, 'target': target,
                                 **label_statistics(part, pool, target)})
        score = score.copy(); score['day_utc'] = day; scoring.append(score)
        table = batch['thresholds'].reset_index()
        table['day_utc'] = day; table['month'] = day.strftime('%Y-%m'); threshold_rows.append(table)
    combined = pd.concat(scoring)
    if not combined.index.is_unique:
        raise ValueError('Scoring orders repeated across daily label cohorts')
    label_rows = []
    for period in ['all'] + protocol['report_months']:
        period_mask = pd.Series(True, index=combined.index) if period == 'all' else combined.day_utc.dt.strftime('%Y-%m').eq(period)
        for cohort in ['risk', 'benchmark_common']:
            mask = period_mask & (True if cohort == 'risk' else pool.metadata.loc[combined.index, 'benchmark_eligible'])
            for side, label in [(0, 'Call'), (1, 'Put')]:
                part = combined.loc[mask & pool.X.loc[combined.index, 'option_type'].eq(side)]
                for target in TARGET_COLUMNS:
                    label_rows.append({'period': period, 'cohort': cohort, 'direction': label, 'target': target,
                        'direction_fallback_orders': int(part.threshold_source.eq('direction_fallback').sum()),
                        'insufficient_history_orders': int(part.threshold_source.eq('insufficient_direction_history').sum()),
                        **label_statistics(part, pool, target)})
    training = pd.DataFrame(training)
    support = training.groupby(['direction', 'target']).agg(
        windows=('day_utc', 'size'), minimum_training_orders=('orders', 'min'),
        median_training_orders=('orders', 'median'), maximum_training_orders=('orders', 'max'),
        minimum_positive_labels=('positive_labels', 'min'), median_positive_labels=('positive_labels', 'median'),
        maximum_positive_labels=('positive_labels', 'max'), minimum_positive_days=('positive_creation_days', 'min'),
        median_positive_days=('positive_creation_days', 'median'),
        windows_below_ten_positives=('positive_labels', lambda s: int(s.lt(10).sum())),
        windows_below_three_positive_days=('positive_creation_days', lambda s: int(s.lt(3).sum()))).reset_index()
    threshold = pd.concat(threshold_rows)
    summary = threshold.groupby(['month', 'option_type', 'product_duration_seconds', 'threshold_source']).agg(
        days=('day_utc', 'size'), minimum_history_orders=('threshold_history_orders', 'min'),
        unit_q975_min=('unit_threshold_q975', 'min'), unit_q975_median=('unit_threshold_q975', 'median'),
        unit_q975_max=('unit_threshold_q975', 'max'),
        cash_q975_min=('cash_threshold_q975', 'min'), cash_q975_median=('cash_threshold_q975', 'median'),
        cash_q975_max=('cash_threshold_q975', 'max')).reset_index()
    feature_rows = []; created = pool.metadata.created_at_utc
    roles = {'warmup_before_april': created.lt(protocol['validation_start']),
             **{month: created.dt.strftime('%Y-%m').eq(month) for month in protocol['report_months']}}
    for role, mask in roles.items():
        for name in features:
            values = features.loc[mask, name]
            feature_rows.append({'role': role, 'feature': name, 'orders': len(values),
                'missing': int(values.isna().sum()), 'constant_nonmissing': values.nunique() <= 1,
                'minimum': values.min(), 'median': values.median(), 'p95': values.quantile(.95), 'maximum': values.max()})
    tables = {'feature_dictionary': dictionary, 'feature_statistics': pd.DataFrame(feature_rows),
              'training_daily': training, 'training_support': support, 'label_coverage': pd.DataFrame(label_rows),
              'threshold_summary': summary, 'daily_cohorts': pd.DataFrame(queues)}
    summary = {'feature_rows': len(features), 'feature_count': features.shape[1], 'scoring_orders': len(combined),
               'scoring_days': len(queues), 'scoring_ids_sha256': fingerprint(sorted(map(str, combined.index))),
               'primary_target': f"high_risk_{settings['primary_target']}_q975",
               'train_classifier': False, 'price_backtest': False,
               'feature_missing_cells': int(features.isna().sum().sum())}
    return tables, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'audit'])
    parser.add_argument('--run-id')
    parser.add_argument('--output-root', type=Path, default=PROJECT/'outputs/feature_engineering')
    args = parser.parse_args(argv); settings = load_settings()
    if args.action == 'plan':
        print(json.dumps(settings | {'features': FEATURE_NAMES, 'output_categories': ['metrics', 'provenance', 'reports']}, indent=2)); return 0
    if not args.run_id: parser.error('audit requires --run-id')
    destination = run_destination(args.output_root, args.run_id)
    started = perf_counter(); guard = install_development_guard(DEFAULT_OUTPUT)
    identity, _ = development_snapshot(); pool = load_rolling_pool('risk')
    market_path, order_path = DEFAULT_DATA/'btc_market_history.csv', DEFAULT_DATA/'pp_transactions_training.csv'
    external_stats = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in [market_path, order_path]}
    market, market_source = read_market_prefix(market_path, '2026-07-01T00:00:00Z')
    events, order_source = read_order_prefix(order_path, '2026-07-01T00:00:00Z')
    features, dictionary = build_high_risk_features(pool, market, events)
    tables, summary = collect_audit(pool, features, dictionary, settings)
    if development_snapshot()[0] != identity or any((p.stat().st_size, p.stat().st_mtime_ns) != stat for p, stat in external_stats.items()):
        raise ValueError('Input sources changed during feature audit')
    names = ['pp_models/high_risk_data.py', 'pp_models/high_risk_features.py', 'pp_models/high_risk_labels.py',
             'pp_models/rolling_protocol.py', 'pp_models/directional_features.py', 'pp_models/market_activity_features.py',
             'pp_models/scoped_data.py', 'scripts/payout_signal_analysis.py', 'config/high_risk_features_v1.json',
             'config/validation_protocol_v2.json', 'docs/plans/2026-10-09-high-risk-features.md',
             'tests/test_high_risk_labels.py', 'tests/test_high_risk_features.py', 'tests/test_high_risk_data.py']
    source = {'settings': settings, 'protocol': load_protocol(), 'development': identity,
              'market_prefix': market_source, 'order_prefix': order_source,
              'code_sha256': {name: file_digest(PROJECT/name) for name in names},
              'test_csv_read': False, 'july_prices_parsed': False, 'blocked_test_opens': guard['blocked_test_opens'],
              'opened_development_files': sorted(guard['opened_development_files']),
              'elapsed_seconds': perf_counter()-started,
              'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform == 'darwin' else 1024)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.high_risk_', dir=destination.parent) as temp:
        root = Path(temp); (root/'metrics').mkdir(); (root/'provenance').mkdir(); (root/'reports').mkdir()
        for name, table in tables.items():
            table.to_csv(root/f'metrics/{name}.csv', index=False, float_format='%.17g')
        atomic_json(root/'provenance/source.json', source)
        atomic_json(root/'metrics/summary.json', summary)
        with ZipFile(root/'provenance/source_snapshot.zip', 'w', ZIP_DEFLATED) as archive:
            for name in names: archive.write(PROJECT/name, name)
        files = {str(p.relative_to(root)): file_digest(p) for p in root.rglob('*') if p.is_file()}
        atomic_json(root/'provenance/manifest.json', {'producer': 'high_risk_feature_audit', 'status': 'complete', 'files': files})
        root.rename(destination)
    print(json.dumps(summary | {'output': str(destination), 'elapsed_seconds': source['elapsed_seconds']}, indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
