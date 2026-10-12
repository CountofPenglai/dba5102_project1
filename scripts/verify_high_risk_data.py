"""Recompute label summaries and selected features independently of their builders."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pp_data.config import PROJECT, DEFAULT_OUTPUT, DEFAULT_DATA
from pp_models.directional_study import install_development_guard
from pp_models.directional_features import read_market_prefix
from pp_models.high_risk_features import build_high_risk_features
from pp_models.market_activity_features import read_order_prefix
from pp_models.rolling_protocol import load_rolling_pool
from pp_models.scoped_data import development_snapshot
from pp_models.study_storage import atomic_json, file_digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(argv); root = args.run_dir.resolve()
    guard = install_development_guard(DEFAULT_OUTPUT)
    manifest = json.loads((root/'provenance/manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert file_digest(root/name) == digest, name
    source = json.loads((root/'provenance/source.json').read_text())
    for name, digest in source['code_sha256'].items():
        assert file_digest(PROJECT/name) == digest, name
    pool = load_rolling_pool('risk'); identity, payload = development_snapshot()
    assert identity == source['development']
    raw_targets = pd.concat([pd.read_csv(BytesIO(payload[f'datasets/{part}/targets.csv']),
        float_precision='round_trip').set_index('order_id') for part in ['train', 'validation']])
    np.testing.assert_allclose(pool.y*pool.X.quantity_btc, raw_targets.loc[pool.X.index, 'payout_usdc'], rtol=1e-12, atol=1e-8)
    y_unit = pool.y; y_cash = raw_targets.loc[pool.X.index, 'payout_usdc']
    training = pd.read_csv(root/'metrics/training_daily.csv')
    coverage = pd.read_csv(root/'metrics/label_coverage.csv')
    independent = []; checks = 0
    for stamp in training.day_utc.unique():
        day = pd.Timestamp(stamp); meta = pool.metadata
        mask = ((meta.created_at_utc >= day-pd.Timedelta(days=30)) & (meta.created_at_utc < day) & (meta.closed_at_utc < day))
        history = pool.X.loc[mask]
        today = pool.X.loc[(meta.created_at_utc >= day) & (meta.created_at_utc < day+pd.Timedelta(days=1)) & (meta.closed_at_utc < '2026-07-01')]
        labels = {}
        for role, query in [('train', history), ('score', today)]:
            frame = pd.DataFrame(index=query.index)
            for (side, tenor), g in query.groupby(['option_type', 'product_duration_seconds']):
                direction = history.loc[history.option_type == side]
                cell = direction.loc[direction.product_duration_seconds == tenor]
                chosen = cell if len(cell) >= 80 else direction
                for kind, target in [('unit', y_unit), ('cash', y_cash)]:
                    for suffix, q in [('q950', .95), ('q975', .975)]:
                        column = f'high_risk_{kind}_{suffix}'
                        threshold = np.quantile(target.loc[chosen.index], q) if len(chosen) >= 60 else np.nan
                        frame.loc[g.index, column] = (target.loc[g.index] > threshold).astype(float) if threshold > 0 else np.nan
            labels[role] = frame
        for _, row in training[training.day_utc.eq(stamp)].iterrows():
            ids = history.index[history.option_type.eq(0 if row.direction == 'Call' else 1)]
            value = labels['train'].loc[ids, row.target]
            positive = ids[value.eq(1)]
            assert len(ids) == row.orders
            assert int(value.notna().sum()) == row.available_labels
            assert len(positive) == row.positive_labels
            assert pool.metadata.loc[positive, 'created_at_utc'].dt.floor('D').nunique() == row.positive_creation_days
            np.testing.assert_allclose(y_cash.loc[positive].sum(), row.positive_payout_usdc, rtol=1e-12, atol=1e-8)
            checks += 1
        labels['score']['month'] = day.strftime('%Y-%m'); independent.append(labels['score'])
    all_labels = pd.concat(independent)
    assert all_labels.index.is_unique and len(all_labels) == 4074
    for _, row in coverage.iterrows():
        mask = pd.Series(True, index=all_labels.index)
        if row.period != 'all': mask &= all_labels.month.eq(row.period)
        if row.cohort == 'benchmark_common': mask &= pool.metadata.loc[all_labels.index, 'benchmark_eligible']
        mask &= pool.X.loc[all_labels.index, 'option_type'].eq(0 if row.direction == 'Call' else 1)
        ids = all_labels.index[mask]; values = all_labels.loc[ids, row.target]; positive = ids[values.eq(1)]
        assert len(ids) == row.orders and values.notna().sum() == row.available_labels
        assert len(positive) == row.positive_labels
        assert pool.metadata.loc[positive, 'created_at_utc'].dt.floor('D').nunique() == row.positive_creation_days
        np.testing.assert_allclose(y_cash.loc[positive].sum(), row.positive_payout_usdc, rtol=1e-12, atol=1e-8)
        checks += 1
    market, market_source = read_market_prefix(DEFAULT_DATA/'btc_market_history.csv', '2026-07-01T00:00:00Z')
    events, event_source = read_order_prefix(DEFAULT_DATA/'pp_transactions_training.csv', '2026-07-01T00:00:00Z')
    assert market_source['prefix_sha256'] == source['market_prefix']['prefix_sha256']
    assert event_source['prefix_sha256'] == source['order_prefix']['prefix_sha256']
    x, _ = build_high_risk_features(pool, market, events)
    prices = np.log(market.close.to_numpy()); trades = market.number_of_trades.to_numpy()
    sample = pool.X.index[np.linspace(0, len(pool.X)-1, 24).astype(int)]
    for order in sample:
        contract = pool.X.loc[order]; t = pool.metadata.loc[order, 'created_at_utc']
        i = int(market.close_time_utc.searchsorted(t, side='right')-1)
        sign = 1 if contract.option_type == 0 else -1
        reference = {'option_type': contract.option_type,
            'log_nominal_seconds': np.log(contract.product_duration_seconds),
            'log_remaining_seconds': np.log(contract.remaining_seconds),
            'log_quantity_btc': np.log(contract.quantity_btc),
            'direction_log_moneyness': sign*np.log(market.close.iloc[i]/contract.strike_price_usdc_per_btc)}
        for w in [60, 240, 1440]:
            r = sign*np.diff(prices[i-w:i+1])
            reference[f'direction_return_{w}m'] = float(r.sum())
            reference[f'risk_rms_{w}m'] = np.sqrt(np.mean(np.maximum(r, 0)**2))
            reference[f'opposite_rms_{w}m'] = np.sqrt(np.mean(np.minimum(r, 0)**2))
        recent = np.diff(prices[i-60:i+1]); prior = np.diff(prices[i-120:i-59]); long = np.diff(prices[i-240:i+1])
        reference['vol_change_60m'] = np.sqrt(np.mean(recent**2))-np.sqrt(np.mean(prior**2))
        reference['vol_ratio_60m_240m'] = np.sqrt(np.mean(recent**2))/max(np.sqrt(np.mean(long**2)), 1e-12)
        path = sign*prices[i-240:i+1]
        reference['risk_max_jump_60m'] = max(0, float((sign*recent).max()))
        reference['opposite_max_jump_60m'] = max(0, float((-sign*recent).max()))
        reference['risk_excursion_240m'] = float((path-np.minimum.accumulate(path)).max())
        reference['opposite_excursion_240m'] = float((np.maximum.accumulate(path)-path).max())
        for w in [15, 60]: reference[f'log_market_trades_per_min_{w}m'] = np.log1p(trades[i-w+1:i+1].mean())
        denominator = trades[i-1499:i-59].mean()
        reference['market_trade_ratio_60m'] = trades[i-59:i+1].mean()/denominator if denominator > 0 else np.nan
        quote = market.quote_asset_volume.iloc[i-59:i+1].sum()
        buy = market.taker_buy_quote_asset_volume.iloc[i-59:i+1].sum()
        reference['market_direction_pressure_60m'] = sign*(2*buy/quote-1) if quote > 0 else np.nan
        for w in [15, 60, 240]:
            complete = events.created_at_utc.min() <= t-pd.Timedelta(minutes=w)
            active = events[(events.created_at_utc >= t-pd.Timedelta(minutes=w)) & (events.created_at_utc < t)]
            reference[f'platform_same_count_{w}m'] = float(active.option_type.eq(contract.option_type).sum()) if complete else np.nan
            reference[f'platform_opposite_count_{w}m'] = float(active.option_type.ne(contract.option_type).sum()) if complete else np.nan
        past = events[(events.created_at_utc >= t-pd.Timedelta(minutes=300)) & (events.created_at_utc < t-pd.Timedelta(minutes=60))]
        reference['platform_activity_ratio_60m'] = (4*(reference['platform_same_count_60m']+reference['platform_opposite_count_60m'])/len(past)
            if len(past) and events.created_at_utc.min() <= t-pd.Timedelta(minutes=300) else np.nan)
        reference['platform_history_complete_240m'] = float(events.created_at_utc.min() <= t-pd.Timedelta(minutes=240))
        assert set(reference) == set(x)
        np.testing.assert_allclose(x.loc[order].to_numpy(), pd.Series(reference).reindex(x.columns).to_numpy(),
                                   rtol=1e-9, atol=1e-12, equal_nan=True)
    result = {'status': 'passed', 'training_and_scoring_aggregate_checks': checks,
              'independently_rebuilt_orders': len(sample), 'features_per_order': x.shape[1],
              'cash_targets_reconciled': True, 'source_hashes_verified': True,
              'test_csv_read': False, 'blocked_test_opens': guard['blocked_test_opens'],
              'verification_script_sha256': file_digest(Path(__file__))}
    atomic_json(root/'provenance/verification.json', result)
    print(json.dumps(result, indent=2)); return 0


if __name__ == '__main__': raise SystemExit(main())
