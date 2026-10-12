"""One price-time feature entry for future peak-payout classification.

The function never reads pool.y or actual closing metadata. Raw sources use the
existing audited pre-July prefix readers. No feature or scaler is fit globally.
"""
import numpy as np
import pandas as pd
from .directional_features import directional_features
from .market_activity_features import price_dynamics
from scripts.payout_signal_analysis import market_signals, platform_signals

FEATURE_NAMES = (
    ['option_type', 'log_nominal_seconds', 'log_remaining_seconds', 'log_quantity_btc', 'direction_log_moneyness']
    + [name for w in (60, 240, 1440) for name in [f'direction_return_{w}m', f'risk_rms_{w}m', f'opposite_rms_{w}m']]
    + ['vol_change_60m', 'vol_ratio_60m_240m']
    + [name for side in ['risk', 'opposite'] for name in [f'{side}_max_jump_60m', f'{side}_excursion_240m']]
    + ['log_market_trades_per_min_15m', 'log_market_trades_per_min_60m', 'market_trade_ratio_60m', 'market_direction_pressure_60m']
    + [f'platform_{side}_count_{w}m' for w in (15, 60, 240) for side in ['same', 'opposite']]
    + ['platform_activity_ratio_60m', 'platform_history_complete_240m'])


def build_high_risk_features(pool, market, events):
    x, meta = pool.X, pool.metadata
    required = ['option_type', 'product_duration_seconds', 'remaining_seconds', 'quantity_btc',
                'log_moneyness'] + [f'{name}_{w}m' for w in (60, 240, 1440) for name in ['log_return', 'realized_vol']]
    if (x.empty or not x.index.is_unique or not x.index.equals(meta.index)
            or not np.isfinite(x[required]).all().all() or not x.option_type.isin([0, 1]).all()
            or x[['product_duration_seconds', 'remaining_seconds', 'quantity_btc']].le(0).any().any()):
        raise ValueError('Invalid pricing-time input columns or IDs')
    created = meta.created_at_utc
    direction = directional_features(created, market)
    dynamics = price_dynamics(created, market)
    for frame in [direction, dynamics]:
        if not frame.feature_last_close_utc.equals(meta.last_candle_close_utc.rename('feature_last_close_utc')):
            raise ValueError('High-risk features must share canonical last completed candle')
    for w in (60, 240, 1440):
        if not np.allclose(direction[f'total_vol_{w}m']*np.sqrt(w), x[f'realized_vol_{w}m'], rtol=1e-9, atol=1e-12):
            raise ValueError('High-risk feature volatility differs from canonical input')
    activity = market_signals(created, market)
    platform = platform_signals(created, events)
    is_call = x.option_type.eq(0); sign = 1-2*x.option_type
    features, dictionary = {}, []

    def add(name, values, family, definition, missing='forbidden'):
        features[name] = values
        dictionary.append({'feature': name, 'family': family, 'definition': definition,
                           'available_at': 'order creation, using only completed market history or strictly earlier creation events',
                           'missing_policy': missing})

    add('option_type', x.option_type, 'contract', '0=Call, 1=Put')
    for source, name in [('product_duration_seconds', 'log_nominal_seconds'), ('remaining_seconds', 'log_remaining_seconds'),
                         ('quantity_btc', 'log_quantity_btc')]:
        add(name, np.log(x[source]), 'contract', f'natural log of {source}; quantity is insured BTC, not premium')
    add('direction_log_moneyness', sign*x.log_moneyness, 'contract', '(1-2*option_type)*log(market price / strike)')
    for w in (60, 240, 1440):
        add(f'direction_return_{w}m', sign*x[f'log_return_{w}m'], 'market_direction', f'direction-aligned past {w}m log return')
        for name, call, put in [('risk', 'up', 'down'), ('opposite', 'down', 'up')]:
            add(f'{name}_rms_{w}m', np.where(is_call, direction[f'{call}_vol_{w}m'], direction[f'{put}_vol_{w}m']),
                'market_volatility', f'RMS of {name}-direction minute log returns over all {w} minutes')
    add('vol_change_60m', dynamics.vol_change_60m, 'market_dynamics', 'last60m RMS minus preceding nonoverlapping60m RMS')
    add('vol_ratio_60m_240m', dynamics.vol_ratio_60m_240m, 'market_dynamics', 'RMS60 / max(RMS240,1e-12); zero on a flat market')
    for name, call, put in [('risk', 'up', 'down'), ('opposite', 'down', 'up')]:
        add(f'{name}_max_jump_60m', np.where(is_call, activity[f'market_max_{call}_jump_60m'], activity[f'market_max_{put}_jump_60m']),
            'market_dynamics', f'largest {name}-direction completed minute log jump over60m')
        add(f'{name}_excursion_240m', np.where(is_call, activity[f'market_draw{call}_240m'], activity[f'market_draw{put}_240m']),
            'market_dynamics', f'ordered {name}-direction maximum log-price excursion over240m')
    for w in (15, 60):
        add(f'log_market_trades_per_min_{w}m', np.log1p(activity[f'market_trades_per_min_{w}m']),
            'market_activity', f'log(1+market executions per minute over{w}m); not order-book submissions')
    add('market_trade_ratio_60m', activity.market_trade_ratio_60m, 'market_activity',
        'recent60m executions/min divided by preceding nonoverlapping24h executions/min', 'NaN if historical denominator is zero')
    add('market_direction_pressure_60m', sign*(1-2*activity.market_sell_share_60m), 'market_activity',
        'direction-aligned taker net-buy quote amount / all quote amount over60m', 'NaN if quote volume is zero')
    for w in (15, 60, 240):
        for name, call, put in [('same', 'call', 'put'), ('opposite', 'put', 'call')]:
            add(f'platform_{name}_count_{w}m', np.where(is_call, platform[f'platform_{call}_count_{w}m'], platform[f'platform_{put}_count_{w}m']),
                'platform_activity', f'{name}-direction records in [t-{w}m,t), excludes same-time records', 'NaN if source prefix begins inside lookback')
    add('platform_activity_ratio_60m', platform.platform_activity_ratio_60m, 'platform_activity',
        'recent60m total order rate divided by preceding240m rate', 'NaN if denominator is zero or less than300m source history')
    add('platform_history_complete_240m', platform.platform_call_count_240m.notna().astype(float), 'platform_activity',
        '1 if raw order source covers whole240m lookback; 0 otherwise')
    result = pd.DataFrame(features, index=x.index)
    if result.columns.tolist() != FEATURE_NAMES:
        raise ValueError('Feature schema differs from fixed high-risk whitelist')
    definitions = pd.DataFrame(dictionary)
    mandatory = definitions.loc[definitions.missing_policy.eq('forbidden'), 'feature']
    if (np.isinf(result.to_numpy(float)).any() or not np.isfinite(result[mandatory]).all().all()
            or not result.index.equals(x.index)):
        raise ValueError('Invalid high-risk features; no silent row exclusion or zero imputation')
    return result, definitions
