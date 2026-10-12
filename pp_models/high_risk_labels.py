"""Conditional peak-payout labels fitted exclusively to settled rolling history.

Targets and threshold diagnostics stay separate from pricing-time features.
Labels are relative to each daily training origin, not permanent order properties.
"""
import json
import numpy as np
import pandas as pd
from pp_data.config import PROJECT
from .rolling_protocol import load_protocol

CONFIG_PATH = PROJECT / 'config/high_risk_features_v1.json'
QUANTILES = {'q950': .95, 'q975': .975}
TARGET_COLUMNS = [f'high_risk_{kind}_{suffix}' for kind in ['unit', 'cash'] for suffix in QUANTILES]


def validate_settings(c):
    if (c.get('version') != 'high_risk_features_v1'
            or c.get('protocol') != 'config/validation_protocol_v2.json'
            or c.get('primary_target') not in ['unit', 'cash']
            or c.get('primary_quantile') != .975 or c.get('auxiliary_quantile') != .95
            or c.get('minimum_cell_orders') != 80 or c.get('minimum_direction_orders') != 60
            or c.get('strictly_above_threshold') is not True
            or c.get('nonpositive_threshold') != 'unavailable'
            or c.get('feature_set') != 'pricing_time_core_v1'
            or c.get('external_features') is not False or c.get('parameter_search') is not False
            or c.get('train_classifier') is not False):
        raise ValueError('Unsupported fixed high-risk feature/label settings')


def load_settings():
    c = json.loads(CONFIG_PATH.read_text()); validate_settings(c); return c


def _check_pool(pool):
    x, y, meta = pool.X, pool.y, pool.metadata
    if (x.empty or not x.index.is_unique or not x.index.equals(y.index) or not x.index.equals(meta.index)
            or not x.option_type.isin([0, 1]).all()
            or not np.isfinite(x[['product_duration_seconds', 'quantity_btc']]).all().all()
            or x.product_duration_seconds.le(0).any() or x.quantity_btc.le(0).any()
            or not np.isfinite(y).all() or y.lt(0).any()
            or meta.created_at_utc.dt.tz is None or meta.closed_at_utc.dt.tz is None
            or meta[['created_at_utc', 'closed_at_utc']].isna().any().any()
            or meta.closed_at_utc.lt(meta.created_at_utc).any()):
        raise ValueError('Invalid high-risk input alignment, contracts or labels')


def fit_peak_thresholds(pool, day, settings):
    """Fit a daily label definition; no classifier or feature preprocessing is fit."""
    validate_settings(settings); _check_pool(pool)
    day = pd.Timestamp(day); protocol = load_protocol()
    if (day.tzinfo is None or day != day.floor('D') or str(day.tz) != 'UTC'
            or day < pd.Timestamp(protocol['validation_start'])
            or day >= pd.Timestamp(protocol['validation_end'])):
        raise ValueError('Daily label origin must be UTC midnight in April--June')
    meta = pool.metadata
    mask = (meta.created_at_utc.ge(day-pd.Timedelta(days=protocol['history_days']))
            & meta.created_at_utc.lt(day) & meta.closed_at_utc.lt(day))
    ids = pool.X.index[mask]; history = pool.X.loc[ids]; rows = []

    def add_row(group, side, tenor, source):
        unit = pool.y.loc[group.index]; cash = unit * group.quantity_btc
        row = {'option_type': side, 'product_duration_seconds': tenor,
               'threshold_source': source, 'threshold_history_orders': len(group)}
        for kind, values in [('unit', unit), ('cash', cash)]:
            row.update({f'{kind}_threshold_{suffix}': float(values.quantile(q)) for suffix, q in QUANTILES.items()})
        rows.append(row)

    for side, direction in history.groupby('option_type'):
        if len(direction) < settings['minimum_direction_orders']:
            continue
        add_row(direction, side, -1, 'direction_fallback')
        for tenor, group in direction.groupby('product_duration_seconds'):
            if len(group) >= settings['minimum_cell_orders']:
                add_row(group, side, tenor, 'direction_tenor')
    columns = ['option_type', 'product_duration_seconds', 'threshold_source', 'threshold_history_orders'] + [
        f'{kind}_threshold_{suffix}' for kind in ['unit', 'cash'] for suffix in QUANTILES]
    thresholds = pd.DataFrame(rows, columns=columns).set_index(['option_type', 'product_duration_seconds'])
    return {'day': day, 'history_ids': ids.copy(), 'thresholds': thresholds,
            'primary_target': settings['primary_target'], 'history_days': protocol['history_days']}


def label_orders(pool, ids, fitted):
    """Apply already fitted thresholds to outcomes; never return an input feature table."""
    _check_pool(pool)
    if not isinstance(ids, pd.Index) or not ids.is_unique or not ids.isin(pool.X.index).all():
        raise ValueError('Labels need unique known order IDs')
    out = pd.DataFrame(index=ids)
    out['threshold_source'] = 'insufficient_direction_history'
    out['threshold_history_orders'] = 0
    for kind in ['unit', 'cash']:
        for suffix in QUANTILES:
            out[f'{kind}_threshold_{suffix}'] = np.nan
    thresholds = fitted['thresholds']
    for (side, tenor), group in pool.X.loc[ids].groupby(['option_type', 'product_duration_seconds']):
        key = (side, tenor) if (side, tenor) in thresholds.index else (side, -1)
        if key in thresholds.index:
            row = thresholds.loc[key]
            out.loc[group.index, row.index] = row.to_numpy()
    targets = {'unit': pool.y.loc[ids], 'cash': pool.y.loc[ids] * pool.X.loc[ids, 'quantity_btc']}
    for kind, target in targets.items():
        for suffix in QUANTILES:
            threshold = out[f'{kind}_threshold_{suffix}']
            valid = threshold.notna() & threshold.gt(0)
            out[f'high_risk_{kind}_{suffix}'] = target.gt(threshold).astype('Int64').where(valid)
        out[f'{kind}_threshold_available'] = out[f'{kind}_threshold_q975'].gt(0)
    out['high_risk'] = out[f"high_risk_{fitted['primary_target']}_q975"].copy()
    out['label_available_at_utc'] = pool.metadata.loc[ids, 'closed_at_utc']
    return out
