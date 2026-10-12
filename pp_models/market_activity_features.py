"""Pricing-time price dynamics and platform creation-event counts, in memory only."""
import csv
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from pp_data.config import load_config
from pp_data.io import ModelSplit
from .directional_features import augment_pool

PRICE_FEATURES = ['vol_change_60m', 'vol_ratio_60m_240m', 'max_up_jump_240m',
                  'max_down_jump_240m', 'max_drawdown_240m', 'max_drawup_240m']
ACTIVITY_FEATURES = [f'platform_{side}_count_{w}m' for w in (60, 240) for side in ('call', 'put')]


def _times(series):
    if series.empty or series.isna().any() or series.dt.tz is None or not series.index.is_unique:
        raise ValueError('Expected nonempty unique timezone-aware pricing times')
    return series.dt.as_unit('ns').astype('int64').to_numpy()


def price_dynamics(created, usable, short_window=60, long_window=240):
    times = _times(created)
    if (any(isinstance(w, bool) or not isinstance(w, int) or w <= 0 for w in (short_window, long_window))
            or long_window < 2 * short_window):
        raise ValueError('Need positive windows and at least two short windows of history')
    history = usable.sort_values('close_time_utc').reset_index(drop=True)
    if (history.empty or history.open_time_utc.duplicated().any()
            or not np.isfinite(history.close).all() or history.close.le(0).any()):
        raise ValueError('Expected unique audited market candles')
    closes = _times(history.close_time_utc)
    opens = _times(history.open_time_utc)
    positions = np.searchsorted(closes, times, side='right') - 1
    if (positions < long_window).any():
        raise ValueError('Missing contiguous price history')
    indices = positions[:, None] - np.arange(long_window, -1, -1)
    if (np.diff(opens[indices], axis=1) != 60_000_000_000).any() or (times-closes[positions] > 60_000_000_000).any():
        raise ValueError('Missing contiguous or fresh price history')
    paths = np.log(history.close.to_numpy()[indices])
    returns = np.diff(paths, axis=1)
    recent = np.sqrt(np.mean(returns[:, -short_window:]**2, axis=1))
    prior = np.sqrt(np.mean(returns[:, -2*short_window:-short_window]**2, axis=1))
    long = np.sqrt(np.mean(returns**2, axis=1))
    result = pd.DataFrame({
        f'vol_change_{short_window}m': recent - prior,
        f'vol_ratio_{short_window}m_{long_window}m': recent / np.maximum(long, 1e-12),
        f'max_up_jump_{long_window}m': np.maximum(returns, 0).max(axis=1),
        f'max_down_jump_{long_window}m': np.maximum(-returns, 0).max(axis=1),
        f'max_drawdown_{long_window}m': (np.maximum.accumulate(paths, axis=1)-paths).max(axis=1),
        f'max_drawup_{long_window}m': (paths-np.minimum.accumulate(paths, axis=1)).max(axis=1),
    }, index=created.index)
    result['feature_last_close_utc'] = pd.Series(history.close_time_utc.iloc[positions].array, index=created.index)
    return result


def read_order_prefix(path, cutoff):
    """Only parse createdAt/optionType before cutoff; outcome fields cannot filter events."""
    end = pd.Timestamp(cutoff)
    if end.tzinfo is None or end > pd.Timestamp(load_config()['splits']['validation']['end']):
        raise ValueError('Order cutoff must be timezone-aware and within validation')
    digest, records, previous = hashlib.sha256(), [], None
    count, invalid_directions, boundary = 0, 0, None
    with Path(path).open('rb') as stream:
        header = stream.readline()
        columns = next(csv.reader([header.decode('utf-8-sig')]))
        time_column, type_column = columns.index('createdAt'), columns.index('optionType')
        digest.update(header)
        for line in stream:
            values = next(csv.reader([line.decode('utf-8')]))
            stamp = pd.to_datetime(values[time_column], utc=True, errors='raise')
            if pd.isna(stamp) or (previous is not None and stamp < previous):
                raise ValueError('Order source must be chronological with valid creation timestamps')
            if stamp >= end:
                boundary = stamp.isoformat()
                break
            previous = stamp
            digest.update(line)
            count += 1
            try:
                direction = float(values[type_column])
            except ValueError:
                direction = np.nan
            if direction not in (0., 1.):
                invalid_directions += 1
                continue
            records.append((stamp, int(direction)))
    if not records:
        raise ValueError('No usable platform creation events')
    events = pd.DataFrame(records, columns=['created_at_utc', 'option_type'])
    return events, {'source_path': str(Path(path).resolve()), 'cutoff_exclusive': end.isoformat(),
        'prefix_sha256': digest.hexdigest(), 'prefix_rows': count, 'usable_events': len(events),
        'invalid_direction_rows': invalid_directions, 'boundary_timestamp_only': boundary,
        'first_created_utc': events.created_at_utc.min().isoformat(),
        'last_created_utc': events.created_at_utc.max().isoformat(),
        'parsed_columns': ['createdAt', 'optionType'],
        'filter': 'valid creation timestamp and direction only; no settlement/outcome/cohort filter',
        'deduplication': 'source has no order ID; count each provided creation record',
        'prefix_ordering_assumption': 'chronological provider file; stop at first boundary'}


def platform_activity(created, events):
    times = _times(created)
    if (events.created_at_utc.isna().any() or events.created_at_utc.dt.tz is None
            or not events.option_type.isin([0, 1]).all()):
        raise ValueError('Invalid creation events')
    counts = {}
    for w in (60, 240):
        for direction, label in ((0, 'call'), (1, 'put')):
            history = events.loc[events.option_type.eq(direction), 'created_at_utc']
            stamps = np.sort(history.dt.as_unit('ns').astype('int64').to_numpy())
            counts[f'platform_{label}_count_{w}m'] = (np.searchsorted(stamps, times, side='left')
                - np.searchsorted(stamps, times - w*60_000_000_000, side='left'))
    return pd.DataFrame(counts, index=created.index)


def augment_market_activity(pool, usable, events):
    augmented, direction_info = augment_pool(pool, usable)
    created = pool.metadata.created_at_utc
    dynamics = price_dynamics(created, usable)
    if not dynamics.feature_last_close_utc.equals(pool.metadata.last_candle_close_utc.rename('feature_last_close_utc')):
        raise ValueError('Dynamics timestamps differ from canonical features')
    extra = pd.concat([dynamics.drop(columns='feature_last_close_utc'), platform_activity(created, events)], axis=1)
    if set(extra) & set(augmented.X) or not np.isfinite(extra).all().all():
        raise ValueError('Invalid or colliding supplemental features')
    X = pd.concat([augmented.X, extra], axis=1)
    statistics = []
    start = pd.Timestamp(load_config()['splits']['validation']['start'])
    for split, mask in [('train', created.lt(start)), ('validation', created.ge(start))]:
        for feature in extra:
            values = extra.loc[mask, feature]
            statistics.append({'split': split, 'feature': feature, 'orders': len(values), 'mean': values.mean(),
                'min': values.min(), 'median': values.median(), 'p90': values.quantile(.9),
                'p99': values.quantile(.99), 'max': values.max(), 'zero_fraction': values.eq(0).mean()})
    return ModelSplit(X, pool.y.copy(), pool.metadata.copy()), {
        'version': 'market_activity_features_v1', 'directional': direction_info,
        'added_features': list(extra), 'orders': len(X), 'canonical_rows_preserved': True,
        'price_path': 'last 241 completed contiguous minute closes; ordered log drawdown/drawup',
        'vol_change': 'last 60 minute RMS minus preceding nonoverlapping 60 minute RMS',
        'vol_ratio': '60 minute RMS / max(240 minute RMS, 1e-12)',
        'order_counts': '[created_at - window, created_at); all recorded directions, regardless of settlement',
        'same_time_events_excluded': True}, pd.DataFrame(statistics)
