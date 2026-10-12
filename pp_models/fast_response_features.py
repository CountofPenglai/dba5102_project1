"""Causal UTC 15-minute Put response state, independent of order outcomes."""
import numpy as np
import pandas as pd
from pp_data.io import ModelSplit

STEP = pd.Timedelta(minutes=15)


def response_bars(usable, lookback=96, *, interval_minutes=15, power=None):
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise ValueError('Invalid response lookback')
    if (type(interval_minutes) is not int or interval_minutes not in (5, 15)
            or (power is not None and (type(power) is not int or power not in (1, 2)))
            or (power is None) != (interval_minutes == 15)):
        raise ValueError('Expected legacy 15-minute or linear/squared 5-minute response')
    step, frequency = pd.Timedelta(minutes=interval_minutes), f'{interval_minutes}min'
    history = usable.sort_values('open_time_utc').copy()
    if (history.empty or history.open_time_utc.dt.tz is None or history.close_time_utc.dt.tz is None
            or history.open_time_utc.duplicated().any() or not np.isfinite(history.close).all()
            or history.close.le(0).any()
            or not history.open_time_utc.eq(history.open_time_utc.dt.floor('min')).all()
            or not history.close_time_utc.eq(history.open_time_utc+pd.Timedelta(milliseconds=59999)).all()):
        raise ValueError('Expected unique audited complete minute candles')
    history['bar'] = history.open_time_utc.dt.tz_convert('UTC').dt.floor(frequency)
    bars = history.groupby('bar').agg(price=('close', 'last'), count=('close', 'size'))
    grid = pd.date_range(bars.index.min(), bars.index.max(), freq=frequency, tz='UTC')
    bars = bars.reindex(grid)
    # Unique aligned minutes and the exact count imply a complete interval.
    price = bars.price.where(bars['count'].eq(interval_minutes))
    returns = np.log(price).diff()
    sigma = np.sqrt(returns.pow(2).shift(1).rolling(lookback, min_periods=lookback).mean()).clip(lower=.0001)
    state, pressure, run = 0., 0., 0
    states, phases = [], []
    for ret, scale in zip(returns, sigma):
        if not np.isfinite(ret) or not np.isfinite(scale):
            state, pressure, run = 0., 0., 0
            states.append(float('nan')); phases.append('unavailable')
            continue
        decline = max(-float(ret)/float(scale)-.1, 0.)
        if decline > 0:
            run += 1
            if power is None:
                state += (1-state)*(-np.expm1(-decline))
            else:
                pressure = min(3., pressure+decline)
                state = (pressure/3.)**power
            phase = 'first_down' if run == 1 else 'continued_down'
        else:
            state *= .5
            if power is not None:
                pressure *= 2.**(-1./power)
            run = 0
            phase = 'recovery' if state > 0 else 'quiet'
        states.append(state); phases.append(phase)
    return pd.DataFrame({'response_available_at_utc': grid+step, 'fast_response_state': states,
                         'fast_response_phase': phases, 'fast_response_sigma': sigma.to_numpy()})


def response_features(created, usable, lookback=96, *, interval_minutes=15, power=None):
    if created.empty or not created.index.is_unique or created.isna().any() or created.dt.tz is None:
        raise ValueError('Invalid response order timestamps')
    right = response_bars(usable, lookback, interval_minutes=interval_minutes, power=power)
    left = pd.DataFrame({'created': created.dt.tz_convert('UTC').array, 'position': np.arange(len(created))})
    matched = pd.merge_asof(left.sort_values('created'), right, left_on='created',
        right_on='response_available_at_utc', direction='backward', allow_exact_matches=True)
    matched = matched.set_index('position').reindex(range(len(created)))
    age = matched.created - matched.response_available_at_utc
    if (not np.isfinite(matched.fast_response_state).all() or not age.ge(pd.Timedelta(0)).all()
            or not age.lt(pd.Timedelta(minutes=interval_minutes)).all()):
        raise ValueError('Missing contiguous latest response state; orders cannot be dropped or forward-filled')
    return matched.drop(columns='created').set_axis(created.index)


def augment_pool(pool, usable):
    if not pool.X.index.equals(pool.y.index) or not pool.X.index.equals(pool.metadata.index):
        raise ValueError('Misaligned canonical pool')
    extra = response_features(pool.metadata.created_at_utc, usable)
    if set(extra) & (set(pool.X) | set(pool.metadata)):
        raise ValueError('Response feature columns collide')
    metadata = pool.metadata.assign(response_available_at_utc=extra.response_available_at_utc)
    X = pd.concat([pool.X, extra.drop(columns='response_available_at_utc')], axis=1)
    return ModelSplit(X, pool.y.copy(), metadata), {
        'version': 'fast_response_features_v1', 'orders': len(X), 'interval_minutes': 15,
        'scale_lookback_bars': 96, 'scale_floor': .0001, 'noise_threshold_sigma': .1,
        'recovery_factor_per_bar': .5, 'max_state': float(extra.fast_response_state.max()),
        'latest_state_age_seconds_max': float((pool.metadata.created_at_utc-extra.response_available_at_utc).dt.total_seconds().max()),
        'state_continues_across_days': True, 'labels_used_for_state': False}
