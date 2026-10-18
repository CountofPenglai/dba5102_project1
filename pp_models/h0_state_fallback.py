"""Market-distribution novelty and strictly historical forecast-error mixing."""
import numpy as np
import pandas as pd

STATE_COLUMNS = ['logv60', 'logv1440', 'imbalance', 'trend']


def market_states(market):
    m = market.sort_values('open_time_utc')
    if (m.empty or m.open_time_utc.dt.tz is None or m.open_time_utc.duplicated().any()
            or not m.open_time_utc.eq(m.open_time_utc.dt.floor('min')).all()
            or not m.close_time_utc.eq(m.open_time_utc+pd.Timedelta(milliseconds=59999)).all()
            or not np.isfinite(m.close).all() or m.close.le(0).any()):
        raise ValueError('Invalid complete minute market')
    times = pd.date_range(m.open_time_utc.min(), m.open_time_utc.max(), freq='min')
    price = m.set_index('open_time_utc').close.reindex(times); logs = np.log(price); r = logs.diff()
    rv60 = np.sqrt(r.pow(2).rolling(60).sum()); v24 = r.pow(2).rolling(1440).sum()
    balance = (r.clip(lower=0).pow(2).rolling(1440).sum()-r.clip(upper=0).pow(2).rolling(1440).sum())/(v24+1e-12)
    raw = pd.DataFrame({'logv60': np.log(rv60.clip(lower=1e-6)), 'logv1440': np.log(np.sqrt(v24).clip(lower=1e-6)),
        'imbalance': balance, 'trend': ((logs-logs.shift(1440))/np.sqrt(v24).clip(lower=1e-6)).clip(-5, 5)})
    raw.index = times+pd.Timedelta(minutes=1)
    # Only whole completed hours, followed by a six-hour trailing state average.
    hourly = raw.loc[raw.index.minute == 0, STATE_COLUMNS]
    return hourly.rolling(6, min_periods=6).mean()


def match_states(created, states):
    if (created.isna().any() or created.dt.tz is None or not created.index.is_unique
            or not states.index.is_unique or states.index.tz is None or list(states) != STATE_COLUMNS):
        raise ValueError('Invalid state matching inputs')
    left = pd.DataFrame({'created': created.array, 'position': np.arange(len(created))})
    right = states.sort_index().reset_index(names='state_at')
    merged = pd.merge_asof(left.sort_values('created'), right, left_on='created', right_on='state_at',
        direction='backward', tolerance=pd.Timedelta(hours=1)-pd.Timedelta(nanoseconds=1))
    return merged.set_index('position').reindex(range(len(created)))[STATE_COLUMNS].set_axis(created.index)


class StateGate:
    # Measures novelty relative to recent market states, not payout probability.
    def fit(self, states, origin, training_start):
        origin = pd.Timestamp(origin); earliest = max(pd.Timestamp(training_start), origin-pd.Timedelta(days=30))
        if (origin.tzinfo is None or states.index.tz is None or not states.index.is_unique or list(states) != STATE_COLUMNS):
            raise ValueError('Invalid background state history')
        h = states.loc[(states.index >= earliest) & (states.index < origin)].dropna()
        if not np.isfinite(h).all().all(): raise ValueError('Nonfinite state history')
        self.state = dict(supported=False, reason='insufficient_hourly_support', hours=len(h),
            latest_hour=h.index.max().isoformat() if len(h) else None)
        if len(h) >= 168:
            raw = h.to_numpy(); center = np.median(raw, axis=0)
            # Median/IQR scaling limits extreme hours' influence on the background.
            scale = (np.quantile(raw, .75, axis=0)-np.quantile(raw, .25, axis=0))/1.349
            scale[scale < 1e-6] = 1.
            d = np.sqrt(np.mean(((raw-center)/scale)**2, axis=1)); lo, hi = np.quantile(d, [.9, .99])
            if hi-lo < 1e-6: self.state['reason'] = 'degenerate_distance'
            else: self.state.update(supported=True, reason='none', center=center.tolist(), scale=scale.tolist(), q90=float(lo), q99=float(hi))
        return self

    def predict(self, features):
        if not hasattr(self, 'state'): raise ValueError('State background not fitted')
        if list(features) != STATE_COLUMNS: raise ValueError('Unexpected state features')
        valid = np.isfinite(features.to_numpy()).all(axis=1); distance = np.full(len(features), np.nan)
        # gate=1 means full fallback to Base; unknown states take that default.
        # The caller multiplies H0's blend weight by (1-gate).
        gate = np.ones(len(features))
        if self.state['supported'] and valid.any():
            z = (features.loc[valid].to_numpy()-self.state['center'])/self.state['scale']
            distance[valid] = np.sqrt(np.mean(z*z, axis=1))
            # Move from no fallback at historical q90 to full fallback at q99.
            gate[valid] = np.clip((distance[valid]-self.state['q90'])/(self.state['q99']-self.state['q90']), 0, 1)
        return gate, distance


def history_weight(past, origin):
    """Fit one direction's convex blend from genuinely earlier settled forecasts."""
    origin = pd.Timestamp(origin)
    required = ['base', 'h0', 'actual', 'created', 'closed', 'forecast_origin']
    if (any(k not in past for k in required) or not past.index.is_unique or origin.tzinfo is None
            or not np.isfinite(past[['base', 'h0', 'actual']]).all().all()
            or past[['base', 'h0', 'actual']].lt(0).any().any()):
        raise ValueError('Invalid prior forecast records')
    for k in ('created', 'closed', 'forecast_origin'):
        if past[k].isna().any() or past[k].dt.tz is None: raise ValueError('Invalid forecast times')
    if not (past.created.ge(origin-pd.Timedelta(days=30)) & past.created.lt(origin) & past.closed.lt(origin)
            & past.forecast_origin.le(past.created)).all():
        raise ValueError('Non-forward or unsettled history')
    n = len(past); days = int(past.created.dt.floor('D').nunique())
    # Insufficient settled forward history uses Base (alpha=0), not a fitted weight.
    par = dict(alpha=0., supported=False, reason='insufficient_oos_support', orders=n, days=days,
        latest_close=past.closed.max().isoformat() if n else None,
        latest_forecast_origin=past.forecast_origin.max().isoformat() if n else None)
    if n >= 100 and days >= 10:
        d = past.h0.to_numpy()-past.base.to_numpy(); delta = past.actual.to_numpy()-past.base.to_numpy()
        d2 = float(d@d); cross = float(d@delta)
        if d2 <= 1e-12: par['reason'] = 'identical_anchors'
        else:
            # Minimize sum((alpha*d-delta)^2) + penalty*(alpha-0.5)^2.
            # The closed-form solution is clipped to a convex blend; no future errors
            # or retrospectively selected monthly weights enter this objective.
            penalty = 100*d2/n; alpha = float(np.clip((cross+.5*penalty)/(d2+penalty), 0, 1))
            par.update(alpha=alpha, supported=True, reason='none', difference_ss=d2, cross=cross, penalty=penalty,
                h0_historical_mse=float(np.mean((past.h0-past.actual)**2)),
                base_historical_mse=float(np.mean((past.base-past.actual)**2)))
    return par
