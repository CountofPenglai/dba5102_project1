"""Causal terminal-return distributions; scenarios and row features stay in memory."""
import numpy as np
import pandas as pd
from pp_data.io import ModelSplit

HORIZONS = (30, 60, 120, 240, 480)
METHODS = ('static', 'dynamic')
LEVELS = (80, 90, 95)


def _finite_samples(samples):
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Expected finite one-dimensional empirical distribution')
    return values


def expected_put(log_returns, spot, strike):
    values = _finite_samples(log_returns)
    if not np.isfinite([spot, strike]).all() or min(spot, strike) <= 0 or np.abs(values).max() > 20:
        raise ValueError('Invalid price or explosive log-return scenarios')
    return float(np.maximum(strike-spot*np.exp(values), 0.).mean())


def distribution_scores(samples, actual):
    """Empirical CRPS plus interpolated central intervals, in input return units."""
    values = np.sort(_finite_samples(samples))
    if not np.isfinite(actual):
        raise ValueError('Nonfinite terminal outcome')
    n = len(values)
    result = {'crps': float(np.abs(values-actual).mean()-np.dot(2*np.arange(1,n+1)-n-1,values)/(n*n))}
    for level in LEVELS:
        alpha = 1-level/100
        lower, upper = np.quantile(values, [alpha/2, 1-alpha/2])
        result.update({f'covered_{level}': float(lower <= actual <= upper),
            f'lower_miss_{level}': float(actual < lower), f'upper_miss_{level}': float(actual > upper),
            f'width_{level}': float(upper-lower),
            f'interval_score_{level}': float(upper-lower+2/alpha*(max(lower-actual,0)+max(actual-upper,0)))})
    return result


class TerminalMarket:
    """Fixed half-life EWMA with complete-window history available strictly before D."""
    def __init__(self, usable):
        if (usable.empty or usable.open_time_utc.duplicated().any()
                or not usable.open_time_utc.is_monotonic_increasing
                or usable.open_time_utc.dt.tz is None
                or not usable.open_time_utc.eq(usable.open_time_utc.dt.floor('min')).all()
                or not np.isfinite(usable.close).all() or usable.close.le(0).any()):
            raise ValueError('Expected sorted audited minute candles')
        available = pd.DatetimeIndex(usable.open_time_utc+pd.Timedelta(minutes=1))
        if (pd.DatetimeIndex(usable.close_time_utc) > available).any():
            raise ValueError('Candle cannot be used before close')
        index = pd.date_range(available[0], available[-1], freq='min')
        self.price = pd.Series(usable.close.to_numpy(float), index=available).reindex(index)
        log_price = np.log(self.price)
        returns = log_price.diff()
        blocks = returns.isna().cumsum()
        variance = returns.pow(2).groupby(blocks).transform(
            lambda s: s.ewm(halflife=60, adjust=False, min_periods=1440).mean())
        self.sigma = np.sqrt(variance).clip(lower=1e-6)
        self.forward = {}
        for horizon in HORIZONS:
            complete = self.price.notna().rolling(horizon+1).sum().shift(-horizon).eq(horizon+1)
            self.forward[horizon] = (log_price.shift(-horizon)-log_price).where(complete)
        self._history, self._samples = {}, {}

    @staticmethod
    def signal_time(stamp):
        stamp = pd.Timestamp(stamp)
        if stamp.tzinfo is None:
            raise ValueError('Timezone-aware signal timestamp required')
        return stamp.tz_convert('UTC').floor('5min')

    def sigma_at(self, stamp):
        value = self.sigma.get(self.signal_time(stamp), float('nan'))
        if not np.isfinite(value):
            raise ValueError('Missing continuous EWMA warmup at pricing time')
        return float(value)

    def history(self, day, horizon):
        day = pd.Timestamp(day)
        if (day.tzinfo is None or str(day.tz) != 'UTC' or day != day.floor('D')
                or isinstance(horizon, bool) or horizon not in HORIZONS):
            raise ValueError('Invalid daily terminal horizon')
        key = day, horizon
        if key not in self._history:
            origins = self.price.index
            eligible = (origins >= day-pd.Timedelta(days=30)) & (origins+pd.Timedelta(minutes=horizon) < day)
            eligible &= origins.minute % 30 == 0
            frame = pd.DataFrame({'return': self.forward[horizon], 'sigma': self.sigma}).loc[eligible].dropna()
            if len(frame) < 100:
                raise ValueError(f'Insufficient complete market history: {day}, H={horizon}, n={len(frame)}')
            self._history[key] = frame
        return self._history[key]

    def samples(self, day, horizon, stamp, method):
        day, stamp = pd.Timestamp(day), pd.Timestamp(stamp)
        if method not in METHODS or stamp.tzinfo is None or not day <= stamp < day+pd.Timedelta(days=1):
            raise ValueError('Distribution must use the forecast day snapshot')
        key = day, horizon, method
        if key not in self._samples:
            history = self.history(day,horizon)
            values = history['return'].to_numpy(float)
            if method == 'dynamic':
                values = values/(history.sigma.to_numpy(float)*np.sqrt(horizon))
            self._samples[key] = np.sort(values-values.mean())
        scale = self.sigma_at(stamp)*np.sqrt(horizon) if method == 'dynamic' else 1.
        return self._samples[key]*scale


def augment_pool(pool, market):
    if not pool.X.index.equals(pool.y.index) or not pool.X.index.equals(pool.metadata.index):
        raise ValueError('Misaligned terminal feature pool')
    names = ['terminal_supported','terminal_static_proxy','terminal_dynamic_proxy']
    if set(names) & set(pool.X):
        raise ValueError('Terminal features already exist')
    X = pool.X.copy()
    remaining = X.remaining_seconds
    if not np.isfinite(remaining).all() or remaining.le(0).any():
        raise ValueError('Nonpositive contractual remaining time')
    supported = X.option_type.eq(1) & remaining.le(480*60)
    X['terminal_supported'] = supported.astype(int)
    X['terminal_static_proxy'], X['terminal_dynamic_proxy'] = 0., 0.
    min_history, max_history, min_scaling, max_age = float('inf'), 0, 1., 0.
    for idx in X.index[supported]:
        stamp = pool.metadata.at[idx,'created_at_utc']
        day = stamp.floor('D')
        tau = X.at[idx,'remaining_seconds']/60
        horizon = next(h for h in HORIZONS if h >= tau)
        scaling = np.sqrt(tau/horizon)
        n = len(market.history(day,horizon))
        min_history, max_history = min(min_history,n), max(max_history,n)
        min_scaling = min(min_scaling,scaling)
        max_age = max(max_age,(stamp-market.signal_time(stamp)).total_seconds())
        for method in METHODS:
            samples = market.samples(day,horizon,stamp,method)*scaling
            X.at[idx,f'terminal_{method}_proxy'] = expected_put(samples,X.at[idx,'market_price_usdc_per_btc'],X.at[idx,'strike_price_usdc_per_btc'])
    if not np.isfinite(X[names]).all().all():
        raise ValueError('Nonfinite terminal features')
    return ModelSplit(X,pool.y.copy(),pool.metadata.copy()), {
        'orders':len(X),'supported_put_orders':int(supported.sum()),
        'call_orders_unchanged':int(X.option_type.eq(0).sum()),
        'unsupported_put_orders':int((X.option_type.eq(1)&~supported).sum()),
        'minimum_history_origins':int(min_history) if supported.any() else 0,'maximum_history_origins':max_history,
        'minimum_horizon_return_scaling':min_scaling,'maximum_signal_age_seconds':max_age,
        'order_outcomes_used_for_distribution':False,'historical_features_replayed_as_of_origin':True}


def evaluate_market(market, phase_days, periods):
    """Only daily/horizon/state aggregates leave this function; outcomes score only."""
    rows = []
    for phase, days in phase_days.items():
        end = pd.Timestamp(periods[phase][1])
        for day in days:
            for horizon in HORIZONS:
                history = market.history(day,horizon)
                threshold = float(history.sigma.quantile(.8))
                origins = pd.date_range(day,day+pd.Timedelta(days=1),freq='30min',inclusive='left')
                origins = origins[origins+pd.Timedelta(minutes=horizon)<end]
                valid = []
                for stamp in origins:
                    outcome = market.forward[horizon].get(stamp,float('nan'))
                    if np.isfinite(outcome) and np.isfinite(market.sigma.get(stamp,float('nan'))):
                        valid.append((stamp,outcome))
                if not valid:
                    raise ValueError(f'No complete market scoring origins: {day}, H={horizon}')
                for method in METHODS:
                    scores = []
                    for stamp, outcome in valid:
                        scale = market.sigma_at(stamp)
                        entry = distribution_scores(market.samples(day,horizon,stamp,method)*10000,outcome*10000)
                        entry['volatility_group'] = 'high' if scale >= threshold else 'normal'
                        entry['scale_ratio'] = (scale/history.sigma).mean() if method == 'dynamic' else 1.
                        scores.append(entry)
                    frame = pd.DataFrame(scores)
                    for group in ('all','normal','high'):
                        subset = frame if group == 'all' else frame.loc[frame.volatility_group.eq(group)]
                        if subset.empty:
                            continue
                        rows.append(dict(phase=phase,day_utc=day.isoformat(),method=method,horizon_minutes=horizon,
                            volatility_group=group,origins=len(subset),history_origins=len(history),
                            scheduled_origins=len(origins) if group == 'all' else float('nan'),
                            excluded_origins=len(origins)-len(valid) if group == 'all' else float('nan'),
                            historical_sigma_q80=threshold,
                            **subset.drop(columns='volatility_group').mean().to_dict()))
    daily = pd.DataFrame(rows)
    if daily.empty:
        raise ValueError('No market scoring origins')
    keys = ['phase','method','horizon_minutes','volatility_group']
    counts = ['scheduled_origins','excluded_origins']
    metrics = [c for c in daily if c not in keys+counts+['day_utc','origins','history_origins','historical_sigma_q80']]
    summaries = []
    for key, frame in daily.groupby(keys,sort=True):
        summaries.append(dict(zip(keys,key)) | {'origins':int(frame.origins.sum()),'days':len(frame),
            **{c:float(frame[c].sum(min_count=1)) for c in counts},
            **{m:float(np.average(frame[m],weights=frame.origins)) for m in metrics}})
    return pd.DataFrame(summaries),daily
