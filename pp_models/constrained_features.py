"""Five-minute contract-aware signals and separately adapted empirical return tails."""
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE
from pp_data.io import ModelSplit
from .terminal_distribution import TerminalMarket, HORIZONS, distribution_scores
from scripts.payout_signal_analysis import market_signals, platform_signals


def expected_payoff(samples, spot, strike, side):
    values = np.asarray(samples, float)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or np.abs(values).max() > 20 or side not in (0, 1)
            or not np.isfinite([spot, strike]).all() or min(spot, strike) <= 0):
        raise ValueError('Invalid payoff scenarios')
    difference = spot * np.exp(values) - strike
    return float(np.maximum(difference if side == 0 else -difference, 0.).mean())


def tail_factors(base, recent):
    """Shrink seven-day 5/95 percentiles toward the 30-day zero-centered law."""
    base, recent = np.asarray(base, float), np.asarray(recent, float)
    if (base.ndim != 1 or recent.ndim != 1 or len(base) < 2
            or not np.isfinite(base).all() or not np.isfinite(recent).all()):
        raise ValueError('Invalid calibration sample')
    if len(recent) < 100: return 1., 1., len(recent)
    low, high = np.quantile(base, [.05, .95])
    rlow, rhigh = np.quantile(recent, [.05, .95])
    weight = len(recent) / (len(recent) + 200.)
    ratios = [max(-rlow, 0.) / max(-low, 1e-8), max(rhigh, 0.) / max(high, 1e-8)]
    factors = [float(np.exp(weight * np.log(np.clip(r, .5, 2.)))) for r in ratios]
    return *factors, len(recent)


def apply_tail_factors(values, lower, upper):
    if not np.isfinite([lower, upper]).all() or min(lower, upper) <= 0:
        raise ValueError('Tail factors must be positive')
    values = np.asarray(values, float)
    return values * np.where(values < 0, lower, upper)


class CalibratedMarket(TerminalMarket):
    def __init__(self, usable):
        super().__init__(usable)
        self._tail_parameters = {}

    def parameters(self, day, horizon):
        key = pd.Timestamp(day), horizon
        if key not in self._tail_parameters:
            history = self.history(day, horizon)
            z = history['return'].to_numpy(float) / (history.sigma.to_numpy(float) * np.sqrt(horizon))
            base = z - z.mean()
            recent = z[history.index >= pd.Timestamp(day) - pd.Timedelta(days=7)]
            self._tail_parameters[key] = tail_factors(base, recent)
        return self._tail_parameters[key]

    def samples(self, day, horizon, stamp, method):
        if method != 'calibrated': return super().samples(day, horizon, stamp, method)
        values = super().samples(day, horizon, stamp, 'dynamic')
        lower, upper, _ = self.parameters(day, horizon)
        return apply_tail_factors(values, lower, upper)


def augment_constraints(pool, market, usable, events):
    X = pool.X.copy()
    # Completed five-minute snapshot for all activity/fast state features.
    times = pool.metadata.created_at_utc.dt.floor('5min')
    signals = market_signals(times, usable)
    activity = platform_signals(times, events)
    tau = X.remaining_seconds / 60
    if tau.le(0).any(): raise ValueError('Nonpositive remaining time')
    X['constraint_supported'] = tau.le(480).astype(int)
    X['contract_dynamic'] = X['contract_calibrated'] = 0.
    X['contract_momentum'] = X['contract_vol_change'] = 0.
    X['flow_market'] = np.log1p(signals.market_trades_per_min_60m)
    X['flow_call'] = np.log1p(activity.platform_call_count_60m)
    X['flow_put'] = np.log1p(activity.platform_put_count_60m)
    for idx in X.index[X.constraint_supported.eq(1)]:
        stamp = pool.metadata.at[idx, 'created_at_utc']; day = stamp.floor('D')
        duration = tau.loc[idx]; horizon = next(h for h in HORIZONS if h >= duration)
        sigma = market.sigma_at(stamp)
        uncertainty = max(sigma * np.sqrt(duration), 1e-6)
        spot, strike = X.at[idx, 'market_price_usdc_per_btc'], X.at[idx, 'strike_price_usdc_per_btc']
        moneyness = np.log(spot / strike) / uncertainty
        gate = np.exp(-.5 * min(moneyness * moneyness, 100.))
        sign = 1. if X.at[idx, 'option_type'] == 0 else -1.
        X.at[idx, 'contract_momentum'] = sign * signals.at[idx, 'market_return_15m'] / uncertainty * gate
        X.at[idx, 'contract_vol_change'] = (signals.at[idx, 'market_vol_change_60m'] /
                                           max(signals.at[idx, 'market_total_rms_240m'], 1e-6)) * gate
        for method in ('dynamic', 'calibrated'):
            values = market.samples(day, horizon, stamp, method) * np.sqrt(duration / horizon)
            proxy = expected_payoff(values, spot, strike, X.at[idx, 'option_type'])
            X.at[idx, f'contract_{method}'] = np.log1p(proxy) - np.log1p(X.at[idx, BASE_FEATURE])
    return ModelSplit(X, pool.y.copy(), pool.metadata.copy()), {
        'orders':len(X), 'supported_orders':int(X.constraint_supported.sum()),
        'unsupported_orders':int(X.constraint_supported.eq(0).sum()),
        'signal_stride_minutes':5, 'max_signal_age_seconds':float((pool.metadata.created_at_utc-times).dt.total_seconds().max()),
        'activity_missing':X[['flow_market','flow_call','flow_put']].isna().sum().to_dict(),
        'unavailable_activity':'training-side median imputation gives zero standardized effect',
        'distribution':'30d FHS; separate 7d standardized 5/95 percentile tail scaling, shrunk n/(n+200)',
        'tail_ratio_limits':[.5,2.], 'minimum_recent_origins':100,
        'early_close_input':False, 'historical_features_replayed_as_of_origin':True}


def evaluate_calibrated_market(market, phase_days, periods):
    """Same origins for static/dynamic/calibrated, with distinct downside/upside misses."""
    rows = []
    for phase, days in phase_days.items():
        end = pd.Timestamp(periods[phase][1])
        for day in days:
            for horizon in HORIZONS:
                history = market.history(day, horizon)
                threshold = float(history.sigma.quantile(.8))
                origins = pd.date_range(day, day+pd.Timedelta(days=1), freq='30min', inclusive='left')
                origins = origins[origins+pd.Timedelta(minutes=horizon) < end]
                valid = [(s, market.forward[horizon].get(s, np.nan)) for s in origins
                         if np.isfinite(market.forward[horizon].get(s, np.nan))
                         and np.isfinite(market.sigma.get(s, np.nan))]
                if not valid: raise ValueError('No complete market scoring origins')
                lower, upper, recent_n = market.parameters(day, horizon)
                for method in ('static', 'dynamic', 'calibrated'):
                    scores = []
                    for stamp, actual in valid:
                        entry = distribution_scores(market.samples(day,horizon,stamp,method)*10000,actual*10000)
                        entry['volatility_group'] = 'high' if market.sigma_at(stamp) >= threshold else 'normal'
                        scores.append(entry)
                    frame = pd.DataFrame(scores)
                    for group in ('all', 'normal', 'high'):
                        sub = frame if group == 'all' else frame.loc[frame.volatility_group.eq(group)]
                        if sub.empty: continue
                        rows.append(dict(phase=phase, day_utc=day.isoformat(), method=method,
                            horizon_minutes=horizon, volatility_group=group, origins=len(sub),
                            scheduled_origins=len(origins) if group=='all' else np.nan,
                            excluded_origins=len(origins)-len(valid) if group=='all' else np.nan,
                            lower_factor=lower if method=='calibrated' else 1.,
                            upper_factor=upper if method=='calibrated' else 1., recent_origins=recent_n,
                            **sub.drop(columns='volatility_group').mean().to_dict()))
    daily = pd.DataFrame(rows); keys = ['phase','method','horizon_minutes','volatility_group']
    counts = ['scheduled_origins','excluded_origins']
    metrics = [c for c in daily if c not in keys+counts+['day_utc','origins','recent_origins']]
    summaries = []
    for key, frame in daily.groupby(keys, sort=True):
        summaries.append(dict(zip(keys,key)) | {'origins':int(frame.origins.sum()),'days':len(frame),
            **{c:float(frame[c].sum(min_count=1)) for c in counts},
            **{m:float(np.average(frame[m],weights=frame.origins)) for m in metrics}})
    return pd.DataFrame(summaries), daily
