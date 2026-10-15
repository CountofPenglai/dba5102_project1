"""Small direction-specific volatility responses of H0's conditional amount."""
import copy
import numpy as np
import pandas as pd

CANDIDATES = ('amount_level', 'amount_joint', 'amount_gate')
MARKET_COLUMNS = ['up_vol_60m', 'down_vol_60m', 'up_vol_240m', 'down_vol_240m']
STRENGTH = 100.


def market_features(created, market):
    """Completed minute returns only; holes remain unknown, with no quote deletion."""
    if (not isinstance(created, pd.Series) or created.empty or not created.index.is_unique
            or created.isna().any() or created.dt.tz is None):
        raise ValueError('Invalid quote creation times')
    m = market.sort_values('open_time_utc')
    if (m.empty or m.open_time_utc.dt.tz is None or m.open_time_utc.duplicated().any()
            or not m.open_time_utc.eq(m.open_time_utc.dt.floor('min')).all()
            or not m.close_time_utc.eq(m.open_time_utc+pd.Timedelta(milliseconds=59999)).all()
            or not np.isfinite(m.close).all() or m.close.le(0).any()):
        raise ValueError('Invalid audited minute candles')
    minutes = pd.date_range(m.open_time_utc.min(), m.open_time_utc.max(), freq='min')
    price = m.set_index('open_time_utc').close.reindex(minutes)
    returns = np.log(price).diff()
    right = pd.DataFrame({'at': minutes+pd.Timedelta(milliseconds=59999)})
    for w in (60, 240):
        for label, r in (('up', returns.clip(lower=0)), ('down', returns.clip(upper=0))):
            right[f'{label}_vol_{w}m'] = np.sqrt(r.pow(2).rolling(w, min_periods=w).mean()).to_numpy()
    left = pd.DataFrame({'created': created.array, 'position': np.arange(len(created))})
    matched = pd.merge_asof(left.sort_values('created'), right, left_on='created', right_on='at',
        direction='backward', tolerance=pd.Timedelta(seconds=60)).set_index('position').reindex(range(len(created)))
    result = matched[MARKET_COLUMNS].set_axis(created.index)
    result['available'] = np.isfinite(result[MARKET_COLUMNS]).all(axis=1)
    result['feature_last_close_utc'] = pd.Series(matched['at'].array, index=created.index)
    return result


def amount_features(x, market, anchor):
    if (not x.index.is_unique or not x.index.equals(market.index) or not x.index.equals(anchor.index)
            or not x.option_type.isin([0, 1]).all() or not np.isfinite(anchor).all() or anchor.lt(0).any()
            or not np.isfinite(x[['remaining_seconds', 'market_price_usdc_per_btc']]).all().all()
            or x[['remaining_seconds', 'market_price_usdc_per_btc']].le(0).any().any()):
        raise ValueError('Invalid aligned amount features/anchor')
    call = x.option_type.eq(0)
    v60 = np.where(call, market.up_vol_60m, market.down_vol_60m)
    v240 = np.where(call, market.up_vol_240m, market.down_vol_240m)
    available = market.available.to_numpy(bool) & np.isfinite(v60) & np.isfinite(v240) & (v60 >= 0) & (v240 >= 0)
    level = np.log1p(x.market_price_usdc_per_btc.to_numpy()*v60*np.sqrt(x.remaining_seconds.to_numpy()/60) /
        np.maximum(anchor.to_numpy(), 1.))
    speed = np.log((v60+1e-8)/(v240+1e-8))
    return pd.DataFrame({'level': np.where(available, level, np.nan),
        'speed': np.where(available, speed, np.nan), 'available': available}, index=x.index)


def check_alignment(side, features, anchor):
    if (not isinstance(side, pd.Series) or not side.index.is_unique or not side.isin([0, 1]).all()
            or not features.index.equals(side.index) or not anchor.index.equals(side.index)
            or not np.isfinite(anchor).all() or anchor.lt(0).any()):
        raise ValueError('Invalid aligned amount inputs')
    return features.available.eq(True) & np.isfinite(features[['level', 'speed']]).all(axis=1)


def gate(features, q75, q95):
    level = np.clip((features.level.to_numpy()-q75)/(q95-q75), 0, 1)
    ratio = np.exp(np.clip(features.speed.to_numpy(), -30, 30))
    return level*np.clip((ratio-1.25)/.75, 0, 1)


class HourlyAmount:
    def __init__(self, name):
        if name not in CANDIDATES: raise ValueError('Unknown fixed hourly amount model')
        self.name = name; self.fitted = False

    def fit(self, side, features, y, created, anchor):
        valid = check_alignment(side, features, anchor)
        if (not side.index.equals(y.index) or not side.index.equals(created.index) or not np.isfinite(y).all()
                or y.lt(0).any() or created.isna().any() or created.dt.tz is None):
            raise ValueError('Invalid amount labels/timestamps')
        self.parameters = {}; self.fitted = False
        for s in (0, 1):
            use = side.eq(s) & valid & y.gt(0) & anchor.gt(0)
            n = int(use.sum()); days = int(created.loc[use].dt.floor('D').nunique())
            par = dict(supported=False, reason='insufficient_positive_support', positive_orders=n, positive_days=days,
                history_orders=int(side.eq(s).sum()), missing_market_orders=int((side.eq(s) & ~valid).sum()))
            if n >= 40 and days >= 5:
                h = features.loc[use]; s0 = anchor.loc[use].to_numpy(float); target = y.loc[use].to_numpy(float)
                if self.name == 'amount_gate':
                    q75, q95 = features.loc[side.eq(s) & valid, 'level'].quantile([.75, .95]).to_numpy()
                    par.update(q75=float(q75), q95=float(q95))
                    if q95-q75 < 1e-6:
                        par['reason'] = 'degenerate_level_quantiles'; self.parameters[str(s)] = par; continue
                    g = gate(h, q75, q95); active = g > 0
                    adays = int(created.loc[use].loc[active].dt.floor('D').nunique())
                    par.update(active_positive_orders=int(active.sum()), active_positive_days=adays)
                    if active.sum() < 15 or adays < 3:
                        par['reason'] = 'insufficient_active_support'; self.parameters[str(s)] = par; continue
                    phi = g[:, None]
                else:
                    columns = ['level'] if self.name == 'amount_level' else ['level', 'speed']
                    raw = h[columns].to_numpy(); weights = s0*s0
                    mean = np.average(raw, axis=0, weights=weights)
                    scale = np.sqrt(np.average((raw-mean)**2, axis=0, weights=weights)); scale[scale < 1e-8] = 1.
                    z = np.clip((raw-mean)/scale, -3, 3)
                    center = np.average(z, axis=0, weights=weights); phi = z-center
                    par.update(columns=columns, mean=mean.tolist(), scale=scale.tolist(), center=center.tolist())
                normalization = float(np.sqrt(np.mean(s0*s0)))
                design = phi*(s0/normalization)[:, None]
                rhs = design.T@((target-s0)/normalization)
                try:
                    coef = np.linalg.solve(design.T@design+STRENGTH*np.eye(phi.shape[1]), rhs)
                    if self.name == 'amount_gate': coef = np.maximum(coef, 0.)
                    if not np.isfinite(coef).all(): raise FloatingPointError('Nonfinite amount coefficient')
                    par.update(supported=True, reason='none', beta=coef.tolist(), fit_scale=normalization)
                except (np.linalg.LinAlgError, FloatingPointError):
                    par['reason'] = 'numerical_failure'
            self.parameters[str(s)] = par
        self.fitted = True
        return self

    def predict(self, side, features, anchor):
        if not self.fitted: raise ValueError('Hourly amount model is not fitted')
        valid = check_alignment(side, features, anchor)
        output = anchor.to_numpy(float).copy(); supported = np.zeros(len(side), bool)
        for s in (0, 1):
            par = self.parameters[str(s)]; use = side.eq(s) & valid & anchor.gt(0)
            if not par['supported'] or not use.any(): continue
            f = features.loc[use]
            if self.name == 'amount_gate':
                phi = gate(f, par['q75'], par['q95'])[:, None]
            else:
                phi = np.clip((f[par['columns']].to_numpy()-par['mean'])/par['scale'], -3, 3)-par['center']
            delta = np.clip(phi@par['beta'], 0. if self.name == 'amount_gate' else -.1, .1)
            output[use] *= 1+delta; supported[use] = True
        if not np.isfinite(output).all() or (output < 0).any(): raise ValueError('Invalid hourly severity')
        return output, supported

    def state(self):
        if not self.fitted: raise ValueError('Hourly amount not fitted')
        return copy.deepcopy(self.parameters)
