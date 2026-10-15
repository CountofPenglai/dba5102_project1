"""Independent arithmetic replay helpers; no estimator fitting or legacy imports."""
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE

def metrics(p):
    e=p.prediction.to_numpy()-p.actual.to_numpy();q=p.quantity.to_numpy()
    y=p.actual.to_numpy();pred=p.prediction.to_numpy()
    return dict(orders=len(p),rmse_per_btc=np.sqrt(np.mean(e**2)),mae_per_btc=np.mean(abs(e)),
        actual_total_usdc=float(y@q),predicted_total_usdc=float(pred@q),total_bias_usdc=float(e@q),
        gross_underprediction_usdc=float(np.maximum(-e,0)@q),gross_overprediction_usdc=float(np.maximum(e,0)@q),
        predicted_actual_ratio=float(pred@q)/float(y@q) if float(y@q) else np.nan)


def h0_values(x, state):
    probability = np.empty(len(x)); severity = np.empty(len(x))
    for side in (0, 1):
        mask = x.option_type.eq(side); xx = x.loc[mask]; p = state[str(side)]
        raw = np.c_[np.log(xx[BASE_FEATURE]), np.log(xx.remaining_seconds),
            np.clip((1-2*side)*xx.log_moneyness / np.maximum(xx.realized_vol_60m*np.sqrt(xx.remaining_seconds/3600), 1e-6), -5, 5)]
        probability[mask] = 1/(1+np.exp(-(np.clip((raw-p['mean'])/p['scale'], -3, 3)@p['coef']+p['intercept']))) if 'coef' in p else p['probability']
        severity[mask] = xx[BASE_FEATURE]*p['severity']
    return probability, severity


def quadratic_check(reference, fixed, macro, pooled):
    daily = pd.read_csv(reference/'metrics/validation/daily.csv', dtype={'value': str})
    daily = daily.loc[daily.group.eq('direction') & daily.model.isin(['official_raw', 'h0', 'half_blend'])].copy()
    daily['sse'] = daily.orders*daily.rmse_per_btc**2
    piv = daily.pivot(index=['day', 'month', 'value'], columns='model', values='sse')
    n = daily.loc[daily.model.eq('official_raw')].set_index(['day', 'month', 'value']).orders.reindex(piv.index)
    d2 = 2*(piv.h0+piv.official_raw)-4*piv.half_blend
    assert (d2 >= -1e-5).all()
    checks = 0
    for name, (c, p) in fixed.items():
        a = np.where(piv.index.get_level_values('value') == '0', c, p)
        sse = (1-a)*piv.official_raw+a*piv.h0-a*(1-a)*d2
        rates = sse.groupby(level='month').sum()/n.groupby(level='month').sum()
        expected_macro = np.sqrt(rates.mean()); expected_pooled = np.sqrt(sse.sum()/n.sum())
        np.testing.assert_allclose(macro.loc[name, 'macro_rmse_per_btc'], expected_macro, rtol=1e-10, atol=1e-7)
        np.testing.assert_allclose(pooled.loc[name, 'rmse_per_btc'], expected_pooled, rtol=1e-10, atol=1e-7)
        checks += 1
    return checks
