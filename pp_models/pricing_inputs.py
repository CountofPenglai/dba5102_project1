"""Shared H0 input contract and transform; no fitting or research-run imports."""
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE

MODEL_INPUTS = ['option_type', BASE_FEATURE, 'remaining_seconds',
                'product_duration_seconds', 'realized_vol_60m', 'log_moneyness']
# Six fields form the input contract, but the logistic model has only three
# transformed features. Nominal product duration is validated, not regressed on.


def check_pricing_inputs(X):
    """Reject missing/invalid quotes instead of silently dropping or imputing them."""
    if (not isinstance(X, pd.DataFrame) or X.empty or not X.index.is_unique
            or list(X) != MODEL_INPUTS):
        raise ValueError('H0 requires a nonempty, uniquely indexed frame with exactly MODEL_INPUTS')
    if (not np.isfinite(X.to_numpy(float)).all() or not X.option_type.isin([0, 1]).all()
            or X[[BASE_FEATURE, 'remaining_seconds', 'product_duration_seconds']].le(0).any().any()
            or X.realized_vol_60m.lt(0).any()):
        raise ValueError('Invalid pricing-time H0 inputs: positive Base/durations and finite market features required')


def hurdle_design(X):
    """Log Base, log remaining time, and direction-aligned volatility-scaled moneyness.

    Volatility is sqrt(sum of squared minute log returns), not annualized.
    The floor and clipping are the original reported H0 safeguards.
    """
    # Scale observed one-hour variation to remaining coverage using sqrt(time).
    # This is a feature normalization, not a forecast of terminal price variance.
    uncertainty = np.maximum(X.realized_vol_60m.to_numpy()*np.sqrt(X.remaining_seconds.to_numpy()/3600), 1e-6)
    # log_moneyness = log(spot/strike). Flipping the Put sign makes positive values
    # mean in-the-money for either direction; the floor handles quiet markets.
    aligned = (1-2*X.option_type.to_numpy())*X.log_moneyness.to_numpy()/uncertainty
    # Clip normalized moneyness here; standardized features are clipped again in H0.
    return np.column_stack([np.log(X[BASE_FEATURE]), np.log(X.remaining_seconds), np.clip(aligned, -5, 5)])
