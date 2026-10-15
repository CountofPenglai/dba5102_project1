"""Official Base times a learned, nonnegative Put response amplitude."""
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE

INPUTS = ['option_type', BASE_FEATURE, 'fast_response_state']


class FastResponseModel:
    def __init__(self, lam):
        if isinstance(lam, bool) or lam not in (.1, 1., 10.):
            raise ValueError('Unsupported fast response penalty')
        self.lam, self.fitted = float(lam), False

    def _validate_X(self, X):
        if (not isinstance(X, pd.DataFrame) or X.empty or not X.index.is_unique or list(X) != INPUTS
                or not np.isfinite(X.to_numpy(dtype=float)).all()
                or not X.option_type.isin([0, 1]).all() or X[BASE_FEATURE].le(0).any()
                or not X.fast_response_state.between(0, 1).all()):
            raise ValueError('Invalid fast response inputs')

    def fit(self, X, y):
        self._validate_X(X)
        self.fitted = False
        if not isinstance(y, pd.Series) or not y.index.equals(X.index) or not np.isfinite(y).all() or y.lt(0).any():
            raise ValueError('Invalid aligned payout targets')
        put = X.option_type.eq(1)
        target = y.loc[put].to_numpy(dtype=float)
        base = X.loc[put, BASE_FEATURE].to_numpy(dtype=float)
        exposure = base*X.loc[put, 'fast_response_state'].to_numpy(dtype=float)
        denominator = float(exposure@exposure + self.lam*(target@target))
        self.beta = max(0., float(exposure@(target-base))/denominator) if denominator > 0 else 0.
        if not np.isfinite(self.beta):
            raise ValueError('Nonfinite response amplitude')
        self.put_training_rows = int(put.sum())
        self.fitted = True
        return self

    def predict_raw(self, X):
        self._validate_X(X)
        if not self.fitted:
            raise ValueError('Fast response model is not fitted')
        factor = 1+self.beta*X.fast_response_state.to_numpy()*X.option_type.eq(1).to_numpy()
        prediction = X[BASE_FEATURE].to_numpy()*factor
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite fast response prediction')
        return prediction

    def diagnostics(self):
        return {'response_beta': self.beta, 'put_training_rows': self.put_training_rows,
                'response_zero_amplitude': self.beta == 0.}
