"""Small nonnegative payout calibration around official Base, with proxy shrinkage."""
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from pp_data.features import BASE_FEATURE

INPUTS = ['option_type',BASE_FEATURE,'terminal_supported','terminal_static_proxy','terminal_dynamic_proxy']
NAMES = {'raw':'official_raw','scaled':'official_put_scaled',
         'static':'official_terminal_static','dynamic':'official_terminal_dynamic'}


class TerminalPayoutModel:
    def __init__(self, mode, lam=None):
        if (mode not in NAMES or (mode in ('raw','scaled') and lam is not None)
                or (mode in ('static','dynamic') and (isinstance(lam,bool) or lam not in (.01,.1,1.)))):
            raise ValueError('Invalid terminal payout mode or penalty')
        self.mode,self.lam,self.fitted = mode,lam,False

    def _validate_X(self,X):
        if (not isinstance(X,pd.DataFrame) or X.empty or not X.index.is_unique or list(X)!=INPUTS
                or not np.isfinite(X.to_numpy(float)).all() or not X.option_type.isin([0,1]).all()
                or not X.terminal_supported.isin([0,1]).all()
                or ((X.terminal_supported==1)&(X.option_type!=1)).any()
                or X[BASE_FEATURE].le(0).any()
                or X[['terminal_static_proxy','terminal_dynamic_proxy']].lt(0).any().any()):
            raise ValueError('Invalid terminal pricing inputs')

    def fit(self,X,y):
        self._validate_X(X)
        if not isinstance(y,pd.Series) or not X.index.equals(y.index) or not np.isfinite(y).all() or y.lt(0).any():
            raise ValueError('Invalid terminal payout targets')
        self.fitted=False
        active=X.terminal_supported.eq(1)
        self.training_rows=int(active.sum());self.a,self.b=1.,0.
        if self.mode!='raw' and active.any():
            base=X.loc[active,BASE_FEATURE].to_numpy(float);target=y.loc[active].to_numpy(float)
            self.a=float(base@target/(base@base))
            normalizer=np.sqrt(target@target)
            if self.mode in ('static','dynamic') and normalizer>0:
                proxy=X.loc[active,f'terminal_{self.mode}_proxy'].to_numpy(float)
                design=np.vstack([np.column_stack([base,proxy])/normalizer,[0.,np.sqrt(self.lam)]])
                self.a,self.b=map(float,nnls(design,np.r_[target/normalizer,0.],maxiter=100)[0])
        self.fitted=True
        self.put_mean,self.put_max=1.,1.
        return self

    def predict_raw(self,X):
        self._validate_X(X)
        if not self.fitted:raise ValueError('Terminal payout model is not fitted')
        prediction=X[BASE_FEATURE].to_numpy(float).copy()
        active=X.terminal_supported.eq(1).to_numpy()
        if self.mode!='raw':
            prediction[active]*=self.a
            if self.mode in ('static','dynamic'):
                prediction[active]+=self.b*X.loc[active,f'terminal_{self.mode}_proxy'].to_numpy(float)
        if not np.isfinite(prediction).all() or (prediction<0).any():
            raise ValueError('Invalid terminal payout prediction')
        put=X.option_type.eq(1).to_numpy()
        if put.any():
            mult=prediction[put]/X.loc[put,BASE_FEATURE].to_numpy(float)
            self.put_mean,self.put_max=float(mult.mean()),float(mult.max())
        return prediction

    def diagnostics(self):
        return {'terminal_base_coefficient':self.a,'terminal_proxy_coefficient':self.b,
                'terminal_training_put_rows':self.training_rows,'terminal_put_multiplier_mean':self.put_mean,
                'terminal_put_multiplier_max':self.put_max}


def build_terminal_model(name,resolved):
    mode=resolved['distribution_mode']
    if (mode not in NAMES or name!=NAMES[mode] or resolved['selected_models']!=[name]
            or resolved['features']!=INPUTS or resolved['cohort']!='benchmark' or resolved['use_base'] is not True):
        raise ValueError('Invalid terminal model factory configuration')
    return TerminalPayoutModel(mode,resolved['distribution_lambda'])
