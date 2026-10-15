"""Fixed, small payout calibration probes; no external signals or margin."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pp_data.features import BASE_FEATURE
from .baselines import RollingMean
from .terminal_payout import TerminalPayoutModel, INPUTS as TERMINAL_INPUTS

MODEL_NAMES=['official_raw','grouped_mean','direction_scaled','terminal_static',
             'hurdle_base','direction_shrunk','day_balanced_shrunk']
MODEL_INPUTS=['option_type',BASE_FEATURE,'remaining_seconds','product_duration_seconds',
              'realized_vol_60m','log_moneyness','terminal_supported','terminal_static_proxy','terminal_dynamic_proxy']


def hurdle_design(X):
    uncertainty=np.maximum(X.realized_vol_60m.to_numpy()*np.sqrt(X.remaining_seconds.to_numpy()/3600),1e-6)
    aligned=(1-2*X.option_type.to_numpy())*X.log_moneyness.to_numpy()/uncertainty
    return np.column_stack([np.log(X[BASE_FEATURE]),np.log(X.remaining_seconds),np.clip(aligned,-5,5)])


class CalibrationModel:
    def __init__(self,name):
        if name not in MODEL_NAMES:raise ValueError('Unknown rolling calibration')
        self.name=name;self.fitted=False

    def _check(self,X):
        if (not isinstance(X,pd.DataFrame) or X.empty or not X.index.is_unique or list(X)!=MODEL_INPUTS
            or not np.isfinite(X.to_numpy(float)).all() or not X.option_type.isin([0,1]).all()
            or X[[BASE_FEATURE,'remaining_seconds','product_duration_seconds']].le(0).any().any()
            or X.realized_vol_60m.lt(0).any()):raise ValueError('Invalid pricing-time calibration inputs')

    def fit(self,X,y,created=None):
        self._check(X)
        if not isinstance(y,pd.Series) or not X.index.equals(y.index) or not np.isfinite(y).all() or y.lt(0).any():
            raise ValueError('Invalid payout target alignment')
        if self.name=='day_balanced_shrunk' and (created is None or not created.index.equals(X.index)
                or created.isna().any() or created.dt.tz is None):raise ValueError('Calendar weighting needs aligned creation times')
        self.parameters={};self.detail={};self.fitted=False
        if self.name=='grouped_mean':self.reference=RollingMean(True,30).fit(X,y)
        if self.name=='terminal_static':self.reference=TerminalPayoutModel('static',.01).fit(X[TERMINAL_INPUTS],y)
        for side,label in [(0,'call'),(1,'put')]:
            mask=X.option_type.eq(side);b=X.loc[mask,BASE_FEATURE].to_numpy(float);target=y.loc[mask].to_numpy(float)
            n=len(target);positive=target>0
            self.detail[label+'_training_orders']=n;self.detail[label+'_positive_orders']=int(positive.sum())
            parameter={'c':1.,'probability':1.,'severity':1.}
            if n:
                weights=np.ones(n)
                if self.name=='day_balanced_shrunk':
                    days=created.loc[mask].dt.floor('D');weights=1/days.map(days.value_counts()).to_numpy(float)
                denominator=float(weights@(b*b));c=float(weights@(b*target))/denominator
                if self.name in ['direction_shrunk','day_balanced_shrunk']:c=(c+1)/2
                parameter['c']=c
                if self.name=='hurdle_base':
                    parameter['probability']=float(positive.mean())
                    parameter['severity']=float(b[positive]@target[positive]/(b[positive]@b[positive])) if positive.any() else 0.
                    if 0<positive.sum()<n:
                        design=hurdle_design(X.loc[mask]);scaler=StandardScaler().fit(design)
                        model=LogisticRegression(C=.1,max_iter=1000,tol=1e-9,solver='lbfgs')
                        model.fit(np.clip(scaler.transform(design),-3,3),positive.astype(int))
                        parameter.update(scaler=scaler,model=model)
            self.parameters[side]=parameter
            self.detail[label+'_coefficient']=parameter['c']
            self.detail[label+'_positive_severity_coefficient']=parameter['severity'] if self.name=='hurdle_base' else np.nan
        self.fitted=True;return self

    def predict_components(self,X):
        self._check(X)
        if not self.fitted or self.name!='hurdle_base':raise ValueError('Hurdle model not fitted')
        probabilities=np.empty(len(X));severity=np.empty(len(X))
        for side in [0,1]:
            mask=X.option_type.eq(side).to_numpy();parameter=self.parameters[side]
            if not mask.any():continue
            probabilities[mask]=parameter['probability']
            if 'model' in parameter:
                z=np.clip(parameter['scaler'].transform(hurdle_design(X.loc[mask])),-3,3)
                probabilities[mask]=parameter['model'].predict_proba(z)[:,1]
            severity[mask]=X.loc[mask,BASE_FEATURE]*parameter['severity']
        return probabilities,severity

    def predict_raw(self,X):
        self._check(X)
        if not self.fitted:raise ValueError('Calibration not fitted')
        if self.name=='official_raw':result=X[BASE_FEATURE].to_numpy(float).copy()
        elif self.name=='grouped_mean':result=self.reference.predict_raw(X)
        elif self.name=='terminal_static':result=self.reference.predict_raw(X[TERMINAL_INPUTS])
        elif self.name=='hurdle_base':
            p,s=self.predict_components(X);result=p*s
        else:result=X[BASE_FEATURE].to_numpy()*X.option_type.map({s:p['c'] for s,p in self.parameters.items()}).to_numpy()
        if not np.isfinite(result).all() or (result<0).any():raise ValueError('Invalid calibrated payout')
        return result

    def diagnostics(self):
        detail=self.detail.copy()
        if self.name=='terminal_static':detail.update(self.reference.diagnostics())
        return detail
