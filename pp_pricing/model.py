"""Inference only: no training, calendar overlay, label access, or parameter search."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pp_audit.market import audit_market
from pp_data.features import BASE_FEATURE, build_features
from pp_models.pricing_inputs import MODEL_INPUTS, check_pricing_inputs, hurdle_design

DEFAULT_MODEL = Path(__file__).with_name('model.json')
ORDER_INPUTS = ['createdAt', 'expirationDate', 'optionType', 'productType',
                'quantity_BTC', 'strikePrice', 'apiBasePrice_per_BTC_USDC']


def with_order_index(frame):
    """Preserve optional order IDs consistently in the Python and CSV interfaces."""
    if 'order_id' in frame:
        frame = frame.set_index('order_id')
    if frame.index.name == 'order_id':
        ids = pd.Series(frame.index, dtype='string')
        if ids.isna().any() or ids.str.strip().eq('').any() or not ids.is_unique:
            raise ValueError('order_id must be nonempty and unique for every row')
        frame = frame.copy()
        frame.index = pd.Index(ids.astype(str), name='order_id')
    return frame


class FinalPricingModel:
    """Load numeric H0 parameters and apply fixed Call/Put blending weights.

    No labels are needed after loading. Parameters remain frozen until the caller
    explicitly loads a different artifact; this does not simulate weekly feedback.
    """

    def __init__(self, artifact):
        # Keep the loaded numerical state independent of the caller's input dict.
        self.artifact = copy.deepcopy(artifact)
        try:
            # This interface serves the selected 40% Call / 15% Put H0 blend only.
            # Reject incompatible artifacts instead of silently changing the recipe.
            if (artifact['schema_version'] != 'h0_base_blend_v1'
                    or artifact['model'] != 'blend_c040_p015'
                    or artifact['input_columns'] != MODEL_INPUTS
                    or artifact['alpha'] != {'0': .4, '1': .15}
                    or set(artifact['parameters']) != {'0', '1'}):
                raise ValueError('Unsupported final-model schema, inputs or weights')
            self.fitted_at = pd.Timestamp(artifact['fitted_at'])
            if pd.isna(self.fitted_at) or self.fitted_at.tzinfo is None:
                raise ValueError('Model fitted_at must be timezone-aware')
            self.parameters = self.artifact['parameters']
            for p in self.parameters.values():
                if (not np.isfinite([p['probability'], p['severity']]).all()
                        or not 0 <= p['probability'] <= 1 or p['severity'] < 0):
                    raise ValueError('Invalid probability/severity parameters')
                logistic = {'mean', 'scale', 'coef', 'intercept'}
                if logistic.intersection(p):
                    if not logistic.issubset(p):
                        raise ValueError('Incomplete logistic parameters')
                    for key in ['mean', 'scale', 'coef']:
                        if np.shape(p[key]) != (3,) or not np.isfinite(p[key]).all():
                            raise ValueError('Expected three finite logistic parameters')
                    if not np.isfinite(p['intercept']) or (np.asarray(p['scale']) <= 0).any():
                        raise ValueError('Invalid logistic intercept/scales')
        except (KeyError, TypeError) as error:
            raise ValueError('Invalid final-model artifact') from error

    @classmethod
    def load(cls, path=DEFAULT_MODEL):
        """Load a portable JSON artifact; no executable pickle or private data."""
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def predict(self, X, *, created_at):
        """Predict a feature frame in input order, in USDC/BTC plus a multiplier.

        `created_at` must be a Series aligned exactly to X. Naive input dates are
        interpreted as UTC, matching the official order-file convention.
        """
        check_pricing_inputs(X)
        if not isinstance(created_at, pd.Series) or not created_at.index.equals(X.index):
            raise ValueError('Quote timestamps must align exactly with feature rows')
        created = pd.to_datetime(created_at, utc=True, format='mixed', errors='coerce')
        # The final frozen state must not be applied retrospectively. Historical
        # evaluation replays the appropriate weekly state through a separate path.
        if created.isna().any() or created.lt(self.fitted_at).any():
            raise ValueError('Quote timestamps must be valid and not precede model fitted_at')

        probability = np.empty(len(X)); coefficient = np.empty(len(X))
        raw = hurdle_design(X)
        for side in (0, 1):
            mask = X.option_type.eq(side).to_numpy()
            p = self.parameters[str(side)]
            # Preserve the training fallback when a direction has no logistic fit.
            probability[mask] = p['probability']
            if 'coef' in p:
                # Replay the exact training transform without importing scikit-learn.
                z = np.clip((raw[mask] - p['mean']) / p['scale'], -3, 3)
                logits = z @ p['coef'] + p['intercept']
                # Stable sigmoid: avoids exp overflow for large negative logits.
                probability[mask] = np.exp(-np.logaddexp(0., -logits))
            coefficient[mask] = p['severity']

        base = X[BASE_FEATURE].to_numpy(float)
        severity = base * coefficient
        h0 = probability * severity
        alpha = np.where(X.option_type.eq(0), .4, .15)
        # Shrink H0's adjustment toward Base, rather than adding another markup.
        # Equivalently, multiplier = 1 + alpha * (probability * coefficient - 1).
        prediction = alpha*h0 + (1-alpha)*base
        if not np.isfinite(prediction).all() or (prediction < 0).any():
            raise ValueError('Nonfinite or negative model output')
        return pd.DataFrame({
            'created_at_utc': created, 'base_per_btc_usdc': base,
            'positive_payout_probability': probability,
            'conditional_payout_per_btc_usdc': severity, 'h0_per_btc_usdc': h0,
            'alpha_h0': alpha, 'prediction_per_btc_usdc': prediction,
            'pricing_multiplier': prediction/base,
            'model_fitted_at_utc': self.fitted_at.isoformat(),
        }, index=X.index)

    def predict_orders(self, orders, market):
        """Adapt official order columns and BTCUSDC minute candles to the model.

        Select only creation-time fields before feature construction. Payout,
        closedAt, paid premium, and markup are neither required nor read.
        Missing Base or an unavailable contiguous 60-minute window is an error.
        """
        missing = sorted(set(ORDER_INPUTS)-set(orders.columns))
        if missing:
            raise ValueError('Missing official order inputs: '+', '.join(missing))
        quotes = with_order_index(orders).loc[:, ORDER_INPUTS].copy()
        # Official exports can mix SQL-style, ISO and fractional-second dates.
        # Normalize before the shared builder's column-wise timestamp conversion.
        for name in ['createdAt', 'expirationDate']:
            quotes[name] = pd.to_datetime(quotes[name], format='mixed', utc=True, errors='coerce')
            if quotes[name].isna().any():
                raise ValueError('Invalid official order timestamp: '+name)
        X, _ = build_features(quotes, audit_market(market).usable)
        # Feature construction matches completed candles backward from each quote.
        # Outputs stay per BTC; quantity does not multiply the model's prediction.
        return self.predict(X[MODEL_INPUTS], created_at=quotes.createdAt)
