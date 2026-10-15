"""Descriptive residual diagnostics; no model selection or correction fitting."""
import numpy as np
import pandas as pd
from .two_stage_study import metric_values


def error_metrics(part):
    if part.empty:
        return {'orders': 0, 'creation_days': 0, 'mean_error_per_btc': np.nan}
    return {**metric_values(part), 'creation_days': int(part.day.nunique()),
            'mean_error_per_btc': float((part.prediction-part.actual).mean()),
            'underprediction_fraction': float(part.prediction.lt(part.actual).mean())}


def known_error_signals(frame, metadata):
    """Previous seven creation days, closed strictly before current UTC day."""
    outputs = []
    for _, model in frame.groupby('model', sort=False):
        if not model.index.is_unique:
            raise ValueError('Residual history has duplicate scoring order IDs')
        model = model.copy()
        model['history_orders'] = 0
        model['history_creation_days'] = 0
        model['history_mean_error'] = np.nan
        model['history_error_state'] = 'insufficient_history'
        closed = metadata.loc[model.index, 'closed_at_utc']
        for (day, side), current in model.groupby(['day', 'side']):
            history = model.loc[model.day.ge(day-pd.Timedelta(days=7)) & model.day.lt(day)
                                & closed.lt(day) & model.side.eq(side)]
            ids = current.index
            model.loc[ids, 'history_orders'] = len(history)
            model.loc[ids, 'history_creation_days'] = history.day.nunique()
            if len(history) >= 20 and history.day.nunique() >= 3:
                error = float((history.prediction-history.actual).mean())
                model.loc[ids, 'history_mean_error'] = error
                model.loc[ids, 'history_error_state'] = ('prior_underprediction' if error < 0 else
                                                         'prior_overprediction' if error > 0 else 'prior_zero_error')
        outputs.append(model)
    return pd.concat(outputs)


def error_tables(frame, components, metadata, raw_model):
    names = list(dict.fromkeys(['official_raw', 'h0', 'grouped_mean', raw_model, 'final_combination']))
    selected = frame.loc[frame.model.isin(names)].copy()
    features = ['acceleration_group', 'absolute_risk_group', 'position_group', 'remaining_bucket']
    for column in features:
        selected[column] = components.loc[selected.index, column].to_numpy()
    selected['observed_payout'] = np.where(selected.actual.gt(0), 'positive', 'zero')
    selected['directional_state'] = selected.acceleration_group + '_' + selected.absolute_risk_group
    selected = known_error_signals(selected, metadata)
    slices = features + ['directional_state', 'tenor', 'tail_group', 'observed_payout', 'history_error_state']
    groups, lag_rows, daily_rows = [], [], []
    for name, model in selected.groupby('model'):
        for month in ['all', '2026-04', '2026-05', '2026-06']:
            period = model if month == 'all' else model.loc[model.month.eq(month)]
            for direction in ['all', 0, 1]:
                part = period if direction == 'all' else period.loc[period.side.eq(direction)]
                groups.append({'model': name, 'month': month, 'direction': direction,
                               'feature': 'overall', 'value': 'all', **error_metrics(part)})
                for feature in slices:
                    for value in sorted(model[feature].unique(), key=str):
                        sub = part.loc[part[feature].eq(value)]
                        groups.append({'model': name, 'month': month, 'direction': direction,
                                       'feature': feature, 'value': str(value), **error_metrics(sub)})
                supported = part.loc[part.history_mean_error.notna()].copy()
                supported['error'] = supported.prediction-supported.actual
                by_day = supported.groupby(['day', 'side']).agg(history=('history_mean_error', 'first'),
                                                               error=('error', 'mean'), orders=('error', 'size'))
                corr = (float(by_day.history.corr(by_day.error)) if len(by_day) >= 4
                        and by_day.history.std() > 0 and by_day.error.std() > 0 else np.nan)
                lag_rows.append({'model': name, 'month': month, 'direction': direction,
                                 'supported_orders': len(supported), 'direction_days': len(by_day),
                                 'daily_error_correlation': corr,
                                 'same_error_sign_fraction': float((np.sign(by_day.history) == np.sign(by_day.error)).mean())
                                                             if len(by_day) else np.nan})
        for (day, side), part in model.groupby(['day', 'side']):
            daily_rows.append({'model': name, 'day': day.isoformat(), 'direction': side, **error_metrics(part)})
    amount_rows = []
    for month in ['all', '2026-04', '2026-05', '2026-06']:
        period = components if month == 'all' else components.loc[components.month.eq(month)]
        for side in ['all', 0, 1]:
            part = period if side == 'all' else period.loc[period.side.eq(side)]
            positive = part.loc[part.actual.gt(0)]
            for name, col in dict([('h0', 's0'), (raw_model, 's0' if raw_model == 'h0' else f's_{raw_model}')]).items():
                amount_rows.append({'model': name, 'month': month, 'direction': side, 'orders': len(part),
                                    'positive_orders': len(positive),
                                    'mean_probability': part.p_logistic.mean(), 'positive_rate': part.actual.gt(0).mean(),
                                    'positive_amount_mean_error': (positive[col]-positive.actual).mean(),
                                    'positive_amount_rmse': np.sqrt(np.mean((positive[col]-positive.actual)**2))})
    return {'error_groups': pd.DataFrame(groups), 'historical_error_relationship': pd.DataFrame(lag_rows),
            'daily_direction_errors': pd.DataFrame(daily_rows), 'component_errors': pd.DataFrame(amount_rows)}
