"""Prequential fitting and compact metrics; row predictions stay in memory."""
import numpy as np
import pandas as pd
from time import perf_counter
import warnings
from threadpoolctl import threadpool_limits
from .data import daily_batches, period_bounds
from .estimators import build_model


def key_metrics(actual, prediction, quantity):
    actual, prediction, quantity = [np.asarray(v, dtype=float) for v in (actual, prediction, quantity)]
    if (actual.ndim != 1 or not len(actual) or actual.shape != prediction.shape
            or actual.shape != quantity.shape or not all(np.isfinite(v).all() for v in (actual, prediction, quantity))
            or (actual < 0).any() or (prediction < 0).any() or (quantity <= 0).any()):
        raise ValueError("Invalid arrays for payout metrics")
    error = prediction - actual
    observed, predicted = float(np.dot(actual, quantity)), float(np.dot(prediction, quantity))
    return {"orders": len(actual), "rmse_per_btc": float(np.sqrt(np.mean(error**2))),
            "mae_per_btc": float(np.mean(np.abs(error))), "actual_total_usdc": observed,
            "predicted_total_usdc": predicted, "total_bias_usdc": predicted - observed,
            "gross_overprediction_usdc": float(np.dot(np.maximum(error, 0), quantity)),
            "gross_underprediction_usdc": float(np.dot(np.maximum(-error, 0), quantity)),
            "predicted_actual_ratio": predicted / observed if observed else None}


def _scopes(frame, cohort):
    yield cohort, frame
    if cohort == "risk":
        common = frame.loc[frame.benchmark_eligible]
        if not common.empty:
            yield "benchmark", common


def _metrics(frame):
    result = key_metrics(frame.actual, frame.prediction, frame.quantity) | {
        "clipped_predictions": int((frame.raw_prediction < 0).sum())}
    if 'fast_response_state' in frame:
        result.update(response_state_mean=float(frame.fast_response_state.mean()),
                      multiplier_mean=float(frame.multiplier.mean()),
                      multiplier_max=float(frame.multiplier.max()))
    if 'response_at_cap' in frame:
        result['multiplier_at_cap_orders'] = int(frame.response_at_cap.sum())
    return result


def evaluation_batches(pool, split, resolved, period):
    batches = list(daily_batches(pool, split, resolved['window_days'], period))
    if not resolved.get('sequential_start'):
        yield from batches
        return
    start = pd.Timestamp(resolved['sequential_start'])
    outer_start, _ = period_bounds(split)
    if (split != 'validation' or start.tzinfo is None or start != start.floor('D')
            or not outer_start <= start <= batches[0][0]):
        raise ValueError('Invalid sequential initialization date')
    scoring = {day: ids for day, _, ids in batches}
    created, closed = pool.metadata.created_at_utc, pool.metadata.closed_at_utc
    for day in pd.date_range(start, batches[-1][0], freq='D'):
        history = created.ge(day-pd.Timedelta(days=resolved['window_days'])) & created.lt(day) & closed.lt(day)
        yield day, pool.X.index[history], scoring.get(day, pool.X.index[:0])


def evaluate(pool, split, resolved, model_factory=build_model, period=None):
    daily_rows, all_rows = [], []
    features = resolved["features"]
    persistent, unscored = {}, {}
    # The factory receives only configuration; fit receives only the day's legal
    # historical rows. Test outcomes are used below solely for scoring.
    for day, train_ids, predict_ids in evaluation_batches(pool, split, resolved, period):
        if len(train_ids) < resolved["min_train_rows"]:
            raise ValueError(f"Insufficient history at {day}: {len(train_ids)} rows")
        X_train, y_train = pool.X.loc[train_ids, features], pool.y.loc[train_ids]
        X_today = pool.X.loc[predict_ids, features]
        threshold = float(pool.X.loc[train_ids, "realized_vol_60m"].quantile(.8))
        details = {"day_utc": day.isoformat(), "training_rows": len(train_ids),
                   "training_positive_rows": int(y_train.gt(0).sum()),
                   "constant_target": bool(y_train.nunique() == 1),
                   "constant_features": "|".join(X_train.columns[X_train.nunique() == 1]),
                   "volatility_q80": threshold}
        if resolved.get('guarded_response_metrics'):
            from .guarded_response import fit_risk_context, calm_mask
            context = fit_risk_context(pool.X.loc[train_ids], y_train)
            details.update(context)
        if resolved.get('activity_metrics'):
            activity = pool.X.platform_call_count_240m + pool.X.platform_put_count_240m
            activity_threshold = float(activity.loc[train_ids].quantile(.8))
            details['platform_activity_q80'] = activity_threshold
        for name in resolved["selected_models"]:
            if resolved.get('sequential_start'):
                if name not in persistent:
                    persistent[name] = model_factory(name, resolved)
                model = persistent[name]
            else:
                model = model_factory(name, resolved)
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                started = perf_counter()
                model.fit(X_train, y_train)
                fit_seconds = perf_counter() - started
                started = perf_counter()
                raw = np.asarray(model.predict_raw(X_today), dtype=float) if len(predict_ids) else None
                predict_seconds = perf_counter() - started
            if not len(predict_ids):
                totals = unscored.setdefault(name, {'unscored_fit_days': 0, 'unscored_fit_seconds': 0., 'unscored_warning_count': 0})
                totals['unscored_fit_days'] += 1
                totals['unscored_fit_seconds'] += fit_seconds
                totals['unscored_warning_count'] += len(caught)
                continue
            runtime = {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                       "warning_count": len(caught),
                       "warning_categories": "|".join(sorted({w.category.__name__ for w in caught}))}
            if resolved.get('sequential_start'):
                runtime.update(unscored.get(name, {'unscored_fit_days': 0, 'unscored_fit_seconds': 0., 'unscored_warning_count': 0}))
            if resolved.get("daily_direction_metrics") and hasattr(model, "diagnostics"):
                runtime.update(model.diagnostics())
            if raw.shape != (len(predict_ids),) or not np.isfinite(raw).all():
                raise ValueError(f"Invalid predictions from {name} on {day}")
            rows = pd.DataFrame({"model": name, "actual": pool.y.loc[predict_ids],
                                 "raw_prediction": raw, "prediction": np.maximum(raw, 0),
                                 "quantity": pool.X.loc[predict_ids, "quantity_btc"],
                                 "option_type": pool.X.loc[predict_ids, "option_type"],
                                 "product_duration_seconds": pool.X.loc[predict_ids, "product_duration_seconds"],
                                 "volatility_group": np.where(pool.X.loc[predict_ids, "realized_vol_60m"] >= threshold,
                                                              "high", "normal"),
                                 "benchmark_eligible": pool.metadata.loc[predict_ids, "benchmark_eligible"]})
            if resolved.get('activity_metrics'):
                rows['platform_activity_group'] = np.where(activity.loc[predict_ids] >= activity_threshold, 'high', 'normal')
            if resolved.get('guarded_response_metrics'):
                calm = calm_mask(pool.X.loc[predict_ids], context)
                rows['pricing_risk_group'] = np.where(rows.option_type.eq(0), 'call_control',
                    np.where(calm, 'calm', 'stress') if context['context_put_rows'] else 'unavailable')
                # Outcome labels classify evaluation rows only, after prediction.
                rows['payout_tail_group'] = np.where(rows.option_type.eq(0), 'call_control',
                    np.where(rows.actual.gt(context['payout_q90']), 'tail', 'body') if context['context_put_rows'] else 'unavailable')
            if resolved.get('fast_response_metrics'):
                rows['fast_response_state'] = pool.X.loc[predict_ids, 'fast_response_state']
                rows['fast_response_phase'] = np.where(rows.option_type.eq(0), 'call_control',
                    pool.X.loc[predict_ids, 'fast_response_phase'])
                rows['multiplier'] = rows.prediction / pool.X.loc[predict_ids, 'api_base_price_per_btc_usdc']
                if resolved.get('response_max_multiplier') is not None:
                    rows['response_at_cap'] = rows.multiplier.ge(resolved['response_max_multiplier']-1e-12)
            all_rows.append(rows)
            for scope, frame in _scopes(rows, resolved["cohort"]):
                entry = {"model": name, "cohort": scope, **details, **runtime, **_metrics(frame)}
                if resolved.get("daily_direction_metrics"):
                    entry.update(group="overall", value="all")
                daily_rows.append(entry)
                if resolved.get("daily_direction_metrics"):
                    daily_groups = ['option_type']
                    if resolved.get('fast_response_metrics'):
                        daily_groups.append('fast_response_phase')
                    if resolved.get('guarded_response_metrics'):
                        daily_groups.extend(['pricing_risk_group', 'payout_tail_group'])
                    for column in daily_groups:
                        for value, group in frame.groupby(column, sort=True):
                            daily_rows.append({"model": name, "cohort": scope, **details, **runtime,
                                               "group": column, "value": str(value), **_metrics(group)})
    predictions = pd.concat(all_rows)
    summary = []
    for name in resolved["selected_models"]:
        for scope, frame in _scopes(predictions.loc[predictions.model == name], resolved["cohort"]):
            summary.append({"model": name, "cohort": scope, "group": "overall", "value": "all", **_metrics(frame)})
            groups = ["option_type", "product_duration_seconds", "volatility_group"]
            if resolved.get('activity_metrics'):
                groups.append('platform_activity_group')
            if resolved.get('fast_response_metrics'):
                groups.append('fast_response_phase')
            if resolved.get('guarded_response_metrics'):
                groups.extend(['pricing_risk_group', 'payout_tail_group'])
            for column in groups:
                for value, group in frame.groupby(column, sort=True):
                    summary.append({"model": name, "cohort": scope, "group": column,
                                    "value": str(value), **_metrics(group)})
    return pd.DataFrame(summary), pd.DataFrame(daily_rows)
