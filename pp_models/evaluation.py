"""Prequential fitting and compact metrics; row predictions stay in memory."""
import numpy as np
import pandas as pd
from time import perf_counter
import warnings
from threadpoolctl import threadpool_limits
from .data import daily_batches
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
            "predicted_actual_ratio": predicted / observed if observed else None}


def _scopes(frame, cohort):
    yield cohort, frame
    if cohort == "risk":
        common = frame.loc[frame.benchmark_eligible]
        if not common.empty:
            yield "benchmark", common


def _metrics(frame):
    return key_metrics(frame.actual, frame.prediction, frame.quantity) | {
        "clipped_predictions": int((frame.raw_prediction < 0).sum())}


def evaluate(pool, split, resolved, model_factory=build_model, period=None):
    daily_rows, all_rows = [], []
    features = resolved["features"]
    # The factory receives only configuration; fit receives only the day's legal
    # historical rows. Test outcomes are used below solely for scoring.
    for day, train_ids, predict_ids in daily_batches(pool, split, resolved["window_days"], period):
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
        for name in resolved["selected_models"]:
            model = model_factory(name, resolved)
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                started = perf_counter()
                model.fit(X_train, y_train)
                fit_seconds = perf_counter() - started
                started = perf_counter()
                raw = np.asarray(model.predict_raw(X_today), dtype=float)
                predict_seconds = perf_counter() - started
            runtime = {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                       "warning_count": len(caught),
                       "warning_categories": "|".join(sorted({w.category.__name__ for w in caught}))}
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
            all_rows.append(rows)
            for scope, frame in _scopes(rows, resolved["cohort"]):
                daily_rows.append({"model": name, "cohort": scope, **details, **runtime, **_metrics(frame)})
    predictions = pd.concat(all_rows)
    summary = []
    for name in resolved["selected_models"]:
        for scope, frame in _scopes(predictions.loc[predictions.model == name], resolved["cohort"]):
            summary.append({"model": name, "cohort": scope, "group": "overall", "value": "all", **_metrics(frame)})
            for column in ("option_type", "product_duration_seconds", "volatility_group"):
                for value, group in frame.groupby(column, sort=True):
                    summary.append({"model": name, "cohort": scope, "group": column,
                                    "value": str(value), **_metrics(group)})
    return pd.DataFrame(summary), pd.DataFrame(daily_rows)
