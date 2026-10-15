"""Common per-BTC, cash and monthly report metrics."""
import numpy as np
import pandas as pd
from .resampling import scope_frames

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


def groups(frame):
    yield "overall", "all", frame
    for side, part in frame.groupby("side"):
        yield "direction", str(int(side)), part
        for value, tail in part.groupby("tail_group"):
            yield "direction_tail", f"{int(side)}|{value}", tail
    for tenor, part in frame.groupby("tenor"):
        yield "tenor", str(int(tenor)), part


def metric_values(frame):
    values = key_metrics(frame.actual, frame.prediction, frame.quantity)
    values["absolute_total_bias_usdc"] = abs(values["total_bias_usdc"])
    return values


def aggregate(frame):
    rows, daily = [], []
    for month in ["all"] + sorted(frame.month.unique()):
        selected = frame if month == "all" else frame.loc[frame.month.eq(month)]
        for model, model_frame in selected.groupby("model"):
            for group, value, part in groups(model_frame):
                rows.append({"month": month, "model": model, "group": group, "value": value,
                             "days": int(part.day.nunique()), **metric_values(part)})
    for (day, model), part in frame.groupby(["day", "model"]):
        for group, value, sub in groups(part):
            if group in ("overall", "direction"):
                daily.append({"day": day.isoformat(), "month": part.month.iloc[0], "model": model,
                              "group": group, "value": value, **metric_values(sub)})
    summary, daily = pd.DataFrame(rows), pd.DataFrame(daily)
    monthly = summary.loc[(summary.month != "all") & summary.group.eq("overall")]
    macro = []
    for model, part in monthly.groupby("model"):
        macro.append({"model": model, "months": len(part), "orders": int(part.orders.sum()),
                      "macro_rmse_per_btc": float(np.sqrt(np.mean(part.rmse_per_btc ** 2))),
                      "macro_mae_per_btc": float(part.mae_per_btc.mean())})
    return summary, daily, pd.DataFrame(macro)


def ratio(a,b):
    return float(a/b) if b else (1. if a==0 else np.inf)


def scoped_macro(frame,months):
    rows=[]
    for scope,part in scope_frames(frame):
        for model,p in part.groupby('model'):
            monthly=[metric_values(p.loc[p.month.eq(m)]) for m in months if p.month.eq(m).any()]
            complete=len(monthly)==len(months)
            rows.append(dict(scope=scope,model=model,months=len(monthly),expected_months=len(months),
                complete=complete,macro_rmse_per_btc=np.sqrt(np.mean([v['rmse_per_btc']**2 for v in monthly])) if complete else np.nan,
                macro_mae_per_btc=np.mean([v['mae_per_btc'] for v in monthly]) if complete else np.nan))
    return pd.DataFrame(rows)
