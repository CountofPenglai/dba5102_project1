"""Paired calendar sensitivity and block bootstrap for branch 5."""
import calendar
import hashlib
import numpy as np
import pandas as pd


def build_paired_daily_statistics(frame, comparisons):
    required = {"day", "month", "model", "actual", "prediction", "side"}
    if not required.issubset(frame):
        raise ValueError("Missing paired prediction columns")
    rows = []
    for candidate, reference in comparisons:
        c = frame.loc[frame.model.eq(candidate)]
        r = frame.loc[frame.model.eq(reference)]
        if (c.empty or r.empty or not c.index.is_unique or not r.index.is_unique
                or not c.index.equals(r.index)
                or not c[["actual", "day", "month", "side"]].equals(r[["actual", "day", "month", "side"]])):
            raise ValueError("Paired models do not share identical ordered rows")
        for direction in ("all", 0, 1):
            mask = np.ones(len(c), dtype=bool) if direction == "all" else c.side.eq(direction).to_numpy()
            for day, positions in c.loc[mask].groupby("day", sort=True).indices.items():
                cc = c.loc[mask].iloc[positions]
                rr = r.loc[mask].iloc[positions]
                ce, re = cc.prediction.to_numpy() - cc.actual.to_numpy(), rr.prediction.to_numpy() - rr.actual.to_numpy()
                rows.append({"candidate": candidate, "reference": reference, "direction": str(direction),
                             "day": pd.Timestamp(day), "month": cc.month.iloc[0], "orders": len(cc),
                             "candidate_sse": float(np.dot(ce, ce)), "reference_sse": float(np.dot(re, re)),
                             "candidate_sae": float(np.abs(ce).sum()), "reference_sae": float(np.abs(re).sum()),
                             "tail_orders": int(cc.tail_group.eq("tail").sum()) if "tail_group" in cc else 0})
    return pd.DataFrame(rows)


def paired_delta(daily):
    monthly = daily.groupby("month").agg(orders=("orders", "sum"), candidate_sse=("candidate_sse", "sum"),
        reference_sse=("reference_sse", "sum"), candidate_sae=("candidate_sae", "sum"),
        reference_sae=("reference_sae", "sum"))
    if set(monthly.index) != {"2026-04", "2026-05", "2026-06"} or (monthly.orders <= 0).any():
        return None
    return {"rmse_delta": float(np.sqrt(np.mean(monthly.candidate_sse / monthly.orders))
                                - np.sqrt(np.mean(monthly.reference_sse / monthly.orders))),
            "mae_delta": float(np.mean(monthly.candidate_sae / monthly.orders)
                               - np.mean(monthly.reference_sae / monthly.orders))}


def calendar_influence(daily):
    results = []
    start, end = pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")
    for (candidate, reference, direction), part in daily.groupby(["candidate", "reference", "direction"]):
        full = paired_delta(part)
        for kind, windows in (("single_day", [(d, d + pd.Timedelta(days=1)) for d in pd.date_range(start, end, inclusive="left")]),
                              ("seven_day", [(d, d + pd.Timedelta(days=7)) for d in pd.date_range(start, end-pd.Timedelta(days=7), inclusive="both")])):
            values = []
            for left, right in windows:
                delta = paired_delta(part.loc[~part.day.ge(left) | ~part.day.lt(right)])
                if delta is not None:
                    values.append((left, delta))
            for metric in ("rmse_delta", "mae_delta"):
                numbers = np.array([v[1][metric] for v in values])
                base = full[metric] if full else np.nan
                extreme_min = int(numbers.argmin()) if len(numbers) else None
                extreme_max = int(numbers.argmax()) if len(numbers) else None
                results.append({"candidate": candidate, "reference": reference, "direction": direction,
                                "window_type": kind, "metric": metric, "full_delta": base,
                                "minimum_delta": float(numbers[extreme_min]) if len(numbers) else np.nan, "maximum_delta": float(numbers[extreme_max]) if len(numbers) else np.nan,
                                "sign_flip_windows": int(np.sum(np.sign(numbers) != np.sign(base))) if full else np.nan,
                                "minimum_start": values[extreme_min][0].date().isoformat() if len(numbers) else None,
                                "maximum_start": values[extreme_max][0].date().isoformat() if len(numbers) else None,
                                "valid_windows": len(numbers), "total_windows": len(windows),
                                "excluded_windows": len(windows)-len(numbers)})
    return pd.DataFrame(results)


def calendar_draw_weights(block_days, repeats=2000, seed=20261012):
    """One month-stratified circular calendar schedule, independent of pair order."""
    if block_days not in (3, 7, 14) or repeats != 2000:
        raise ValueError("Unsupported fixed bootstrap settings")
    days = pd.date_range("2026-04-01", "2026-06-30", tz="UTC")
    weights = np.zeros((repeats, len(days)), dtype=np.int64)
    rng = np.random.default_rng(seed + block_days * 1000)
    for repeat in range(repeats):
        offset = 0
        for month in (4, 5, 6):
            n = calendar.monthrange(2026, month)[1]
            starts = rng.integers(0, n, size=int(np.ceil(n / block_days)))
            chosen = ((starts[:, None] + np.arange(block_days)) % n).ravel()[:n]
            weights[repeat, offset:offset+n] = np.bincount(chosen, minlength=n)
            offset += n
    return days, weights


def interval_summary(values, repeats=2000, minimum_valid=1900):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    valid = len(values)
    estimable = valid >= minimum_valid
    return {"repeats": repeats, "valid_repeats": valid, "excluded_repeats": repeats-valid,
            "estimable": estimable,
            "lower_95": float(np.quantile(values, .025)) if estimable else np.nan,
            "median": float(np.quantile(values, .5)) if estimable else np.nan,
            "upper_95": float(np.quantile(values, .975)) if estimable else np.nan}


def _calendar_bootstrap(daily, block_days, repeats, seed, minimum_valid, probability=False):
    if minimum_valid != 1900:
        raise ValueError("Unsupported fixed bootstrap settings")
    days, weights = calendar_draw_weights(block_days, repeats, seed)
    draw_hash = hashlib.sha256(weights.astype("<i8").tobytes()).hexdigest()
    loss_columns = (["candidate_brier_sum", "reference_brier_sum"] if probability else
                    ["candidate_sse", "reference_sse", "candidate_sae", "reference_sae"])
    output = []
    for (candidate, reference, direction), part in daily.groupby(["candidate", "reference", "direction"], sort=True):
        if (part.day.duplicated().any() or not part.day.isin(days).all()
                or not np.isfinite(part[["orders"] + loss_columns].to_numpy(float)).all()
                or part[["orders"] + loss_columns].lt(0).any().any()):
            raise ValueError("Invalid paired calendar statistics")
        calendar_rows = part.set_index("day").reindex(days, fill_value=0)
        monthly, valid = [], np.ones(repeats, dtype=bool)
        for month in (4, 5, 6):
            mask = days.month == month
            totals = weights[:, mask] @ calendar_rows.loc[mask, ["orders"] + loss_columns].to_numpy(float)
            counts = totals[:, 0]
            valid &= counts > 0
            losses = np.full((repeats, len(loss_columns)), np.nan)
            np.divide(totals[:, 1:], counts[:, None], out=losses, where=counts[:, None] > 0)
            monthly.append(losses)
        means = np.mean(monthly, axis=0)
        differences = ({"brier_delta": means[:, 0]-means[:, 1]} if probability else
                       {"rmse_delta": np.sqrt(means[:, 0])-np.sqrt(means[:, 1]),
                        "mae_delta": means[:, 2]-means[:, 3]})
        for metric, values in differences.items():
            output.append({"candidate": candidate, "reference": reference, "direction": direction,
                           "block_days": block_days, "metric": metric,
                           "calendar_draws_sha256": draw_hash,
                           **interval_summary(values[valid], repeats, minimum_valid)})
    return pd.DataFrame(output)


def paired_calendar_bootstrap(daily, block_days, repeats=2000, seed=20261012, minimum_valid=1900):
    return _calendar_bootstrap(daily, block_days, repeats, seed, minimum_valid)


def build_probability_daily_statistics(frame, candidate_column, reference_column, candidate, reference):
    required = {"day", "month", "actual", "side", candidate_column, reference_column}
    if not required.issubset(frame) or frame.index.duplicated().any():
        raise ValueError("Invalid probability pairing frame")
    rows = []
    actual = frame.actual.gt(0).astype(float)
    for direction in ("all", 0, 1):
        selected = frame if direction == "all" else frame.loc[frame.side.eq(direction)]
        for day, part in selected.groupby("day", sort=True):
            y = actual.loc[part.index].to_numpy(float)
            cp, rp = part[candidate_column].to_numpy(float), part[reference_column].to_numpy(float)
            rows.append({"candidate": candidate, "reference": reference, "direction": str(direction),
                         "day": pd.Timestamp(day), "month": part.month.iloc[0], "orders": len(part),
                         "candidate_brier_sum": float(np.sum((cp-y)**2)),
                         "reference_brier_sum": float(np.sum((rp-y)**2))})
    return pd.DataFrame(rows)


def paired_calendar_bootstrap_brier(daily, block_days, repeats=2000, seed=20261012, minimum_valid=1900):
    return _calendar_bootstrap(daily, block_days, repeats, seed, minimum_valid, probability=True)
