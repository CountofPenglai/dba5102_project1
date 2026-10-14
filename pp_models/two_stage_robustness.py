"""Paired calendar sensitivity and block bootstrap for branch 5."""
import calendar
import numpy as np
import pandas as pd


def build_paired_daily_statistics(frame, comparisons):
    required = {"day", "month", "model", "actual", "prediction", "side"}
    if not required.issubset(frame) or frame.index.has_duplicates is False:
        # Prediction frames intentionally repeat order IDs once per model.
        if not required.issubset(frame):
            raise ValueError("Missing paired prediction columns")
    rows = []
    for candidate, reference in comparisons:
        c = frame.loc[frame.model.eq(candidate)]
        r = frame.loc[frame.model.eq(reference)]
        if c.empty or r.empty or not c.index.equals(r.index) or not np.array_equal(c.actual, r.actual):
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
                             "candidate_sae": float(np.abs(ce).sum()), "reference_sae": float(np.abs(re).sum())})
    return pd.DataFrame(rows)


def paired_delta(daily):
    monthly = daily.groupby("month").agg(orders=("orders", "sum"), candidate_sse=("candidate_sse", "sum"),
        reference_sse=("reference_sse", "sum"), candidate_sae=("candidate_sae", "sum"),
        reference_sae=("reference_sae", "sum"))
    if len(monthly) != 3 or (monthly.orders <= 0).any():
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
                              ("seven_day", [(d, d + pd.Timedelta(days=7)) for d in pd.date_range(start, end-pd.Timedelta(days=7), inclusive="left")])):
            values = []
            for left, right in windows:
                delta = paired_delta(part.loc[~part.day.ge(left) | ~part.day.lt(right)])
                if delta is not None:
                    values.append((left, delta))
            for metric in ("rmse_delta", "mae_delta"):
                numbers = np.array([v[1][metric] for v in values])
                base = full[metric]
                extreme_min, extreme_max = int(numbers.argmin()), int(numbers.argmax())
                results.append({"candidate": candidate, "reference": reference, "direction": direction,
                                "window_type": kind, "metric": metric, "full_delta": base,
                                "minimum_delta": float(numbers[extreme_min]), "maximum_delta": float(numbers[extreme_max]),
                                "sign_flip_windows": int(np.sum(np.sign(numbers) != np.sign(base))),
                                "minimum_start": values[extreme_min][0].date().isoformat(),
                                "maximum_start": values[extreme_max][0].date().isoformat(),
                                "valid_windows": len(numbers)})
    return pd.DataFrame(results)


def paired_calendar_bootstrap(daily, block_days, repeats=2000, seed=20261012, minimum_valid=1900):
    if block_days not in (3, 7, 14) or repeats != 2000 or minimum_valid != 1900:
        raise ValueError("Unsupported fixed bootstrap settings")
    output = []
    for pair_index, ((candidate, reference, direction), part) in enumerate(
            daily.groupby(["candidate", "reference", "direction"], sort=True)):
        rng = np.random.default_rng(seed + block_days * 1000 + pair_index)
        day_rows = {pd.Timestamp(row.day).normalize(): row for row in part.itertuples()}
        draws = []
        for _ in range(repeats):
            sampled = []
            for month in (4, 5, 6):
                days = pd.date_range(pd.Timestamp(2026, month, 1, tz="UTC"), periods=calendar.monthrange(2026, month)[1])
                chosen = []
                while len(chosen) < len(days):
                    start = int(rng.integers(0, len(days)))
                    chosen.extend(days[(start + offset) % len(days)] for offset in range(block_days))
                selected = [day_rows[d] for d in chosen[:len(days)] if d in day_rows]
                if not selected:
                    sampled = []
                    break
                sampled.extend(selected)
            if not sampled:
                continue
            draw = paired_delta(pd.DataFrame([row._asdict() for row in sampled]))
            if draw is not None:
                draws.append(draw)
        for metric in ("rmse_delta", "mae_delta"):
            values = np.array([draw[metric] for draw in draws], float)
            valid = len(values)
            output.append({"candidate": candidate, "reference": reference, "direction": direction,
                           "block_days": block_days, "metric": metric, "repeats": repeats,
                           "valid_repeats": valid, "excluded_repeats": repeats-valid,
                           "lower_95": float(np.quantile(values, .025)) if valid >= minimum_valid else np.nan,
                           "median": float(np.quantile(values, .5)) if valid >= minimum_valid else np.nan,
                           "upper_95": float(np.quantile(values, .975)) if valid >= minimum_valid else np.nan})
    return pd.DataFrame(output)


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
    if block_days not in (3, 7, 14) or repeats != 2000 or minimum_valid != 1900:
        raise ValueError("Unsupported fixed bootstrap settings")
    output = []
    for pair_index, ((candidate, reference, direction), part) in enumerate(
            daily.groupby(["candidate", "reference", "direction"], sort=True)):
        rng = np.random.default_rng(seed + 50000 + block_days * 1000 + pair_index)
        day_rows = {pd.Timestamp(row.day).normalize(): row for row in part.itertuples()}
        values = []
        for _ in range(repeats):
            monthly = []
            for month in (4, 5, 6):
                days = pd.date_range(pd.Timestamp(2026, month, 1, tz="UTC"), periods=calendar.monthrange(2026, month)[1])
                chosen = []
                while len(chosen) < len(days):
                    start = int(rng.integers(0, len(days)))
                    chosen.extend(days[(start + offset) % len(days)] for offset in range(block_days))
                selected = [day_rows[d] for d in chosen[:len(days)] if d in day_rows]
                orders = sum(row.orders for row in selected)
                if not orders:
                    monthly = []
                    break
                monthly.append(sum(row.candidate_brier_sum for row in selected)/orders
                               - sum(row.reference_brier_sum for row in selected)/orders)
            if len(monthly) == 3:
                values.append(float(np.mean(monthly)))
        values = np.asarray(values)
        valid = len(values)
        output.append({"candidate": candidate, "reference": reference, "direction": direction,
                       "block_days": block_days, "metric": "brier_delta", "repeats": repeats,
                       "valid_repeats": valid, "excluded_repeats": repeats-valid,
                       "lower_95": float(np.quantile(values, .025)) if valid >= minimum_valid else np.nan,
                       "median": float(np.quantile(values, .5)) if valid >= minimum_valid else np.nan,
                       "upper_95": float(np.quantile(values, .975)) if valid >= minimum_valid else np.nan})
    return pd.DataFrame(output)
