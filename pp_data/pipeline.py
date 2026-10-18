"""Shared quality masks and a chronological pool with final settled labels."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from pp_audit.orders import audit_orders
from pp_audit.market import audit_market
from .config import load_config, load_protocol
from .features import build_features, MARKET_FEATURES, RISK_FEATURES


@dataclass
class PreparedData:
    features: pd.DataFrame
    targets: pd.DataFrame
    metadata: pd.DataFrame
    statistics: pd.DataFrame
    exclusions: pd.DataFrame
    config: dict


def pool_statistics(metadata):
    """Summarize the pool without assigning any model evaluation roles."""
    dates = metadata.copy()
    for name in ("created_at_utc", "closed_at_utc"):
        dates[name] = pd.to_datetime(dates[name], utc=True, format="mixed")
    rows = []
    for cohort in ("risk", "benchmark"):
        selected = dates if cohort == "risk" else dates.loc[dates.benchmark_eligible]
        rows.append({
            "cohort": cohort, "rows": len(selected),
            "first_created_utc": selected.created_at_utc.min(),
            "last_created_utc": selected.created_at_utc.max(),
            "last_label_available_utc": selected.closed_at_utc.max(),
            "base_missing_rows": int(selected.base_missing.sum()),
        })
    return pd.DataFrame(rows)


def prepare(orders_raw: pd.DataFrame, market_raw: pd.DataFrame) -> PreparedData:
    config = load_config()
    start = pd.Timestamp(config["research_start"])
    end = pd.Timestamp(load_protocol()["validation_end"])
    audited = audit_orders(orders_raw)
    market = audit_market(market_raw)
    d, cohorts = audited.rows, audited.cohorts
    x, last_close = build_features(d, market.usable)
    ids = pd.Series([f"training_{i + 1:06d}" for i in range(len(d))], index=d.index)
    reason = pd.Series("invalid_order", index=d.index)
    reason.loc[cohorts.technical] = "outside_research_window"
    belongs = cohorts.technical & d.createdAt.ge(start) & d.createdAt.lt(end)
    reason.loc[belongs] = "label_not_known_before_research_end"
    reason.loc[belongs & d.closedAt.lt(end)] = "retained"
    history_ready = last_close.notna() & x[MARKET_FEATURES].notna().all(axis=1)
    reason.loc[reason.eq("retained") & ~history_ready] = "insufficient_market_history"
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        unit_payout = d.realizedProfit_USDC / d.quantity_BTC
    finite = np.isfinite(x[RISK_FEATURES]).all(axis=1) & np.isfinite(unit_payout)
    reason.loc[reason.eq("retained") & ~finite] = "invalid_derived_values"
    targets = pd.DataFrame({
        "order_id": ids, "payout_usdc": d.realizedProfit_USDC,
        "payout_per_btc_usdc": unit_payout,
        "positive_payout": d.realizedProfit_USDC.gt(0).astype(int),
    })
    x.insert(0, "order_id", ids)
    metadata = pd.DataFrame({
        "order_id": ids, "source_row_number": np.arange(1, len(d) + 1),
        "created_at_utc": d.createdAt, "closed_at_utc": d.closedAt,
        "expiration_at_utc": d.expirationDate, "last_candle_close_utc": last_close,
        "benchmark_eligible": cohorts.common,
        "base_missing": d.apiBasePrice_per_BTC_USDC.isna(),
        "nominal_duration_deviation_seconds": (d.expirationDate - d.createdAt).dt.total_seconds() - d.productType,
        "recorded_premium_usdc": d.premiumAmount_USDC,
        "recorded_price_per_btc_usdc": d.orderPrice_per_BTC_USDC,
        "recorded_markup_per_btc_usdc": d.markup_per_BTC_USDC,
    })
    positions = metadata.loc[reason.eq("retained")].sort_values(
        ["created_at_utc", "source_row_number"], kind="stable").index
    features, targets, metadata = (table.loc[positions].reset_index(drop=True)
                                    for table in (x, targets, metadata))
    definitions = {
        "invalid_order": "Failed shared order quality checks, including missing labels, time anomalies, and duplicates",
        "outside_research_window": "Creation outside the research interval",
        "label_not_known_before_research_end": "Settlement at or after the final research cutoff",
        "insufficient_market_history": "No recent complete sequence of 1441 closed minute candles",
        "invalid_derived_values": "Nonfinite features or unit payout",
        "retained": "Retained in the chronological pool; benchmark is its common-price subset",
    }
    exclusions = pd.DataFrame([
        {"reason": key, "rows": int(reason.eq(key).sum()), "definition": value}
        for key, value in definitions.items()
    ])
    return PreparedData(features, targets, metadata, pool_statistics(metadata), exclusions, config)
