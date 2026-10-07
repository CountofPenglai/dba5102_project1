"""Shared quality masks, label availability, and one chronological assignment."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from pp_audit.orders import audit_orders
from pp_audit.market import audit_market
from .config import load_config
from .features import build_features, MARKET_FEATURES, RISK_FEATURES


@dataclass
class SplitTables:
    features: pd.DataFrame
    targets: pd.DataFrame
    metadata: pd.DataFrame


@dataclass
class PreparedData:
    splits: dict[str, SplitTables]
    statistics: pd.DataFrame
    exclusions: pd.DataFrame
    config: dict


def prepare(orders_raw: pd.DataFrame, market_raw: pd.DataFrame) -> PreparedData:
    config = load_config()
    audited = audit_orders(orders_raw)
    market = audit_market(market_raw)
    d, cohorts = audited.rows, audited.cohorts
    x, last_close = build_features(d, market.usable)
    ids = pd.Series([f"training_{i + 1:06d}" for i in range(len(d))], index=d.index)
    assignments = pd.Series("", index=d.index)
    reason = pd.Series("invalid_order", index=d.index)
    reason.loc[cohorts.technical] = "outside_split_window"
    for split, bounds in config["splits"].items():
        belongs = cohorts.technical & d.createdAt.ge(pd.Timestamp(bounds["start"])) & d.createdAt.lt(pd.Timestamp(bounds["end"]))
        assignments.loc[belongs] = split
        reason.loc[belongs] = "label_not_known_before_split_end"
        reason.loc[belongs & d.closedAt.lt(pd.Timestamp(bounds["end"]))] = "retained"
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
    splits, stats = {}, []
    for split, bounds in config["splits"].items():
        mask = reason.eq("retained") & assignments.eq(split)
        positions = metadata.loc[mask].sort_values(["created_at_utc", "source_row_number"], kind="stable").index
        splits[split] = SplitTables(*(table.loc[positions].reset_index(drop=True) for table in (x, targets, metadata)))
        for cohort in ("risk", "benchmark"):
            selected = mask if cohort == "risk" else mask & cohorts.common
            n = int(selected.sum())
            stats.append({
                "split": split, "cohort": cohort, "rows": n,
                "window_start_utc": bounds["start"], "window_end_exclusive_utc": bounds["end"],
                "first_created_utc": d.loc[selected, "createdAt"].min(),
                "last_created_utc": d.loc[selected, "createdAt"].max(),
                "last_label_available_utc": d.loc[selected, "closedAt"].max(),
                "base_missing_rows": int((selected & d.apiBasePrice_per_BTC_USDC.isna()).sum()),
            })
    definitions = {
        "invalid_order": "未通过共享订单技术条件；含缺失标签、时间异常与多余重复记录",
        "outside_split_window": "通过技术条件，但创建时间不在固定研究区间",
        "label_not_known_before_split_end": "创建时间在本组内，但结算不早于本组结束；不移入其他组",
        "insufficient_market_history": "最近有效行情过旧或缺少连续1441根已结束分钟线",
        "invalid_derived_values": "其余条件满足，但特征或单位赔付出现非有限值",
        "retained": "保留到唯一分割；benchmark是其中满足基础价条件的子集",
    }
    exclusions = pd.DataFrame([
        {"reason": key, "rows": int(reason.eq(key).sum()), "definition": value}
        for key, value in definitions.items()
    ])
    return PreparedData(splits, pd.DataFrame(stats), exclusions, config)
