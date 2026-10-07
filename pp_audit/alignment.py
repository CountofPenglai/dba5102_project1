"""Point-in-time market coverage and research cutoff audit."""

import numpy as np
import pandas as pd
from .common import Metrics
from .orders import OrderAudit
from .market import MarketAudit


def audit_alignment(
    orders: OrderAudit, market: MarketAudit, metadata: dict,
    cutoffs: list[str], validation_present: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = orders.rows
    history = market.usable
    valid_time = d.createdAt.notna().to_numpy()
    n_valid = int(valid_time.sum())
    found = np.zeros(len(d), dtype=bool)
    fresh = np.zeros(len(d), dtype=bool)
    latest = np.full(len(d), -1, dtype=np.int64)
    if len(history):
        end_ns = history.close_time_utc.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        order_ns = d.createdAt.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        latest[valid_time] = np.searchsorted(end_ns, order_ns[valid_time], side="right") - 1
        found = valid_time & (latest >= 0)
        # Never use the current incomplete minute or a future candle.
        age = (order_ns[found] - end_ns[latest[found]]) / 1e9
        fresh[found] = (age >= 0) & (age <= 60)
    m = Metrics()
    m.add("orders_with_valid_creation_time", n_valid, "rows", "行情匹配分母，仅创建时间可解析的订单")
    m.count("orders_without_completed_candle", valid_time & ~found,
            "创建时刻之前没有通过质量检查的已结束K线", n_valid)
    m.count("orders_with_stale_completed_candle", found & ~fresh,
            "最近已结束有效K线距下单超过60秒", n_valid)
    m.count("orders_with_fresh_completed_candle", fresh,
            "最近有效K线结束时间不晚于下单，间隔不超过60秒", n_valid)
    open_ns = history.open_time_utc.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    for window in (60, 240, 1440):
        complete = np.zeros(len(d), dtype=bool)
        sufficient = fresh & (latest >= window - 1)
        positions = latest[sufficient]
        if len(positions):
            # Unique grid-aligned minute starts plus exact span prove continuity.
            spans = open_ns[positions] - open_ns[positions - window + 1]
            complete[sufficient] = spans == (window - 1) * 60 * 1_000_000_000
        m.count(f"orders_with_complete_{window}m_history", complete,
                f"最近{window}根已结束一分钟K线全部有效且连续", n_valid)
        if window == 1440:
            m.count("common_candidates_with_complete_1440m_history",
                    complete & orders.cohorts.common.to_numpy(),
                    "基础价共同候选中具有连续24小时事前行情的数量",
                    int(orders.cohorts.common.sum()))
    m.add("metadata_available", bool(metadata), "boolean", "是否读取到提供方元数据")
    m.add("validation_file_present", validation_present, "boolean", "输入目录是否有官方验证CSV，不代表已审核其内容")
    expected = metadata.get("training_rows")
    m.add("declared_training_rows", expected, "rows", "提供方元数据声明的训练订单数量")
    m.add("observed_minus_declared_training_rows", len(d) - expected if expected is not None else None,
          "rows", "实际训练数量减声明数量；无声明则不可计算")
    filter_metadata = metadata.get("negative_markup_filter")
    removed = filter_metadata.get("training_rows_removed") if isinstance(filter_metadata, dict) else None
    before_filter = metadata.get("pre_negative_markup_filter_training_rows")
    m.add("source_negative_markup_removed_training_rows", removed, "rows",
          "提供方在交付前删除的负markup训练记录；本程序未重新删除", before_filter)
    m.add("future_api_quote_match_share", None, "fraction",
          "不可核验：源订单不含匹配报价时间，不能证明基础价不存在未来信息")
    last_order = pd.to_datetime(metadata.get("validation_last_createdAt_utc"), utc=True, errors="coerce")
    last_close = history.close_time_utc.max()
    shortfall = None
    if pd.notna(last_order) and pd.notna(last_close):
        shortfall = max(0.0, (last_order - last_close).total_seconds())
    m.add("validation_market_end_shortfall_seconds", shortfall, "seconds",
          "元数据声明的最后验证订单时间超过可用行情结束时间的秒数；任一端缺少或不可解析则不可计算")
    return m.frame(), _cutoff_statistics(orders, metadata, cutoffs)


def _cutoff_statistics(orders, metadata, cutoffs):
    requested = [(value, "internal_research") for value in cutoffs]
    official = metadata.get("validation_first_createdAt_utc")
    if official is not None:
        requested.append((official, "official_validation"))
    records = []
    d = orders.rows
    for text, source in requested:
        cutoff = pd.to_datetime(text, utc=True, errors="raise")
        if pd.isna(cutoff):
            raise ValueError(f"Invalid cutoff: {text}")
        prior = d.createdAt.lt(cutoff)
        known = d.closedAt.lt(cutoff)
        records.append(dict(
            cutoff_utc=cutoff, cutoff_type=source,
            orders_created_before_cutoff=int(prior.sum()),
            outcome_unavailable_rows=int((prior & ~orders.cohorts.labeled).sum()),
            recorded_outcome_not_yet_settled_rows=int((prior & orders.cohorts.labeled & d.closedAt.ge(cutoff)).sum()),
            recorded_outcome_without_close_time_rows=int((prior & orders.cohorts.labeled & d.closedAt.isna()).sum()),
            technical_candidates_known_before_cutoff=int((prior & known & orders.cohorts.technical).sum()),
            common_candidates_known_before_cutoff=int((prior & known & orders.cohorts.common).sum()),
        ))
    return pd.DataFrame(records, columns=[
        "cutoff_utc", "cutoff_type", "orders_created_before_cutoff",
        "outcome_unavailable_rows", "recorded_outcome_not_yet_settled_rows",
        "recorded_outcome_without_close_time_rows", "technical_candidates_known_before_cutoff",
        "common_candidates_known_before_cutoff",
    ]).drop_duplicates().reset_index(drop=True)
