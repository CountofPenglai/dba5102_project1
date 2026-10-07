"""Order quality, outcome concentration and common-sample audit."""

from dataclasses import dataclass
import math
import pandas as pd
from .common import ORDER_COLUMNS, ORDER_DATES, ORDER_NUMBERS, Metrics, parse_frame, ratio, total


@dataclass
class OrderAudit:
    metrics: pd.DataFrame
    segments: pd.DataFrame
    rows: pd.DataFrame
    cohorts: pd.DataFrame


def audit_orders(raw: pd.DataFrame) -> OrderAudit:
    d, missing, bad_num, bad_dates = parse_frame(
        raw, ORDER_COLUMNS, ORDER_DATES, ORDER_NUMBERS,
    )
    n = len(d)
    duplicate = raw.reset_index(drop=True)[ORDER_COLUMNS].duplicated()
    quantity_valid = d.quantity_BTC.gt(0)
    premium_valid = d.premiumAmount_USDC.ge(0)
    outcome_valid = d.realizedProfit_USDC.ge(0)
    labeled = outcome_valid & premium_valid & quantity_valid
    direction_valid = d.optionType.isin([0, 1])
    label_valid = d.optionType.map({0: "Call", 1: "Put"}).eq(d.optionType_label)
    tenor_valid = d.productType.gt(0) & d.productType.mod(1).eq(0)
    expired = d.expirationDate.le(d.createdAt)
    closed_before = d.closedAt.lt(d.createdAt)
    time_anomaly = expired | closed_before
    duration = (d.expirationDate - d.createdAt).dt.total_seconds()
    duration_deviation = duration - d.productType
    actual_unit = d.premiumAmount_USDC / d.quantity_BTC.where(quantity_valid)
    normalized_error = (actual_unit - d.orderPrice_per_BTC_USDC).abs()
    normalized_ok = normalized_error.le(1e-6) & d.orderPrice_per_BTC_USDC.ge(0)
    identity_eligible = actual_unit.notna() & d.orderPrice_per_BTC_USDC.notna()
    markup_error = (d.orderPrice_per_BTC_USDC - d.apiBasePrice_per_BTC_USDC
                    - d.markup_per_BTC_USDC).abs()
    markup_identity_eligible = d[[
        "orderPrice_per_BTC_USDC", "apiBasePrice_per_BTC_USDC", "markup_per_BTC_USDC",
    ]].notna().all(axis=1)
    technical = (
        labeled & d[ORDER_DATES].notna().all(axis=1) & ~time_anomaly
        & direction_valid & label_valid & tenor_valid & d.strikePrice.gt(0)
        & normalized_ok & ~duplicate
    )
    common = (technical & d.apiBasePrice_per_BTC_USDC.gt(0)
              & d.markup_per_BTC_USDC.ge(0) & markup_error.le(1e-6))
    cohorts = pd.DataFrame(dict(labeled=labeled, technical=technical, common=common))
    m = Metrics()
    m.add("rows", n, "rows", "输入订单总数，未执行清洗")
    m.add("first_created_utc", d.createdAt.min(), "UTC", "可解析创建时间的最早值")
    m.add("last_created_utc", d.createdAt.max(), "UTC", "可解析创建时间的最晚值")
    checks = [
        ("duplicate_rows", duplicate, "完全重复的多余记录数量，仅候选样本排除重复首条以外的记录"),
        ("numeric_invalid_rows", bad_num.any(axis=1), "至少一个非空数值字段无法解析或为无穷值；不等于缺失"),
        ("timestamp_invalid_rows", bad_dates.any(axis=1), "至少一个非空时间字段无法解析"),
        ("creation_or_expiry_missing_rows", missing[["createdAt", "expirationDate"]].any(axis=1), "创建或到期时间原始缺失"),
        ("base_missing_rows", missing.apiBasePrice_per_BTC_USDC, "原始 API Base 为空，未填补"),
        ("base_invalid_rows", bad_num.apiBasePrice_per_BTC_USDC | d.apiBasePrice_per_BTC_USDC.le(0), "非空基础价不可解析、非有限或不为正"),
        ("markup_missing_rows", missing.markup_per_BTC_USDC, "原始 markup 为空"),
        ("outcome_missing_rows", missing.realizedProfit_USDC, "原始结果字段为空，不当作零赔付"),
        ("outcome_invalid_rows", bad_num.realizedProfit_USDC | d.realizedProfit_USDC.lt(0), "非空结果不可解析、非有限或为负；负值需确认业务口径"),
        ("closed_missing_rows", missing.closedAt, "原始结算时间为空"),
        ("outcome_close_availability_mismatch_rows", d.realizedProfit_USDC.notna() ^ d.closedAt.notna(), "结果与结算时间仅一项可解析"),
        ("quantity_invalid_rows", ~quantity_valid, "数量缺失、非有限或不为正"),
        ("premium_input_invalid_rows", ~premium_valid | ~d.orderPrice_per_BTC_USDC.ge(0), "总保费或单位保费缺失、非有限、不可解析或为负"),
        ("direction_or_label_invalid_rows", ~direction_valid | ~label_valid, "方向不是0/1，或与Call/Put文字不一致"),
        ("contract_input_invalid_rows", ~tenor_valid | ~d.strikePrice.gt(0), "期限不是正整数秒或执行价不为正，含缺失"),
        ("closed_before_created_rows", closed_before, "实际结算早于创建，不直接修复时间"),
        ("expired_at_or_before_creation_rows", expired, "到期不晚于创建，剩余保护时间非正"),
        ("time_order_anomaly_rows", time_anomaly, "上述两类时间异常的并集，不重复计数"),
        ("closed_before_expiry_rows", d.closedAt.lt(d.expirationDate), "提前结算的描述统计，不自动视为错误"),
        ("nominal_duration_deviation_over_60s_rows", duration_deviation.abs().gt(60), "实际剩余期限与标称期限相差超过60秒；仅标记"),
        ("negative_markup_rows", d.markup_per_BTC_USDC.lt(0), "与提供方负markup过滤规则不符的记录"),
        ("technical_candidate_rows", technical, "通过README技术条件的候选订单，不代表业务语义已确认"),
        ("common_candidate_rows", common, "技术候选中基础价和markup有效、价格差额一致的共同候选"),
    ]
    for name, mask, definition in checks:
        m.count(name, mask, definition)
    m.count("normalized_price_identity_error_rows", identity_eligible & normalized_error.gt(1e-6),
            "总保费/数量与单位保费不一致，绝对容差1e-6；仅可计算记录",
            int(identity_eligible.sum()))
    m.count("markup_identity_error_rows", markup_identity_eligible & markup_error.gt(1e-6),
            "单位保费-基础价与markup不一致，绝对容差1e-6；仅三项完整记录",
            int(markup_identity_eligible.sum()))
    m.add("nominal_duration_deviation_median_seconds", duration_deviation.median(),
          "seconds", "到期-创建-标称期限的中位数；可解析记录")
    m.count("labeled_rows", labeled, "有限非负结果、有效保费和正数量；仍包含时间异常")
    nl = int(labeled.sum())
    m.count("zero_outcome_rows", labeled & d.realizedProfit_USDC.eq(0),
            "有结果记录中结果为0的数量", nl)
    m.count("positive_outcome_rows", labeled & d.realizedProfit_USDC.gt(0),
            "有结果记录中结果为正的数量", nl)
    payouts = d.loc[labeled, "realizedProfit_USDC"]
    payout_total = total(payouts)
    for name, value, definition in [
        ("labeled_recorded_premium_usdc", total(d.loc[labeled, "premiumAmount_USDC"]), "有结果记录的实际总保费，与结果金额使用相同样本"),
        ("labeled_outcome_usdc", payout_total, "有结果记录的结果金额总和，业务口径待确认"),
        ("common_recorded_premium_usdc", total(d.loc[common, "premiumAmount_USDC"]), "共同候选的实际总保费"),
        ("common_base_premium_usdc", total(d.loc[common, "apiBasePrice_per_BTC_USDC"] * d.loc[common, "quantity_BTC"]), "共同候选的基础价乘BTC数量总和，不是正式baseline得分"),
        ("common_outcome_usdc", total(d.loc[common, "realizedProfit_USDC"]), "共同候选的结果金额总和"),
    ]:
        m.add(name, value, "USDC", definition)
    top_n = math.ceil(nl * .01)
    m.add("top_1pct_outcome_rows", top_n, "rows", "有结果记录数的1%向上取整")
    share = float(payouts.nlargest(top_n).sum() / payout_total) if nl and payout_total > 0 else None
    m.add("top_1pct_outcome_share", share, "fraction", "最高赔付1%记录占全部有结果记录赔付总额的比例")
    multiplier = d.loc[common, "orderPrice_per_BTC_USDC"] / d.loc[common, "apiBasePrice_per_BTC_USDC"]
    m.add("common_recorded_to_base_price_ratio_median", multiplier.median(), "multiple", "共同候选的实际单位保费/API Base中位数，并非需求弹性")
    return OrderAudit(m.frame(), _segments(d, cohorts, missing, time_anomaly), d, cohorts)


def _segments(d, cohorts, missing, time_anomaly):
    definitions = {
        "month": d.createdAt.dt.strftime("%Y-%m").fillna("unknown"),
        "direction": d.optionType.map({0: "Call", 1: "Put"}).fillna("unknown"),
        "duration_hours": (d.productType / 3600).map(lambda v: f"{v:g}" if pd.notna(v) else "unknown"),
    }
    records = []
    for dimension, groups in definitions.items():
        for group in sorted(groups.unique()):
            mask = groups.eq(group)
            labeled = mask & cohorts.labeled
            n, nl = int(mask.sum()), int(labeled.sum())
            records.append(dict(
                dimension=dimension, group=group, rows=n, share_of_all_rows=n / len(d),
                base_missing_rows=int((mask & missing.apiBasePrice_per_BTC_USDC).sum()),
                base_missing_rate=float(missing.loc[mask, "apiBasePrice_per_BTC_USDC"].mean()),
                outcome_missing_rows=int((mask & missing.realizedProfit_USDC).sum()),
                time_order_anomaly_rows=int((mask & time_anomaly).sum()),
                technical_candidate_rows=int((mask & cohorts.technical).sum()),
                common_candidate_rows=int((mask & cohorts.common).sum()),
                labeled_rows=nl,
                positive_outcome_rate=ratio(int((labeled & d.realizedProfit_USDC.gt(0)).sum()), nl),
                labeled_quantity_btc=total(d.loc[labeled, "quantity_BTC"]),
                labeled_recorded_premium_usdc=total(d.loc[labeled, "premiumAmount_USDC"]),
                labeled_outcome_usdc=total(d.loc[labeled, "realizedProfit_USDC"]),
            ))
    return pd.DataFrame(records)
