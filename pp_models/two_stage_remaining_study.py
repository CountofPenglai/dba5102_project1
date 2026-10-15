"""Execute confirmed two-stage branches 2--5, stopping before post-development work."""
import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from pp_data.config import DEFAULT_OUTPUT, PROJECT
from pp_data.features import BASE_FEATURE
from .baselines import RollingMean
from .calibration_models import CalibrationModel
from .directional_features import augment_pool, read_market_prefix
from .directional_study import install_development_guard
from .rolling_protocol import load_protocol, rolling_batches, protocol_coverage
from .runs import fingerprint, run_destination
from .scoped_data import load_validation_pool
from .study import study_snapshot
from .study_storage import atomic_json, file_digest
from .two_stage_ablation import HistoricalPayoutProbability
from .two_stage_comparisons import build_comparison_registry
from .two_stage_error_analysis import error_tables
from .two_stage_robustness import (build_paired_daily_statistics, build_probability_daily_statistics,
                                   calendar_influence, paired_calendar_bootstrap,
                                   paired_calendar_bootstrap_brier)
from .two_stage_severity import (DIRECTIONAL_INPUTS, DIRECTIONAL_NAMES, SEVERITY_INPUTS,
                                 SEVERITY_NAMES, TwoStageSeverity, directional_acceleration)
from .two_stage_shrinkage import WEIGHTS, blend_predictions, combine_direction_weights, select_direction_weight
from .two_stage_study import aggregate, metric_values, probability_frame, tail_context

CONFIGS = {
    "directional": PROJECT / "config/two_stage_directional_v1.json",
    "shrinkage": PROJECT / "config/two_stage_shrinkage_v1.json",
    "ablation": PROJECT / "config/two_stage_ablation_v1.json",
    "robustness": PROJECT / "config/two_stage_robustness_v1.json",
}


def load_configs():
    values = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in CONFIGS.items()}
    if (values["directional"] != {"version": "two_stage_directional_v1",
            "protocol": "config/validation_protocol_v2.json", "candidates": {"g2a": 1.0, "g2b": 10.0},
            "paired_controls": {"g2a": "g1a", "g2b": "g1b"}, "minimum_positive_orders": 40,
            "minimum_creation_days": 5, "standardized_clip": 3.0, "risk_tolerance_vs_base": 1.02}
            or values["shrinkage"]["weights"] != [1.0, .5, .25, 0.0]
            or values["ablation"]["tenor_smoothing"] != 30
            or values["robustness"]["block_days"] != [3, 7, 14]
            or values["robustness"]["repeats"] != 2000
            or values["robustness"]["seed"] != 20261012
            or values["robustness"]["minimum_valid_repeats"] != 1900):
        raise ValueError("Unsupported remaining-branch configuration")
    return values


def directional_risk(X):
    acceleration = directional_acceleration(X)
    absolute = pd.Series(np.where(X.option_type.eq(0), X.up_vol_60m, X.down_vol_60m), index=X.index)
    return acceleration, absolute


def evaluate_components(pool, protocol, progress=None):
    predictions, components, parameters, cohorts = [], [], [], []
    for day, history, today in rolling_batches(pool, protocol):
        Xh_prob, Xq_prob = probability_frame(pool.X.loc[history]), probability_frame(pool.X.loc[today])
        y = pool.y.loc[history]
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error")
            hurdle = CalibrationModel("hurdle_base").fit(Xh_prob, y, pool.metadata.loc[history, "created_at_utc"])
            logistic, _ = hurdle.predict_components(Xq_prob)
            direction_scaled = CalibrationModel("direction_scaled").fit(Xh_prob, y)
            grouped = RollingMean(True, 30).fit(pool.X.loc[history], y)
        probability_models = {}
        for level in ("direction_frequency", "direction_tenor_frequency"):
            model = HistoricalPayoutProbability(level).fit(
                pool.X.loc[history, ["option_type", "product_duration_seconds"]], y)
            probability_models[level] = model.predict_probability(
                pool.X.loc[today, ["option_type", "product_duration_seconds"]])
        threshold, source = tail_context(pool, history, today)
        acceleration, absolute = directional_risk(pool.X.loc[today])
        historical_acceleration, historical_absolute = directional_risk(pool.X.loc[history])
        low_absolute = pd.Series(False, index=today)
        for side in (0, 1):
            hm = pool.X.loc[history, "option_type"].eq(side)
            qm = pool.X.loc[today, "option_type"].eq(side)
            low_absolute.loc[qm] = absolute.loc[qm].le(float(historical_absolute.loc[hm].median()))
        current = pd.DataFrame({"day": day, "month": day.strftime("%Y-%m"), "actual": pool.y.loc[today],
            "quantity": pool.X.loc[today, "quantity_btc"], "side": pool.X.loc[today, "option_type"].astype(int),
            "tenor": pool.X.loc[today, "product_duration_seconds"].astype(int),
            "tail_group": np.where(threshold.isna(), "unavailable",
                np.where(pool.y.loc[today].gt(threshold), "tail", "body")),
            "acceleration_group": np.where(acceleration.gt(0), "positive", "nonpositive"),
            "relative_positive_low_absolute": acceleration.gt(0) & low_absolute,
            "absolute_risk_group": np.where(low_absolute, "low", "high"),
            "position_group": np.where(((1-2*pool.X.loc[today, "option_type"])*pool.X.loc[today, "log_moneyness"]).gt(0),
                                       "in_payout_direction", "at_or_opposite_direction"),
            "remaining_bucket": pd.cut(pool.X.loc[today, "remaining_seconds"], [0, 3600, 14400, np.inf],
                                       labels=["up_to_1h", "1_to_4h", "over_4h"]).astype(str)}, index=today)
        controls = {"official_raw": pool.X.loc[today, BASE_FEATURE].to_numpy(float),
                    "grouped_mean": grouped.predict_raw(pool.X.loc[today]),
                    "direction_scaled": direction_scaled.predict_raw(Xq_prob)}
        for name, values in controls.items():
            predictions.append(current.assign(model=name, prediction=values))
        severities = {}
        for name in SEVERITY_NAMES + DIRECTIONAL_NAMES:
            columns = DIRECTIONAL_INPUTS if name in DIRECTIONAL_NAMES else SEVERITY_INPUTS
            model = TwoStageSeverity(name).fit(pool.X.loc[history, columns], y,
                                                pool.metadata.loc[history, "created_at_utc"])
            severities[name] = model.predict_severity(pool.X.loc[today, columns])
            predictions.append(current.assign(model=name, prediction=logistic * severities[name]))
            parameters.append({"day": day.isoformat(), "month": day.strftime("%Y-%m"), "model": name,
                               **model.diagnostics()})
        components.append(current.assign(p_logistic=logistic,
            p_direction=probability_models["direction_frequency"],
            p_direction_tenor=probability_models["direction_tenor_frequency"],
            s0=severities["h0"], s_g0=severities["g0"], s_g1a=severities["g1a"], s_g1b=severities["g1b"],
            s_g2a=severities["g2a"], s_g2b=severities["g2b"], base=pool.X.loc[today, BASE_FEATURE]))
        cohorts.append({"day": day.isoformat(), "training_orders": len(history), "scoring_orders": len(today),
                        "score_ids_sha256": fingerprint([str(i) for i in today]),
                        "tail_threshold_fallbacks": int(source.eq("direction_fallback").sum()),
                        "shared_probability_sha256": fingerprint([float(v) for v in logistic])})
        if progress and (day.day == 1 or day.day % 10 == 0):
            progress(f"{day.date()}: branches 2--4 components available")
    return pd.concat(predictions), pd.concat(components), pd.DataFrame(parameters), pd.DataFrame(cohorts)


def candidate_risk(summary, models, tolerance=1.02):
    rows = []
    for model in models:
        passed = []
        for side in (0, 1):
            direction, tail = str(side), f"{side}|tail"
            def one(which, group, value):
                result = summary.query("month == 'all' and model == @which and group == @group and value == @value")
                if len(result) != 1:
                    raise ValueError("Missing risk comparison group")
                return result.iloc[0]
            own, base = one(model, "direction", direction), one("official_raw", "direction", direction)
            own_tail, base_tail = one(model, "direction_tail", tail), one("official_raw", "direction_tail", tail)
            checks = {"tail_rmse_pass": own_tail.rmse_per_btc <= tolerance * base_tail.rmse_per_btc,
                      "gross_underprediction_pass": own.gross_underprediction_usdc
                                                    <= tolerance * base.gross_underprediction_usdc,
                      "absolute_total_bias_pass": own.absolute_total_bias_usdc <= base.absolute_total_bias_usdc}
            direction_pass = bool(all(checks.values()))
            passed.append(direction_pass)
            rows.append({"model": model, "direction": side,
                         "tail_rmse_ratio_vs_base": own_tail.rmse_per_btc / base_tail.rmse_per_btc,
                         "gross_underprediction_ratio_vs_base": own.gross_underprediction_usdc/base.gross_underprediction_usdc,
                         "absolute_total_bias_ratio_vs_base": own.absolute_total_bias_usdc/base.absolute_total_bias_usdc,
                         **checks, "direction_pass": direction_pass})
        for row in rows[-2:]:
            row["model_all_directions_pass"] = bool(all(passed))
    return pd.DataFrame(rows)


def branch2(frame):
    models = ["official_raw", "grouped_mean", "h0", "g0", "g1a", "g1b", "g2a", "g2b"]
    selected = frame.loc[frame.model.isin(models)]
    summary, daily, macro = aggregate(selected)
    risk = candidate_risk(summary, ["g2a", "g2b"])
    lookup, risk_lookup = macro.set_index("model"), risk.groupby("model").model_all_directions_pass.first()
    comparisons, decisions = [], []
    for candidate, reference in (("g2a", "g1a"), ("g2b", "g1b")):
        comparisons.append({"candidate": candidate, "reference": reference,
            "rmse_change": lookup.loc[candidate, "macro_rmse_per_btc"] / lookup.loc[reference, "macro_rmse_per_btc"] - 1,
            "mae_change": lookup.loc[candidate, "macro_mae_per_btc"] / lookup.loc[reference, "macro_mae_per_btc"] - 1})
        accuracy = (lookup.loc[candidate, "macro_rmse_per_btc"] < lookup.loc["h0", "macro_rmse_per_btc"]
                    and lookup.loc[candidate, "macro_mae_per_btc"] <= lookup.loc["h0", "macro_mae_per_btc"])
        paired = (lookup.loc[candidate, "macro_rmse_per_btc"] < lookup.loc[reference, "macro_rmse_per_btc"]
                  and lookup.loc[candidate, "macro_mae_per_btc"] <= lookup.loc[reference, "macro_mae_per_btc"])
        decisions.append({"model": candidate, "accuracy_vs_h0_pass": accuracy,
                          "paired_increment_pass": paired, "risk_pass": bool(risk_lookup.loc[candidate]),
                          "retained": bool(accuracy and paired and risk_lookup.loc[candidate])})
    diagnostics = []
    for model in ("g1a", "g2a", "g1b", "g2b"):
        part = selected.loc[selected.model.eq(model)]
        for side in (0, 1):
            for group, mask in (("positive_acceleration", part.acceleration_group.eq("positive")),
                                ("nonpositive_acceleration", part.acceleration_group.eq("nonpositive")),
                                ("positive_low_absolute", part.relative_positive_low_absolute)):
                sub = part.loc[part.side.eq(side) & mask]
                if sub.empty:
                    diagnostics.append({"model": model, "direction": side, "slice": group, "orders": 0})
                else:
                    diagnostics.append({"model": model, "direction": side, "slice": group,
                                        "creation_days": sub.day.nunique(), **metric_values(sub)})
    return {"summary": summary, "daily": daily, "macro_monthly": macro, "risk_constraints": risk,
            "paired_increment": pd.DataFrame(comparisons), "selection": pd.DataFrame(decisions),
            "diagnostic_slices": pd.DataFrame(diagnostics)}


def coefficient_summary(parameters):
    rows = []
    coefficient_columns = [column for column in parameters if column.endswith("_coefficient")]
    for model, part in parameters.loc[parameters.model.isin(["g1a", "g1b", "g2a", "g2b"])].groupby("model"):
        for column in coefficient_columns:
            values = part[column].dropna()
            if len(values):
                rows.append({"model": model, "parameter": column, "days": len(values),
                             "minimum": values.min(), "median": values.median(), "maximum": values.max()})
        for side in ("call", "put"):
            reasons = part[f"{side}_fallback_reason"].fillna("none").value_counts()
            for reason, count in reasons.items():
                rows.append({"model": model, "parameter": f"{side}_fallback:{reason}",
                             "days": int(count), "minimum": np.nan, "median": np.nan, "maximum": np.nan})
    return pd.DataFrame(rows)


def direction_metrics(part):
    body, tail = part.loc[part.tail_group.eq("body")], part.loc[part.tail_group.eq("tail")]
    if body.empty or tail.empty:
        raise ValueError("Direction tail/body metrics unavailable")
    all_metrics, body_metrics, tail_metrics = metric_values(part), metric_values(body), metric_values(tail)
    return {"all_mae": all_metrics["mae_per_btc"], "body_mae": body_metrics["mae_per_btc"],
            "tail_rmse": tail_metrics["rmse_per_btc"],
            "gross_underprediction": all_metrics["gross_underprediction_usdc"],
            "absolute_total_bias": all_metrics["absolute_total_bias_usdc"]}


def branch3(frame, branch1_decisions, branch2_tables):
    # R enters on predictive increment alone; direction risk is checked after
    # blending below. Raw risk failures may be repaired by shrinkage.
    second = branch2_tables["selection"]
    retained = second.loc[second.accuracy_vs_h0_pass & second.paired_increment_pass, "model"].tolist()
    retained += branch1_decisions.loc[branch1_decisions.replacement_accuracy_pass, "model"].tolist()
    if retained:
        macro = branch2_tables["macro_monthly"].set_index("model")
        raw_model = min(retained, key=lambda name: macro.loc[name, "macro_rmse_per_btc"])
    else:
        raw_model = "h0"
    base = frame.loc[frame.model.eq("official_raw")].copy()
    raw = frame.loc[frame.model.eq(raw_model)].copy()
    if not base.index.equals(raw.index):
        raise ValueError("Shrinkage source rows differ")
    base_series = pd.Series(base.prediction.to_numpy(), index=base.index)
    raw_series = pd.Series(raw.prediction.to_numpy(), index=raw.index)
    metrics_rows, choices = [], {}
    for side in (0, 1):
        mask = base.side.eq(side).to_numpy()
        base_part = base.loc[mask]
        base_metrics = direction_metrics(base_part)
        by_weight = {}
        for weight in WEIGHTS:
            prediction = blend_predictions(base_series.loc[mask], raw_series.loc[mask], weight)
            part = base_part.copy(); part["prediction"] = prediction.to_numpy()
            by_weight[weight] = direction_metrics(part)
            metrics_rows.append({"direction": side, "record_type": "candidate", "weight": weight,
                                 **by_weight[weight]})
        tail = base_part.loc[base_part.tail_group.eq("tail")]
        decision = select_direction_weight(by_weight, base_metrics, len(tail), tail.day.nunique())
        choices[side] = decision["weight"]
        decision.update(direction=side, tail_orders=len(tail), tail_creation_days=tail.day.nunique())
        selected_weight = decision.pop("weight")
        metrics_rows.append({"direction": side, "record_type": "selection",
                             "selected_weight": selected_weight, **decision})
    final_prediction = combine_direction_weights(base_series, raw_series, base.side, choices)
    frames = [base.assign(model="official_raw"), frame.loc[frame.model.eq("h0")].copy(), raw.assign(model="raw_R")]
    for weight, name in ((.5, "uniform_0_5"), (.25, "uniform_0_25")):
        frames.append(base.assign(model=name, prediction=blend_predictions(base_series, raw_series, weight).to_numpy()))
    frames.append(base.assign(model="final_combination", prediction=final_prediction.to_numpy()))
    result = pd.concat(frames)
    summary, daily, macro = aggregate(result)
    lookup = macro.set_index("model")
    overall = summary.query("month == 'all' and group == 'overall'").set_index("model")
    selected_rows = [r for r in metrics_rows if r.get("record_type") == "selection"]
    final_pass = (lookup.loc["final_combination", "macro_rmse_per_btc"] < lookup.loc["h0", "macro_rmse_per_btc"]
                  and lookup.loc["final_combination", "macro_mae_per_btc"] <= lookup.loc["h0", "macro_mae_per_btc"]
                  and all(r["passed"] for r in selected_rows)
                  and overall.loc["final_combination", "absolute_total_bias_usdc"]
                      <= overall.loc["official_raw", "absolute_total_bias_usdc"])
    decision = pd.DataFrame([{"raw_model": raw_model, "call_weight": choices[0], "put_weight": choices[1],
                              "final_adopted": bool(final_pass),
                              "reason": "all_conditions_pass" if final_pass else "final_conditions_failed"}])
    return {"summary": summary, "daily": daily, "macro_monthly": macro,
            "direction_weight_metrics": pd.DataFrame(metrics_rows), "selection": decision}, result


def probability_metrics(components):
    rows, bins = [], np.linspace(0, 1, 11)
    actual = components.actual.gt(0).to_numpy()
    for name, column in (("direction_frequency", "p_direction"),
                         ("direction_tenor_frequency", "p_direction_tenor"),
                         ("logistic_probability", "p_logistic")):
        for month in ["all"] + sorted(components.month.unique()):
            chosen = components if month == "all" else components.loc[components.month.eq(month)]
            for side in ("all", 0, 1):
                part = chosen if side == "all" else chosen.loc[chosen.side.eq(side)]
                p, y = part[column].to_numpy(float), part.actual.gt(0).to_numpy()
                rows.append({"probability": name, "month": month, "direction": side, "orders": len(part),
                             "positive_orders": int(y.sum()), "mean_probability": float(p.mean()),
                             "actual_positive_rate": float(y.mean()), "brier": float(np.mean((p-y)**2)),
                             "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1-1e-15), labels=[False, True]))})
    calibration = []
    for name, column in (("direction_frequency", "p_direction"),
                         ("direction_tenor_frequency", "p_direction_tenor"),
                         ("logistic_probability", "p_logistic")):
        groups = pd.cut(components[column], bins, include_lowest=True, duplicates="drop")
        for group, part in components.groupby(groups, observed=True):
            calibration.append({"probability": name, "bin": str(group), "orders": len(part),
                                "mean_probability": part[column].mean(),
                                "actual_positive_rate": part.actual.gt(0).mean()})
    return pd.DataFrame(rows), pd.DataFrame(calibration)


def branch4(components, branch3_tables, shrink_frame):
    raw_model = branch3_tables["selection"].raw_model.iloc[0]
    severity_column = {"h0": "s0", "g0": "s_g0", "g1a": "s_g1a", "g1b": "s_g1b",
                       "g2a": "s_g2a", "g2b": "s_g2b"}[raw_model]
    matrix = []
    severity_specs = [("s0", "s0")]
    if severity_column != "s0":
        severity_specs.append(("sR", severity_column))
    for severity_name, column in severity_specs:
        for probability_name, pcol in (("direction_frequency", "p_direction"),
                                       ("direction_tenor_frequency", "p_direction_tenor"),
                                       ("logistic_probability", "p_logistic")):
            name = "h0" if probability_name == "logistic_probability" and severity_name == "s0" else (
                   raw_model if probability_name == "logistic_probability" and severity_name == "sR" else
                   f"{probability_name}_{severity_name}")
            part = components.copy()
            part["model"] = name
            part["prediction"] = part[pcol] * part[column]
            matrix.append(part)
    frame = pd.concat(matrix)
    summary, daily, macro = aggregate(frame)
    probabilities, calibration = probability_metrics(components)
    positive = components.loc[components.actual.gt(0)]
    amount_rows = []
    for name, column in severity_specs:
        error = positive[column] - positive.actual
        amount_rows.append({"severity": name, "orders": len(positive),
                            "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error)))})
    paired = []
    lookup = macro.set_index("model")
    def contrast(layer, candidate, reference):
        paired.append({"layer": layer, "candidate": candidate, "reference": reference,
                       "rmse_delta": lookup.loc[candidate, "macro_rmse_per_btc"]-lookup.loc[reference, "macro_rmse_per_btc"],
                       "mae_delta": lookup.loc[candidate, "macro_mae_per_btc"]-lookup.loc[reference, "macro_mae_per_btc"]})
    for severity_name, _ in severity_specs:
        logistic = "h0" if severity_name == "s0" else raw_model
        for level in ("direction_frequency", "direction_tenor_frequency"):
            contrast("probability", logistic, f"{level}_{severity_name}")
    if severity_column != "s0":
        contrast("amount", raw_model, "h0")
        for level in ("direction_frequency", "direction_tenor_frequency"):
            contrast("amount", f"{level}_sR", f"{level}_s0")
    final = branch3_tables["selection"].iloc[0]
    final_macro = branch3_tables["macro_monthly"].set_index("model")
    paired.append({"layer": "shrinkage", "candidate": "final_combination", "reference": raw_model,
                   "rmse_delta": final_macro.loc["final_combination", "macro_rmse_per_btc"]
                                 - final_macro.loc["raw_R", "macro_rmse_per_btc"],
                   "mae_delta": final_macro.loc["final_combination", "macro_mae_per_btc"]
                                - final_macro.loc["raw_R", "macro_mae_per_btc"]})
    return {"summary": summary, "daily": daily, "macro_monthly": macro,
            "probability_metrics": probabilities, "probability_calibration": calibration,
            "positive_severity_metrics": pd.DataFrame(amount_rows), "paired_components": pd.DataFrame(paired),
            "matrix_definition": pd.DataFrame([{"model": f.model.iloc[0]} for f in matrix])}, frame


def monthly_direction_deltas(daily):
    rows = []
    for (candidate, reference), pair in daily.groupby(["candidate", "reference"]):
        for direction in ("all", "0", "1"):
            for month in ("2026-04", "2026-05", "2026-06"):
                part = pair.loc[pair.direction.eq(direction) & pair.month.eq(month)]
                n = part.orders.sum()
                rows.append({"candidate": candidate, "reference": reference, "direction": direction, "month": month,
                             "orders": int(n), "creation_days": len(part),
                             "tail_orders": int(part.tail_orders.sum()),
                             "tail_creation_days": int(part.tail_orders.gt(0).sum()),
                             "rmse_delta": np.sqrt(part.candidate_sse.sum()/n)-np.sqrt(part.reference_sse.sum()/n) if n else np.nan,
                             "mae_delta": (part.candidate_sae.sum()-part.reference_sae.sum())/n if n else np.nan})
    return pd.DataFrame(rows)


def branch5(base_frame, components, branch1_decisions, branch2_tables, branch3_tables, ablation_frame, branch4_tables):
    macro = pd.concat([branch2_tables["macro_monthly"], branch4_tables["macro_monthly"],
                       branch3_tables["macro_monthly"].query("model == 'final_combination'")]).drop_duplicates("model")
    registry = build_comparison_registry(macro, branch1_decisions, branch2_tables["selection"],
                                         branch3_tables["selection"].iloc[0], branch4_tables["paired_components"])
    # Registration is complete before any interval is computed. One canonical
    # copy per model prevents duplicate order IDs when controls overlap.
    sources = [base_frame]
    sources.append(ablation_frame.loc[~ablation_frame.model.isin(base_frame.model.unique())])
    sources.append(branch3_tables["prediction_frame"].query("model == 'final_combination'"))
    combined = pd.concat(sources)
    pairs = list(registry.loc[registry.included, ["candidate", "reference"]].itertuples(index=False, name=None))
    daily = build_paired_daily_statistics(combined, pairs)
    overall = daily.loc[daily.direction.eq("all")]
    influence = calendar_influence(overall)
    bootstrap = pd.concat([paired_calendar_bootstrap(overall, block) for block in (3, 7, 14)], ignore_index=True)
    probability_parts = []
    for column, reference in (("p_direction", "direction_frequency"),
                              ("p_direction_tenor", "direction_tenor_frequency")):
        probability_parts.append(build_probability_daily_statistics(
            components, "p_logistic", column, "logistic_probability", reference))
    probability_daily = pd.concat(probability_parts, ignore_index=True)
    probability_overall = probability_daily.loc[probability_daily.direction.eq("all")]
    probability_intervals = pd.concat(
        [paired_calendar_bootstrap_brier(probability_overall, block) for block in (3, 7, 14)], ignore_index=True)
    bootstrap = pd.concat([bootstrap, probability_intervals], ignore_index=True)
    probability_monthly = []
    for (candidate, reference, direction, month), part in probability_daily.groupby(["candidate", "reference", "direction", "month"]):
        n = part.orders.sum()
        probability_monthly.append({"candidate": candidate, "reference": reference,
            "direction": direction, "month": month,
            "orders": int(n), "creation_days": len(part),
            "brier_delta": (part.candidate_brier_sum.sum()-part.reference_brier_sum.sum())/n})
    return {"comparison_registry": registry,
            "monthly_direction_deltas": monthly_direction_deltas(daily),
            "probability_monthly_direction_deltas": pd.DataFrame(probability_monthly),
            "calendar_influence": influence, "paired_intervals": bootstrap}


def write_branch(root, run_id, prefix, tables, report_lines, provenance):
    path = run_destination(root, f"{prefix}_{run_id}")
    path.mkdir(parents=True)
    for name in ("metrics", "reports", "provenance"):
        (path / name).mkdir()
    files = []
    for name, frame in tables.items():
        if name == "prediction_frame":
            continue
        destination = path / f"metrics/{name}.csv"
        frame.to_csv(destination, index=False, float_format="%.17g")
        files.append(destination)
    report = path / "reports/evaluation.md"
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    files.append(report)
    source = path / "provenance/source.json"
    atomic_json(source, provenance)
    files.append(source)
    atomic_json(path / "provenance/manifest.json", {"status": "complete",
                "files": {str(p.relative_to(path)): file_digest(p) for p in files}})
    return path


def load_branch1_reference(path, source):
    path = Path(path)
    manifest = json.loads((path / "provenance/manifest.json").read_text(encoding="utf-8"))
    names = ["metrics/selection.csv", "metrics/macro_monthly.csv", "metrics/daily_cohorts.csv", "provenance/source.json"]
    if manifest.get("status") != "complete":
        raise ValueError("Branch 1 must be complete before remaining studies")
    for name in names:
        if manifest.get("files", {}).get(name) != file_digest(path / name):
            raise ValueError("Branch-1 reference checksum mismatch")
    provenance = json.loads((path / "provenance/source.json").read_text(encoding="utf-8"))
    if provenance.get("development") != source or provenance.get("test_csv_read") is not False:
        raise ValueError("Branch-1 code/data/environment or development scope differs")
    tables = {name: pd.read_csv(path / f"metrics/{name}.csv", float_precision="round_trip")
              for name in ("selection", "macro_monthly", "daily_cohorts")}
    tables["identity"] = {"path": str(path.resolve()), "manifest_sha256": file_digest(path / "provenance/manifest.json")}
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run"])
    parser.add_argument("--run-id", default="20261014")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--market-csv", type=Path, required=False)
    parser.add_argument("--branch1-run", type=Path, help="Completed branch-1 run with matching code/data")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "outputs/model_studies")
    args = parser.parse_args(argv)
    configs, protocol = load_configs(), load_protocol()
    plan = {"branches": [2, 3, 4, 5], "configs": configs, "test_used": False,
            "post_development_started": False, "parameters_tuned": False}
    if args.action == "plan":
        print(json.dumps(plan, indent=2)); return 0
    if args.market_csv is None or args.branch1_run is None:
        parser.error("run requires --market-csv and --branch1-run")
    started = perf_counter(); guard = install_development_guard(args.data_dir)
    source = study_snapshot(args.data_dir)
    config_hashes = {name: file_digest(path) for name, path in CONFIGS.items()}
    reference = load_branch1_reference(args.branch1_run, source)
    market_stat = args.market_csv.stat()
    pool = load_validation_pool({"cohort": "benchmark", "use_base": True}, args.data_dir)
    coverage = protocol_coverage(pool, protocol)
    if coverage.scoring_orders.sum() != 2953 or coverage.scoring_days.sum() != 72:
        raise ValueError("Remaining branches require the fixed common cohort")
    market, market_info = read_market_prefix(args.market_csv, protocol["validation_end"])
    pool, feature_info = augment_pool(pool, market)
    frame, components, parameters, cohorts = evaluate_components(
        pool, protocol, progress=lambda line: print(line, flush=True))
    branch2_tables = branch2(frame)
    branch2_tables["coefficient_summary"] = coefficient_summary(parameters)
    branch1_decisions = reference["selection"]
    old = reference["macro_monthly"].set_index("model")
    new = branch2_tables["macro_monthly"].set_index("model")
    shared = old.index.intersection(new.index)
    if not np.allclose(old.loc[shared, ["macro_rmse_per_btc", "macro_mae_per_btc"]],
                       new.loc[shared, ["macro_rmse_per_btc", "macro_mae_per_btc"]], rtol=1e-10, atol=1e-10):
        raise ValueError("Rebuilt branch-1 scores differ from the registered reference")
    if (cohorts.day.tolist() != reference["daily_cohorts"].day.tolist()
            or cohorts.score_ids_sha256.tolist() != reference["daily_cohorts"].score_ids_sha256.tolist()):
        raise ValueError("Rebuilt branch-1 scoring IDs differ")
    branch3_tables, shrink_frame = branch3(frame, branch1_decisions, branch2_tables)
    branch3_tables["prediction_frame"] = shrink_frame
    branch4_tables, ablation_frame = branch4(components, branch3_tables, shrink_frame)
    branch5_tables = branch5(frame, components, branch1_decisions, branch2_tables, branch3_tables, ablation_frame, branch4_tables)
    residual_frame = pd.concat([frame, shrink_frame.query("model == 'final_combination'")])
    residuals = error_tables(residual_frame, components, pool.metadata, branch3_tables["selection"].raw_model.iloc[0])
    current_stat = args.market_csv.stat()
    if (study_snapshot(args.data_dir) != source
            or config_hashes != {name: file_digest(path) for name, path in CONFIGS.items()}
            or (market_stat.st_size, market_stat.st_mtime_ns) != (current_stat.st_size, current_stat.st_mtime_ns)
            or file_digest(args.branch1_run / "provenance/manifest.json") != reference["identity"]["manifest_sha256"]):
        raise ValueError("Study sources changed during execution")
    provenance = {"development": source, "branch1_reference": reference["identity"], "coverage": coverage.to_dict("records"), "market_prefix": market_info,
                  "directional_features": feature_info,
                  "config_sha256": config_hashes,
                  "test_csv_read": False, "blocked_test_opens": guard["blocked_test_opens"],
                  "post_development_started": False, "july_used": False}
    outputs = []
    b2macro = branch2_tables["macro_monthly"].set_index("model")
    outputs.append(write_branch(args.output_root, args.run_id, "two_stage_directional", branch2_tables,
        ["# 分支2：方向波动信号", "",
         f"G2a相对G1a RMSE变化：{b2macro.loc['g2a','macro_rmse_per_btc']/b2macro.loc['g1a','macro_rmse_per_btc']-1:.3%}。",
         f"G2b相对G1b RMSE变化：{b2macro.loc['g2b','macro_rmse_per_btc']/b2macro.loc['g1b','macro_rmse_per_btc']-1:.3%}。",
         "采用结论以metrics/selection.csv为准；未启用收缩之外的新搜索，未读取7月。"], provenance | {"branch": 2}))
    decision3 = branch3_tables["selection"].iloc[0]
    outputs.append(write_branch(args.output_root, args.run_id, "two_stage_shrinkage", branch3_tables,
        ["# 分支3：收缩与方向回退", "",
         f"固定原始配方R：{decision3.raw_model}；Call权重：{decision3.call_weight:g}；Put权重：{decision3.put_weight:g}。",
         f"最终组合采用：{'是' if decision3.final_adopted else '否'}（{decision3.reason}）。",
         "未搜索16种组合；未启动正式冻结或7月。"], provenance | {"branch": 3}))
    probability_all = branch4_tables["probability_metrics"].query("month == 'all' and direction == 'all'").set_index("probability")
    outputs.append(write_branch(args.output_root, args.run_id, "two_stage_ablation", branch4_tables,
        ["# 分支4：概率与金额消融", "",
         f"方向×期限频率Brier：{probability_all.loc['direction_tenor_frequency','brier']:.6f}。",
         f"Logistic Brier：{probability_all.loc['logistic_probability','brier']:.6f}。",
         "概率、正金额和最终输出分别报告；本分支不重新选择主模型。"], provenance | {"branch": 4}))
    primary = branch5_tables["paired_intervals"].query("block_days == 7 and direction == 'all'")
    outputs.append(write_branch(args.output_root, args.run_id, "two_stage_robustness", branch5_tables,
        ["# 分支5：稳定性与不确定性", "",
         f"固定登记比较：{branch5_tables['comparison_registry'].included.sum()}项；7日块主区间：{len(primary)}项指标。",
         "区间仅为开发期探索证据；未据区间重新选择模型，未启动后期阶段。"], provenance | {"branch": 5}))
    outputs.append(write_branch(args.output_root, args.run_id, "two_stage_errors", residuals,
        ["# 两阶段模型误差诊断", "", "误差定义为预测减实际；正值高估、负值低估。",
         "定价时分组与事后赔付分组分别报告。历史误差只取此前7天创建且当日之前已结算订单，至少20笔/3个创建日。",
         "这些是开发期描述性规律；未训练误差修正器，未重新调参，未启用7月。"], provenance | {"branch": "error_diagnostics"}))
    execution = {"outputs": [str(path) for path in outputs], "elapsed_seconds": perf_counter()-started,
                 "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                 /(1024**2 if sys.platform == "darwin" else 1024),
                 "scoring_orders": int(coverage.scoring_orders.sum()), "scoring_days": int(coverage.scoring_days.sum()),
                 "july_used": False, "post_development_started": False}
    print(json.dumps(execution, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
