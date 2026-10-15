"""Fixed branch-1 two-stage severity study: python -m pp_models.two_stage_study plan|run."""
import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
import warnings
from zipfile import ZIP_DEFLATED, ZipFile
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_data.config import DEFAULT_OUTPUT, PROJECT
from pp_data.features import BASE_FEATURE
from .baselines import RollingMean
from .calibration_models import CalibrationModel, MODEL_INPUTS
from .directional_study import install_development_guard
from .evaluation import key_metrics
from .rolling_protocol import load_protocol, rolling_batches, protocol_coverage, PROTOCOL_PATH
from .runs import fingerprint, run_destination
from .scoped_data import load_validation_pool
from .study import study_snapshot
from .study_storage import atomic_json, file_digest
from .two_stage_severity import SEVERITY_INPUTS, SEVERITY_NAMES, TwoStageSeverity

CONFIG_PATH = PROJECT / "config/two_stage_severity_v1.json"
MODEL_NAMES = ("official_raw", "grouped_mean", "direction_scaled") + SEVERITY_NAMES


def load_settings(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "version": "two_stage_severity_v1", "protocol": "config/validation_protocol_v2.json",
        "models": list(MODEL_NAMES), "features": ["log_remaining_seconds", "aligned_volatility_position"],
        "minimum_positive_orders": 40, "minimum_creation_days": 5, "standardized_clip": 3.0,
        "aligned_position_clip": 5.0, "tail_quantile": .9, "tail_minimum_history": 30,
        "risk_tolerance_vs_base": 1.02, "group_smoothing": 30, "hurdle_logistic_C": .1,
    }
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported fixed two-stage severity settings")
    candidates = config.get("severity_candidates", {})
    if (list(candidates) != list(SEVERITY_NAMES)
            or candidates["h0"] != {"kind": "positive_base_least_squares", "alpha": None}
            or candidates["g0"] != {"kind": "positive_ratio_mean", "alpha": None}
            or candidates["g1a"] != {"kind": "gamma_log_link", "alpha": 1.0}
            or candidates["g1b"] != {"kind": "gamma_log_link", "alpha": 10.0}):
        raise ValueError("Severity candidate grid changed")
    selection = config.get("selection", {})
    if selection != {"macro_rmse_strictly_below_h0": True, "macro_mae_not_above_h0": True,
                      "risk_constraints_are_working_proposal": True}:
        raise ValueError("Selection rule changed")
    return config


def probability_frame(X):
    result = X[["option_type", BASE_FEATURE, "remaining_seconds", "product_duration_seconds",
                "realized_vol_60m", "log_moneyness"]].copy()
    result["terminal_supported"] = 0.0
    result["terminal_static_proxy"] = 0.0
    result["terminal_dynamic_proxy"] = 0.0
    return result[MODEL_INPUTS]


def tail_context(pool, history, today):
    h, q = pool.X.loc[history], pool.X.loc[today]
    threshold = pd.Series(np.nan, index=today, dtype=float)
    source = pd.Series("unavailable", index=today, dtype=object)
    for (side, tenor), part in q.groupby(["option_type", "product_duration_seconds"]):
        side_ids = h.index[h.option_type.eq(side)]
        cell = h.index[h.option_type.eq(side) & h.product_duration_seconds.eq(tenor)]
        ids = cell if len(cell) >= 30 else side_ids
        if len(ids) >= 30:
            threshold.loc[part.index] = pool.y.loc[ids].quantile(.9)
            source.loc[part.index] = "direction_tenor" if len(cell) >= 30 else "direction_fallback"
    return threshold, source


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


def risk_constraints(summary, tolerance=1.02):
    rows = []
    for model in ("h0", "g0", "g1a", "g1b"):
        direction_passes = []
        for side in (0, 1):
            value, tail_value = str(side), f"{side}|tail"
            own = summary.query("month == 'all' and model == @model and group == 'direction' and value == @value")
            base = summary.query("month == 'all' and model == 'official_raw' and group == 'direction' and value == @value")
            own_tail = summary.query("month == 'all' and model == @model and group == 'direction_tail' and value == @tail_value")
            base_tail = summary.query("month == 'all' and model == 'official_raw' and group == 'direction_tail' and value == @tail_value")
            if any(len(x) != 1 for x in (own, base, own_tail, base_tail)):
                row = {"model": model, "direction": side, "available": False,
                       "tail_rmse_pass": False, "gross_underprediction_pass": False,
                       "absolute_total_bias_pass": False, "direction_pass": False}
            else:
                tail_ratio = float(own_tail.rmse_per_btc.iloc[0] / base_tail.rmse_per_btc.iloc[0])
                under_ratio = float(own.gross_underprediction_usdc.iloc[0]
                                    / base.gross_underprediction_usdc.replace(0, np.nan).iloc[0])
                bias_ratio = float(own.absolute_total_bias_usdc.iloc[0]
                                   / base.absolute_total_bias_usdc.replace(0, np.nan).iloc[0])
                row = {"model": model, "direction": side, "available": True,
                       "tail_rmse_ratio_vs_base": tail_ratio,
                       "gross_underprediction_ratio_vs_base": under_ratio,
                       "absolute_total_bias_ratio_vs_base": bias_ratio,
                       "tail_rmse_pass": tail_ratio <= tolerance,
                       "gross_underprediction_pass": under_ratio <= tolerance,
                       "absolute_total_bias_pass": bias_ratio <= 1.0}
                row["direction_pass"] = bool(row["tail_rmse_pass"] and row["gross_underprediction_pass"]
                                             and row["absolute_total_bias_pass"])
            direction_passes.append(row["direction_pass"])
            rows.append(row)
        for row in rows[-2:]:
            row["model_all_directions_pass"] = bool(all(direction_passes))
    return pd.DataFrame(rows)


def selection_table(macro, constraints):
    lookup = macro.set_index("model")
    h0 = lookup.loc["h0"]
    risk = constraints.groupby("model").model_all_directions_pass.first()
    rows = []
    for model in ("g0", "g1a", "g1b"):
        row = lookup.loc[model]
        rmse_pass = bool(row.macro_rmse_per_btc < h0.macro_rmse_per_btc)
        mae_pass = bool(row.macro_mae_per_btc <= h0.macro_mae_per_btc)
        rows.append({"model": model,
                     "rmse_change_vs_h0": float(row.macro_rmse_per_btc / h0.macro_rmse_per_btc - 1),
                     "mae_change_vs_h0": float(row.macro_mae_per_btc / h0.macro_mae_per_btc - 1),
                     "rmse_pass": rmse_pass, "mae_pass": mae_pass,
                     "replacement_accuracy_pass": bool(rmse_pass and mae_pass),
                     "risk_working_proposal_pass": bool(risk.loc[model]),
                     "directly_deliverable": bool(rmse_pass and mae_pass and risk.loc[model])})
    return pd.DataFrame(rows)


def contract_increment(macro):
    lookup = macro.set_index("model")
    rows = []
    for model in ("g1a", "g1b"):
        rows.append({"model": model, "reference": "g0",
                     "rmse_change_vs_g0": float(lookup.loc[model, "macro_rmse_per_btc"]
                                                 / lookup.loc["g0", "macro_rmse_per_btc"] - 1),
                     "mae_change_vs_g0": float(lookup.loc[model, "macro_mae_per_btc"]
                                                / lookup.loc["g0", "macro_mae_per_btc"] - 1)})
    return pd.DataFrame(rows)


def leave_one_day_out(frame):
    rows = []
    for excluded in sorted(frame.day.unique()):
        selected = frame.loc[frame.day.ne(excluded)]
        _, _, macro = aggregate(selected)
        lookup = macro.set_index("model")
        for model in ("g0", "g1a", "g1b"):
            rows.append({"excluded_day": pd.Timestamp(excluded).date().isoformat(), "model": model,
                         "rmse_change_vs_h0": float(lookup.loc[model, "macro_rmse_per_btc"]
                                                     / lookup.loc["h0", "macro_rmse_per_btc"] - 1),
                         "mae_change_vs_h0": float(lookup.loc[model, "macro_mae_per_btc"]
                                                    / lookup.loc["h0", "macro_mae_per_btc"] - 1)})
    return pd.DataFrame(rows)


def evaluate_two_stage(pool, protocol, settings, progress=None):
    predictions, parameters, cohorts = [], [], []
    batches = list(rolling_batches(pool, protocol))
    for day, history, today in batches:
        if len(history) < protocol["minimum_history_orders"]:
            raise ValueError(f"Insufficient history at {day}")
        Xh_prob, Xq_prob = probability_frame(pool.X.loc[history]), probability_frame(pool.X.loc[today])
        Xh_sev, Xq_sev = pool.X.loc[history, SEVERITY_INPUTS], pool.X.loc[today, SEVERITY_INPUTS]
        y = pool.y.loc[history]
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error")
            hurdle = CalibrationModel("hurdle_base").fit(
                Xh_prob, y, pool.metadata.loc[history, "created_at_utc"])
            probability, _ = hurdle.predict_components(Xq_prob)
            direction_scaled = CalibrationModel("direction_scaled").fit(Xh_prob, y)
            grouped = RollingMean(True, settings["group_smoothing"]).fit(pool.X.loc[history], y)
        threshold, source = tail_context(pool, history, today)
        current = pd.DataFrame({
            "day": day, "month": day.strftime("%Y-%m"), "actual": pool.y.loc[today],
            "quantity": pool.X.loc[today, "quantity_btc"], "side": pool.X.loc[today, "option_type"].astype(int),
            "tenor": pool.X.loc[today, "product_duration_seconds"].astype(int),
            "tail_group": np.where(threshold.isna(), "unavailable",
                                   np.where(pool.y.loc[today].gt(threshold), "tail", "body")),
        }, index=today)
        controls = {
            "official_raw": pool.X.loc[today, BASE_FEATURE].to_numpy(float),
            "grouped_mean": grouped.predict_raw(pool.X.loc[today]),
            "direction_scaled": direction_scaled.predict_raw(Xq_prob),
        }
        for name, prediction in controls.items():
            predictions.append(current.assign(model=name, prediction=prediction))
        for name in SEVERITY_NAMES:
            started = perf_counter()
            model = TwoStageSeverity(name, settings["severity_candidates"][name]["alpha"],
                                     settings["minimum_positive_orders"], settings["minimum_creation_days"])
            with threadpool_limits(limits=1), warnings.catch_warnings():
                warnings.simplefilter("error")
                model.fit(Xh_sev, y, pool.metadata.loc[history, "created_at_utc"])
                severity = model.predict_severity(Xq_sev)
            prediction = probability * severity
            if not np.isfinite(prediction).all() or (prediction < 0).any():
                raise ValueError("Invalid two-stage prediction")
            predictions.append(current.assign(model=name, prediction=prediction))
            parameters.append({"day": day.isoformat(), "month": day.strftime("%Y-%m"), "model": name,
                               "fit_predict_seconds": perf_counter() - started, **model.diagnostics()})
        cohorts.append({"day": day.isoformat(), "month": day.strftime("%Y-%m"),
                        "training_orders": len(history), "scoring_orders": len(today),
                        "train_ids_sha256": fingerprint([str(i) for i in history]),
                        "score_ids_sha256": fingerprint([str(i) for i in today]),
                        "latest_training_close": pool.metadata.loc[history, "closed_at_utc"].max().isoformat(),
                        "tail_threshold_fallbacks": int(source.eq("direction_fallback").sum()),
                        "tail_threshold_unavailable": int(source.eq("unavailable").sum()),
                        "shared_probability_sha256": fingerprint([float(v) for v in probability])})
        if progress and (day.day == 1 or day.day % 10 == 0):
            progress(f"{day.date()}: {len(history)} history, {len(today)} scored")
    frame = pd.concat(predictions)
    summary, daily, macro = aggregate(frame)
    constraints = risk_constraints(summary, settings["risk_tolerance_vs_base"])
    tables = {"summary": summary, "daily": daily, "macro_monthly": macro,
              "daily_parameters": pd.DataFrame(parameters), "daily_cohorts": pd.DataFrame(cohorts),
              "risk_constraints": constraints, "selection": selection_table(macro, constraints),
              "contract_increment": contract_increment(macro), "leave_one_day_out": leave_one_day_out(frame)}
    info = {"models": list(MODEL_NAMES), "unique_scoring_orders": int(frame.index.nunique()),
            "scoring_rows_per_model": int(len(frame) / len(MODEL_NAMES)), "scoring_days": len(batches),
            "scoring_ids_sha256": fingerprint([str(i) for i in frame.loc[frame.model.eq("h0")].index]),
            "probability_fits": len(batches), "severity_fits": len(parameters), "no_july": True,
            "parameters_tuned": False}
    return tables, info


def render_report(tables, info):
    macro = tables["macro_monthly"].set_index("model")
    selection = tables["selection"].set_index("model")
    increment = tables["contract_increment"].set_index("model")
    lines = ["# 两阶段赔付模型：分支1结果", "", "## 运行范围", "",
             f"- 共同评分订单：{info['unique_scoring_orders']:,}笔；评分日：{info['scoring_days']}天。",
             "- 评分期为2026年4月至6月；逐日仅使用前30天且当日前已结算记录；未读取7月分区。", "",
             "## 核心指标（USDC/BTC）", "",
             "| 模型 | 月等权RMSE | 月等权MAE |", "|---|---:|---:|"]
    for model in MODEL_NAMES:
        row = macro.loc[model]
        lines.append(f"| {model} | {row.macro_rmse_per_btc:.4f} | {row.macro_mae_per_btc:.4f} |")
    lines += ["", "## 判定", ""]
    for model in ("g0", "g1a", "g1b"):
        row = selection.loc[model]
        lines.append(f"- {model}：相对H0的RMSE变化{row.rmse_change_vs_h0:.3%}，MAE变化{row.mae_change_vs_h0:.3%}；"
                     f"替换精度条件={'通过' if row.replacement_accuracy_pass else '未通过'}；"
                     f"Base风险约束工作提案={'通过' if row.risk_working_proposal_pass else '未通过'}。")
    lines += ["", "## 合同变量增量", ""]
    for model in ("g1a", "g1b"):
        row = increment.loc[model]
        lines.append(f"- {model}相对G0：RMSE变化{row.rmse_change_vs_g0:.3%}，MAE变化{row.mae_change_vs_g0:.3%}。")
    lines += ["", "## 证据边界", "",
              "- Gamma回归是正值均值工作模型，不是对赔付分布的Gamma假设检验。",
              "- 4–6月已用于开发研究；结论不是独立盲测或因果证据。",
              "- 风险约束中的2%为研究工作容忍度，不是官方业务规则。",
              "- 本次只执行分支1；未执行方向信号、收缩、消融、稳定性、模型冻结或7月评价。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run"])
    parser.add_argument("--run-id")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=PROJECT / "outputs/model_studies")
    args = parser.parse_args(argv)
    if args.action == "run" and not args.run_id:
        parser.error("run requires --run-id")
    protocol, settings = load_protocol(), load_settings()
    plan = {"study": settings["version"], "protocol": protocol, "models": list(MODEL_NAMES),
            "probability_model": "one shared CalibrationModel('hurdle_base') fit per scoring day",
            "training_performed": False, "test_used": False, "parameters_tuned": False}
    if args.action == "plan":
        print(json.dumps(plan, indent=2))
        return 0
    started = perf_counter()
    guard = install_development_guard(args.data_dir)
    source = study_snapshot(args.data_dir)
    dataset_manifest_path = args.data_dir / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    data_provenance = {
        "canonical_manifest_sha256": file_digest(dataset_manifest_path),
        "input_sha256": dataset_manifest.get("input_sha256"),
        "artifact_rows": {name: details.get("rows") for name, details in dataset_manifest["artifacts"].items()},
    }
    hashes = {str(path.relative_to(PROJECT)): file_digest(path) for path in (CONFIG_PATH, PROTOCOL_PATH)}
    pool = load_validation_pool({"cohort": "benchmark", "use_base": True}, args.data_dir)
    coverage = protocol_coverage(pool, protocol)
    if coverage.scoring_orders.sum() != 2953 or coverage.scoring_days.sum() != 72:
        raise ValueError("Branch-1 scoring cohort differs from the fixed 2,953-order/72-day benchmark")
    root = run_destination(args.output_root, args.run_id, args.data_dir)
    root.mkdir(parents=True, exist_ok=False)
    for name in ("metrics", "reports", "provenance"):
        (root / name).mkdir()
    atomic_json(root / "provenance/plan.json", plan | {"coverage": coverage.to_dict("records")})
    atomic_json(root / "provenance/manifest.json", {"status": "running"})
    tables, info = evaluate_two_stage(pool, protocol, settings, progress=lambda line: print(line, flush=True))
    tables["coverage"] = coverage
    if study_snapshot(args.data_dir) != source or any(file_digest(PROJECT / name) != digest for name, digest in hashes.items()):
        raise ValueError("Source changed during two-stage severity study")
    paths = [root / "provenance/plan.json"]
    for name, frame in tables.items():
        path = root / f"metrics/{name}.csv"
        frame.to_csv(path, index=False, float_format="%.17g")
        paths.append(path)
    report = root / "reports/evaluation.md"
    report.write_text(render_report(tables, info), encoding="utf-8")
    paths.append(report)
    provenance = {"study": settings["version"], "development": source,
                  "canonical_preprocessing": data_provenance, "config_sha256": hashes,
                  "execution": info, "elapsed_seconds": perf_counter() - started,
                  "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                  / (1024 ** 2 if sys.platform == "darwin" else 1024),
                  "test_csv_read": False, "blocked_test_opens": guard["blocked_test_opens"],
                  "opened_development_files": sorted(guard["opened_development_files"]),
                  "limitations": ["Apr--Jun retrospective development replay, not independent confirmation",
                                  "No July artifacts, raw July order labels, or July market prices used",
                                  "No target clipping, candidate search, margin, demand, or deployment decision"]}
    atomic_json(root / "provenance/source.json", provenance)
    paths.append(root / "provenance/source.json")
    with ZipFile(root / "provenance/source_snapshot.zip", "w", ZIP_DEFLATED) as archive:
        for name in sorted(set(source["code_sha256"]) | set(hashes)
                           | {"pp_models/two_stage_severity.py", "pp_models/two_stage_study.py",
                              "tests/test_two_stage_severity.py", "tests/test_two_stage_study.py",
                              "scripts/verify_two_stage_study.py"}):
            archive.write(PROJECT / name, name)
    paths.append(root / "provenance/source_snapshot.zip")
    atomic_json(root / "provenance/manifest.json", {"status": "complete",
                "files": {str(path.relative_to(root)): file_digest(path) for path in paths}})
    print(json.dumps({"output": str(root), **info, "elapsed_seconds": provenance["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
