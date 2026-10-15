"""Independent H0 component studies, Apr--Jun only; python -m pp_models.two_stage_component_study."""
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
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from pp_data.config import DEFAULT_OUTPUT, PROJECT
from pp_data.features import BASE_FEATURE
from .calibration_models import CalibrationModel
from .directional_features import augment_pool, read_market_prefix
from .directional_study import install_development_guard
from .rolling_protocol import PROTOCOL_PATH, load_protocol, protocol_coverage, rolling_batches
from .runs import fingerprint, run_destination
from .scoped_data import load_validation_pool
from .study import study_snapshot
from .study_storage import atomic_json, file_digest
from .two_stage_components import INPUTS, AmountRefinement, ProbabilityRefinement
from .two_stage_remaining_study import direction_metrics, monthly_direction_deltas
from .two_stage_robustness import (build_paired_daily_statistics, build_probability_daily_statistics,
    calendar_influence, paired_calendar_bootstrap, paired_calendar_bootstrap_brier)
from .two_stage_study import aggregate, probability_frame, tail_context

CONFIG_PATH = PROJECT / "config/two_stage_components_v1.json"
AMOUNT_SPECS = {f"s_{features}_{strength}": (features, strength)
               for features in ("contract", "market") for strength in (30, 100)}
PROBABILITY_SPECS = {f"p_{features}_{label}": (features, strength)
                    for features in ("contract", "market") for label, strength in (("c003", .03), ("c01", .1))}
CANDIDATES = tuple(AMOUNT_SPECS) + tuple(PROBABILITY_SPECS)


def load_settings(path=CONFIG_PATH):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"version": "two_stage_components_v1", "protocol": "config/validation_protocol_v2.json",
        "amount_candidates": {k: list(v) for k, v in AMOUNT_SPECS.items()},
        "probability_candidates": {k: list(v) for k, v in PROBABILITY_SPECS.items()},
        "amount_loss": "positive_currency_squared_error_plus_h0_anchored_ridge",
        "amount_bounds": [.5, 1.5], "minimum_positive_amount_orders": 40,
        "minimum_probability_orders": 80, "minimum_probability_class_orders": 20,
        "minimum_creation_days": 5, "standardized_clip": 3., "probability_spline_degree": 2,
        "probability_standardized_knots": [-3., 0., 3.], "bootstrap_block_days": [3, 7, 14],
        "bootstrap_repeats": 2000, "bootstrap_seed": 20261012, "minimum_valid_repeats": 1900,
        "risk_tolerance_vs_base": 1.02, "combine_components": False}
    if result != expected:
        raise ValueError("Unsupported fixed independent-component study settings")
    return result


def evaluate_component_candidates(pool, protocol, progress=None, *, amount_specs=AMOUNT_SPECS,
                                  probability_factories=None):
    if probability_factories is None:
        probability_factories = {name: (lambda f=features, c=strength: ProbabilityRefinement(f, c))
                                 for name, (features, strength) in PROBABILITY_SPECS.items()}
    candidates = tuple(amount_specs) + tuple(probability_factories)
    batches = list(rolling_batches(pool, protocol))
    predictions, parameters, cohorts = [], [], []
    for day, history, today in batches:
        if len(history) < protocol["minimum_history_orders"]:
            raise ValueError("Insufficient fixed-window training history")
        Xh, Xq = pool.X.loc[history, INPUTS], pool.X.loc[today, INPUTS]
        y, created = pool.y.loc[history], pool.metadata.loc[history, "created_at_utc"]
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error")
            h0 = CalibrationModel("hurdle_base").fit(probability_frame(pool.X.loc[history]), y, created)
            p0, s0 = h0.predict_components(probability_frame(pool.X.loc[today]))
            _, sh = h0.predict_components(probability_frame(pool.X.loc[history]))
            probabilities, severities = {"h0": p0}, {"h0": s0}
            for name, (features, strength) in amount_specs.items():
                model = AmountRefinement(features, strength).fit(Xh, y, created, pd.Series(sh, index=history))
                severities[name] = model.predict_severity(Xq, pd.Series(s0, index=today))
                probabilities[name] = p0.copy()
                parameters.extend({"day": day, "month": day.strftime("%Y-%m"), "model": name,
                                   "branch": "amount", **row} for row in model.diagnostics())
            for name, factory in probability_factories.items():
                model = factory().fit(Xh, y, created)
                probabilities[name] = model.predict_probability(Xq, pd.Series(p0, index=today))
                severities[name] = s0.copy()
                parameters.extend({"day": day, "month": day.strftime("%Y-%m"), "model": name,
                                   "branch": "probability", **row} for row in model.diagnostics())
        threshold, source = tail_context(pool, history, today)
        current = pd.DataFrame({"day": day, "month": day.strftime("%Y-%m"), "actual": pool.y.loc[today],
            "quantity": pool.X.loc[today, "quantity_btc"], "side": Xq.option_type.astype(int),
            "tenor": pool.X.loc[today, "product_duration_seconds"].astype(int),
            "tail_group": np.where(threshold.isna(), "unavailable", np.where(pool.y.loc[today].gt(threshold), "tail", "body"))}, index=today)
        predictions.append(current.assign(model="official_raw", prediction=pool.X.loc[today, BASE_FEATURE],
                                          probability=np.nan, severity=np.nan))
        for name in ("h0",) + candidates:
            predictions.append(current.assign(model=name, probability=probabilities[name], severity=severities[name],
                                               prediction=probabilities[name]*severities[name]))
        cohorts.append({"day": day.isoformat(), "month": day.strftime("%Y-%m"), "training_orders": len(history),
            "scoring_orders": len(today), "train_ids_sha256": fingerprint([str(i) for i in history]),
            "score_ids_sha256": fingerprint([str(i) for i in today]),
            "latest_training_close": pool.metadata.loc[history, "closed_at_utc"].max().isoformat(),
            "h0_probability_sha256": fingerprint(p0.tolist()), "h0_severity_sha256": fingerprint(s0.tolist()),
            "tail_threshold_unavailable": int(source.eq("unavailable").sum())})
        if progress and (day.day == 1 or day.day % 10 == 0):
            progress(f"{day.date()}: {len(history)} history, {len(today)} scored; "
                     f"{len(amount_specs)} amount + {len(probability_factories)} probability models")
    return pd.concat(predictions), pd.DataFrame(parameters), pd.DataFrame(cohorts)


def component_integrity(frame, candidates=CANDIDATES, amount_candidates=AMOUNT_SPECS, combined_candidates=()):
    if not set(combined_candidates).issubset(candidates) or set(combined_candidates) & set(amount_candidates):
        raise ValueError("Invalid combined component declarations")
    if set(frame.model) != {"official_raw", "h0", *candidates}:
        raise ValueError("Unexpected or missing model, including possible combined candidate")
    h0, rows = frame.loc[frame.model.eq("h0")], []
    for name in candidates:
        part = frame.loc[frame.model.eq(name)]
        combined = name in combined_candidates
        frozen = None if combined else "probability" if name in amount_candidates else "severity"
        aligned = part.index.is_unique and part.index.equals(h0.index)
        same = aligned and (combined or np.array_equal(part[frozen].to_numpy(), h0[frozen].to_numpy()))
        product = np.array_equal(part.prediction.to_numpy(), (part.probability*part.severity).to_numpy())
        valid = (np.isfinite(part[['probability', 'severity', 'prediction']]).all().all()
                 and part.probability.between(0, 1).all() and part.severity.ge(0).all())
        if not same or not product or not valid:
            raise ValueError(f"Frozen component or product differs: {name}")
        rows.append({"model": name, "branch": "combined" if combined else "amount" if name in amount_candidates else "probability",
            "orders": len(part), "days": part.day.nunique(), "frozen_component": frozen,
            "frozen_component_exact": None if combined else bool(same), "product_exact": bool(product)})
    return pd.DataFrame(rows)


def component_metrics(frame, parameters, candidates=CANDIDATES, amount_candidates=AMOUNT_SPECS, combined_candidates=()):
    metrics, calibration, changes = [], [], []
    for name in ("h0",) + tuple(candidates):
        model = frame.loc[frame.model.eq(name)]
        h0 = frame.loc[frame.model.eq("h0")]
        for month in ["all"] + sorted(model.month.unique()):
            period = model if month == "all" else model.loc[model.month.eq(month)]
            for side in ("all", 0, 1):
                part = period if side == "all" else period.loc[period.side.eq(side)]
                if part.empty:
                    continue
                p, z = part.probability, part.actual.gt(0).astype(int)
                positive = part.loc[z.eq(1)]
                err = positive.severity-positive.actual
                metrics.append({"model": name, "month": month, "direction": str(side), "orders": len(part),
                    "positive_orders": len(positive), "mean_probability": p.mean(), "positive_rate": z.mean(),
                    "brier": float(np.mean((p-z)**2)), "log_loss": log_loss(z, p, labels=[0, 1]),
                    "roc_auc": roc_auc_score(z, p) if z.nunique() == 2 else np.nan,
                    "average_precision": average_precision_score(z, p) if z.nunique() == 2 else np.nan,
                    "positive_amount_bias": err.mean(), "positive_amount_rmse": float(np.sqrt(np.mean(err**2))),
                    "positive_amount_mae": float(np.abs(err).mean())})
                if name != "h0":
                    kinds = ('severity_multiplier', 'probability_difference') if name in combined_candidates else (
                        ('severity_multiplier',) if name in amount_candidates else ('probability_difference',))
                    for kind in kinds:
                        if kind == 'severity_multiplier':
                            anchor = h0.loc[part.index, "severity"]
                            valid = anchor.gt(0)
                            values = part.loc[valid, "severity"] / anchor.loc[valid]
                            at_bound = np.isclose(values, .5) | np.isclose(values, 1.5)
                        else:
                            values = part.probability-h0.loc[part.index, "probability"]
                            at_bound = np.zeros(len(values), dtype=bool)
                        changes.append({"model": name, "month": month, "direction": str(side), "kind": kind,
                            "orders": len(part), "valid_changes": len(values), "mean": values.mean(),
                            "p10": values.quantile(.1), "median": values.median(), "p90": values.quantile(.9),
                            "bound_fraction": float(np.mean(at_bound)) if len(values) else np.nan})
        if name not in amount_candidates:
            for side in ("all", 0, 1):
                part = model if side == "all" else model.loc[model.side.eq(side)]
                bins = pd.cut(part.probability, np.linspace(0, 1, 11), include_lowest=True)
                for label, group in part.groupby(bins, observed=True):
                    calibration.append({"model": name, "direction": str(side), "bin": str(label), "orders": len(group),
                        "mean_probability": group.probability.mean(), "positive_rate": group.actual.gt(0).mean()})
    fits = []
    for (name, branch, side), part in parameters.groupby(["model", "branch", "side"]):
        row = {"model": name, "branch": branch, "side": side, "scoring_days": len(part),
            "fallback_days": int(part.fallback.ne("none").sum()),
            "minimum_training_orders": part.training_orders.min(), "minimum_positive_orders": part.positive_orders.min(),
            "minimum_creation_days": part.creation_days.min(),
            "median_coefficient_norm": part.coefficient_norm.median() if part.coefficient_norm.notna().any() else np.nan}
        for reason in ("insufficient_positive_support", "insufficient_class_support", "zero_anchor", "numerical_failure"):
            row[reason + "_days"] = int(part.fallback.eq(reason).sum())
        fits.append(row)
    return {"component_metrics": pd.DataFrame(metrics), "probability_calibration": pd.DataFrame(calibration),
            "prediction_changes": pd.DataFrame(changes), "fit_support": pd.DataFrame(fits)}


def risk_constraints(frame, candidates=CANDIDATES):
    rows = []
    for side in (0, 1):
        base = frame.loc[frame.model.eq("official_raw") & frame.side.eq(side)]
        for name in ("h0",) + tuple(candidates):
            part = frame.loc[frame.model.eq(name) & frame.side.eq(side)]
            tail = part.loc[part.tail_group.eq("tail")]
            support = len(tail) >= 30 and tail.day.nunique() >= 5
            row = {"model": name, "side": side, "tail_orders": len(tail), "tail_days": tail.day.nunique(),
                   "tail_support_pass": bool(support)}
            if base.empty or part.empty or not support or not part.tail_group.eq("body").any():
                rows.append(row | {"direction_pass": False})
                continue
            own, baseline = direction_metrics(part), direction_metrics(base)
            for metric in own:
                limit = 1. if metric in ("all_mae", "absolute_total_bias") else 1.02
                row.update({metric: own[metric], metric+"_base": baseline[metric],
                            metric+"_pass": bool(own[metric] <= limit*baseline[metric])})
            rows.append(row | {"direction_pass": bool(all(row[m+"_pass"] for m in own))})
    return pd.DataFrame(rows)


def evaluate_tables(frame, parameters, candidates=CANDIDATES, amount_candidates=AMOUNT_SPECS, extra_pairs=(), combined_candidates=()):
    integrity = component_integrity(frame, candidates, amount_candidates, combined_candidates)
    summary, daily, macro = aggregate(frame)
    risk = risk_constraints(frame, candidates)
    lookup, decisions = macro.set_index("model"), []
    biases = summary.query("month == 'all' and group == 'overall'").set_index("model").absolute_total_bias_usdc
    for name in candidates:
        rmse = float(lookup.loc[name, "macro_rmse_per_btc"]-lookup.loc["h0", "macro_rmse_per_btc"])
        mae = float(lookup.loc[name, "macro_mae_per_btc"]-lookup.loc["h0", "macro_mae_per_btc"])
        risk_pass = bool(risk.loc[risk.model.eq(name)].direction_pass.all() and biases.loc[name] <= biases.loc["official_raw"])
        decisions.append({"model": name, "branch": "combined" if name in combined_candidates else "amount" if name in amount_candidates else "probability",
            "rmse_delta_vs_h0": rmse, "mae_delta_vs_h0": mae, "strict_both_improve": rmse < 0 and mae < 0,
            "replacement_accuracy_pass": rmse < 0 and mae <= 0,
            "risk_working_proposal_pass": risk_pass, "overall_bias_pass": bool(biases.loc[name] <= biases.loc["official_raw"]),
            "accuracy_and_risk_pass": bool(rmse < 0 and mae <= 0 and risk_pass)})
    pairs = [("h0", "official_raw")] + [(n, r) for n in candidates for r in ("h0", "official_raw")]
    pairs += [(n, n.replace("market", "contract")) for n in candidates
              if "market" in n and n.replace("market", "contract") in candidates]
    pairs = list(dict.fromkeys(pairs + list(extra_pairs)))
    paired = build_paired_daily_statistics(frame, pairs)
    probability_daily = []
    h0 = frame.loc[frame.model.eq("h0")]
    probability_names = [n for n in candidates if n not in amount_candidates]
    for name in probability_names:
        part = frame.loc[frame.model.eq(name)].assign(p0=h0.probability.to_numpy())
        probability_daily.append(build_probability_daily_statistics(part, "probability", "p0", name, "h0"))
    probability_daily = pd.concat(probability_daily, ignore_index=True)
    with threadpool_limits(limits=1):
        intervals = pd.concat([paired_calendar_bootstrap(paired, b) for b in (3, 7, 14)], ignore_index=True)
        probability_intervals = pd.concat([paired_calendar_bootstrap_brier(probability_daily, b) for b in (3, 7, 14)], ignore_index=True)
    return {"summary": summary, "daily": daily, "macro_monthly": macro, "risk_constraints": risk,
        "selection": pd.DataFrame(decisions), "component_integrity": integrity,
        "comparison_registry": pd.DataFrame(pairs, columns=["candidate", "reference"]),
        "monthly_direction_deltas": monthly_direction_deltas(paired), "paired_intervals": intervals,
        "probability_intervals": probability_intervals,
        "calendar_influence": calendar_influence(paired.loc[paired.direction.eq("all")]),
        **component_metrics(frame, parameters, candidates, amount_candidates, combined_candidates)}


def verify_h0_reference(path, tables, cohorts):
    path = Path(path)
    manifest = json.loads((path / "provenance/manifest.json").read_text())
    required = ["metrics/macro_monthly.csv", "metrics/daily_cohorts.csv"]
    if manifest.get("status") != "complete" or any(manifest["files"].get(n) != file_digest(path/n) for n in required):
        raise ValueError("Invalid H0 reference artifacts")
    old = pd.read_csv(path / required[0], float_precision="round_trip").set_index("model")
    new = tables["macro_monthly"].set_index("model")
    cols = ["macro_rmse_per_btc", "macro_mae_per_btc"]
    if not np.allclose(old.loc[["h0", "official_raw"], cols], new.loc[["h0", "official_raw"], cols], atol=1e-10, rtol=1e-12):
        raise ValueError("H0/Base metric reproduction failed")
    old_cohorts = pd.read_csv(path / required[1])
    for col in ("day", "train_ids_sha256", "score_ids_sha256"):
        if old_cohorts[col].tolist() != cohorts[col].tolist():
            raise ValueError("H0 reference cohort mismatch")
    return {"path": str(path.resolve()), "manifest_sha256": file_digest(path / "provenance/manifest.json"),
            "metric_and_training_scoring_cohorts_match": True}


def report_lines(tables):
    lines = ["# 两阶段独立组件实验", "", "4–6月逐日滚动，2953单/72日；过去30天创建、严格当日前结算。",
        "金额分支固定H0概率；概率分支固定H0金额。8个固定候选，没有组合、追加搜索或使用7月。", "",
        "| 模型 | 月等权RMSE | 月等权MAE |", "|---|---:|---:|"]
    for row in tables["macro_monthly"].itertuples():
        lines.append(f"| {row.model} | {row.macro_rmse_per_btc:.6f} | {row.macro_mae_per_btc:.6f} |")
    lines += ["", "单位USDC/BTC，RMSE为三月MSE等权平均后开方，MAE为三月MAE等权平均。", ""]
    for row in tables["selection"].itertuples():
        lines.append(f"- {row.model}：ΔRMSE={row.rmse_delta_vs_h0:+.4f}、ΔMAE={row.mae_delta_vs_h0:+.4f}；"
                     f"严格双降={row.strict_both_improve}，风险约束={row.risk_working_proposal_pass}。")
    lines += ["", "完整概率/正金额指标、方向/月份/尾部、倍率与支持度、3/7/14日块区间、删日/删周在metrics。",
        "金额模型在合法正赔付历史上拟合原单位平方损失，预测倍率裁剪[0.5,1.5]；概率用小型正则样条Logistic。",
        "固定组件逐笔等同H0已核验；本研究没有开启两分支组合。开发期多次探索，区间未校正多重比较，也不代表独立测试。",
        "没有模型冻结、7月评价或部署兼容性回放；风险约束仍为工作提案。"]
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--run-id")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=PROJECT / "outputs/model_studies")
    parser.add_argument("--h0-reference", type=Path)
    parser.add_argument("--market-csv", type=Path, default=PROJECT.parent / "docs/Jasper Vault Options Pricing Optimisation(Included Base Premium)/btc_market_history.csv")
    args = parser.parse_args(argv)
    settings, protocol = load_settings(), load_protocol()
    if args.action == "plan":
        print(json.dumps({"settings": settings, "protocol": protocol, "training_performed": False}, indent=2))
        return 0
    if not args.run_id:
        parser.error("run requires --run-id")
    destination = run_destination(args.output_root, "two_stage_components_"+args.run_id, args.data_dir)
    started = perf_counter()
    guard = install_development_guard(args.data_dir)
    source = study_snapshot(args.data_dir)
    hashes = {str(p.relative_to(PROJECT)): file_digest(p) for p in (CONFIG_PATH, PROTOCOL_PATH)}
    pool = load_validation_pool({"cohort": "benchmark", "use_base": True}, args.data_dir)
    coverage = protocol_coverage(pool, protocol)
    if coverage.scoring_orders.sum() != 2953 or coverage.scoring_days.sum() != 72:
        raise ValueError("Fixed development cohort changed")
    market_stat = args.market_csv.stat()
    market, market_info = read_market_prefix(args.market_csv, protocol["validation_end"])
    pool, feature_info = augment_pool(pool, market)
    frame, parameters, cohorts = evaluate_component_candidates(pool, protocol, lambda x: print(x, flush=True))
    tables = evaluate_tables(frame, parameters) | {"daily_cohorts": cohorts, "coverage": coverage}
    reference = verify_h0_reference(args.h0_reference, tables, cohorts) if args.h0_reference else None
    current_stat = args.market_csv.stat()
    if (study_snapshot(args.data_dir) != source
            or hashes != {str(p.relative_to(PROJECT)): file_digest(p) for p in (CONFIG_PATH, PROTOCOL_PATH)}
            or (market_stat.st_size, market_stat.st_mtime_ns) != (current_stat.st_size, current_stat.st_mtime_ns)):
        raise ValueError("Sources changed during component study")
    provenance = {"study": settings["version"], "development": source, "config": settings, "protocol": protocol,
        "config_sha256": hashes, "market_prefix": market_info, "directional_features": feature_info,
        "h0_reference": reference, "coverage": coverage.to_dict("records"), "test_csv_read": False, "july_used": False,
        "blocked_test_opens": guard["blocked_test_opens"], "components_combined": False,
        "parameters_tuned": False, "post_development_started": False}
    for directory in ("metrics", "provenance", "reports"):
        (destination / directory).mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(destination / f"metrics/{name}.csv", index=False, float_format="%.17g")
    (destination / "reports/evaluation.md").write_text("\n".join(report_lines(tables))+"\n", encoding="utf-8")
    atomic_json(destination / "provenance/source.json", provenance)
    with ZipFile(destination / "provenance/source_code.zip", "w", ZIP_DEFLATED) as archive:
        for path in sorted((PROJECT / "pp_models").glob("*.py")) + [CONFIG_PATH, PROTOCOL_PATH]:
            archive.write(path, str(path.relative_to(PROJECT)))
    execution = {"output": str(destination), "elapsed_seconds": perf_counter()-started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform == "darwin" else 1024),
        "scoring_orders": 2953, "scoring_days": 72, "july_used": False, "components_combined": False}
    atomic_json(destination / "provenance/execution.json", execution)
    atomic_json(destination / "provenance/manifest.json", {"status": "complete",
        "files": {str(p.relative_to(destination)): file_digest(p) for p in sorted(destination.rglob("*")) if p.is_file()}})
    print(json.dumps(execution, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
