"""Validation-only directional GAM experiment: python -m pp_models.directional_study."""
import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
import pandas as pd
from pp_data.config import DEFAULT_DATA, DEFAULT_OUTPUT, PROJECT
from .baselines import BASELINES, build_baseline
from .config import load_model_config
from .data import coverage, daily_batches, period_bounds
from .directional_features import augment_pool, read_market_prefix
from .directional_gam import VARIANTS, build_directional_model, variant_config
from .evaluation import evaluate
from .runs import fingerprint
from .scoped_data import load_validation_pool
from .study import choose_candidate, study_snapshot
from .study_storage import atomic_json, file_digest, study_workspace

CONFIG_PATH = PROJECT / "config/gam_directional_v1.json"


def load_study_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_study(config)
    return config


def validate_study(config):
    if (set(config) != {"version", "periods", "lambdas", "rmse_tolerance"}
            or config["version"] != "gam_directional_v1"
            or config["lambdas"] != [10, 100, 1000] or config["rmse_tolerance"] != .01
            or set(config["periods"]) != {"tuning", "followup"}):
        raise ValueError("Unsupported directional study configuration")
    first = period_bounds("validation", config["periods"]["tuning"])
    second = period_bounds("validation", config["periods"]["followup"])
    if first[1] != second[0] or any(t != t.floor("D") for t in first + second):
        raise ValueError("Study phases must be contiguous UTC day boundaries")


def plan_study(pool, base, config):
    validate_study(config)
    resolved = variant_config(base, "original", 100)
    periods = {name: coverage(pool, "validation", resolved, bounds) for name, bounds in config["periods"].items()}
    if any(p["insufficient_history_days"] for p in periods.values()):
        raise ValueError("Insufficient history for directional study")
    for phase, bounds in config["periods"].items():
        ids = [str(i) for _, _, rows in daily_batches(pool, "validation", resolved["window_days"], bounds) for i in rows]
        periods[phase]["prediction_ids_sha256"] = fingerprint(ids)
    return {"study": "directional_gam_followup_exploration", "periods": periods,
            "variants": list(VARIANTS), "tuning_candidates": 10,
            "max_gam_fits": 10 * periods["tuning"]["prediction_days"] + 7 * periods["followup"]["prediction_days"],
            "baseline_updates": 2 * sum(p["prediction_days"] for p in periods.values()),
            "test_used": False, "margin_used": False, "threads": 1,
            "fixed_parameters": {"n_splines": 4, "max_iter": 100, "tol": 1e-4},
            "selection": "within 1% of minimum tuning RMSE, prefer larger lambda",
            "fixed_control": "also evaluate lam=100 in followup, without reselecting"}


def _tables(records, selected, phase):
    summary, daily = [], []
    for result in records.values():
        if result["phase"] != phase:
            continue
        cfg = result["resolved"]
        variant = cfg.get("gam_variant", cfg["selected_models"][0])
        baseline = variant in BASELINES
        chosen = cfg == selected.get(variant)
        lam = None if baseline else cfg["models"]["gam"]["lam"]
        role = "baseline" if baseline else ("selected_fixed" if chosen and lam == 100 else
                                            "selected" if chosen else "fixed_lam100" if lam == 100 else "candidate")
        extra = dict(variant=variant, phase=phase, role=role, lam=lam, candidate_id=result["id"])
        summary.append(result["summary"].assign(**extra))
        daily.append(result["daily"].assign(**extra))
    summary, daily = pd.concat(summary, ignore_index=True), pd.concat(daily, ignore_index=True)
    keys = ["cohort", "group", "value"]
    for baseline in BASELINES:
        ref = summary.loc[summary.variant.eq(baseline), keys + ["rmse_per_btc"]]
        label = "rmse_change_vs_" + baseline
        ref = ref.rename(columns={"rmse_per_btc": "baseline_rmse"})
        summary = summary.merge(ref, on=keys, how="left", validate="many_to_one")
        summary[label] = summary.rmse_per_btc / summary.baseline_rmse.replace(0, float("nan")) - 1
        summary = summary.drop(columns="baseline_rmse")
    return summary, daily


def run_directional_study(root, run_id, base, config, pool, source, resume=False, evaluator=evaluate,
                          progress=None, source_check=None, dataset_dir=DEFAULT_OUTPUT):
    plan = plan_study(pool, base, config)
    identity = {"base": base, "study": config, "source": source, "plan": plan}
    with study_workspace(root, run_id, identity, resume, dataset_dir) as store:
        records, selected = {}, {}
        atomic_json(store.path / "provenance/plan.json", plan)
        def trial(resolved, phase, baseline=False):
            if source_check:
                source_check()
            result = store.trial(resolved, phase, config["periods"][phase],
                lambda: evaluator(pool, "validation", resolved,
                                  model_factory=build_baseline if baseline else build_directional_model,
                                  period=config["periods"][phase]))
            overall = result["summary"].query("cohort == 'risk' and group == 'overall'")
            if int(overall.orders.iloc[0]) != plan["periods"][phase]["prediction_rows"]:
                raise ValueError("Candidate scoring cohort changed")
            if source_check:
                source_check()
            records[result["id"]] = result
            if progress:
                progress(f"{phase}: {resolved.get('gam_variant', resolved['selected_models'][0])} "
                         f"lambda={resolved['models']['gam']['lam']} RMSE={result['score']:.4f}")
            return result

        references = []
        for name in BASELINES:
            cfg = variant_config(base, "original", 100)
            cfg.pop("gam_variant")
            cfg.update(selected_models=[name], features=["option_type", "product_duration_seconds"], baseline_smoothing=30)
            references.append(cfg)
            trial(cfg, "tuning", True)
        for variant in VARIANTS:
            lambdas = [100] if variant == "original" else config["lambdas"]
            candidates = [trial(variant_config(base, variant, lam), "tuning") for lam in lambdas]
            selected[variant] = choose_candidate(candidates, "gam", config["rmse_tolerance"])["resolved"]
        # Both chosen models and the fixed-lambda ablation were specified before followup.
        atomic_json(store.path / "provenance/selected_models.json", selected)
        for variant in VARIANTS:
            trial(selected[variant], "followup")
            if selected[variant]["models"]["gam"]["lam"] != 100:
                trial(variant_config(base, variant, 100), "followup")
        for cfg in references:
            trial(cfg, "followup", True)
        paths = [store.path / "provenance/plan.json", store.path / "provenance/selected_models.json"]
        (store.path / "metrics").mkdir(exist_ok=True)
        for phase in config["periods"]:
            summary, daily = _tables(records, selected, phase)
            for suffix, frame in (("", summary), ("_daily", daily)):
                path = store.path / f"metrics/{phase}{suffix}.csv"
                frame.to_csv(path, index=False, float_format="%.17g")
                paths.append(path)
        store.status("complete", completed_candidates=len(records),
                     files={str(p.relative_to(store.path)): file_digest(p) for p in paths},
                     limitations=["Repeatedly observed June: followup exploration, not independent confirmation",
                                  "No test CSV or margin; aggregate metrics only; no real profit inference"])
        return store.path


def install_development_guard(dataset_dir):
    root = Path(dataset_dir).resolve()
    state = {"opened_development_files": set(), "blocked_test_opens": 0}
    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
        if path.is_relative_to(root / "datasets/test"):
            state["blocked_test_opens"] += 1
            raise PermissionError("Directional study cannot open test artifacts")
        if path.is_relative_to(root / "datasets"):
            state["opened_development_files"].add(str(path.relative_to(root)))
    sys.addaudithook(guard)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run"])
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--market-csv", type=Path, default=DEFAULT_DATA / "btc_market_history.csv")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "outputs/model_studies")
    args = parser.parse_args(argv)
    if args.action == "run" and not args.run_id:
        parser.error("run requires --run-id")
    guard = install_development_guard(args.data_dir)
    started = perf_counter()
    base, config = load_model_config(), load_study_config()
    source = study_snapshot(args.data_dir)
    pool = load_validation_pool(variant_config(base, "original", 100), args.data_dir)
    market_stat = args.market_csv.stat()
    market, market_info = read_market_prefix(args.market_csv, "2026-07-01T00:00:00Z")
    pool, feature_info = augment_pool(pool, market)
    del market
    audit_code = {str(p.relative_to(PROJECT)): file_digest(p) for p in sorted((PROJECT / "pp_audit").glob("*.py"))}
    def source_check():
        current = args.market_csv.stat()
        if (current.st_size, current.st_mtime_ns) != (market_stat.st_size, market_stat.st_mtime_ns):
            raise ValueError("Market source changed during experiment")
        if study_snapshot(args.data_dir) != source:
            raise ValueError("Development code/data/environment changed during experiment")
        if any(file_digest(PROJECT / p) != digest for p, digest in audit_code.items()):
            raise ValueError("Market audit code changed during experiment")
    source_check()
    provenance = {"development": source, "market_prefix": market_info,
                  "features": feature_info, "market_audit_code": audit_code}
    if args.action == "plan":
        print(json.dumps(plan_study(pool, base, config) | {"source": provenance}, indent=2))
        return 0
    path = run_directional_study(args.output_root, args.run_id, base, config, pool, provenance,
                                 resume=args.resume, progress=lambda line: print(line, flush=True),
                                 source_check=source_check, dataset_dir=args.data_dir)
    execution = {"elapsed_seconds_including_data_loading": perf_counter() - started,
                 "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024),
                 "test_csv_read": False, "blocked_test_opens": guard["blocked_test_opens"],
                 "opened_development_files": sorted(guard["opened_development_files"])}
    atomic_json(path / "provenance/execution.json", execution)
    manifest_path = path / "provenance/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["provenance/execution.json"] = file_digest(path / "provenance/execution.json")
    atomic_json(manifest_path, manifest)
    print(json.dumps({"output": str(path), **execution}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
