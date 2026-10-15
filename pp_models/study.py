"""Sequential validation-only tuning; confirmation never selects parameters."""
import copy
from pathlib import Path
import pandas as pd
from pp_data.config import DEFAULT_OUTPUT
from .baselines import BASELINES, build_baseline
from .config import MODEL_NAMES, resolve_config
from .data import coverage
from .estimators import build_model
from .evaluation import evaluate
from .runs import fingerprint, snapshot
from .scoped_data import development_snapshot
from .study_config import candidate_config, validate_study
from .study_storage import atomic_json, file_digest, study_workspace


def study_snapshot(dataset_dir):
    development, _ = development_snapshot(dataset_dir)
    source = snapshot(dataset_dir)  # Only manifest/code/version reads, no test CSV reads.
    source["dataset_sha256"] = fingerprint(development)
    source["dataset_scope"] = ["train", "validation"]
    return source


def plan_study(pool, base, study, models=None):
    resolved = validate_study(base, study, models)
    periods = {phase: coverage(pool, "validation", resolved, period)
               for phase, period in study["periods"].items()}
    if any(v["insufficient_history_days"] for v in periods.values()):
        raise ValueError("Insufficient history for this study")
    rows = []
    for model in resolved["selected_models"]:
        n = 1 + sum(len(s["values"]) - 1 for s in study["stages"][model])
        tuning = n * periods["tuning"]["prediction_days"]
        confirmation = periods["confirmation"]["prediction_days"]
        seconds = [(tuning + confirmation) * t for t in study["seconds_per_fit_assumption"][model]]
        rows.append({"model": model, "max_candidates": n, "tuning_fits": tuning,
                     "confirmation_fits": confirmation, "estimated_fit_seconds": seconds})
    return {"training_performed": False, "test_csv_read": False, "periods": periods,
            "selected_models": resolved["selected_models"],
            "models": rows, "max_model_fits": sum(r["tuning_fits"] + r["confirmation_fits"] for r in rows),
            "baseline_updates": 2 * sum(p["prediction_days"] for p in periods.values()),
            "estimated_total_fit_seconds": [sum(r["estimated_fit_seconds"][i] for r in rows) for i in (0, 1)],
            "timing_is_measured": False, "timing_basis": "Unmeasured per-fit engineering assumptions in study config",
            "execution_threads": 1, "gpu_required": False,
            "estimated_process_memory_gib": [.5, 2.]}


def choose_candidate(candidates, model, tolerance):
    best = min(c["score"] for c in candidates)
    near = [c for c in candidates if c["score"] <= best * (1 + tolerance)]
    def complexity(c):
        p = c["resolved"]["models"][model]
        if model == "gam":
            return p["n_splines"], -p["lam"], c["id"]
        if model == "ebm":
            return len(p["interactions"]), -p["min_samples_leaf"], p["max_rounds"], c["id"]
        return p["max_depth"], -p["min_samples_leaf"], p["max_iter"], c["id"]
    return min(near, key=complexity)


def _comparisons(records, cohort):
    summaries = pd.concat([r["summary"] for r in records], ignore_index=True)
    daily = pd.concat([r["daily"] for r in records], ignore_index=True)
    for baseline in BASELINES:
        reference = summaries.loc[summaries.model == baseline,
                                  ["cohort", "group", "value", "rmse_per_btc"]].rename(
                                      columns={"rmse_per_btc": f"{baseline}_rmse"})
        summaries = summaries.merge(reference, on=["cohort", "group", "value"], how="left", validate="many_to_one")
        denom = summaries[f"{baseline}_rmse"].replace(0, float("nan"))
        summaries[f"relative_rmse_change_vs_{baseline}"] = summaries.rmse_per_btc / denom - 1
        ref_days = daily.loc[(daily.model == baseline) & (daily.cohort == cohort), ["day_utc", "rmse_per_btc"]]
        rates = {}
        for model in summaries.model.unique():
            own = daily.loc[(daily.model == model) & (daily.cohort == cohort), ["day_utc", "rmse_per_btc"]]
            joined = own.merge(ref_days, on="day_utc", suffixes=("_model", "_baseline"), validate="one_to_one")
            rates[model] = float((joined.rmse_per_btc_model < joined.rmse_per_btc_baseline).mean())
        summaries[f"daily_win_rate_vs_{baseline}"] = summaries.model.map(rates).where(
            (summaries.group == "overall") & (summaries.cohort == cohort))
    return summaries


def run_study(root, run_id, base, study, pool, source, resume=False, evaluator=evaluate,
              progress=None, source_check=None, dataset_dir=DEFAULT_OUTPUT, models=None):
    plan = plan_study(pool, base, study, models)
    identity = {"base": base, "study": study, "source": source,
                "selected_models": plan["selected_models"]}
    with study_workspace(root, run_id, identity, resume, dataset_dir) as store:
        records, selected = {}, {}
        def trial(raw, model, phase, baseline=False):
            if source_check is not None and source_check() != source:
                raise ValueError("Study source changed during execution")
            if baseline:
                resolved = resolve_config(raw, study["cohort"], study["use_base"])
                resolved["selected_models"] = [model]
                resolved["features"] = ["option_type", "product_duration_seconds"]
                resolved["baseline_smoothing"] = study["baseline_smoothing"]
            else:
                resolved = resolve_config(raw, study["cohort"], study["use_base"], [model])
            period = study["periods"][phase]
            result = store.trial(resolved, phase, period,
                lambda: evaluator(pool, "validation", resolved,
                                  model_factory=build_baseline if baseline else build_model, period=period))
            if source_check is not None and source_check() != source:
                raise ValueError("Study source changed during candidate evaluation")
            result["raw"] = copy.deepcopy(raw)
            records[result["id"]] = result
            if progress:
                progress(f"{phase}: {model} candidate {result['id'][:10]} available")
            return result

        # Establish the cheap tuning references before fitting any complex model.
        for model in BASELINES:
            trial(base, model, "tuning", baseline=True)
        for model in plan["selected_models"]:
            current, candidates = copy.deepcopy(base), {}
            for stage in study["stages"][model]:
                for value in stage["values"]:
                    raw, _ = candidate_config(current, model, stage["parameter"], value, study)
                    result = trial(raw, model, "tuning")
                    candidates[result["id"]] = result
                winner = choose_candidate(list(candidates.values()), model, study["rmse_tolerance"])
                current = winner["raw"]
            selected[model] = winner["resolved"]
        # Persist selection before any confirmation score is computed.
        atomic_json(store.path / "provenance/selected_models.json", selected)
        for model, resolved in selected.items():
            raw = {k: copy.deepcopy(resolved[k]) for k in base}
            trial(raw, model, "confirmation")
        for model in BASELINES:
            trial(base, model, "confirmation", baseline=True)
        metrics_dir = store.path / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        tuning = [r for r in records.values() if r["phase"] == "tuning"]
        leaderboard = []
        for r in tuning:
            row = r["summary"].loc[(r["summary"].cohort == study["cohort"]) & (r["summary"].group == "overall")].iloc[0].to_dict()
            row.update(candidate_id=r["id"], selected=r["resolved"] == selected.get(row["model"]))
            leaderboard.append(row)
        pd.DataFrame(leaderboard).sort_values(["model", "rmse_per_btc"]).to_csv(metrics_dir / "tuning_leaderboard.csv", index=False)
        confirmation = [r for r in records.values() if r["phase"] == "confirmation"]
        _comparisons(confirmation, study["cohort"]).to_csv(metrics_dir / "confirmation.csv", index=False)
        atomic_json(store.path / "provenance/plan.json", plan)
        files = {str(p.relative_to(store.path)): file_digest(p) for p in
                 [metrics_dir / "tuning_leaderboard.csv", metrics_dir / "confirmation.csv",
                  store.path / "provenance/selected_models.json", store.path / "provenance/plan.json"]}
        store.status("complete", files=files, completed_candidates=len(records),
                     limitations=["Validation-only study; confirmation never selects hyperparameters",
                                  "No test CSV read, no row predictions or model checkpoints persisted"])
        return store.path
