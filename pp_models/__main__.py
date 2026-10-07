"""Explicit check/train commands; no arguments never imply training."""
import argparse
from pathlib import Path
import sys
from pp_data.config import DEFAULT_OUTPUT, PROJECT
from .config import MODEL_NAMES, load_model_config, resolve_config
from .data import coverage, load_pool
from .estimators import build_model
from .evaluation import evaluate
from .runs import OUTPUT_ROOT, canonical_json, run_destination, snapshot, validate_reference, write_run


def main(argv=None):
    parser = argparse.ArgumentParser(description="PP payout models: check configuration or explicitly train")
    commands = parser.add_subparsers(dest="command")
    for command in ("check", "train"):
        sub = commands.add_parser(command, help="No fitting or files" if command == "check" else "Run daily forward training")
        sub.add_argument("--config", type=Path)
        sub.add_argument("--models", nargs="+", choices=MODEL_NAMES)
        sub.add_argument("--cohort", choices=("risk", "benchmark"), default="risk")
        sub.add_argument("--use-base", action="store_true")
        sub.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUTPUT, help="Canonical pp_data output directory")
        sub.add_argument("--split", choices=("validation", "test"), default="validation", required=command == "train")
        sub.add_argument("--validation-run", type=Path)
        if command == "train":
            sub.add_argument("--run-id", required=True)
            sub.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    for command in ("study-plan", "study-run"):
        sub = commands.add_parser(command, help="Validation-only planning (no fit)" if command == "study-plan"
                                  else "Explicit validation-only tuning and confirmation")
        sub.add_argument("--config", type=Path)
        sub.add_argument("--study-config", type=Path)
        sub.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUTPUT)
        if command == "study-run":
            sub.add_argument("--run-id", required=True)
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/model_studies")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command in ("study-plan", "study-run"):
            from .scoped_data import load_validation_pool
            from .study import plan_study, run_study, study_snapshot
            from .study_config import load_study_config, validate_study
            base, study = load_model_config(args.config), load_study_config(args.study_config)
            resolved = validate_study(base, study)
            before = study_snapshot(args.dataset_dir)
            for name in MODEL_NAMES:
                build_model(name, resolved)
            pool = load_validation_pool(resolved, args.dataset_dir)
            if study_snapshot(args.dataset_dir) != before:
                raise ValueError("Development sources changed during loading")
            plan = plan_study(pool, base, study)
            if args.command == "study-plan":
                print(canonical_json(plan).rstrip())
                return 0
            destination = run_study(args.output_dir, args.run_id, base, study, pool, before,
                resume=args.resume, progress=lambda msg: print(msg, file=sys.stderr),
                source_check=lambda: study_snapshot(args.dataset_dir), dataset_dir=args.dataset_dir)
            print(f"Completed validation-only study: {destination}")
            return 0
        if args.command == "train" and args.split == "test" and args.validation_run is None:
            raise ValueError("Test training requires --validation-run from a completed validation evaluation")
        if args.validation_run is not None and args.split != "test":
            raise ValueError("--validation-run is only used with --split test")
        resolved = resolve_config(load_model_config(args.config), args.cohort, args.use_base, args.models)
        if args.command == "train":
            run_destination(args.output_dir, args.run_id, args.dataset_dir)
        before = snapshot(args.dataset_dir)
        reference = validate_reference(args.validation_run, resolved, before) if args.validation_run else None
        # Instantiation is deliberately separate from fit, also catching missing deps.
        for name in resolved["selected_models"]:
            build_model(name, resolved)
        pool = load_pool(args.split, resolved, args.dataset_dir)
        if snapshot(args.dataset_dir) != before:
            raise ValueError("Sources changed while loading canonical splits; retry")
        counts = coverage(pool, args.split, resolved)
        if counts["insufficient_history_days"]:
            raise ValueError(f"Insufficient history on {counts['insufficient_history_days']} prediction days")
        if args.command == "check":
            print(canonical_json({"training_performed": False, "split": args.split,
                                  "models": resolved["selected_models"], "features": resolved["features"],
                                  "cohort": resolved["cohort"], "use_base": resolved["use_base"],
                                  "validation_reference_checked": reference is not None,
                                  "window_days": resolved["window_days"], **counts}).rstrip())
            return 0
        summary, daily = evaluate(pool, args.split, resolved)
        if snapshot(args.dataset_dir) != before:
            raise ValueError("Sources changed during training; results were not published")
        destination = write_run(args.output_dir, args.run_id, args.split, resolved, before,
                                summary, daily, reference, args.dataset_dir)
        print(f"Completed {args.split} run: {destination}")
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
