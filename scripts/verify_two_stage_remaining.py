"""Independent aggregate verification for confirmed two-stage branches 2--5."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path):
    manifest = json.loads((path / "provenance/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete branch: {path.name}")
    for name, expected in manifest["files"].items():
        if digest(path / name) != expected:
            raise ValueError(f"Checksum mismatch: {path.name}/{name}")
    source = json.loads((path / "provenance/source.json").read_text(encoding="utf-8"))
    if source.get("test_csv_read") is not False or source.get("july_used") is not False:
        raise ValueError("Reserved July scope was accessed")


def verify(root, run_id):
    root = Path(root).resolve()
    paths = {name: root / f"two_stage_{name}_{run_id}" for name in
             ("directional", "shrinkage", "ablation", "robustness")}
    for path in paths.values():
        checked(path)
    macro2 = pd.read_csv(paths["directional"] / "metrics/macro_monthly.csv").set_index("model")
    pair2 = pd.read_csv(paths["directional"] / "metrics/paired_increment.csv").set_index("candidate")
    for candidate, reference in (("g2a", "g1a"), ("g2b", "g1b")):
        expected = macro2.loc[candidate, "macro_rmse_per_btc"] / macro2.loc[reference, "macro_rmse_per_btc"] - 1
        if not np.isclose(expected, pair2.loc[candidate, "rmse_change"], atol=1e-12):
            raise ValueError("Directional paired change mismatch")
    decision3 = pd.read_csv(paths["shrinkage"] / "metrics/selection.csv").iloc[0]
    weights3 = pd.read_csv(paths["shrinkage"] / "metrics/direction_weight_metrics.csv")
    selected = weights3.loc[weights3.record_type.eq("selection")].set_index("direction")
    if (decision3.call_weight != selected.loc[0, "selected_weight"]
            or decision3.put_weight != selected.loc[1, "selected_weight"]):
        raise ValueError("Shrinkage selected weights do not reconcile")
    probability4 = pd.read_csv(paths["ablation"] / "metrics/probability_metrics.csv", dtype={"direction": str})
    overall4 = probability4.query("month == 'all' and direction == 'all'")
    if set(overall4.probability) != {"direction_frequency", "direction_tenor_frequency", "logistic_probability"}:
        raise ValueError("Probability ablation is incomplete")
    if not (overall4.orders == 2953).all():
        raise ValueError("Probability cohort changed")
    intervals5 = pd.read_csv(paths["robustness"] / "metrics/paired_intervals.csv", dtype={"direction": str})
    if set(intervals5.block_days) != {3, 7, 14} or intervals5.valid_repeats.min() < 1900:
        raise ValueError("Robustness interval support failed")
    brier = intervals5.query("metric == 'brier_delta' and direction == 'all'")
    if len(brier) != 3:
        raise ValueError("Probability Brier intervals missing")
    return {"verified": True, "branches": [2, 3, 4, 5], "scoring_orders": 2953,
            "directional_models": len(macro2), "selected_raw_model": decision3.raw_model,
            "call_weight": float(decision3.call_weight), "put_weight": float(decision3.put_weight),
            "robustness_interval_rows": len(intervals5), "minimum_valid_repeats": int(intervals5.valid_repeats.min()),
            "july_used": False, "post_development_started": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.root, args.run_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
