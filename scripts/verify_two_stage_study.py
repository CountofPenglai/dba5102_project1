"""Independent aggregate checks for one completed branch-1 severity study."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "provenance/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not manifest.get("files"):
        raise ValueError("Study is not complete")
    for name, expected in manifest["files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    summary = pd.read_csv(root / "metrics/summary.csv", dtype={"value": str})
    macro = pd.read_csv(root / "metrics/macro_monthly.csv").set_index("model")
    selection = pd.read_csv(root / "metrics/selection.csv").set_index("model")
    cohorts = pd.read_csv(root / "metrics/daily_cohorts.csv")
    expected_models = {"official_raw", "grouped_mean", "direction_scaled", "h0", "g0", "g1a", "g1b"}
    if set(macro.index) != expected_models or cohorts.scoring_orders.sum() != 2953 or len(cohorts) != 72:
        raise ValueError("Unexpected models or scoring cohort")
    monthly = summary.loc[(summary.month != "all") & summary.group.eq("overall")]
    for model, part in monthly.groupby("model"):
        rmse = float(np.sqrt(np.mean(part.rmse_per_btc ** 2)))
        mae = float(part.mae_per_btc.mean())
        if not np.isclose(rmse, macro.loc[model, "macro_rmse_per_btc"], rtol=0, atol=1e-10):
            raise ValueError(f"Macro RMSE mismatch: {model}")
        if not np.isclose(mae, macro.loc[model, "macro_mae_per_btc"], rtol=0, atol=1e-10):
            raise ValueError(f"Macro MAE mismatch: {model}")
    if not (abs(macro.loc["official_raw", "macro_rmse_per_btc"] - 292.57) < .03
            and abs(macro.loc["h0", "macro_rmse_per_btc"] - 288.20) < .03):
        raise ValueError("H0/Base historical reference was not reproduced")
    for model in ("g0", "g1a", "g1b"):
        rmse_pass = macro.loc[model, "macro_rmse_per_btc"] < macro.loc["h0", "macro_rmse_per_btc"]
        mae_pass = macro.loc[model, "macro_mae_per_btc"] <= macro.loc["h0", "macro_mae_per_btc"]
        if bool(selection.loc[model, "replacement_accuracy_pass"]) != bool(rmse_pass and mae_pass):
            raise ValueError(f"Selection rule mismatch: {model}")
    source = json.loads((root / "provenance/source.json").read_text(encoding="utf-8"))
    if source.get("test_csv_read") is not False or source.get("blocked_test_opens") != 0:
        raise ValueError("Reserved test scope was accessed")
    return {"verified": True, "models": len(expected_models), "scoring_orders": 2953,
            "scoring_days": 72, "base_macro_rmse": macro.loc["official_raw", "macro_rmse_per_btc"],
            "h0_macro_rmse": macro.loc["h0", "macro_rmse_per_btc"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.run_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
