import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from test_preprocessing import dated_order, history_for
from pp_data import load_split
from pp_data.io import build_dataset


def write_inputs(root):
    source = root / "source"
    source.mkdir()
    orders = pd.DataFrame([
        dated_order("2026-05-15 12:00:00", realizedProfit_USDC=20),
        dated_order("2026-05-15 12:01:00", apiBasePrice_per_BTC_USDC=None, markup_per_BTC_USDC=None),
        dated_order("2026-06-15 12:00:00", realizedProfit_USDC=30),
        dated_order("2026-07-15 12:00:00", realizedProfit_USDC=40),
    ])
    orders.to_csv(source / "pp_transactions_training.csv", index=False)
    history_for(orders).to_csv(source / "btc_market_history.csv", index=False)
    return source


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


class PreprocessingIOTests(unittest.TestCase):
    def test_rebuild_between_csv_reads_cannot_mix_features_and_targets(self):
        # Interleave a real rebuild with a real reader; no mocked data or results.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = write_inputs(root), root / "output"
            build_dataset(source, output)
            real_read = pd.read_csv
            calls = 0

            def interleaved_read(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    path = source / "pp_transactions_training.csv"
                    changed = real_read(path)
                    changed.loc[0, ["quantity_BTC", "premiumAmount_USDC", "realizedProfit_USDC"]] = [2, 20, 80]
                    changed.to_csv(path, index=False)
                    build_dataset(source, output)
                return real_read(*args, **kwargs)

            with patch("pp_data.io.pd.read_csv", side_effect=interleaved_read):
                loaded = load_split("train", output_dir=output)
            self.assertEqual(loaded.X.quantity_btc.iloc[0], 1)
            self.assertEqual(loaded.y.iloc[0], 20)
            self.assertEqual(load_split("train", output_dir=output).y.iloc[0], 40)

    def test_empty_split_or_cohort_raises_actionable_value_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = write_inputs(root), root / "output"
            path = source / "pp_transactions_training.csv"
            raw = pd.read_csv(path)
            raw.loc[:, "apiBasePrice_per_BTC_USDC"] = float("nan")
            raw.loc[:, "markup_per_BTC_USDC"] = float("nan")
            raw.to_csv(path, index=False)
            build_dataset(source, output)
            with self.assertRaisesRegex(ValueError, "No eligible rows"):
                load_split("train", output_dir=output)
            self.assertEqual(len(load_split("train", cohort="risk", output_dir=output).X), 2)
            raw.loc[:, "realizedProfit_USDC"] = float("nan")
            raw.to_csv(path, index=False)
            build_dataset(source, output)
            for split in ("train", "validation", "test"):
                with self.assertRaisesRegex(ValueError, "No eligible rows"):
                    load_split(split, cohort="risk", output_dir=output)

    def test_cli_reproducibility_source_preservation_and_shared_cohort_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = write_inputs(root), root / "output"
            source_hashes = hashes(source)
            command = [sys.executable, "-m", "pp_data", "--data-dir", str(source), "--output-dir", str(output)]
            for run in range(2):
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                if run == 0:
                    first = hashes(output)
                else:
                    self.assertEqual(first, hashes(output))
            self.assertEqual(source_hashes, hashes(source))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest["artifacts"]), 11)
            common = load_split("train", output_dir=output)
            risk = load_split("train", cohort="risk", output_dir=output)
            ablation = load_split("train", use_base=False, output_dir=output)
            self.assertEqual(common.X.index.tolist(), ["training_000001"])
            self.assertEqual(risk.X.index.tolist(), ["training_000001", "training_000002"])
            self.assertEqual(common.y.tolist(), [20])
            self.assertIn("api_base_price_per_btc_usdc", common.X)
            self.assertNotIn("api_base_price_per_btc_usdc", risk.X)
            self.assertNotIn("closed_at_utc", common.X)
            self.assertNotIn("recorded_premium_usdc", common.X)
            self.assertEqual(ablation.X.index.tolist(), common.X.index.tolist())
            with self.assertRaises(ValueError):
                load_split("train", cohort="risk", use_base=True, output_dir=output)
            self.assertEqual(load_split("validation", output_dir=output).X.index.tolist(), ["training_000003"])
            self.assertEqual(load_split("test", output_dir=output).X.index.tolist(), ["training_000004"])

    def test_tampered_artifact_and_modified_split_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = write_inputs(root), root / "output"
            build_dataset(source, output)
            path = output / "datasets/train/features.csv"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_split("train", output_dir=output)
            build_dataset(source, output)
            path = output / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["config"]["splits"]["train"]["end"] = "2026-07-01T00:00:00Z"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "config"):
                load_split("train", output_dir=output)

    def test_foreign_output_and_source_overlap_are_rejected_without_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_inputs(root)
            before = hashes(source)
            for output in (source, source / "nested", root):
                with self.assertRaises(ValueError):
                    build_dataset(source, output)
            output = root / "other"
            output.mkdir()
            (output / "keep.txt").write_text("keep")
            with self.assertRaises(ValueError):
                build_dataset(source, output)
            self.assertEqual((output / "keep.txt").read_text(), "keep")
            self.assertEqual(before, hashes(source))

    def test_invalid_source_or_conflicting_metadata_writes_no_partial_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = write_inputs(root), root / "output"
            meta = source / "pp_research_datasets_metadata.json"
            meta.write_text(json.dumps({"validation_first_createdAt_utc": "2026-08-01T00:00:00Z"}))
            with self.assertRaisesRegex(ValueError, "official"):
                build_dataset(source, output)
            self.assertFalse(output.exists())
            meta.write_text("[]")
            with self.assertRaises(ValueError):
                build_dataset(source, output)
            self.assertFalse(output.exists())
            meta.unlink()
            orders = source / "pp_transactions_training.csv"
            pd.read_csv(orders).drop(columns="quantity_BTC").to_csv(orders, index=False)
            with self.assertRaisesRegex(ValueError, "quantity_BTC"):
                build_dataset(source, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
