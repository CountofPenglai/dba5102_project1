"""Flat report pool tests: identity, safety, chronology, and lossless migration."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from fixtures import candles, order
from pp_data import io
from pp_data.pipeline import prepare


def dated_order(created, closed=None, **changes):
    at = pd.Timestamp(created, tz="UTC")
    end = at + pd.Timedelta(minutes=30)
    return order(createdAt=at, closedAt=closed or end, expirationDate=end, **changes)


def history_for(raw):
    blocks = [candles(value.floor("min") - pd.Timedelta(minutes=1442), 1444)
              for value in pd.to_datetime(raw.createdAt, utc=True)]
    return pd.concat(blocks, ignore_index=True).drop_duplicates("open_time_utc")


def write_inputs(root):
    source = root / "source"
    source.mkdir()
    orders = pd.DataFrame([
        dated_order("2026-03-15 12:00:00", realizedProfit_USDC=20.123456789012345),
        dated_order("2026-05-31 23:45:00", apiBasePrice_per_BTC_USDC=None,
                    markup_per_BTC_USDC=None),
        dated_order("2026-06-30 23:45:00", realizedProfit_USDC=30),
        dated_order("2026-07-15 12:00:00", realizedProfit_USDC=40),
    ])
    orders.to_csv(source / "pp_transactions_training.csv", index=False)
    history_for(orders).to_csv(source / "btc_market_history.csv", index=False)
    return source


def fingerprints(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def alter_table(root, kind, change):
    """Re-sign a changed table, so validation beyond its digest is exercised."""
    path = root / f"datasets/{kind}.csv"
    table = pd.read_csv(path, float_precision="round_trip")
    change(table)
    table.to_csv(path, index=False, float_format="%.17g")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][f"datasets/{kind}.csv"]["sha256"] = io.checksum(path)
    manifest_path.write_text(json.dumps(manifest))


def legacy_pool(root, prepared):
    """Physical bundles deliberately do not follow their former calendar roles."""
    root.mkdir()
    artifacts = {}
    for bundle, positions in (("train", [3]), ("validation", [0, 2]), ("test", [1])):
        for kind in ("features", "targets", "metadata"):
            table = getattr(prepared, kind).iloc[positions].reset_index(drop=True)
            name = f"datasets/{bundle}/{kind}.csv"
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=False, float_format="%.17g")
            artifacts[name] = {"sha256": io.checksum(path), "rows": len(table),
                               "columns": list(table)}
    for name, table in {
        "reports/split_statistics.csv": pd.DataFrame({"split": ["train"], "rows": [4]}),
        "reports/exclusion_statistics.csv": pd.DataFrame({"reason": ["retained"], "rows": [4],
                                                          "definition": ["historical pool"]}),
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        artifacts[name] = {"sha256": io.checksum(path), "rows": len(table), "columns": list(table)}
    (root / "manifest.json").write_text(json.dumps({
        "producer": "pp_data", "version": "preprocessing_v1", "artifacts": artifacts,
        "code_sha256": {"retired.py": "a" * 64}, "config": {"unused": "old roles"},
    }))
    return root


class FlatDataTests(unittest.TestCase):
    def require_flat_loader(self):
        self.assertTrue(callable(getattr(io, "load_dataset", None)),
                        "Flat dataset loading must replace split selection")
        return io.load_dataset

    def test_roundtrip_keeps_all_dates_and_cohort_rows_without_metadata_features(self):
        load = self.require_flat_loader()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_inputs(root)
            before = fingerprints(source)
            result = io.build_dataset(source, root / "out")
            risk = load(root / "out", cohort="risk")
            benchmark = load(root / "out")
            self.assertEqual(risk.X.index.tolist(), [f"training_{i:06d}" for i in range(1, 5)])
            self.assertEqual(benchmark.X.index.tolist(), ["training_000001", "training_000003", "training_000004"])
            self.assertEqual(benchmark.y.iloc[0], 20.123456789012345)
            self.assertEqual(list(benchmark.X.index), list(load(root / "out", use_base=False).X.index))
            self.assertNotIn("closed_at_utc", benchmark.X)
            self.assertNotIn("split", result.metadata)
            self.assertNotIn("api_base_price_per_btc_usdc", risk.X)
            self.assertEqual(before, fingerprints(source))
            self.assertEqual(len(io.canonical_identity(root / "out")["artifacts"]), 5)
            with self.assertRaises(ValueError):
                load(root / "out", cohort="risk", use_base=True)
            first = fingerprints(root / "out")
            io.build_dataset(source, root / "out")
            self.assertEqual(first, fingerprints(root / "out"))

    def test_raw_prepare_keeps_month_boundary_labels_until_only_final_cutoff(self):
        raw = pd.DataFrame([
            dated_order("2026-05-31 23:45:00"),
            dated_order("2026-06-30 23:45:00"),
            dated_order("2026-07-31 17:11:47", closed="2026-07-31 17:11:48Z"),
            dated_order("2026-07-31 17:11:48"),
        ])
        result = prepare(raw, history_for(raw))
        self.assertTrue(hasattr(result, "features"), "Prepared data must be one flat pool")
        self.assertEqual(result.features.order_id.tolist(), ["training_000001", "training_000002"])
        self.assertEqual(result.exclusions.set_index("reason").rows.sum(), 4)

    def test_tamper_and_resigned_alignment_future_candle_and_target_fail(self):
        load = self.require_flat_loader()
        for kind, change, message in [
            ("targets", lambda t: t.__setitem__("order_id", t.order_id.iloc[::-1].to_numpy()), "aligned"),
            ("metadata", lambda t: t.loc.__setitem__((0, "last_candle_close_utc"), "2026-04-01 00:00:00+00:00"), "temporal"),
            ("targets", lambda t: t.loc.__setitem__((0, "payout_per_btc_usdc"), -1), "target"),
        ]:
            with self.subTest(kind=kind, message=message), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                io.build_dataset(write_inputs(root), root / "out")
                alter_table(root / "out", kind, change)
                with self.assertRaisesRegex(ValueError, message):
                    load(root / "out")
                with self.assertRaises(ValueError):
                    io.canonical_identity(root / "out")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            io.build_dataset(write_inputs(root), root / "out")
            path = root / "out/reports/exclusion_statistics.csv"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load(root / "out")

    def test_report_schema_is_validated_even_when_manifest_is_resigned(self):
        load = self.require_flat_loader()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            io.build_dataset(write_inputs(root), root / "out")
            artifact = "reports/dataset_statistics.csv"
            report = root / "out" / artifact
            report.write_text("foreign\n1\n")
            path = root / "out/manifest.json"
            manifest = json.loads(path.read_text())
            manifest["artifacts"][artifact] = {
                "sha256": io.checksum(report), "columns": ["foreign"], "rows": 1,
                "dtypes": {"foreign": "int64"},
            }
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "schema"):
                load(root / "out")

    def test_producer_code_fingerprint_is_provenance_not_load_requirement(self):
        load = self.require_flat_loader()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            io.build_dataset(write_inputs(root), root / "out")
            path = root / "out/manifest.json"
            manifest = json.loads(path.read_text())
            manifest["code_sha256"] = {"previous_producer.py": "0" * 64}
            path.write_text(json.dumps(manifest))
            self.assertEqual(len(load(root / "out").X), 3)

    def test_migration_preserves_all_table_values_and_verifies_every_artifact(self):
        self.require_flat_loader()
        migrate = importlib.import_module("scripts.migrate_report_data").migrate_report_data
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = io.build_dataset(write_inputs(root), root / "flat")
            legacy = legacy_pool(root / "legacy", prepared)
            before = fingerprints(legacy)
            migrate(legacy, root / "migrated")
            self.assertEqual(before, fingerprints(legacy))
            for kind in ("features", "targets", "metadata"):
                original = pd.read_csv(root / f"flat/datasets/{kind}.csv", float_precision="round_trip")
                migrated = pd.read_csv(root / f"migrated/datasets/{kind}.csv", float_precision="round_trip")
                pd.testing.assert_frame_equal(original, migrated, check_exact=True)
            old_report = legacy / "reports/split_statistics.csv"
            old_report.write_bytes(old_report.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                migrate(legacy, root / "bad")
            self.assertFalse((root / "bad").exists())

    def test_migration_rejects_foreign_manifest_paths_overlap_and_foreign_output(self):
        self.require_flat_loader()
        migrate = importlib.import_module("scripts.migrate_report_data").migrate_report_data
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = io.build_dataset(write_inputs(root), root / "flat")
            legacy = legacy_pool(root / "legacy", prepared)
            for output in (legacy, legacy / "nested", root):
                with self.assertRaises(ValueError):
                    migrate(legacy, output)
            foreign = root / "foreign"
            foreign.mkdir()
            (foreign / "keep.txt").write_text("keep")
            with self.assertRaises(ValueError):
                migrate(legacy, foreign)
            self.assertEqual((foreign / "keep.txt").read_text(), "keep")
            path = legacy / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["artifacts"]["../foreign/keep.txt"] = {"sha256": io.checksum(foreign / "keep.txt")}
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "artifact"):
                migrate(legacy, root / "out")
            self.assertFalse((root / "out").exists())

    def test_explicit_cohort_reference_reproduces_subset_and_marks_its_origin(self):
        load = self.require_flat_loader()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_inputs(root)
            io.build_dataset(source, root / "reference")
            path = source / "pp_transactions_training.csv"
            raw = pd.read_csv(path)
            extra = raw.iloc[[0]].copy()
            extra["realizedProfit_USDC"] = 1
            pd.concat([raw, extra], ignore_index=True).to_csv(path, index=False)
            io.build_dataset(source, root / "expanded")
            io.build_dataset(source, root / "same_pool", cohort_reference=root / "reference")
            self.assertEqual(len(load(root / "expanded", cohort="risk").X), 5)
            self.assertEqual(len(load(root / "same_pool", cohort="risk").X), 4)
            manifest = json.loads((root / "same_pool/manifest.json").read_text())
            self.assertEqual(manifest["cohort_origin"], "reference_source_ids")
            self.assertIn("cohort_reference", manifest)

    def test_reference_rebuild_rejects_reused_row_ids_with_different_content(self):
        for change in ("reorder", "target"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = write_inputs(root)
                io.build_dataset(source, root / "reference")
                path = source / "pp_transactions_training.csv"
                raw = pd.read_csv(path, float_precision="round_trip")
                if change == "reorder":
                    raw = raw.iloc[::-1]
                else:
                    raw.loc[0, "realizedProfit_USDC"] += 100
                raw.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, "reference data"):
                    io.build_dataset(source, root / "rejected", cohort_reference=root / "reference")
                self.assertFalse((root / "rejected").exists())

    def test_cli_uses_one_flat_store(self):
        self.require_flat_loader()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = subprocess.run([sys.executable, "-m", "pp_data", "--data-dir", str(write_inputs(root)),
                                     "--output-dir", str(root / "out")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "out/datasets/features.csv").is_file())
            self.assertFalse((root / "out/datasets/train").exists())


if __name__ == "__main__":
    unittest.main()
