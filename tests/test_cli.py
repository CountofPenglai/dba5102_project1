import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import pandas as pd
from fixtures import order, candles


class CliTests(unittest.TestCase):
    def test_run_preserves_sources_and_exports_only_categorized_aggregates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input", root / "output"
            source.mkdir()
            pd.DataFrame([order()]).to_csv(source / "pp_transactions_training.csv", index=False)
            candles().to_csv(source / "btc_market_history.csv", index=False)
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
            result = subprocess.run([
                sys.executable, "-m", "pp_audit", "--data-dir", str(source),
                "--output-dir", str(output), "--cutoff", "2026-05-01",
            ], capture_output=True, text=True, env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual({str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}, {
                "orders/key_metrics.csv", "orders/segment_statistics.csv",
                "market/key_metrics.csv", "alignment/key_metrics.csv",
                "alignment/cutoff_statistics.csv",
            })
            after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
            self.assertEqual(before, after)
            metrics = pd.read_csv(output / "orders/key_metrics.csv")
            self.assertEqual(metrics.loc[metrics.metric.eq("outcome_missing_rows"), "value"].item(), "0")
            self.assertNotIn("createdAt", pd.read_csv(output / "orders/segment_statistics.csv").columns)

    def test_missing_required_column_fails_without_partial_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            pd.DataFrame([order()]).drop(columns="quantity_BTC").to_csv(source / "pp_transactions_training.csv", index=False)
            candles(periods=2).to_csv(source / "btc_market_history.csv", index=False)
            result = subprocess.run([
                sys.executable, "-m", "pp_audit", "--data-dir", str(source), "--output-dir", str(root / "output"),
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("quantity_BTC", result.stderr)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
