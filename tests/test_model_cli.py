import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from pp_models.config import load_model_config, resolve_config
from test_model_evaluation import pool_fixture


class ModelCliTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("pp_models.__main__"), "training CLI missing")
        from pp_models.__main__ import main
        from pp_models.runs import write_run, validate_reference
        return main, write_run, validate_reference

    def test_no_arguments_only_help(self):
        main, _, _ = self.api()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main([]), 0)
        self.assertIn("check", output.getvalue())
        self.assertIn("train", output.getvalue())

    def test_check_does_not_train_or_write(self):
        main, _, _ = self.api()
        from pygam import LinearGAM
        from interpret.glassbox import ExplainableBoostingRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps(load_model_config() | {"min_train_rows": 1}))
            with patch("pp_models.__main__.load_pool", return_value=pool_fixture()), \
                 patch("pp_models.__main__.snapshot", return_value={"fixture": True}), \
                 patch.object(LinearGAM, "fit", side_effect=AssertionError("fit forbidden")), \
                 patch.object(ExplainableBoostingRegressor, "fit", side_effect=AssertionError("fit forbidden")), \
                 patch.object(HistGradientBoostingRegressor, "fit", side_effect=AssertionError("fit forbidden")), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["check", "--config", str(path)]), 0)
            payload = json.loads(output.getvalue())
            self.assertFalse(payload["training_performed"])
            self.assertEqual(payload["prediction_rows"], 3)
            self.assertEqual(list(Path(temp).iterdir()), [path])

    def test_test_training_requires_validation_reference_before_data_access(self):
        main, _, _ = self.api()
        with patch("pp_models.__main__.load_pool", side_effect=AssertionError("must reject early")), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["train", "--split", "test", "--run-id", "test_one"]), 2)

    def test_run_storage_no_overwrite_and_reference_integrity(self):
        _, write, validate = self.api()
        cfg = resolve_config(load_model_config())
        provenance = {"dataset_sha256": "fixture", "code_sha256": {}, "environment": {}}
        # These are artificial records for storage tests, not a model evaluation.
        table = pd.DataFrame({"orders": [2], "rmse_per_btc": [1.]})
        with tempfile.TemporaryDirectory() as temp:
            folder = write(Path(temp), "val_fixture", "validation", cfg, provenance, table, table)
            expected = {"metrics/summary.csv", "metrics/daily.csv", "provenance/config.json", "provenance/manifest.json"}
            self.assertEqual({str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}, expected)
            validate(folder, cfg, provenance)
            with self.assertRaises(ValueError):
                validate(folder, cfg | {"window_days": 60}, provenance)
            with self.assertRaises(FileExistsError):
                write(Path(temp), "val_fixture", "validation", cfg, provenance, table, table)
            (folder / "metrics/summary.csv").write_text("tampered")
            with self.assertRaises(ValueError):
                validate(folder, cfg, provenance)
            with self.assertRaises(ValueError):
                write(Path(temp), "../escape", "validation", cfg, provenance, table, table)


if __name__ == "__main__":
    unittest.main()
