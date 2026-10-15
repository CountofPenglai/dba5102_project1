import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pp_models.__main__ import main
from pp_models.config import load_model_config
from pp_models.estimators import PayoutRegressor
import numpy as np
import pandas as pd
from test_model_study import study_pool


class StudyCliTests(unittest.TestCase):
    def test_gam_selection_reaches_plan_and_execution_without_other_estimators(self):
        class MeanProbe:
            def fit(self, X, y):
                self.mean = float(y.mean())
            def predict_raw(self, X):
                return np.full(len(X), self.mean)
        def gam_only(name, resolved):
            if name != "gam" or resolved["selected_models"] != ["gam"]:
                raise AssertionError("Unselected estimator constructed")
            return MeanProbe()
        with tempfile.TemporaryDirectory() as temp:
            cfg = Path(temp) / "base.json"
            cfg.write_text(json.dumps(load_model_config() | {"min_train_rows": 1}))
            for command in ("study-plan", "study-run"):
                argv = [command, "--config", str(cfg), "--models", "gam"]
                if command == "study-run":
                    argv += ["--run-id", "gam_only", "--output-dir", str(Path(temp) / "runs")]
                with patch("pp_models.scoped_data.load_validation_pool", return_value=study_pool()), \
                     patch("pp_models.study.study_snapshot", return_value={"fixture": True}), \
                     patch("pp_models.__main__.build_model", side_effect=gam_only), \
                     patch("pp_models.study.build_model", side_effect=gam_only), \
                     contextlib.redirect_stdout(io.StringIO()) as output, \
                     contextlib.redirect_stderr(io.StringIO()):
                    try:
                        self.assertEqual(main(argv), 0)
                    except SystemExit:
                        self.fail("Study model selection is not accepted")
                if command == "study-plan":
                    self.assertEqual(json.loads(output.getvalue())["max_model_fits"], 11)
            result = pd.read_csv(Path(temp) / "runs/gam_only/metrics/confirmation.csv")
            self.assertEqual(set(result.model), {"gam", "rolling_mean_30d", "rolling_group_mean_30d"})

    def test_custom_dataset_rejects_study_outputs_for_new_and_resumed_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = Path(temp) / "base.json"
            cfg.write_text(json.dumps(load_model_config() | {"min_train_rows": 1}))
            dataset = Path(temp) / "custom-canonical"
            dataset.mkdir()
            for nested in (False, True):
                for resume in (False, True):
                    with self.subTest(nested=nested, resume=resume):
                        root = dataset / "results" if nested else dataset
                        argv = ["study-run", "--config", str(cfg), "--dataset-dir", str(dataset),
                                "--output-dir", str(root), "--run-id", "protected"]
                        if resume:
                            argv.append("--resume")
                        with patch("pp_models.scoped_data.load_validation_pool", return_value=study_pool()), \
                             patch("pp_models.study.study_snapshot", return_value={"fixture": True}), \
                             patch("pp_models.study_storage.TrialStore.trial", side_effect=AssertionError("evaluation forbidden")), \
                             contextlib.redirect_stderr(io.StringIO()) as errors:
                            self.assertEqual(main(argv), 2)
                        self.assertIn("overlaps", errors.getvalue())
                        self.assertEqual(list(dataset.iterdir()), [])

    def test_plan_only_has_no_test_option_and_no_fitting(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = Path(temp) / "base.json"
            cfg.write_text(json.dumps(load_model_config() | {"min_train_rows": 1}))
            try:
                with patch("pp_models.scoped_data.load_validation_pool", return_value=study_pool()), \
                     patch("pp_models.study.study_snapshot", return_value={"fixture": True}), \
                     patch.object(PayoutRegressor, "fit", side_effect=AssertionError("fit forbidden")), \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    status = main(["study-plan", "--config", str(cfg)])
            except SystemExit:
                self.fail("study-plan command missing")
            self.assertEqual(status, 0)
            result = json.loads(output.getvalue())
            self.assertFalse(result["training_performed"])
            self.assertEqual(result["max_model_fits"], 37)
            self.assertEqual(list(Path(temp).iterdir()), [cfg])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["study-plan", "--split", "test"])

    def test_nonconvergent_solver_must_not_be_marked_fitted(self):
        class Nonconvergent:
            tol = .001
            def fit(self, X, y):
                self.logs_ = {"diffs": [1., .1]}
        model = PayoutRegressor(Nonconvergent(), ["remaining_seconds"])
        with self.assertRaisesRegex(RuntimeError, "converge"):
            model.fit(pd.DataFrame({"remaining_seconds": [10., 20.]}), pd.Series([0., 2.]))
        self.assertFalse(model.fitted)


if __name__ == "__main__":
    unittest.main()
