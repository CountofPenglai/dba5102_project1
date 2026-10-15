import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from pp_models.config import load_model_config
from test_model_evaluation import pool_fixture


class DirectionalStudyTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("pp_models.directional_study"), "directional study is missing")
        from pp_models import directional_study
        return directional_study

    def test_selection_is_saved_before_any_followup_and_has_equal_cohorts(self):
        api = self.api()
        cfg = api.load_study_config()
        cfg["periods"] = {"tuning": ["2026-06-01T00:00:00Z", "2026-06-02T00:00:00Z"],
                          "followup": ["2026-06-02T00:00:00Z", "2026-06-03T00:00:00Z"]}
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "example"
            def probe(pool, split, resolved, model_factory, period):
                phase = "tuning" if period == cfg["periods"]["tuning"] else "followup"
                if phase == "followup":
                    locked = json.loads((path / "provenance/selected_models.json").read_text())
                    self.assertEqual(locked["directional"]["models"]["gam"]["lam"], 1000)
                    self.assertEqual(locked["semivol"]["models"]["gam"]["lam"], 1000)
                calls.append(phase)
                lam = resolved["models"]["gam"]["lam"]
                score = {10: 30., 100: 20., 1000: 20.1}[lam] if phase == "tuning" else 1./lam
                metrics = {"model": resolved["selected_models"][0], "cohort": "risk", "group": "overall",
                           "value": "all", "orders": 1, "rmse_per_btc": score, "mae_per_btc": score,
                           "actual_total_usdc": 1., "predicted_total_usdc": 2., "total_bias_usdc": 1.,
                           "predicted_actual_ratio": 2., "clipped_predictions": 0}
                return pd.DataFrame([metrics]), pd.DataFrame([metrics | {"day_utc": period[0]}])
            result = api.run_directional_study(tmp, "example", load_model_config() | {"min_train_rows": 1},
                                               cfg, pool_fixture(), {"synthetic": True}, evaluator=probe)
            self.assertEqual(result, path.resolve())
            self.assertEqual(calls, sorted(calls, key=lambda x: x == "followup"))
            manifest = json.loads((path / "provenance/manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["completed_candidates"], 21)
            table = pd.read_csv(path / "metrics/followup.csv")
            self.assertEqual(table.loc[table.variant.eq("directional"), "role"].tolist(), ["selected", "fixed_lam100"])
            from pp_models.study_storage import file_digest
            for name, digest in manifest["files"].items():
                self.assertEqual(file_digest(path / name), digest)

    def test_periods_cannot_cross_into_test(self):
        api = self.api()
        cfg = api.load_study_config()
        cfg["periods"]["followup"][1] = "2026-07-02T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "fixed split"):
            api.validate_study(cfg)
