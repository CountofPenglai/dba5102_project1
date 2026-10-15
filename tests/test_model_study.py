import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from pp_models.config import load_model_config
from test_model_evaluation import pool_fixture


def study_pool():
    p = pool_fixture()
    p.X.loc["confirm"] = p.X.loc["tomorrow"]
    p.y.loc["confirm"] = 12345.
    p.metadata.loc["confirm"] = [pd.Timestamp("2026-06-21T01:00Z"), pd.Timestamp("2026-06-21T02:00Z"), True]
    return p


def evaluator_probe(calls, fail_after=None):
    def evaluate(pool, split, resolved, model_factory=None, period=None):
        if fail_after is not None and len(calls) >= fail_after:
            raise RuntimeError("injected interruption")
        name = resolved["selected_models"][0]
        calls.append((name, period[0], resolved))
        assert split == "validation"
        score = 10. if pd.Timestamp(period[0]).day == 1 else 999.
        row = {"model": name, "cohort": "risk", "group": "overall", "value": "all",
               "orders": 1, "rmse_per_btc": score, "mae_per_btc": score,
               "actual_total_usdc": 10., "predicted_total_usdc": 11., "total_bias_usdc": 1.,
               "predicted_actual_ratio": 1.1, "clipped_predictions": 0}
        daily = row | {"day_utc": period[0], "training_rows": 30, "fit_seconds": 0., "predict_seconds": 0.}
        return pd.DataFrame([row]), pd.DataFrame([daily])
    return evaluate


class StudyTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("pp_models.study"))
        from pp_models.study_config import load_study_config
        from pp_models.study import plan_study, run_study, choose_candidate
        return load_study_config, plan_study, run_study, choose_candidate

    def test_budget_is_staged_not_full_grid(self):
        load, plan, _, _ = self.api()
        base = load_model_config() | {"min_train_rows": 1}
        result = plan(study_pool(), base, load())
        self.assertEqual([r["max_candidates"] for r in result["models"]], [5, 6, 6])
        self.assertEqual(result["max_model_fits"], 37)
        self.assertEqual(result["baseline_updates"], 6)
        self.assertFalse(result["training_performed"])

    def test_selected_model_only_runs_and_cannot_resume_as_another_selection(self):
        load, plan, run, _ = self.api()
        base = load_model_config() | {"min_train_rows": 1}
        budget = plan(study_pool(), base, load(), models=["gam"])
        self.assertEqual(budget["max_model_fits"], 11)  # Five candidates x two days + one confirmation.
        self.assertEqual([r["model"] for r in budget["models"]], ["gam"])
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            path = run(Path(temp), "gam_only", base, load(), study_pool(), {"fixture": True},
                       models=["gam"], evaluator=evaluator_probe(calls))
            self.assertEqual(len(calls), 10)  # Five tuning + one confirmation + four baseline periods.
            self.assertEqual({r[0] for r in calls}, {"gam", "rolling_mean_30d", "rolling_group_mean_30d"})
            self.assertEqual(set(json.loads((path / "provenance/selected_models.json").read_text())), {"gam"})
            run(Path(temp), "gam_only", base, load(), study_pool(), {"fixture": True},
                models=["gam"], resume=True, evaluator=evaluator_probe(calls))
            self.assertEqual(len(calls), 10)
            with self.assertRaisesRegex(ValueError, "changed"):
                run(Path(temp), "gam_only", base, load(), study_pool(), {"fixture": True},
                    models=["ebm"], resume=True, evaluator=evaluator_probe(calls))
        for invalid in ([], ["gam", "gam"], ["unknown"]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                plan(study_pool(), base, load(), models=invalid)

    def test_complete_workflow_deduplicates_and_freezes_before_confirmation(self):
        load, _, run, _ = self.api()
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            path = run(Path(temp), "study", load_model_config() | {"min_train_rows": 1}, load(),
                       study_pool(), {"fixture": True}, evaluator=evaluator_probe(calls))
            self.assertEqual(len(calls), 24)  # 17 candidates, 3 confirmations, 4 baseline periods
            selected = json.loads((path / "provenance/selected_models.json").read_text())
            self.assertEqual(selected["gam"]["models"]["gam"]["lam"], 100)
            self.assertEqual(selected["gam"]["models"]["gam"]["n_splines"], 4)
            self.assertEqual(selected["ebm"]["models"]["ebm"]["interactions"], [])
            self.assertEqual(selected["shallow_gbdt"]["models"]["shallow_gbdt"]["max_depth"], 2)
            before = len(calls)
            run(Path(temp), "study", load_model_config() | {"min_train_rows": 1}, load(),
                study_pool(), {"fixture": True}, resume=True, evaluator=evaluator_probe(calls))
            self.assertEqual(len(calls), before)
            manifest = json.loads((path / "provenance/manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")

    def test_interrupted_study_resumes_completed_candidates_only(self):
        load, _, run, _ = self.api()
        calls = []
        base = load_model_config() | {"min_train_rows": 1}
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                run(Path(temp), "study", base, load(), study_pool(), {"fixture": True},
                    evaluator=evaluator_probe(calls, fail_after=2))
            self.assertEqual(len(calls), 2)
            path = run(Path(temp), "study", base, load(), study_pool(), {"fixture": True},
                       resume=True, evaluator=evaluator_probe(calls))
            self.assertEqual(len(calls), 24)
            with self.assertRaises(ValueError):
                run(Path(temp), "study", base, load(), study_pool(), {"fixture": "changed"}, resume=True,
                    evaluator=evaluator_probe(calls))
            next(path.glob("trials/*/*/summary.csv")).write_text("corrupted")
            with self.assertRaises(ValueError):
                run(Path(temp), "study", base, load(), study_pool(), {"fixture": True}, resume=True,
                    evaluator=evaluator_probe(calls))

    def test_one_percent_tie_rule_cannot_select_a_much_worse_model(self):
        load, _, _, choose = self.api()
        base = load_model_config()
        a = {"id": "a", "score": 10., "resolved": {"models": {"gam": {"n_splines": 8, "lam": 1}}}}
        b = {"id": "b", "score": 10.09, "resolved": {"models": {"gam": {"n_splines": 4, "lam": 100}}}}
        c = {"id": "c", "score": 11., "resolved": {"models": {"gam": {"n_splines": 4, "lam": 1000}}}}
        self.assertEqual(choose([a, b, c], "gam", .01)["id"], "b")


if __name__ == "__main__":
    unittest.main()
