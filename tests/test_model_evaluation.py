import importlib.util
import unittest
import numpy as np
import pandas as pd
from pp_data.io import ModelSplit
from pp_models.config import load_model_config, resolve_config


def pool_fixture():
    ids = ["too_old", "left_edge", "ready", "at_cutoff", "today_early", "today_late", "tomorrow"]
    created = ["05-01 23:59", "05-02 00:00", "05-31 12:00", "05-31 23:00",
               "06-01 01:00", "06-01 10:00", "06-02 10:00"]
    closed = ["05-02 01:00", "05-02 01:00", "05-31 13:00", "06-01 00:00",
              "06-01 02:00", "06-02 00:00", "06-02 11:00"]
    meta = pd.DataFrame({"created_at_utc": pd.to_datetime(["2026-"+t for t in created], utc=True),
                         "closed_at_utc": pd.to_datetime(["2026-"+t for t in closed], utc=True),
                         "benchmark_eligible": [True, True, True, False, True, False, True]}, index=ids)
    X = pd.DataFrame(1., index=ids, columns=load_model_config()["features"])
    X["option_type"] = 0
    X["quantity_btc"] = [1., 1., 1., 1., 2., 1., 1.]
    X["product_duration_seconds"] = 1800
    return ModelSplit(X, pd.Series([999., 4., 2., 100., 6., 10., 8.], index=ids), meta)


class RecordedMean:
    """Artificial scheduling probe, never a production model or saved fit."""
    def fit(self, X, y):
        self.mean = float(y.mean())
    def predict_raw(self, X):
        return np.full(len(X), self.mean)


class EvaluationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("pp_models.evaluation"), "evaluation is missing")
        from pp_models.data import daily_batches
        from pp_models.evaluation import evaluate, key_metrics
        return daily_batches, evaluate, key_metrics

    def test_daily_window_and_settlement_boundaries(self):
        batches, _, _ = self.api()
        days = list(batches(pool_fixture(), "validation", 30))
        self.assertEqual(list(days[0][1]), ["left_edge", "ready"])
        self.assertEqual(list(days[0][2]), ["today_early", "today_late"])
        self.assertEqual(list(days[1][1]), ["ready", "at_cutoff", "today_early"])
        self.assertEqual(list(days[1][2]), ["tomorrow"])

    def test_future_labels_only_affect_later_eligible_days(self):
        _, evaluate, _ = self.api()
        cfg = resolve_config(load_model_config() | {"min_train_rows": 1}, models=["gam"])
        pool = pool_fixture()
        factory = lambda name, config: RecordedMean()
        _, daily = evaluate(pool, "validation", cfg, model_factory=factory)
        self.assertEqual(daily[daily.cohort == "risk"].predicted_total_usdc.tolist(), [9., 36.])
        pool.y.loc["today_early"] = 60.
        _, changed = evaluate(pool, "validation", cfg, model_factory=factory)
        self.assertEqual(changed[changed.cohort == "risk"].predicted_total_usdc.tolist(), [9., 54.])
        pool.y.loc["today_late"] = 1000000.
        _, changed_again = evaluate(pool, "validation", cfg, model_factory=factory)
        self.assertEqual(changed_again[changed_again.cohort == "risk"].predicted_total_usdc.tolist(), [9., 54.])

    def test_metrics_use_quantity_and_handle_zero_total(self):
        _, _, metrics = self.api()
        result = metrics(np.array([0., 10.]), np.array([2., 6.]), np.array([3., 1.]))
        self.assertAlmostEqual(result["rmse_per_btc"], np.sqrt(10))
        self.assertEqual(result["mae_per_btc"], 3.)
        self.assertEqual(result["predicted_total_usdc"], 12.)
        self.assertEqual(result["actual_total_usdc"], 10.)
        self.assertEqual(result["total_bias_usdc"], 2.)
        self.assertEqual(result.get("gross_overprediction_usdc"), 6.)
        self.assertEqual(result.get("gross_underprediction_usdc"), 4.)
        self.assertIsNone(metrics(np.zeros(2), np.ones(2), np.ones(2))["predicted_actual_ratio"])

    def test_sparse_history_is_not_silently_replaced(self):
        _, evaluate, _ = self.api()
        with self.assertRaisesRegex(ValueError, "history"):
            evaluate(pool_fixture(), "validation", resolve_config(load_model_config()),
                     model_factory=lambda *args: RecordedMean())

    def test_opt_in_daily_direction_metrics_reconcile_with_overall(self):
        _, evaluate, _ = self.api()
        cfg = resolve_config(load_model_config() | {"min_train_rows": 1}, models=["gam"])
        cfg["daily_direction_metrics"] = True
        pool = pool_fixture()
        pool.X.loc["today_late", "option_type"] = 1
        _, daily = evaluate(pool, "validation", cfg, model_factory=lambda *args: RecordedMean())
        self.assertIn("group", daily)
        day = daily[(daily.cohort == "risk") & daily.day_utc.eq("2026-06-01T00:00:00+00:00")]
        overall = day[day.group.eq("overall")].iloc[0]
        groups = day[day.group.eq("option_type")]
        self.assertEqual(groups.predicted_total_usdc.tolist(), [6., 3.])
        self.assertEqual(groups.actual_total_usdc.sum(), overall.actual_total_usdc)
        self.assertEqual(groups.orders.sum(), overall.orders)

    def test_platform_activity_groups_use_only_legal_training_thresholds(self):
        _, evaluate, _ = self.api()
        cfg = resolve_config(load_model_config() | {"min_train_rows": 1}, models=["gam"])
        cfg['activity_metrics'] = True
        pool = pool_fixture()
        pool.X['platform_call_count_240m'] = [9999, 0, 10, 9999, 9, 1, 9999]
        pool.X['platform_put_count_240m'] = 0
        summary, _ = evaluate(pool, 'validation', cfg, model_factory=lambda *args: RecordedMean(),
                              period=['2026-06-01T00:00:00Z', '2026-06-03T00:00:00Z'])
        groups = summary[(summary.cohort == 'risk') & summary.group.eq('platform_activity_group')]
        self.assertEqual(groups.set_index('value').orders.to_dict(), {'high': 2, 'normal': 1})
        self.assertEqual(groups.actual_total_usdc.sum(), 30.)


if __name__ == "__main__":
    unittest.main()
