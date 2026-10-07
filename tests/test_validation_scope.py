import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pp_data.io import build_dataset
from pp_models.config import load_model_config, resolve_config
from pp_models.data import daily_batches
from pp_models.evaluation import evaluate
from test_preprocessing_io import write_inputs
from test_model_evaluation import pool_fixture, RecordedMean


class ValidationScopeTests(unittest.TestCase):
    def test_scoped_loader_never_opens_test_files(self):
        self.assertIsNotNone(importlib.util.find_spec("pp_models.scoped_data"))
        from pp_models.scoped_data import load_validation_pool
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "data"
            build_dataset(write_inputs(root), output)
            original = Path.open
            def guarded(path, *args, **kwargs):
                if "datasets/test/" in str(path):
                    raise AssertionError("test data must not be opened")
                return original(path, *args, **kwargs)
            with patch.object(Path, "open", guarded):
                pool = load_validation_pool(resolve_config(load_model_config()), output)
            self.assertEqual(pool.X.index.tolist(), ["training_000001", "training_000002", "training_000003"])
            self.assertNotIn("api_base_price_per_btc_usdc", pool.X)
            (output / "datasets/train/targets.csv").write_text("corrupt")
            with self.assertRaises(ValueError):
                load_validation_pool(resolve_config(load_model_config()), output)

    def test_period_excludes_labels_not_available_for_selection(self):
        import inspect
        self.assertIn("period", inspect.signature(daily_batches).parameters)
        pool = pool_fixture()
        batches = list(daily_batches(pool, "validation", 30,
                                    period=("2026-06-01T00:00Z", "2026-06-02T00:00Z")))
        self.assertEqual(list(batches[0][2]), ["today_early"])
        cfg = resolve_config(load_model_config() | {"min_train_rows": 1}, models=["gam"])
        _, daily = evaluate(pool, "validation", cfg, model_factory=lambda *a: RecordedMean(),
                            period=("2026-06-01T00:00Z", "2026-06-02T00:00Z"))
        self.assertEqual(daily.loc[daily.cohort == "risk", "orders"].tolist(), [1])
        with self.assertRaises(ValueError):
            list(daily_batches(pool, "validation", 30, period=("2026-06-01", "2026-07-02")))

    def test_baseline_mean_and_group_shrinkage(self):
        self.assertIsNotNone(importlib.util.find_spec("pp_models.baselines"))
        from pp_models.baselines import RollingMean
        X = pd.DataFrame({"option_type": [0, 0, 1], "product_duration_seconds": [1800]*3})
        y = pd.Series([0., 10., 20.])
        overall = RollingMean(grouped=False).fit(X, y)
        grouped = RollingMean(grouped=True, smoothing=2).fit(X, y)
        query = pd.DataFrame({"option_type": [0, 1, 0], "product_duration_seconds": [1800, 1800, 3600]})
        np.testing.assert_allclose(overall.predict_raw(query), [10, 10, 10])
        np.testing.assert_allclose(grouped.predict_raw(query), [7.5, 40/3, 10])

    def test_volatility_threshold_is_historical_only(self):
        pool = pool_fixture()
        pool.X.loc[:, "realized_vol_60m"] = [999, 1, 3, 999, 3, 2, 99]
        cfg = resolve_config(load_model_config() | {"min_train_rows": 1}, models=["gam"])
        summary, daily = evaluate(pool, "validation", cfg, model_factory=lambda *a: RecordedMean())
        self.assertIn("volatility_group", summary.group.tolist())
        self.assertAlmostEqual(daily.loc[daily.cohort == "risk", "volatility_q80"].iloc[0], 2.6)


if __name__ == "__main__":
    unittest.main()
