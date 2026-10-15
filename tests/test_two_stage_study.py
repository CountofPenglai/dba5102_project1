import unittest
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE
from pp_data.io import ModelSplit
from pp_models.calibration_models import MODEL_INPUTS
from pp_models.rolling_protocol import load_protocol, rolling_batches
from pp_models.two_stage_study import aggregate, evaluate_two_stage, load_settings, probability_frame


class TwoStageStudyTests(unittest.TestCase):
    def pool(self):
        index = pd.RangeIndex(124)
        created = pd.Series(list(pd.date_range("2026-03-10", periods=120, freq="h", tz="UTC")) + [
            pd.Timestamp("2026-04-01T12:00Z"), pd.Timestamp("2026-04-01T15:00Z"),
            pd.Timestamp("2026-04-02T12:00Z"), pd.Timestamp("2026-04-02T15:00Z")], index=index)
        X = pd.DataFrame({
            "option_type": np.arange(124) % 2,
            BASE_FEATURE: np.resize([80., 100., 120.], 124),
            "remaining_seconds": np.resize([1800., 3600., 7200.], 124),
            "product_duration_seconds": np.resize([1800., 3600.], 124),
            "realized_vol_60m": .02,
            "log_moneyness": np.linspace(-.02, .02, 124),
            "quantity_btc": .1,
        }, index=index)
        y = pd.Series(np.resize([0., 90., 140., 0., 200.], 124), index=index)
        metadata = pd.DataFrame({"created_at_utc": created,
                                 "closed_at_utc": created + pd.Timedelta(minutes=30),
                                 "expiration_at_utc": created + pd.Timedelta(hours=1)}, index=index)
        return ModelSplit(X, y, metadata)

    def test_probability_frame_uses_exact_existing_hurdle_interface(self):
        X = self.pool().X
        result = probability_frame(X)
        self.assertEqual(result.columns.tolist(), MODEL_INPUTS)
        self.assertTrue((result[["terminal_supported", "terminal_static_proxy", "terminal_dynamic_proxy"]] == 0).all().all())

    def test_closed_at_equal_day_is_not_training_history(self):
        pool = self.pool()
        pool.metadata.loc[0, "closed_at_utc"] = pd.Timestamp("2026-04-01", tz="UTC")
        day, history, _ = next(rolling_batches(pool, load_protocol()))
        self.assertEqual(day, pd.Timestamp("2026-04-01", tz="UTC"))
        self.assertNotIn(0, history)

    def test_future_labels_do_not_change_prior_predictions_and_models_share_rows(self):
        pool = self.pool()
        first, info = evaluate_two_stage(pool, load_protocol(), load_settings())
        daily = first["daily"]
        overall = daily.loc[daily.group.eq("overall")]
        self.assertEqual(overall.groupby("model").orders.sum().nunique(), 1)
        self.assertEqual(info["scoring_rows_per_model"], 4)
        old = overall.loc[overall.day.eq("2026-04-01T00:00:00+00:00")].set_index("model").predicted_total_usdc
        pool.y.iloc[-1] = 1e12
        second, _ = evaluate_two_stage(pool, load_protocol(), load_settings())
        new = second["daily"].query("group == 'overall' and day == '2026-04-01T00:00:00+00:00'").set_index("model").predicted_total_usdc
        pd.testing.assert_series_equal(old, new)

    def test_macro_monthly_is_equal_month_weighted(self):
        rows = []
        for month, actual, prediction in (("2026-04", 0., 0.), ("2026-05", 0., 10.)):
            rows.append(pd.DataFrame({"day": [pd.Timestamp(month + "-01", tz="UTC")], "month": [month],
                                      "actual": [actual], "prediction": [prediction], "quantity": [1.],
                                      "side": [0], "tenor": [1800], "tail_group": ["body"], "model": ["x"]}))
        _, _, macro = aggregate(pd.concat(rows, ignore_index=True))
        self.assertAlmostEqual(macro.iloc[0].macro_rmse_per_btc, np.sqrt(50))
        self.assertEqual(macro.iloc[0].macro_mae_per_btc, 5)


if __name__ == "__main__":
    unittest.main()
