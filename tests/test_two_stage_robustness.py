import unittest
import numpy as np
import pandas as pd
from pp_models.two_stage_robustness import (build_paired_daily_statistics, build_probability_daily_statistics,
                                            paired_calendar_bootstrap, paired_calendar_bootstrap_brier,
                                            paired_delta, calendar_influence, calendar_draw_weights, interval_summary)


class RobustnessTests(unittest.TestCase):
    def frame(self):
        days = pd.date_range("2026-04-01", "2026-06-30", freq="D", tz="UTC")
        rows = []
        for model, shift in (("a", 0.), ("b", 1.)):
            for i, day in enumerate(days):
                rows.append({"id": i, "day": day, "month": day.strftime("%Y-%m"), "model": model,
                             "actual": 10., "prediction": 10.+shift, "side": i%2})
        return pd.DataFrame(rows).set_index("id")

    def test_daily_pairing_and_equal_predictions(self):
        frame = self.frame(); frame.loc[frame.model.eq("b"), "prediction"] = 10.
        daily = build_paired_daily_statistics(frame, [("a", "b")])
        self.assertEqual(paired_delta(daily.query("direction == 'all'"))["rmse_delta"], 0)

    def test_fixed_seed_bootstrap_and_pair_reversal(self):
        daily = build_paired_daily_statistics(self.frame(), [("a", "b")]).query("direction == 'all'")
        first = paired_calendar_bootstrap(daily, 7)
        second = paired_calendar_bootstrap(daily, 7)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue((first.valid_repeats == 2000).all())
        self.assertTrue((first.upper_95 < 0).all())

    def test_probability_brier_interval_uses_paired_calendar(self):
        frame = self.frame().loc[lambda x: x.model.eq("a")].copy()
        frame["p_candidate"] = .5; frame["p_reference"] = .4
        frame["actual"] = (np.arange(len(frame)) % 2).astype(float)
        daily = build_probability_daily_statistics(
            frame, "p_candidate", "p_reference", "candidate", "reference").query("direction == 'all'")
        first = paired_calendar_bootstrap_brier(daily, 7)
        second = paired_calendar_bootstrap_brier(daily, 7)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.metric.iloc[0], "brier_delta")
        self.assertEqual(first.valid_repeats.iloc[0], 2000)

    def variable_daily(self):
        frame = self.frame()
        mask = frame.model.eq("a")
        frame.loc[mask, "prediction"] = 10. + (np.arange(mask.sum()) % 13) / 2
        return build_paired_daily_statistics(frame, [("a", "b")]).query("direction == 'all'")

    def test_last_seven_day_window_is_included(self):
        frame = self.frame()
        frame.loc[frame.model.eq("a") & frame.day.eq(pd.Timestamp("2026-06-30", tz="UTC")), "prediction"] = 110.
        daily = build_paired_daily_statistics(frame, [("a", "b")]).query("direction == 'all'")
        result = calendar_influence(daily).query("window_type == 'seven_day' and metric == 'rmse_delta'").iloc[0]
        self.assertEqual(result.valid_windows, 85)
        self.assertEqual(result.minimum_start, "2026-06-24")
        self.assertEqual(result.minimum_delta, -1.)
        self.assertEqual(result.sign_flip_windows, 1)

    def test_adding_comparison_does_not_change_existing_interval(self):
        daily = self.variable_daily()
        added = daily.assign(candidate="0_extra")
        alone = paired_calendar_bootstrap(daily, 7)
        together = paired_calendar_bootstrap(pd.concat([added, daily]), 7).query("candidate == 'a'")
        pd.testing.assert_frame_equal(alone.reset_index(drop=True), together.reset_index(drop=True))

    def test_reversed_comparisons_use_identical_draws(self):
        daily = self.variable_daily()
        reverse = daily.rename(columns={"candidate_sse": "reference_sse", "reference_sse": "candidate_sse",
                                        "candidate_sae": "reference_sae", "reference_sae": "candidate_sae"})
        reverse = reverse.assign(candidate="b", reference="a")
        result = paired_calendar_bootstrap(pd.concat([daily, reverse]), 7)
        forward = result.query("candidate == 'a'").set_index("metric")
        backward = result.query("candidate == 'b'").set_index("metric")
        np.testing.assert_allclose(forward.lower_95, -backward.upper_95, atol=1e-12)
        np.testing.assert_allclose(forward.upper_95, -backward.lower_95, atol=1e-12)

    def test_brier_and_amount_use_the_same_calendar_draws(self):
        daily = self.variable_daily()
        probability = daily.rename(columns={"candidate_sae": "candidate_brier_sum",
                                             "reference_sae": "reference_brier_sum"})
        amount = paired_calendar_bootstrap(daily, 7).query("metric == 'mae_delta'").iloc[0]
        brier = paired_calendar_bootstrap_brier(probability, 7).iloc[0]
        np.testing.assert_allclose(amount[["lower_95", "median", "upper_95"]].to_numpy(float),
                                   brier[["lower_95", "median", "upper_95"]].to_numpy(float), atol=1e-12)

    def test_missing_month_is_reported_as_unestimable(self):
        daily = self.variable_daily().query("month != '2026-05'")
        influence = calendar_influence(daily)
        self.assertTrue((influence.valid_windows == 0).all())
        self.assertTrue(influence.minimum_delta.isna().all())
        result = paired_calendar_bootstrap(daily, 7)
        self.assertTrue((result.valid_repeats == 0).all())
        self.assertTrue(result.lower_95.isna().all())

    def test_duplicate_order_ids_are_rejected(self):
        frame = self.frame()
        frame = pd.concat([frame, frame.loc[[0]]])
        with self.assertRaises(ValueError):
            build_paired_daily_statistics(frame, [("a", "b")])

    def test_calendar_draws_preserve_month_lengths_and_duplicate_weights(self):
        days, weights = calendar_draw_weights(7)
        for month, count in ((4, 30), (5, 31), (6, 30)):
            self.assertTrue((weights[:, days.month == month].sum(axis=1) == count).all())
        self.assertGreater(weights.max(), 1)
        # Independent scalar reconstruction of the first circular draw.
        rng = np.random.default_rng(20261012 + 7000)
        expected = []
        for count in (30, 31, 30):
            selected = []
            while len(selected) < count:
                start = int(rng.integers(count))
                selected.extend((start + i) % count for i in range(7))
            expected.extend([selected[:count].count(i) for i in range(count)])
        np.testing.assert_array_equal(weights[0], expected)

    def test_minimum_valid_repeat_boundary_and_sparse_dates(self):
        self.assertTrue(interval_summary(np.ones(1900))["estimable"])
        self.assertFalse(interval_summary(np.ones(1899))["estimable"])
        daily = self.variable_daily().loc[lambda x: x.day.dt.day.eq(1)]
        result = paired_calendar_bootstrap(daily, 7)
        self.assertTrue((result.valid_repeats < 1900).all())
        self.assertTrue((result.valid_repeats + result.excluded_repeats == 2000).all())
        self.assertTrue(result.lower_95.isna().all())


if __name__ == "__main__": unittest.main()
