import unittest
import numpy as np
import pandas as pd
from pp_models.two_stage_robustness import (build_paired_daily_statistics, build_probability_daily_statistics,
                                            paired_calendar_bootstrap, paired_calendar_bootstrap_brier,
                                            paired_delta)


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


if __name__ == "__main__": unittest.main()
