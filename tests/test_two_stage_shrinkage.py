import unittest
import pandas as pd
from pp_models.two_stage_shrinkage import blend_predictions, combine_direction_weights, select_direction_weight


class ShrinkageTests(unittest.TestCase):
    def test_four_fixed_weights_blend_up_and_down(self):
        base = pd.Series([100., 100.], index=["down", "up"])
        raw = pd.Series([60., 160.], index=base.index)
        self.assertEqual(blend_predictions(base, raw, 1).tolist(), [60., 160.])
        self.assertEqual(blend_predictions(base, raw, .5).tolist(), [80., 130.])
        self.assertEqual(blend_predictions(base, raw, .25).tolist(), [90., 115.])
        self.assertEqual(blend_predictions(base, raw, 0).tolist(), [100., 100.])

    def test_direction_combination_and_first_passing_weight(self):
        base = pd.Series([100., 100.]); raw = pd.Series([60., 160.]); side = pd.Series([0, 1])
        self.assertEqual(combine_direction_weights(base, raw, side, {0: 0, 1: .5}).tolist(), [100., 130.])
        keys = ("all_mae", "body_mae", "tail_rmse", "gross_underprediction", "absolute_total_bias")
        base_metrics = dict.fromkeys(keys, 100.)
        quarter = dict.fromkeys(keys, 102.)
        quarter["all_mae"] = quarter["absolute_total_bias"] = 100.
        rows = {1.: dict.fromkeys(keys, 200.), .5: dict.fromkeys(keys, 110.),
                .25: quarter, 0.: dict.fromkeys(keys, 100.)}
        self.assertEqual(select_direction_weight(rows, base_metrics, 30, 5)["weight"], .25)
        rows[1.] = dict.fromkeys(keys, 100.)
        self.assertEqual(select_direction_weight(rows, base_metrics, 30, 5)["weight"], 1.)
        self.assertEqual(select_direction_weight(rows, base_metrics, 29, 5)["weight"], 0.)


if __name__ == "__main__": unittest.main()
