import unittest
import numpy as np
import pandas as pd
from pp_models.two_stage_ablation import HistoricalPayoutProbability, combine_components


class AblationTests(unittest.TestCase):
    def test_direction_and_smoothed_tenor_probabilities(self):
        X = pd.DataFrame({"option_type": [0]*10 + [1]*2,
                          "product_duration_seconds": [1800]*2 + [3600]*8 + [1800]*2})
        y = pd.Series([100., 0., 100., 100., 100., 0., 0., 0., 0., 0., 0., 0.])
        direction = HistoricalPayoutProbability("direction_frequency").fit(X, y)
        tenor = HistoricalPayoutProbability("direction_tenor_frequency").fit(X, y)
        q = pd.DataFrame({"option_type": [0, 0, 1], "product_duration_seconds": [1800, 7200, 1800]})
        self.assertAlmostEqual(direction.predict_probability(q)[0], .4)
        self.assertAlmostEqual(tenor.predict_probability(q)[0], 13/32)
        self.assertAlmostEqual(tenor.predict_probability(q)[1], .4)
        self.assertEqual(tenor.predict_probability(q)[2], 0.)

    def test_component_combination_keeps_zero_orders(self):
        result = combine_components([0., .5, 1.], [100., 100., 100.], pd.Index([1, 2, 3]))
        self.assertEqual(result.tolist(), [0., 50., 100.])
        with self.assertRaises(ValueError): combine_components([1.2], [1.], pd.Index([1]))


if __name__ == "__main__": unittest.main()
