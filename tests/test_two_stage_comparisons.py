import unittest
import pandas as pd
from pp_models.two_stage_comparisons import build_comparison_registry


class ComparisonRegistryTests(unittest.TestCase):
    def inputs(self):
        scores = {'official_raw': 30., 'grouped_mean': 25., 'h0': 20.,
                  'g0': 18., 'g1a': 22., 'g1b': 17., 'g2a': 21., 'g2b': 16.,
                  'final_combination': 17.5,
                  'direction_frequency_s0': 23., 'direction_tenor_frequency_s0': 19.,
                  'direction_frequency_sR': 20., 'direction_tenor_frequency_sR': 18.}
        macro = pd.DataFrame([{'model': k, 'macro_rmse_per_btc': v, 'macro_mae_per_btc': v} for k, v in scores.items()])
        first = pd.DataFrame({'model': ['g0', 'g1a', 'g1b'],
                              'replacement_accuracy_pass': [True, False, True]})
        second = pd.DataFrame({'model': ['g2a', 'g2b'], 'accuracy_vs_h0_pass': [False, True],
                               'paired_increment_pass': [True, True]})
        final = pd.Series({'raw_model': 'g2b', 'final_adopted': True})
        ablation = pd.DataFrame([
            {'layer': 'probability', 'candidate': 'h0', 'reference': 'direction_frequency_s0'},
            {'layer': 'probability', 'candidate': 'h0', 'reference': 'direction_tenor_frequency_s0'},
            {'layer': 'amount', 'candidate': 'direction_frequency_sR', 'reference': 'direction_frequency_s0'}])
        return macro, first, second, final, ablation

    def test_all_good_branches_keep_their_correct_comparators(self):
        result = build_comparison_registry(*self.inputs())
        included = set(map(tuple, result.loc[result.included, ['candidate', 'reference']].to_numpy()))
        expected = {('g0', 'h0'), ('g1b', 'g0'), ('g1b', 'h0'),
                    ('g2b', 'g1b'), ('g2b', 'h0'), ('g2b', 'g0'),
                    ('final_combination', 'g2b'), ('final_combination', 'h0'),
                    ('final_combination', 'official_raw'), ('final_combination', 'grouped_mean'),
                    ('h0', 'direction_frequency_s0'),
                    ('direction_frequency_sR', 'direction_frequency_s0')}
        self.assertTrue(expected.issubset(included))
        self.assertFalse(result.duplicated(['candidate', 'reference']).any())
        rejected = result.query("candidate == 'g1a'")
        self.assertEqual(len(rejected), 2)
        self.assertFalse(rejected.included.any())
        local = result.query("candidate == 'g2a'")
        self.assertTrue(local.included.all())
        self.assertTrue(local.status.eq('local_evidence').all())

    def test_failed_final_is_not_promoted_by_local_evidence(self):
        args = list(self.inputs())
        args[3] = args[3].copy()
        args[3]['final_adopted'] = False
        result = build_comparison_registry(*args)
        final = result.query("candidate == 'final_combination'")
        self.assertTrue(final.included.all())
        self.assertTrue(final.status.eq('local_evidence').all())

    def test_registration_does_not_require_an_adopted_new_model(self):
        args = list(self.inputs())
        args[1]['replacement_accuracy_pass'] = False
        args[2][['accuracy_vs_h0_pass', 'paired_increment_pass']] = False
        args[3] = pd.Series({'raw_model': 'h0', 'final_adopted': False})
        result = build_comparison_registry(*args)
        self.assertIn(('h0', 'official_raw'), set(map(tuple, result[['candidate', 'reference']].to_numpy())))
        self.assertFalse((result.candidate == result.reference).any())

if __name__ == '__main__':
    unittest.main()
