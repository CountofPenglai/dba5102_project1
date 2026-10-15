import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pp_data.io import ModelSplit
from pp_models.calibration_models import CalibrationModel
from pp_models.rolling_protocol import load_protocol, rolling_batches
from pp_models.two_stage_study import probability_frame
from pp_models.two_stage_component_study import (AMOUNT_SPECS, PROBABILITY_SPECS, component_integrity,
                                                evaluate_component_candidates, load_settings)
from research_fixtures import fixture


class ComponentStudyTests(unittest.TestCase):
    def test_fixed_config_has_no_combination(self):
        config = load_settings()
        self.assertFalse(config['combine_components'])
        self.assertEqual(len(config['amount_candidates']), 4)
        self.assertEqual(len(config['probability_candidates']), 4)

    def test_frozen_components_equal_h0_and_original_model_reproduces(self):
        pool, protocol = fixture(), load_protocol()
        frame, _, cohorts = evaluate_component_candidates(pool, protocol)
        self.assertEqual(set(frame.model), {'official_raw', 'h0', *AMOUNT_SPECS, *PROBABILITY_SPECS})
        h0 = frame.loc[frame.model.eq('h0')]
        for name in AMOUNT_SPECS:
            part = frame.loc[frame.model.eq(name)]
            np.testing.assert_array_equal(part.probability, h0.probability)
        for name in PROBABILITY_SPECS:
            part = frame.loc[frame.model.eq(name)]
            np.testing.assert_array_equal(part.severity, h0.severity)
        for day, history, today in rolling_batches(pool, protocol):
            model = CalibrationModel('hurdle_base').fit(probability_frame(pool.X.loc[history]), pool.y.loc[history])
            np.testing.assert_allclose(h0.loc[today,'prediction'], model.predict_raw(probability_frame(pool.X.loc[today])))
        self.assertTrue(component_integrity(frame).frozen_component_exact.all())
        self.assertTrue((pd.to_datetime(cohorts.latest_training_close) < pd.to_datetime(cohorts.day)).all())

    def test_integrity_check_rejects_hidden_joint_change(self):
        frame, _, _ = evaluate_component_candidates(fixture(), load_protocol())
        frame.loc[frame.model.eq('s_contract_30'), 'probability'] += .01
        with self.assertRaises(ValueError): component_integrity(frame)

    def test_future_and_unsettled_labels_do_not_affect_earlier_predictions(self):
        pool, protocol = fixture(), load_protocol()
        idx = pool.metadata.index[pool.metadata.created_at_utc.eq(pd.Timestamp('2026-03-25T12:00Z'))][0]
        pool.metadata.loc[idx,'closed_at_utc'] = pd.Timestamp('2026-04-03T00:00Z')
        first, _, _ = evaluate_component_candidates(pool, protocol)
        y = pool.y.copy()
        y.loc[pool.metadata.created_at_utc.ge(pd.Timestamp('2026-04-02T00:00Z'))] = 100000.
        y.loc[idx] = 100000.
        second, _, _ = evaluate_component_candidates(ModelSplit(pool.X,y,pool.metadata),protocol)
        early = first.day.lt(pd.Timestamp('2026-04-03T00:00Z'))
        np.testing.assert_array_equal(first.loc[early,'prediction'], second.loc[early,'prediction'])

    def test_july_rejected_before_fitting(self):
        pool = fixture()
        pool.metadata.loc[pool.metadata.index[-1], 'created_at_utc'] = pd.Timestamp('2026-07-01T00:00Z')
        with patch('pp_models.two_stage_component_study.CalibrationModel') as fitter:
            with self.assertRaises(ValueError): evaluate_component_candidates(pool, load_protocol())
            fitter.assert_not_called()


if __name__ == '__main__': unittest.main()
