import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pp_data.features import BASE_FEATURE
from pp_data.io import ModelSplit
from pp_models.rolling_protocol import load_protocol
from pp_models.two_stage_remaining_study import (branch2, branch3, branch4, branch5, evaluate_components,
                                                 load_branch1_reference)
from pp_models.study_storage import file_digest
from pp_models.two_stage_study import aggregate, risk_constraints, selection_table


def prediction_fixture(candidate_errors=None, h0_error=20.0):
    """Fixed scoring rows with 30 tail orders across 15 dates per direction."""
    rows = []
    for month in (4, 5, 6):
        for day in range(1, 6):
            for side in (0, 1):
                for tail_group, actual, base_error in (("body", 100., -10.), ("tail", 200., 30.)):
                    for _ in range(2):
                        rows.append({"day": pd.Timestamp(2026, month, day, tz="UTC"),
                                     "month": f"2026-{month:02d}", "side": side,
                                     "actual": actual, "quantity": 1., "tenor": 1800,
                                     "tail_group": tail_group, "base_error": base_error,
                                     "acceleration_group": "positive",
                                     "relative_positive_low_absolute": False})
    orders = pd.DataFrame(rows)
    errors = dict.fromkeys(("grouped_mean", "h0", "g0", "g1a", "g1b", "g2a", "g2b"), h0_error)
    errors.update(candidate_errors or {})
    frames = [orders.assign(model="official_raw", prediction=orders.actual + orders.base_error)]
    frames += [orders.assign(model=name, prediction=orders.actual + error) for name, error in errors.items()]
    return pd.concat(frames).drop(columns="base_error")


def selection_inputs(frame):
    summary, _, macro = aggregate(frame)
    first = selection_table(macro, risk_constraints(summary))
    return first, branch2(frame)


class TwoStageRemainingStudyTests(unittest.TestCase):
    def test_branch1_accuracy_candidate_can_be_rescued_by_shrinkage(self):
        frame = prediction_fixture({"g1a": -10.5})
        first, second = selection_inputs(frame)
        raw = first.set_index("model").loc["g1a"]
        self.assertTrue(raw.replacement_accuracy_pass)
        self.assertFalse(raw.directly_deliverable)

        tables, predictions = branch3(frame, first, second)
        decision = tables["selection"].iloc[0]
        self.assertEqual(decision.raw_model, "g1a")
        self.assertEqual((decision.call_weight, decision.put_weight), (.25, .25))
        self.assertTrue(decision.final_adopted)
        final = predictions.loc[predictions.model.eq("final_combination")]
        # At lambda=.25 the body/tail errors are -10.125/+19.875.
        np.testing.assert_allclose(final.loc[final.tail_group.eq("body"), "prediction"], 89.875)
        np.testing.assert_allclose(final.loc[final.tail_group.eq("tail"), "prediction"], 219.875)
        self.assertAlmostEqual(tables["macro_monthly"].set_index("model").loc[
            "final_combination", "macro_mae_per_btc"], 15.)

    def test_branch2_increment_candidate_is_not_excluded_by_raw_risk(self):
        frame = prediction_fixture({"g1a": 15., "g2a": -10.5})
        first, second = selection_inputs(frame)
        raw = second["selection"].set_index("model").loc["g2a"]
        self.assertTrue(raw.accuracy_vs_h0_pass and raw.paired_increment_pass)
        self.assertFalse(raw.retained)

        tables, _ = branch3(frame, first, second)
        decision = tables["selection"].iloc[0]
        self.assertEqual(decision.raw_model, "g2a")
        self.assertEqual((decision.call_weight, decision.put_weight), (.25, .25))
        self.assertTrue(decision.final_adopted)

    def test_shrinkage_still_must_beat_h0_after_rescuing_raw_candidate(self):
        frame = prediction_fixture({"g1a": -10.5}, h0_error=11.)
        first, second = selection_inputs(frame)
        tables, _ = branch3(frame, first, second)
        decision = tables["selection"].iloc[0]
        self.assertEqual(decision.raw_model, "g1a")
        self.assertEqual((decision.call_weight, decision.put_weight), (.25, .25))
        self.assertFalse(decision.final_adopted)

    def test_no_accuracy_increment_keeps_h0_as_raw_model(self):
        frame = prediction_fixture()
        first, second = selection_inputs(frame)
        tables, _ = branch3(frame, first, second)
        self.assertEqual(tables["selection"].iloc[0].raw_model, "h0")

    def test_g0_metrics_are_available_for_raw_model_selection(self):
        frame = prediction_fixture({"g0": -10.5})
        tables = branch2(frame)
        macro = tables["macro_monthly"].set_index("model")
        self.assertIn("g0", macro.index)
        self.assertAlmostEqual(macro.loc["g0", "macro_rmse_per_btc"], 10.5)
        self.assertAlmostEqual(macro.loc["g0", "macro_mae_per_btc"], 10.5)

    def test_g0_survives_shrinkage_and_all_three_probability_ablations(self):
        frame = prediction_fixture({"g0": -10.5})
        first, second = selection_inputs(frame)
        tables, shrink_frame = branch3(frame, first, second)
        self.assertEqual(tables["selection"].iloc[0].raw_model, "g0")
        self.assertTrue(tables["selection"].iloc[0].final_adopted)

        components = frame.loc[frame.model.eq("official_raw")].drop(columns=["model", "prediction"])
        components = components.assign(p_logistic=.5, p_direction=.25, p_direction_tenor=.75,
                                       s0=2 * (components.actual + 20.), s_g0=2 * (components.actual - 10.5))
        ablation, predictions = branch4(components, tables, shrink_frame)
        contrasts = set(map(tuple, ablation["paired_components"][["candidate", "reference"]].to_numpy()))
        self.assertTrue({("h0", "direction_frequency_s0"), ("g0", "direction_frequency_sR"),
                         ("g0", "h0"), ("direction_frequency_sR", "direction_frequency_s0"),
                         ("direction_tenor_frequency_sR", "direction_tenor_frequency_s0")}.issubset(contrasts))
        self.assertEqual(set(ablation["matrix_definition"].model), {
            "h0", "g0", "direction_frequency_s0", "direction_tenor_frequency_s0",
            "direction_frequency_sR", "direction_tenor_frequency_sR"})
        for name, body, tail in (("g0", 89.5, 189.5),
                                 ("direction_frequency_sR", 44.75, 94.75),
                                 ("direction_tenor_frequency_sR", 134.25, 284.25)):
            with self.subTest(model=name):
                result = predictions.loc[predictions.model.eq(name)]
                self.assertTrue(result.index.equals(components.index))
                np.testing.assert_allclose(result.loc[result.tail_group.eq("body"), "prediction"], body)
                np.testing.assert_allclose(result.loc[result.tail_group.eq("tail"), "prediction"], tail)
        tables["prediction_frame"] = shrink_frame
        stability = branch5(frame, components, first, second, tables, predictions, ablation)
        registry = stability["comparison_registry"]
        self.assertTrue(registry.query("candidate == 'g0' and reference == 'h0'").included.all())
        self.assertEqual(set(stability["calendar_influence"].query("window_type == 'seven_day'").total_windows), {85})
        intervals = stability["paired_intervals"]
        self.assertTrue(intervals.groupby("block_days").calendar_draws_sha256.nunique().eq(1).all())

    def test_branch1_reference_rejects_changed_provenance_or_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics").mkdir()
            (root / "provenance").mkdir()
            for name in ("selection", "macro_monthly", "daily_cohorts"):
                (root / f"metrics/{name}.csv").write_text("model\ng0\n")
            source = {"dataset_sha256": "dataset", "code_sha256": {"model.py": "code"}}
            (root / "provenance/source.json").write_text(json.dumps({"development": source, "test_csv_read": False}))
            files = list(root.glob("metrics/*.csv")) + [root / "provenance/source.json"]
            (root / "provenance/manifest.json").write_text(json.dumps({"status": "complete",
                "files": {str(p.relative_to(root)): file_digest(p) for p in files}}))
            result = load_branch1_reference(root, source)
            self.assertEqual(result["selection"].model.tolist(), ["g0"])
            with self.assertRaises(ValueError):
                load_branch1_reference(root, source | {"dataset_sha256": "changed"})
            (root / "metrics/selection.csv").write_text("model\ng1a\n")
            with self.assertRaises(ValueError):
                load_branch1_reference(root, source)

    def test_evaluate_components_preserves_g0_amount_separately_from_h0(self):
        created = pd.Series(list(pd.date_range("2026-03-10", periods=120, freq="h", tz="UTC"))
                            + [pd.Timestamp("2026-04-01T12:00Z"), pd.Timestamp("2026-04-01T13:00Z")])
        X = pd.DataFrame({"option_type": np.arange(122) % 2,
                          BASE_FEATURE: list(np.tile([80., 80., 160., 160.], 30)) + [100., 100.],
                          "remaining_seconds": 3600., "product_duration_seconds": 3600.,
                          "realized_vol_60m": .02, "log_moneyness": 0., "quantity_btc": 1.,
                          "up_vol_60m": .01, "up_vol_240m": .02,
                          "down_vol_60m": .02, "down_vol_240m": .01})
        metadata = pd.DataFrame({"created_at_utc": created,
                                 "closed_at_utc": created + pd.Timedelta(minutes=30),
                                 "expiration_at_utc": created + pd.Timedelta(hours=1)})
        pool = ModelSplit(X, pd.Series(120., index=X.index), metadata)
        predictions, components, _, _ = evaluate_components(pool, load_protocol())
        self.assertIn("s_g0", components.columns)
        # G0 mean ratio=(120/80+120/160)/2=1.125; H0 coefficient=.9.
        np.testing.assert_allclose(components.s_g0, [112.5, 112.5])
        np.testing.assert_allclose(components.s0, [90., 90.])
        np.testing.assert_allclose(predictions.loc[predictions.model.eq("g0"), "prediction"], [112.5, 112.5])


if __name__ == "__main__":
    unittest.main()
