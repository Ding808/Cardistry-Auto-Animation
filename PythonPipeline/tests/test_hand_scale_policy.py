"""Synthetic scale-route contracts; not measured camera or human accuracy."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cardcap.solve.hand_scale_policy import resolve_hand_scale
from test_scale import analytic_anchor


def neutral_arrays():
    vertices = np.zeros((778, 3))
    vertices[[4, 5, 6, 443], 1] = [.05, .10, .14, .18]
    shapes = np.zeros((778, 3, 10))
    shapes[443, 1, 0] = .02
    regressor = np.zeros((16, 778))
    regressor[np.arange(16), np.arange(16)] = 1.
    weights = np.zeros((778, 16)); weights[:, 0] = 1.
    parents = np.zeros((2, 16), dtype=np.int64)
    parents[0, [4, 5, 6]] = [0, 4, 5]
    parents[1] = np.arange(16)
    return {"v_template": vertices, "shapedirs": shapes, "J_regressor": regressor,
            "weights": weights, "kintree_table": parents}


class HandScalePolicyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="hand-scale-policy-synthetic-")
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        patcher = patch("cardcap.solve.hand_scale_policy.CONFIG_DIRECTORY", self.folder)
        patcher.start(); self.addCleanup(patcher.stop)
        (self.folder / "HandSolve.json").write_text(json.dumps({}), encoding="utf-8")
        self.arrays = neutral_arrays()
        self.shape = np.zeros(10)
        self.camera = {"calibrated": False, "intrinsics_source": "prior-based"}

    def resolve(self, **kwargs):
        return resolve_hand_scale(self.arrays, self.shape, "a" * 64, "b" * 64,
                                  kwargs.pop("camera", self.camera), **kwargs)

    def assert_payload_contract(self, result):
        self.assertEqual(result["applied"], result["meters_per_unit"] is not None)
        expected_units = "relative_model_units_ue100" if result["meters_per_unit"] is None else "cm_already_scaled"
        self.assertEqual(result["payload_position_units"], expected_units)
        self.assertIsNone(result["confidence"])
        self.assertFalse(result["consumer_must_multiply_scale_again"])
        self.assertFalse(result["independent_metric_accuracy_measured"])
        self.assertFalse(result["ten_percent_accuracy_verified"])
        self.assertIsNone(result["relative_metric_error"])
        if result["meters_per_unit"] is not None:
            self.assertAlmostEqual(result["meters_per_unit"], result["global_scale_factor_applied"])
        else:
            self.assertEqual(result["global_scale_factor_applied"], 1.)
        self.assertEqual(result["source_observations_sha256"], "a" * 64)
        json.dumps(result, allow_nan=False)

    def test_unknown_scale_preserves_model_geometry_without_claiming_personal_length(self):
        original = {name: value.copy() for name, value in self.arrays.items()}
        result = self.resolve()
        self.assert_payload_contract(result)
        self.assertIsNone(result["anchor_method"])
        self.assertIsNone(result["anchor_source"])
        self.assertEqual(result["scale_confidence"], "unknown")
        self.assertIsNone(result["meters_per_unit"])
        self.assertIsNone(result["target_hand_length_mm"])
        self.assertEqual(result["provenance"]["kind"], "unobservable")
        self.assertEqual(result["neutral_hand_measurement"]["provenance"]["kind"], "inferred")
        self.assertEqual(result["anchor_frames"], [])
        self.assertEqual(result["global_scale_factor_applied"], 1.)
        self.assertEqual(result["anthropometry_provenance"]["status"], "optional_file_absent")
        for name in original:
            np.testing.assert_array_equal(self.arrays[name], original[name])
        self.assertEqual(result["neutral_hand_measurement"]["shape_source_sha256"], "a" * 64)

    def test_auto_user_length_uses_shared_shape_and_is_medium_not_independent_validation(self):
        self.shape[0] = .5  # The actual neutral tip becomes 190 mm, not 180 mm.
        path = self.folder / "UserAnthropometry.json"
        path.write_text(json.dumps({"hand_length_mm": 210., "measurement_definition": "Synthetic test assertion only"}), encoding="utf-8")
        result = self.resolve()
        self.assert_payload_contract(result)
        self.assertEqual(result["anchor_method"], "user_measured")
        self.assertEqual(result["anchor_source"], "user_anthropometry")
        self.assertEqual(result["scale_confidence"], "medium")
        self.assertIsNone(result["confidence"])
        self.assertEqual(result["provenance"]["kind"], "user_measured")
        self.assertAlmostEqual(result["global_scale_factor_applied"], .210 / .190)
        self.assertEqual(result["target_hand_length_mm"], 210.)
        self.assertEqual(result["camera_confidence_cap"], "medium")
        self.assertEqual(result["anthropometry_provenance"]["path"], str(path))
        self.assertEqual(result["m3_estimate"]["diagnostics"]["measured_hand_length_m"], .210)
        self.assertEqual(result["provenance"]["sha256"], result["anthropometry_provenance"]["sha256"])
        self.assertIsNone(result["provenance"]["uncertainty"])

    def test_supported_card_anchor_has_priority_and_uncalibrated_camera_never_yields_high(self):
        anchors = [analytic_anchor(2, 1.2), analytic_anchor(5, 1.3), analytic_anchor(8, 1.4)]
        result = self.resolve(card_anchors=anchors, anthropometry_path=self.folder / "unused_missing.json")
        self.assert_payload_contract(result)
        self.assertEqual(result["anchor_method"], "card_pnp")
        self.assertEqual(result["anchor_source"], "supported_card_pnp")
        self.assertEqual(result["anchor_frames"], [2, 5, 8])
        self.assertAlmostEqual(result["global_scale_factor_applied"], 1.3)
        self.assertEqual(result["scale_confidence"], "medium")
        self.assertIsNone(result["confidence"])
        self.assertEqual(result["camera_confidence_cap"], "medium")
        self.assertIsNone(result["target_hand_length_mm"])
        self.assertEqual(len(result["m3_estimate"]["card_anchor_evaluations"]), 3)
        self.assertTrue(all(row["accepted"] for row in result["m3_estimate"]["card_anchor_evaluations"]))
        calibrated = self.resolve(card_anchors=anchors, camera={"calibrated": True})
        self.assertIsNone(calibrated["camera_confidence_cap"])
        self.assertEqual(calibrated["scale_confidence"], "medium")

    def test_rejected_card_and_null_length_ignore_legacy_population_config(self):
        (self.folder / "UserAnthropometry.json").write_text('{"hand_length_mm": null}', encoding="utf-8")
        anchor = replace(analytic_anchor(), source_geometry_is_independent=False)
        result = self.resolve(card_anchors=[anchor], config={"population_hand_length_mm": 195.})
        self.assert_payload_contract(result)
        self.assertIsNone(result["anchor_source"])
        self.assertEqual(result["anchor_frames"], [])
        self.assertEqual(result["global_scale_factor_applied"], 1.)
        self.assertIsNone(result["meters_per_unit"])
        evaluation = result["m3_estimate"]["card_anchor_evaluations"][0]
        self.assertFalse(evaluation["accepted"])
        self.assertIn("circular_or_unverified_source_geometry", evaluation["rejection_reasons"])
        self.assertEqual(result["scale_confidence"], "unknown")

    def test_invalid_declared_measurement_or_plain_pose_is_rejected(self):
        (self.folder / "UserAnthropometry.json").write_text('{"hand_length_mm": true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hand_length_mm"):
            self.resolve()
        (self.folder / "UserAnthropometry.json").write_text('{"hand_length_mm": 210}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "measurement_definition"):
            self.resolve()
        with self.assertRaisesRegex(ValueError, "CardPnPScaleAnchor"):
            self.resolve(card_anchors=[{"accepted": True}])


if __name__ == "__main__":
    unittest.main()
