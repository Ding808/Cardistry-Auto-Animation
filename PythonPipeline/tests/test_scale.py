"""Analytic scale-contract tests; fixtures are not video evidence or anchors."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardcap.solve.scale import (CardPnPScaleAnchor, ScaleEstimate, evaluate_card_anchor,
    estimate_scene_scale, measure_neutral_mano_hand, scale_reconstruction_geometry)


def analytic_anchor(frame=0, ratio=2., kind="same_segment"):
    metric = np.array([[-.03175, 0., 2.], [.03175, 0., 2.]])
    local = metric - [0., 0., 2.]
    if kind == "same_camera_point": metric, local = metric[:1], local[:1]
    evidence = {"description": "ANALYTIC TEST FIXTURE ONLY, not a real measurement", "sha256": "a" * 64}
    return CardPnPScaleAnchor(frame, "test", "b" * 64, "analytic", {
        "accepted": True, "calibrated": True, "full_face_verified": True, "scale_anchor_eligible": True,
        "ambiguities": {"near_best_candidate_ids": ["test_pose"]},
        "dimensions_m": [.0635, .0889], "candidates": [{"candidate_id": "test_pose", "valid": True,
            "rotation_matrix": np.eye(3).tolist(), "tvec_m": [0, 0, 2], "input_corner_reprojection_rmse_px": 0}]},
        "test_pose", metric / ratio, local, kind, "c" * 64, "independent_scene_reconstruction", True,
        "test_camera", "test_camera", dict(evidence, physical_correspondence_verified=True),
        dict(evidence, physical_complete_face_verified=True), dict(evidence, calibrated=True),
        dict(evidence, dimensions_apply_to_this_card_verified=True))


class ScaleTests(unittest.TestCase):
    def test_card_median_is_per_frame_and_robust_to_one_outlier(self):
        anchors = [analytic_anchor(0, 1.), analytic_anchor(1, 1.02), analytic_anchor(2, 100.)]
        anchors += [analytic_anchor(0, 1.) for _ in range(8)]
        result = estimate_scene_scale(scale_group="analytic", card_anchors=anchors)
        self.assertAlmostEqual(result.meters_per_source_unit, 1.02)
        self.assertEqual(result.anchor_frames, [0, 1, 2])
        self.assertEqual(len(result.diagnostics["outlier_diagnostic_frames"]), 1)
        self.assertFalse(result.to_dict()["ten_percent_accuracy_verified"])

    def test_perfect_pnp_fit_does_not_authorize_circular_or_uncalibrated_scale(self):
        anchor = analytic_anchor()
        bad = replace(anchor, source_geometry_is_independent=False)
        self.assertIn("circular_or_unverified_source_geometry", evaluate_card_anchor(bad)["rejection_reasons"])
        pose = dict(anchor.pose, calibrated=False)
        self.assertFalse(evaluate_card_anchor(replace(anchor, pose=pose))["accepted"])
        estimate = estimate_scene_scale(scale_group="analytic", card_anchors=[bad])
        self.assertIsNone(estimate.meters_per_source_unit)
        self.assertIsNone(estimate.anchor_method)
        self.assertEqual(estimate.global_scale_factor, 1.)
        self.assertEqual(estimate.provenance["kind"], "unobservable")
        self.assertFalse(estimate.card_anchor_evaluations[0]["accepted"])

    def test_point_requires_shared_origin_direction_and_scale_group(self):
        anchor = analytic_anchor(kind="same_camera_point")
        self.assertAlmostEqual(evaluate_card_anchor(anchor)["meters_per_source_unit"], 2.)
        self.assertFalse(evaluate_card_anchor(replace(anchor, source_camera_frame="different"))["accepted"])
        self.assertFalse(evaluate_card_anchor(replace(anchor, source_points_units=anchor.source_points_units + [1, 0, 0]))["accepted"])
        estimate = estimate_scene_scale(scale_group="different", card_anchors=[anchor])
        self.assertIsNone(estimate.meters_per_source_unit)
        self.assertIn("different_source_scale_group", estimate.card_anchor_evaluations[0]["rejection_reasons"])

    def test_distant_valid_pnp_candidate_cannot_borrow_near_best_scale_eligibility(self):
        anchor = analytic_anchor(kind="same_camera_point")
        result = evaluate_card_anchor(anchor)
        self.assertTrue(result["accepted"] and result["source_geometry_is_independent"])
        # Scalar alignment still fits perfectly; eligibility belongs to another
        # candidate set. This must reject before treating the ratio as an anchor.
        pose = dict(anchor.pose, ambiguities={"near_best_candidate_ids": ["another_pose"]})
        result = evaluate_card_anchor(replace(anchor, pose=pose))
        self.assertFalse(result["accepted"])
        self.assertIn("selected_pnp_candidate_outside_scale_qualified_near_best_set", result["rejection_reasons"])

    def test_missing_neutral_geometry_and_zero_segment_fail(self):
        with self.assertRaisesRegex(ValueError, "neutral MANO geometry unavailable"):
            measure_neutral_mano_hand({}, np.zeros(10), scale_group="test", model_sha256="a" * 64, shape_source_sha256="b" * 64)
        a = analytic_anchor()
        bad = replace(a, source_points_units=np.ones((2, 3)))
        self.assertIn("degenerate_reference_segment", evaluate_card_anchor(bad)["rejection_reasons"])

    def test_uniform_root_local_and_packet_scale_preserves_projection_and_lengths(self):
        s = ScaleEstimate("user_measured", "test", 1.8, 1., "medium")
        roots = np.array([[.3, -.2, 3.], [-.4, .1, 2.]])
        local = np.array([[[0., 0., 0.], [.1, .02, -.05]], [[0., 0., 0.], [-.08, .04, .01]]])
        packets = np.array([[.4, .1, 2.5]])
        original = local.copy()
        result = scale_reconstruction_geometry(s, scale_group="test", hand_roots_units=roots, hand_local_joints_units=local,
            packet_positions_units=packets, packet_space="same_unscaled_camera_scale_group")
        before, after = roots[:, None] + local, result["hand_joints_camera_m"]
        np.testing.assert_allclose(before[..., :2] / before[..., [2]], after[..., :2] / after[..., [2]], atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(np.diff(result["hand_local_joints_m"], axis=1), axis=-1),
                                   1.8 * np.linalg.norm(np.diff(local, axis=1), axis=-1))
        np.testing.assert_array_equal(local, original)
        np.testing.assert_allclose(result["packet_positions_m"], packets * 1.8)
        with self.assertRaisesRegex(ValueError, "do not scale metric PnP again"):
            scale_reconstruction_geometry(s, scale_group="test", hand_roots_units=roots, hand_local_joints_units=local,
                packet_positions_units=packets, packet_space="already_metric")

    def test_unknown_scale_cannot_export_metre_arrays_or_accept_unproven_length(self):
        unknown = estimate_scene_scale(scale_group="test")
        with self.assertRaisesRegex(ValueError, "Metric scale unavailable"):
            scale_reconstruction_geometry(unknown, scale_group="test", hand_roots_units=np.zeros((1, 3)),
                                          hand_local_joints_units=np.zeros((1, 2, 3)))
        with self.assertRaisesRegex(ValueError, "measurement_provenance"):
            estimate_scene_scale(scale_group="test", measured_hand_length_m=.21)
        with self.assertRaises(TypeError):
            estimate_scene_scale(scale_group="test", hand_prior_m=.21)


if __name__ == "__main__": unittest.main()
