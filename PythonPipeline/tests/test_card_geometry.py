"""Algebra/API tests only: synthetic geometry is not M3 real-video evidence."""

import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from cardcap.cards.pose_pnp import CameraModel, card_object_points, estimate_card_pose
from cardcap.cards.quad_fit import QuadFitResult, fit_quad, validate_quad


DIMENSIONS = {"width_m": .0635, "height_m": .0889, "dimensions_provenance": {"kind": "user_measured", "description": "Synthetic algebra fixture declaration, not an actual card measurement", "source_sha256": "a" * 64, "is_synthetic": True}}


class CardGeometryTests(unittest.TestCase):
    def setUp(self):
        self.camera = CameraModel(np.array([[900., 0, 320.], [0, 900., 240.], [0, 0, 1.]]),
                                  np.zeros(5), True, "synthetic camera for algebra validation only")
        self.rvec = np.array([.3, -.4, .17])
        self.tvec = np.array([.014, -.023, .52])
        self.objects = card_object_points(.0635, .0889)
        self.pixels = cv2.projectPoints(self.objects, self.rvec, self.tvec,
                                        self.camera.matrix, self.camera.distortion)[0].reshape(4, 2)

    def observation(self, pixels=None):
        return QuadFitResult(self.pixels if pixels is None else pixels, True, "full",
                             "synthetic explicitly known corners; not real footage", True)

    def polygon_mask(self, points=None):
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(self.pixels if points is None else points).astype(np.int32), 255)
        return mask

    def test_rectangle_ippe_recovers_known_metric_transform_and_keeps_both_solutions(self):
        self.assertAlmostEqual(np.ptp(self.objects[:, 0]), .0635)
        self.assertAlmostEqual(np.ptp(self.objects[:, 1]), .0889)
        result = estimate_card_pose(self.observation(), self.camera, **DIMENSIONS, known_corner_order=True)
        self.assertEqual(len(result.candidates), 2)
        self.assertTrue(result.accepted)
        best = next(c for c in result.candidates if c.candidate_id == result.best_candidate_id)
        self.assertLess(np.linalg.norm(np.array(best.tvec_m) - self.tvec), 1e-9)
        target_rotation = cv2.Rodrigues(self.rvec)[0]
        self.assertLess(np.max(np.abs(np.array(best.rotation_matrix) - target_rotation)), 1e-9)
        self.assertLess(best.input_corner_reprojection_rmse_px, 1e-8)
        self.assertTrue(best.positive_depth)
        self.assertEqual(result.solver, "cv2.solvePnPGeneric/SOLVEPNP_IPPE")

    def test_unlabelled_rectangle_keeps_all_correspondences_and_semantic_ambiguities(self):
        result = estimate_card_pose(self.observation(), self.camera, **DIMENSIONS)
        self.assertEqual(len(result.candidates), 16)
        self.assertEqual(len({tuple(c.corner_indices) for c in result.candidates}), 8)
        self.assertTrue(result.ambiguities["in_plane_180_degree_semantic_ambiguity"])
        self.assertTrue(result.ambiguities["front_back_semantic_ambiguity"])
        self.assertGreaterEqual(len(result.ambiguities["near_best_candidate_ids"]), 4)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_perspective_quad_is_enclosing_and_not_forced_to_right_angles(self):
        result = fit_quad(self.polygon_mask())
        self.assertTrue(result.accepted, result.to_dict())
        hull = cv2.convexHull(cv2.findContours(self.polygon_mask(), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)[0][0]).reshape(-1, 2)
        edges = np.roll(result.corners_px, -1, axis=0) - result.corners_px
        for index, edge in enumerate(edges):
            delta = hull - result.corners_px[index]
            self.assertGreaterEqual(float(np.min(edge[0] * delta[:, 1] - edge[1] * delta[:, 0])), -1e-5)
        cosines = np.sum(edges * np.roll(edges, -1, axis=0), axis=1) / (
            np.linalg.norm(edges, axis=1) * np.roll(np.linalg.norm(edges, axis=1), -1))
        self.assertGreater(np.max(np.abs(cosines)), .04)

    def test_clipped_corner_pentagon_gets_bounded_support_quad_without_visibility_promotion(self):
        points = np.array([[100, 100], [220, 100], [300, 180], [300, 300], [100, 300]], np.int32)
        mask = self.polygon_mask(points)
        hull = cv2.convexHull(points)
        # This meaningful clipped-corner fixture exposes the old 4-vertex-only search gap.
        perimeter = cv2.arcLength(hull, True)
        self.assertNotIn(4, [len(cv2.approxPolyDP(hull, float(e * perimeter), True)) for e in np.linspace(.002, .05, 25)])
        result = fit_quad(mask, visibility="unknown")
        self.assertIsNotNone(result.corners_px, result.to_dict())
        self.assertTrue(result.metrics["support_line_fallback"]["used"])
        self.assertLessEqual(result.metrics["support_line_fallback"]["tested_four_line_combinations"], 1820)
        self.assertTrue(result.accepted, result.to_dict())
        self.assertFalse(result.full_face_verified)
        edges = np.roll(result.corners_px, -1, axis=0) - result.corners_px
        for index, edge in enumerate(edges):
            delta = points - result.corners_px[index]
            self.assertGreaterEqual(float(np.min(edge[0] * delta[:, 1] - edge[1] * delta[:, 0])), -1e-5)
        self.assertFalse(estimate_card_pose(result, self.camera, **DIMENSIONS).scale_anchor_eligible)

    def test_unknown_partial_and_border_faces_cannot_be_scale_anchors(self):
        for visibility in ("unknown", "partial"):
            quad = fit_quad(self.polygon_mask(), visibility=visibility)
            self.assertTrue(quad.accepted)
            self.assertFalse(quad.full_face_verified)
            self.assertFalse(estimate_card_pose(quad, self.camera, **DIMENSIONS).scale_anchor_eligible)
        border = self.polygon_mask(np.array([[0, 50], [90, 50], [90, 180], [0, 180]]))
        quad = fit_quad(border, visibility="full", visibility_evidence="synthetic assertion")
        self.assertTrue(quad.metrics["touches_image_border"])
        self.assertFalse(quad.full_face_verified)
        with self.assertRaisesRegex(ValueError, "evidence"):
            fit_quad(self.polygon_mask(), visibility="full")

    def test_calibration_flag_is_preserved_and_fit_residual_is_not_validation(self):
        camera = CameraModel(self.camera.matrix, self.camera.distortion, False,
                             "assumed intrinsics; no calibration")
        result = estimate_card_pose(self.observation(), camera, **DIMENSIONS, known_corner_order=True)
        self.assertTrue(result.accepted)
        self.assertFalse(result.calibrated)
        self.assertFalse(result.scale_anchor_eligible)
        self.assertIsNone(result.to_dict()["independent_corner_validation_error_px"])
        self.assertFalse(result.to_dict()["physical_scale_ground_truth_validated"])

    def test_degenerate_masks_holes_components_and_probabilities_rejected(self):
        for mask in (np.zeros((40, 40), np.uint8), np.eye(40, dtype=np.uint8)):
            self.assertFalse(fit_quad(mask).accepted)
        hole = self.polygon_mask()
        hole[210:245, 315:345] = 0
        self.assertFalse(fit_quad(hole).accepted)
        components = self.polygon_mask()
        components[30:70, 30:70] = 255
        self.assertFalse(fit_quad(components).accepted)
        with self.assertRaisesRegex(ValueError, "threshold"):
            fit_quad(np.full((40, 40), .9))
        with self.assertRaises(ValueError):
            fit_quad(np.full((40, 40), np.nan))

    def test_dimensions_cannot_default_or_claim_scale_when_inferred(self):
        with self.assertRaises(TypeError):
            card_object_points()
        with self.assertRaises(TypeError):
            estimate_card_pose(self.observation(), self.camera)
        with self.assertRaisesRegex(ValueError, "provenance"):
            estimate_card_pose(self.observation(), self.camera, width_m=.07, height_m=.1, dimensions_provenance={})
        data = dict(DIMENSIONS, dimensions_provenance=dict(DIMENSIONS["dimensions_provenance"], kind="inferred"))
        result = estimate_card_pose(self.observation(), self.camera, **data, known_corner_order=True)
        self.assertTrue(result.accepted)
        self.assertFalse(result.scale_anchor_eligible)
        self.assertEqual(result.to_dict()["dimensions_provenance"]["kind"], "inferred")

    def test_bad_corner_order_camera_and_dimensions_rejected(self):
        for points in (self.pixels[[0, 2, 1, 3]], np.zeros((4, 2)),
                       np.array([[0., 0], [1., 0], [2., 0], [3., 0]])):
            with self.assertRaises(ValueError):
                validate_quad(points)
        with self.assertRaises(ValueError):
            CameraModel(np.eye(3), np.zeros(5), "false", "invalid bool")
        with self.assertRaises(ValueError):
            CameraModel(np.diag([-1, 1, 1]), np.zeros(5), False, "invalid focal")
        with self.assertRaises(ValueError):
            CameraModel(self.camera.matrix, [0, np.nan, 0, 0], False, "invalid distortion")
        with self.assertRaises(ValueError):
            card_object_points(0, .08)

    def test_nonfinite_and_negative_depth_solver_candidates_retained_but_invalid(self):
        # Controlled invalid solver output verifies gates, not a real pose result.
        rotations = (np.full((3, 1), np.nan), np.zeros((3, 1)))
        translations = (np.ones((3, 1)), np.array([[0.], [0.], [-.5]]))
        with patch("cardcap.cards.pose_pnp.cv2.solvePnPGeneric", return_value=(2, rotations, translations, None)):
            result = estimate_card_pose(self.observation(), self.camera, **DIMENSIONS, known_corner_order=True)
        self.assertEqual(len(result.candidates), 2)
        self.assertFalse(result.accepted)
        reasons = [reason for c in result.candidates for reason in c.rejection_reasons]
        self.assertIn("nonfinite_solver_result", reasons)
        self.assertIn("one_or_more_corners_not_in_front_of_camera", reasons)
        json.dumps(result.to_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
