"""Analytic image-motion checks only; fixtures are not physical observations."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cardcap.cards.motion_relation import compare_image_models, fit_homography, track_mask_points


def track(source, target):
    return {"eligible_for_model_comparison": True, "rejection_reasons": [],
        "features": [{"source_px": a.tolist(), "target_px": b.tolist(), "accepted": True} for a, b in zip(source, target)]}


def transform(points, matrix):
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, [2]]


class MotionRelationTests(unittest.TestCase):
    def test_shared_heldout_image_transform_does_not_promote_physical_membership(self):
        grid = np.mgrid[0:5, 0:5].T.reshape(-1, 2) * 20. + 100.
        other = grid + [200., 50.]
        matrix = np.array([[1.02, .03, 4.], [-.02, .98, -3.], [.0001, -.0001, 1.]])
        result = compare_image_models({"a": track(grid, transform(grid, matrix)), "b": track(other, transform(other, matrix))})
        self.assertEqual(result["image_motion_evidence"], "shared_planar_image_motion_compatible")
        self.assertEqual(result["physical_group_relation"], "unknown")
        self.assertFalse(result["is_lifecycle_event"])
        for score in result["heldout"].values(): self.assertLess(score["common_model_errors"]["rms_px"], .001)

    def test_different_image_transforms_preserve_unknown_physical_relation(self):
        first = np.mgrid[0:5, 0:5].T.reshape(-1, 2) * 20. + 100.
        second = first + [200., 30.]
        a = np.array([[1., .5, 10.], [-.3, 1., -7.], [0, 0, 1.]])
        b = np.array([[1., -.5, -20.], [.3, 1., 14.], [0, 0, 1.]])
        result = compare_image_models({"a": track(first, transform(first, a)), "b": track(second, transform(second, b))})
        self.assertEqual(result["image_motion_evidence"], "separate_planar_image_models_better_supported")
        self.assertEqual(result["physical_group_relation"], "unknown")
        self.assertFalse(result["physical_identity_verified"])

    def test_flat_texture_empty_target_and_collinear_points_are_unknown(self):
        gray = np.full((128, 128), 128, np.uint8); mask = np.ones_like(gray)
        flat = track_mask_points(gray, gray, mask, mask)
        self.assertIn("no_internal_image_corners", flat["rejection_reasons"])
        missing = track_mask_points(gray, gray, mask, np.zeros_like(mask))
        self.assertEqual(missing["features"], [])
        result = compare_image_models({"a": flat, "b": missing})
        self.assertEqual(result["image_motion_evidence"], "unknown")
        self.assertIsNone(result["common_model"])
        line = np.c_[np.arange(12), np.arange(12)]
        self.assertFalse(fit_homography(line, line + 1)["available"])

    def test_actual_lk_on_translated_texture_recovers_only_image_displacement(self):
        image = np.zeros((256, 256), np.uint8)
        rng = np.random.default_rng(76)
        image[50:206, 50:206] = rng.integers(0, 256, (156, 156), dtype=np.uint8)
        shifted = cv2.warpAffine(image, np.array([[1., 0., 4.], [0., 1., -3.]]), (256, 256))
        mask = np.zeros_like(image); mask[55:200, 55:200] = 255
        moved_mask = cv2.warpAffine(mask, np.array([[1., 0., 4.], [0., 1., -3.]]), (256, 256))
        result = track_mask_points(image, shifted, mask, moved_mask)
        self.assertGreaterEqual(result["accepted_correspondences"], 12)
        np.testing.assert_allclose(result["median_displacement_px"], [4., -3.], atol=.05)
        self.assertFalse(result["tracking_quality_is_pose_or_identity_confidence"])


if __name__ == "__main__": unittest.main()
