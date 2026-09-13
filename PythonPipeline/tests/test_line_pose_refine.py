"""Synthetic mathematical fixtures only; no real-video accuracy claim."""
import json
import unittest

import cv2
import numpy as np

from cardcap.cards.line_pose_refine import refine_pose_from_segments


class LinePoseMathTests(unittest.TestCase):
    def setUp(self):
        self.dimensions = [.0635, .0889]
        self.k = np.array([[900., 0, 320.], [0, 900., 240.], [0, 0, 1.]])
        self.d = np.zeros(5)
        self.truth_rvec = np.array([.3, -.4, .17])
        self.truth_tvec = np.array([.014, -.023, .52])
        w, h = self.dimensions
        local = np.array([[-w/2, -h/2, 0.], [w/2, -h/2, 0.], [w/2, h/2, 0.], [-w/2, h/2, 0.]])
        projected = cv2.projectPoints(local, self.truth_rvec, self.truth_tvec, self.k, self.d)[0].reshape(4, 2)
        self.segments = [{"edge_index": i, "points_px": [(projected[i]*.8+projected[(i+1)%4]*.2).tolist(),
                            (projected[i]*.2+projected[(i+1)%4]*.8).tolist()], "weight": 1.} for i in range(4)]
        self.initial = {"rvec": (self.truth_rvec+np.array([.025, -.02, .01])).tolist(),
                        "tvec_m": (self.truth_tvec+np.array([.003, -.002, .012])).tolist()}

    def test_perturbed_known_pose_converges_without_changing_inputs(self):
        before = json.dumps([self.initial, self.segments])
        result = refine_pose_from_segments(self.initial, self.dimensions, self.k, self.d, self.segments)
        self.assertTrue(result["accepted"], result)
        self.assertTrue(result["converged"])
        self.assertEqual(result["jacobian_rank"], 6)
        self.assertLess(np.max(np.abs(np.asarray(result["rotation_matrix"])-cv2.Rodrigues(self.truth_rvec)[0])), 1e-5)
        self.assertLess(np.linalg.norm(np.asarray(result["tvec_m"])-self.truth_tvec), 1e-6)
        self.assertLess(np.max(np.abs(result["optimized_endpoint_residuals_px"])), 1e-5)
        self.assertLess(result["optimized_huber_cost"], result["source_huber_cost"])
        self.assertLessEqual(result["local_constraint_diagnostics"]["max_projected_corner_movement_px"], 40.)
        self.assertFalse(result["ambiguity_resolved"])
        self.assertEqual(json.dumps([self.initial, self.segments]), before)
        json.dumps(result, allow_nan=False)
        regularized = refine_pose_from_segments(self.initial, self.dimensions, self.k, self.d,
                                               self.segments, corner_prior_sigma_px=12.)
        self.assertTrue(regularized["accepted"], regularized)
        self.assertEqual(regularized["jacobian_rank"], 6)
        self.assertFalse(regularized["prior_strength"]["prior_included_in_observation_rank_or_condition"])
        self.assertGreater(regularized["optimized_prior_cost"], 0.)
        self.assertAlmostEqual(regularized["optimized_total_cost"], regularized["optimized_data_cost"]+regularized["optimized_prior_cost"])
        self.assertLess(regularized["optimized_total_cost"], regularized["source_total_cost"])

    def test_only_two_edges_rejected_without_output_pose(self):
        result = refine_pose_from_segments(self.initial, self.dimensions, self.k, self.d, self.segments[:2])
        self.assertFalse(result["accepted"])
        self.assertIn("fewer_than_three_distinct_model_edges", result["rejection_reasons"])
        self.assertIsNone(result["rvec"])
        self.assertIsNone(result["tvec_m"])
        self.assertIsNone(result["rotation_matrix"])
        self.assertEqual(result["iterations"], 0)
        self.assertFalse(result["ambiguity_resolved"])
        with_prior = refine_pose_from_segments(self.initial, self.dimensions, self.k, self.d,
                                              self.segments[:2], corner_prior_sigma_px=12.)
        self.assertFalse(with_prior["accepted"])
        self.assertIn("fewer_than_three_distinct_model_edges", with_prior["rejection_reasons"])
        self.assertIsNone(with_prior["rvec"])


if __name__ == "__main__":
    unittest.main()
