"""Analytic projection checks, not real-camera calibration measurements."""
import unittest
import cv2
import numpy as np

from cardcap.cards.pose_pnp import card_object_points
from cardcap.cards.rectangle_camera import estimate_rectangle_focal, analyze_edge_assignments


def projected(rotation=(.6, -.5, .2)):
    matrix = np.array([[1800., 0, 640], [0, 1800., 360], [0, 0, 1]])
    points, _ = cv2.projectPoints(card_object_points(.0635, .0889), np.array(rotation, np.float64),
                                  np.array([.01, -.02, .8]), matrix, np.zeros(5))
    return points.reshape(4, 2)


class RectangleCameraTests(unittest.TestCase):
    def test_known_perspective_recovers_focal_without_calibrated_claim(self):
        result = estimate_rectangle_focal(projected(), width=.0635, height=.0889, resolution=(1280, 720))
        self.assertTrue(result['accepted_under_assumptions'])
        self.assertAlmostEqual(result['focal_px'], 1800., places=7)
        self.assertFalse(result['camera']['calibrated'])
        self.assertFalse(result['scale_anchor_eligible'])

    def test_frontoparallel_does_not_invent_focal(self):
        result = estimate_rectangle_focal(projected((0., 0., 0.)), width=.0635, height=.0889, resolution=(1280, 720))
        self.assertFalse(result['accepted_under_assumptions'])
        self.assertIsNone(result['camera'])

    def test_absolute_object_scale_cancels(self):
        first = estimate_rectangle_focal(projected(), width=.0635, height=.0889, resolution=(1280, 720))
        second = estimate_rectangle_focal(projected(), width=63.5, height=88.9, resolution=(1280, 720))
        self.assertAlmostEqual(first['focal_px'], second['focal_px'], places=7)

    def test_unordered_edges_and_perturbations_do_not_select_camera(self):
        result = analyze_edge_assignments(projected(), width=.0635, height=.0889, resolution=(1280, 720), perturbations=16)
        self.assertEqual(len(result['branches']), 2)
        self.assertIsNone(result['selected_branch'])
        self.assertFalse(result['calibrated'])
        self.assertFalse(result['branches'][0]['perturbation_sensitivity']['is_statistical_confidence_interval'])

    def test_invalid_quad_and_numeric_inputs_rejected(self):
        with self.assertRaises(ValueError):
            estimate_rectangle_focal(projected()[[0, 2, 1, 3]], width=.0635, height=.0889, resolution=(1280, 720))
        with self.assertRaises(ValueError):
            estimate_rectangle_focal(projected(), width=True, height=.0889, resolution=(1280, 720))

    def test_inconsistent_positive_focal_does_not_return_camera(self):
        corners = projected()
        corners[0, 1] += 25.
        result = estimate_rectangle_focal(corners, width=.0635, height=.0889, resolution=(1280, 720))
        self.assertFalse(result['accepted_under_assumptions'])
        self.assertIn('rectangle_axis_length_inconsistent', result['rejection_reasons'])
        self.assertGreater(result['focal_px'], 0.)  # Retained diagnostic only.
        self.assertIsNone(result['camera'])
        self.assertEqual(result['focal_reliability'], 'unknown')
        self.assertEqual(result['focal_identifiability'], 'unknown')

    def test_near_frontoparallel_noise_is_explicitly_unstable(self):
        corners = projected((0., 0., 0.))
        corners[0, 0] -= .1  # Analytic 1800px camera with a bounded corner error.
        result = analyze_edge_assignments(corners, width=.0635, height=.0889, resolution=(1280, 720),
                                          corner_uncertainty_px=.1, perturbations=32)
        branch = result['branches'][0]
        self.assertTrue(branch['base']['accepted_under_assumptions'])
        self.assertEqual(branch['base']['focal_reliability'], 'unknown')
        self.assertGreater(branch['base']['focal_px'], 2 * 1800.)
        self.assertEqual(branch['focal_reliability'], 'unstable')
        sensitivity = branch['perturbation_sensitivity']
        self.assertEqual(sensitivity['stability_status'], 'unstable')
        self.assertLess(sensitivity['consistent_trial_fraction'], .9)
        self.assertGreater(sensitivity['consistent_focal_relative_spread'], .5)
        self.assertEqual(set(sensitivity['instability_reasons']), {
            'consistent_trial_fraction_below_threshold', 'relative_focal_spread_above_threshold'})
        self.assertFalse(sensitivity['is_statistical_confidence_interval'])
        self.assertIsNone(result['selected_branch'])
        self.assertFalse(result['calibrated'])
        stable_probe = analyze_edge_assignments(projected(), width=.0635, height=.0889, resolution=(1280, 720),
                                                corner_uncertainty_px=.1, perturbations=32)['branches'][0]
        self.assertEqual(stable_probe['perturbation_sensitivity']['stability_status'], 'no_instability_flag')
        self.assertEqual(stable_probe['focal_reliability'], 'unknown')


if __name__ == '__main__':
    unittest.main()
