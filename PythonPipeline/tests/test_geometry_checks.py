import unittest
import numpy as np

from cardcap.geometry_checks import assess_display_samples, assess_geometry


class GeometryChecksTests(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[[0., 0., 5.], [1., 0., 5.]]])

    def test_uniform_geometry_and_denominator_scaling_invariant(self):
        a = assess_geometry(self.p, 1., global_geometry_known=True)
        b = assess_geometry(self.p * 73., 73., global_geometry_known=True)
        self.assertEqual(a, b)
        self.assertEqual(a['status'], 'passed')

    def test_depth_focal_ambiguity_detected_without_reprojection(self):
        p = self.p.copy(); p[:, :, 2] *= 9
        self.assertEqual(assess_geometry(p, 1., global_geometry_known=True)['status'], 'failed')

    def test_wrist_strict_upper_boundary(self):
        self.p[0, 1, 0] = 1.5
        r = assess_geometry(self.p, 1., global_geometry_known=True)
        self.assertEqual(r['checks']['wrist_distance_over_hand_length']['failed_frame_indices'], [0])

    def test_depth_inclusive_boundaries_and_negative_depth(self):
        for depth in (2., 10.):
            self.p[:, :, 2] = depth
            self.assertEqual(assess_geometry(self.p, 1., global_geometry_known=True)['status'], 'passed')
        self.p[:, :, 2] = -5.
        self.assertEqual(assess_geometry(self.p, 1., global_geometry_known=True)['status'], 'failed')

    def test_depth_difference_strict_boundary(self):
        self.p[0, 1, 2] = 7.
        r = assess_geometry(self.p, 1., global_geometry_known=True)
        self.assertEqual(r['checks']['depth_difference_over_hand_length']['status'], 'failed')

    def test_one_bad_frame_cannot_be_hidden_by_median(self):
        p = np.repeat(self.p, 99, axis=0); p[50, 1, 0] = 5
        r = assess_geometry(p, 1., global_geometry_known=True)
        self.assertEqual(r['checks']['wrist_distance_over_hand_length']['median'], 1.)
        self.assertEqual(r['status'], 'failed')

    def test_unknown_never_passes_even_with_placeholder_coordinates(self):
        r = assess_geometry(self.p, 1., global_geometry_known=False)
        self.assertEqual(r['status'], 'unassessable')
        self.assertIsNone(r['checks']['left_depth_over_hand_length']['values'])

    def test_unverified_hypothesis_can_fail_but_cannot_establish_acceptance(self):
        r = assess_geometry(self.p, 1., global_geometry_known=False, numeric_hypothesis_available=True)
        self.assertEqual(r['status'], 'unassessable')
        self.assertEqual(r['numeric_hypothesis_status'], 'passed')
        self.p[0, 1, 2] = 30
        self.assertEqual(assess_geometry(self.p, 1., global_geometry_known=False,
                         numeric_hypothesis_available=True)['status'], 'failed')

    def test_missing_nan_invalid_and_empty_remain_unassessable(self):
        for p, length in ((None, 1), (self.p, None), (self.p, 0), (np.empty((0, 2, 3)), 1)):
            self.assertEqual(assess_geometry(p, length, global_geometry_known=True)['status'], 'unassessable')
        self.p[0, 1, 0] = np.nan
        self.assertEqual(assess_geometry(self.p, 1., global_geometry_known=True)['status'], 'unassessable')
        self.assertEqual(assess_geometry(np.nan_to_num(self.p), 1., global_geometry_known=True,
                         frame_validity=[[True, False]])['status'], 'unassessable')

    def test_display_small_or_missing_hand_does_not_pass(self):
        samples = {s: [{'frame': 0, 'long_edge_px': 24., 'foreground_pixels': 64., 'assessable': True}]
                   for s in ('left', 'right')}
        self.assertEqual(assess_display_samples(samples)['status'], 'passed')
        samples['right'][0]['long_edge_px'] = 2
        self.assertEqual(assess_display_samples(samples)['status'], 'failed')
        samples['right'][0]['assessable'] = False
        self.assertEqual(assess_display_samples(samples)['status'], 'unassessable')
        self.assertEqual(assess_display_samples(None)['status'], 'unassessable')


if __name__ == '__main__':
    unittest.main()
