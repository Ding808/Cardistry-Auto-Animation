import unittest
from copy import deepcopy
from unittest.mock import patch
import cv2
import numpy as np
from cardcap.prepare_animation import (CAMERA_TO_UE, matrix_to_quaternion,
    quaternion_to_matrix, slerp, fill_samples, calibrated_translation, frame_ranges, camera_configuration,
    estimated_full_translation, select_animation_observations)


class IdentitySelectionTests(unittest.TestCase):
    @staticmethod
    def hand(side, ambiguous=False, value=0.):
        return {"side": side, "mano": {"shape": [value] * 10},
                "diagnostics": {"hand_identity": {"ambiguous": ambiguous, "native_side": side}}}

    def frames(self, candidates):
        return [{"frame": 0, "views": {"primary": {"hands": [self.hand("left"), self.hand("right")]}}},
                {"frame": 1, "views": {"primary": {"hands": candidates}}}]

    def test_unique_observations_remain_unchanged_even_when_uncertain(self):
        frames = self.frames([self.hand("left", True), self.hand("right")])
        before = deepcopy(frames)
        retained, excluded = select_animation_observations(frames, "primary")
        self.assertEqual(excluded, [])
        self.assertEqual([len(retained[s]) for s in ("left", "right")], [2, 2])
        self.assertIs(retained["left"][1][0], frames[1]["views"]["primary"]["hands"][0])
        self.assertEqual(frames, before)

    def test_conflicted_uncertain_hand_does_not_replace_reliable_hand_or_change_side(self):
        reliable, uncertain = self.hand("right", False, 2.), self.hand("right", True, 9.)
        for candidates in ([reliable, uncertain], [uncertain, reliable]):
            frames = self.frames(candidates)
            before = deepcopy(frames)
            retained, excluded = select_animation_observations(frames, "primary")
            self.assertIs(retained["right"][1][0], reliable)
            self.assertNotIn(1, retained["left"])
            self.assertEqual(len(excluded), 1)
            self.assertEqual(excluded[0]["source_hand_index"], candidates.index(uncertain))
            self.assertEqual(excluded[0]["reason"], "unresolved_same_side_identity")
            self.assertEqual(frames, before)

    def test_both_ambiguous_conflicting_samples_become_missing_not_arbitrary_winner(self):
        frames = self.frames([self.hand("left", True, 1.), self.hand("left", True, 2.)])
        retained, excluded = select_animation_observations(frames, "primary")
        self.assertEqual(len(excluded), 2)
        self.assertEqual([list(retained[s]) for s in ("left", "right")], [[0], [0]])
        self.assertEqual([entry["source_hand_index"] for entry in excluded], [0, 1])

    def test_unannotated_or_still_conflicting_side_is_rejected(self):
        for candidates in ([self.hand("right"), self.hand("right")],
                           [self.hand("right"), self.hand("right"), self.hand("right", True)],
                           [self.hand("right", "true"), self.hand("right")]):
            with self.assertRaisesRegex(ValueError, "duplicate hand side"):
                select_animation_observations(self.frames(candidates), "primary")

    def test_unknown_side_and_invalid_mano_are_not_hidden_by_filter(self):
        invalid = self.hand("left", True)
        invalid["mano"]["shape"][0] = float("nan")
        for candidates, message in (([self.hand("unknown", True)], "unknown hand side"),
                                    ([self.hand("left"), invalid], "finite MANO")):
            with self.assertRaisesRegex(ValueError, message):
                select_animation_observations(self.frames(candidates), "primary")

    def test_entirely_missing_hand_cannot_be_synthesized_after_filter(self):
        frames = [{"frame": 0, "views": {"primary": {"hands":
                    [self.hand("left", True), self.hand("left", True)]}}}]
        with self.assertRaisesRegex(ValueError, "Both hands need"):
            select_animation_observations(frames, "primary")


class AnimationMathTests(unittest.TestCase):
    def test_unknown_camera_cannot_promote_checkpoint_focal_to_full_image(self):
        source = {"intrinsics":{"fx":1024.,"fy":1024.,"cx":640.,"cy":360.},
                  "distortion":[0.]*5,"calibrated":False,"image_space":"original_distorted",
                  "intrinsics_source":"prior-based"}
        before = deepcopy(source)
        hands = [{"diagnostics": {"crop_focal_length_px": 4096., "crop_image_size_px": 256, "box_size_px": size}}
                 for size in (320, 400, 480)]
        resolved = camera_configuration({"resolution":[1280,720],"camera":source}, hands)
        self.assertIsNone(resolved['intrinsics'])
        self.assertFalse(resolved['calibrated'])
        self.assertIsNone(resolved['distortion'])
        self.assertIsNone(resolved['provenance']['physical_intrinsics'])
        self.assertEqual(resolved['intrinsics_source'],'unobservable')
        self.assertEqual(source,before)
        self.assertEqual(resolved,camera_configuration({"resolution":[1280,720],"camera":source}))
        for hand in hands:
            hand['diagnostics']['crop_focal_length_px'] *= 1000
        self.assertEqual(resolved,camera_configuration({"resolution":[1280,720],"camera":source},hands))
        with self.assertRaisesRegex(ValueError,'requires camera intrinsics'):
            estimated_full_translation(hands[0],resolved,'right')

    def test_focal_changes_depth_without_uniformly_scaling_crop_xy(self):
        hand = {'diagnostics':{'crop_weak_perspective_camera':[1.2,.05,-.08],
                              'box_xyxy_px':[460.,230.,640.,470.],'box_size_px':360.}}
        k = {'intrinsics':{'fx':1024.,'fy':1024.,'cx':640.,'cy':360.}}
        a = estimated_full_translation(hand,k,'right')
        k['intrinsics']['fx'] *= 1.5
        b = estimated_full_translation(hand,k,'right')
        np.testing.assert_allclose(a[:2],b[:2])
        self.assertAlmostEqual(b[2]/a[2],1.5)

    def test_legacy_crop_decode_matches_explicit_weak_camera_and_left_reflection(self):
        s,tx,ty = 1.2,.05,-.08
        model_size, model_focal = 256, 4096.
        base = {'box_xyxy_px':[460.,230.,640.,470.],'box_size_px':360.}
        weak = {'diagnostics':dict(base,crop_weak_perspective_camera=[s,tx,ty])}
        cached = {'diagnostics':dict(base,crop_camera_translation_m=[tx,ty,2*model_focal/(model_size*s)],
                                    crop_focal_length_px=model_focal,crop_image_size_px=model_size)}
        camera = {'intrinsics':{'fx':1024.,'fy':1024.,'cx':640.,'cy':360.}}
        with patch('cardcap.prepare_animation._legacy_crop_convention',return_value=(model_focal,model_size)):
            for side in ('left','right'):
                np.testing.assert_allclose(estimated_full_translation(weak,camera,side),estimated_full_translation(cached,camera,side),atol=1e-12)
        self.assertAlmostEqual(estimated_full_translation(weak,camera,'right')[0]-estimated_full_translation(weak,camera,'left')[0],2*tx)

    def test_shortest_path_across_180_degrees(self):
        a = matrix_to_quaternion(cv2.Rodrigues(np.array([0., 0., np.deg2rad(170)]))[0])
        b = matrix_to_quaternion(cv2.Rodrigues(np.array([0., 0., np.deg2rad(-170)]))[0])
        midpoint = quaternion_to_matrix(slerp(a, b, .5))
        np.testing.assert_allclose(midpoint, np.diag([-1., -1., 1.]), atol=1e-12)

    def test_antipodal_quaternions_are_same_rotation(self):
        q = np.array([.2, .3, .4, .5]); q /= np.linalg.norm(q)
        np.testing.assert_allclose(quaternion_to_matrix(slerp(q, -q, .3)), quaternion_to_matrix(q), atol=1e-12)

    def test_missing_slerp_and_linear_root_and_endpoint_hold(self):
        a = np.tile([0., 0., 0., 1.], (16, 1))
        b = np.tile(matrix_to_quaternion(cv2.Rodrigues(np.array([0., 0., np.pi / 2]))[0]), (16, 1))
        observed = {1: {"rotations": a, "translation": np.array([0., 2., 4.])},
                    3: {"rotations": b, "translation": np.array([10., 4., 8.])}}
        result = fill_samples(observed, 5)
        self.assertEqual([item["sample_kind"] for item in result],
                         ["endpoint_hold", "observed", "interpolated", "observed", "endpoint_hold"])
        np.testing.assert_allclose(result[2]["translation"], [5., 3., 6.])
        expected = cv2.Rodrigues(np.array([0., 0., np.pi / 4]))[0]
        np.testing.assert_allclose(quaternion_to_matrix(result[2]["rotations"][0]), expected, atol=1e-12)
        np.testing.assert_array_equal(observed[1]["rotations"], a)
        self.assertEqual(result[2]["source_frames"], [1, 3])

    def test_no_hand_observations_cannot_create_animation(self):
        with self.assertRaises(ValueError):
            fill_samples({}, 3)

    def test_camera_to_ue_conjugation_and_units(self):
        np.testing.assert_array_equal(CAMERA_TO_UE @ [1., 2., 3.] * 100, [300., 100., -200.])
        rotation = cv2.Rodrigues(np.array([.4, -.3, .2]))[0]
        point = np.array([.3, .8, -.2])
        converted = CAMERA_TO_UE @ rotation @ CAMERA_TO_UE.T
        np.testing.assert_allclose(converted @ (CAMERA_TO_UE @ point), CAMERA_TO_UE @ (rotation @ point), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(converted), 1.)

    def test_actual_intrinsics_translation_fit(self):
        points = np.array([[0., 0., 0.], [.1, .1, .03], [-.1, .1, -.02], [.03, -.1, .05]])
        translation = np.array([.2, -.1, 2.])
        camera = {"intrinsics": {"fx": 1200., "fy": 1000., "cx": 650., "cy": 350.},
                  "distortion": [0.] * 5, "calibrated": True}
        transformed = points + translation
        pixels = transformed[:, :2] / transformed[:, 2:] * [1200., 1000.] + [650., 350.]
        np.testing.assert_allclose(calibrated_translation(points, pixels, camera), translation, atol=1e-12)

    def test_ranges_are_ordered_nonoverlapping(self):
        self.assertEqual(frame_ranges([6, 1, 3, 2, 8, 8]), [[1, 3], [6, 6], [8, 8]])

    def test_nonzero_lens_distortion_applied_exactly_once_for_both_pixel_spaces(self):
        # Synthetic projection regression, not an actual-camera accuracy test.
        points = np.array([[0., 0., 0.], [.1, .1, .03], [-.1, .1, -.02], [.03, -.1, .05]])
        translation = np.array([.3, -.2, 1.2])
        matrix = np.array([[1200., 0., 640.], [0., 1100., 360.], [0., 0., 1.]])
        distortion = [.2, -.05, .003, -.002, .01]
        source = {"intrinsics": {"fx": 1200., "fy": 1100., "cx": 640., "cy": 360.},
                  "distortion": distortion, "calibrated": True}
        distorted, _ = cv2.projectPoints(points, np.zeros(3), translation, matrix, np.array(distortion))
        undistorted, _ = cv2.projectPoints(points, np.zeros(3), translation, matrix, np.zeros(5))
        for space, pixels in (("original_distorted", distorted), ("undistorted_same_intrinsics", undistorted)):
            camera = camera_configuration({"resolution": [1280, 720], "camera": {**source, "image_space": space}})
            np.testing.assert_allclose(calibrated_translation(points, pixels, camera), translation, atol=2e-8)
            self.assertEqual(camera["source_lens_distortion"], distortion)
            self.assertEqual(camera["distortion"], distortion if space == "original_distorted" else [0.] * 5)
        self.assertEqual(source["distortion"], distortion)
        # The previous double-correction produces a measurable wrong translation.
        self.assertGreater(np.linalg.norm(calibrated_translation(points, undistorted, source) - translation), .001)

    def test_calibrated_pixel_space_must_be_explicit(self):
        source = {"intrinsics": {"fx": 1200., "fy": 1100., "cx": 640., "cy": 360.},
                  "distortion": [0.] * 5, "calibrated": True}
        for space in (None, "cropped_undistorted"):
            with self.assertRaisesRegex(ValueError, "image_space"):
                camera_configuration({"resolution": [1280, 720], "camera": {**source, "image_space": space}})

    def test_rectified_pixels_use_effective_k_and_zero_distortion(self):
        source_k = {"fx": 1200., "fy": 1000., "cx": 640., "cy": 360.}
        focal = np.sqrt(source_k["fx"] * source_k["fy"])
        effective = {**source_k, "fx": focal, "fy": focal}
        lens = [.2, -.05, .003, -.002, .01]
        camera = camera_configuration({"resolution": [1280, 720], "camera": {
            "intrinsics": effective, "source_intrinsics": source_k,
            "distortion": lens, "source_lens_distortion": lens, "calibrated": True,
            "image_space": "undistorted_rectified_intrinsics"}})
        points = np.array([[0., 0., 0.], [.1, .1, .03], [-.1, .1, -.02], [.03, -.1, .05]])
        translation = np.array([.3, -.2, 1.2])
        transformed = points + translation
        pixels = transformed[:, :2] / transformed[:, 2:] * focal + [640., 360.]
        np.testing.assert_allclose(calibrated_translation(points, pixels, camera), translation, atol=1e-12)
        self.assertEqual(camera["distortion"], [0.] * 5)
        self.assertEqual(camera["source_intrinsics"], source_k)
        self.assertEqual(camera["source_lens_distortion"], lens)


if __name__ == "__main__":
    unittest.main()
