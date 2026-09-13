"""Synthetic display contracts; no camera calibration or model accuracy claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cardcap.display_space import (CAMERA_TO_UE, DisplayInputError, build_display_space,
    crop_display_affine, select_display_focal, source_crop_diagnostics)
from cardcap.geometry_checks import assess_geometry


def observation(side, center=(640., 360.), weak=(1.2, .02, -.03)):
    x, y = center
    return {"side": side, "handedness_confidence": .9, "image_landmarks_px": [[x, y]] * 21,
        "mano": {"shape": [0.] * 10, "global_orient_axis_angle": [0.] * 3,
                 "hand_pose_axis_angle": [[0.] * 3] * 15},
        "diagnostics": {"crop_weak_perspective_camera": list(weak),
            "crop_camera_translation_m": [weak[1], weak[2], 2. * 4096. / (256 * weak[0])],
            "crop_focal_length_px": 4096., "crop_image_size_px": 256,
            "box_xyxy_px": [x - 110., y - 90., x + 110., y + 90.], "box_size_px": 330.}}


class DisplaySpaceTests(unittest.TestCase):
    def source(self):
        samples = {side: {frame: (observation(side, (560. + si * 150. + frame * 4., 310. + si * 70.),
                    (1.2 + si * .1, .02 + frame * .001, -.03)), {}) for frame in (1, 3)}
                   for si, side in enumerate(("left", "right"))}
        wrists = {side: {frame: np.array([.008 * frame, -.006, .009]) for frame in (1, 3)}
                  for side in samples}
        return samples, wrists

    def test_native_crop_depth_and_left_reflection_are_retained(self):
        hand = observation("left")
        before = deepcopy(hand)
        c_left, slope = crop_display_affine(hand, "left", (1280, 720), [0., 0., .02])
        c_right, _ = crop_display_affine(hand, "right", (1280, 720), [0., 0., .02])
        self.assertAlmostEqual(c_right[0] - c_left[0], .04)
        np.testing.assert_array_equal(slope[:2], [0., 0.])
        np.testing.assert_array_equal((c_left + 1024. * slope)[:2], (c_left + 2048. * slope)[:2])
        self.assertEqual(hand, before)
        kept = source_crop_diagnostics(hand)
        self.assertEqual(kept["source_crop_camera_translation_model_units"], before["diagnostics"]["crop_camera_translation_m"])
        self.assertNotEqual(kept["source_crop_camera_translation_model_units"][2], (c_left + 1024. * slope)[2])
        kept["source_crop_camera_translation_model_units"][2] = -1.
        self.assertEqual(hand, before)

    def test_legacy_native_depth_decodes_same_affine_without_becoming_full_image_depth(self):
        hand = observation("right")
        expected = crop_display_affine(hand, "right", (1280, 720), [0., 0., 0.])
        del hand["diagnostics"]["crop_weak_perspective_camera"]
        actual = crop_display_affine(hand, "right", (1280, 720), [0., 0., 0.])
        np.testing.assert_allclose(actual, expected)
        del hand["diagnostics"]["crop_focal_length_px"]
        with self.assertRaisesRegex(ValueError, "recorded model convention"):
            crop_display_affine(hand, "right", (1280, 720), [0., 0., 0.])
        np.testing.assert_allclose(crop_display_affine(hand, "right", (1280, 720), [0., 0., 0.],
            legacy_crop_convention=(4096., 256)), expected)

    def test_rotating_image_crop_and_geometry_preserves_focal_and_ratios(self):
        source, wrists = self.source()
        before = deepcopy(source)
        normal = build_display_space(source, wrists, 5, (1280, 720), 30., .18)
        rotated, rotated_wrists = deepcopy(source), deepcopy(wrists)
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        for side, samples in rotated.items():
            for frame, (hand, _) in samples.items():
                d = hand["diagnostics"]
                x0, y0, x1, y1 = d["box_xyxy_px"]
                d["box_xyxy_px"] = [720 - y1, x0, 720 - y0, x1]
                scale, tx, ty = d["crop_weak_perspective_camera"]
                xy = rotation @ [(-tx if side == "left" else tx), ty, 0.]
                d["crop_weak_perspective_camera"] = [scale, -xy[0] if side == "left" else xy[0], xy[1]]
                d["crop_camera_translation_m"][:2] = d["crop_weak_perspective_camera"][1:]
                hand["image_landmarks_px"] = [[720 - y, x] for x, y in hand["image_landmarks_px"]]
                rotated_wrists[side][frame] = rotation @ wrists[side][frame]
        portrait = build_display_space(rotated, rotated_wrists, 5, (720, 1280), 30., .18)
        self.assertAlmostEqual(normal["display_assumed_focal_px"], portrait["display_assumed_focal_px"], places=10)
        self.assertEqual(normal["selection"]["failed_frame_indices"], portrait["selection"]["failed_frame_indices"])
        for a, b in zip(normal["frames"], portrait["frames"]):
            self.assertAlmostEqual(a["wrist_distance_over_hand_length"], b["wrist_distance_over_hand_length"], places=12)
            for side in source:
                t = a["hands"][side]["wrist_translation_display_units"]
                transformed = CAMERA_TO_UE @ rotation @ CAMERA_TO_UE.T @ t
                np.testing.assert_allclose(b["hands"][side]["wrist_translation_display_units"], transformed, atol=1e-10)
        self.assertEqual(source, before)

    def test_gate_selection_checks_every_frame_instead_of_accepting_a_median(self):
        constants = np.zeros((3, 2, 3))
        constants[:, 1, 0] = .4
        slopes = np.zeros_like(constants)
        slopes[:2, :, 2] = .001
        slopes[2, :, 2] = .01
        before = constants.copy(), slopes.copy()
        focal, selection = select_display_focal(constants, slopes, 1.)
        self.assertEqual(selection["failed_frame_count"], 1)
        self.assertEqual(selection["frame_gate_failure_count"], 2)
        report = assess_geometry(constants + focal * slopes, 1., global_geometry_known=False, numeric_hypothesis_available=True)
        self.assertEqual(report["numeric_hypothesis_status"], "failed")
        self.assertFalse(report["physical_geometry_validated"])
        for alternative in np.geomspace(10., 100000., 600):
            trial = assess_geometry(constants + alternative * slopes, 1., global_geometry_known=False, numeric_hypothesis_available=True)
            failures = [x["failed_frame_indices"] for x in trial["checks"].values()]
            self.assertGreaterEqual((len(set(sum(failures, []))), sum(map(len, failures))),
                                    (selection["failed_frame_count"], selection["frame_gate_failure_count"]))
        np.testing.assert_array_equal(constants, before[0])
        np.testing.assert_array_equal(slopes, before[1])

    def test_analytic_selection_finds_a_narrow_feasible_interval(self):
        constants = np.zeros((2, 2, 3))
        constants[:, 1, 0] = .2
        slopes = np.zeros_like(constants)
        slopes[0, :, 2] = 2. / 1234.567
        slopes[1, :, 2] = 10. / 1234.568
        focal, report = select_display_focal(constants, slopes, 1.)
        self.assertEqual(report["failed_frame_count"], 0)
        self.assertGreaterEqual(focal, 1234.567 - 1e-10)
        self.assertLessEqual(focal, 1234.568 + 1e-10)
        self.assertFalse(report["selected_at_gate_boundary"])

    def test_infeasible_lateral_distance_is_reported_without_hand_attraction(self):
        constants = np.zeros((2, 2, 3))
        constants[:, 1, 0] = 1.5
        slopes = np.zeros_like(constants)
        slopes[:, :, 2] = .001
        focal, report = select_display_focal(constants, slopes, 1.)
        self.assertEqual(report["failed_frame_count"], 2)
        self.assertEqual(report["per_gate_failed_frame_count"]["wrist_distance_over_hand_length"], 2)
        np.testing.assert_array_equal((constants + focal * slopes)[:, 1, 0], [1.5, 1.5])

    def test_nonpositive_depth_center_reference_does_not_exclude_feasible_focals(self):
        constants = np.zeros((1, 2, 3))
        constants[:, :, 2] = 7.
        constants[:, 1, 0] = .4
        slopes = np.zeros_like(constants)
        slopes[:, :, 2] = .001
        focal, report = select_display_focal(constants, slopes, 1.)
        self.assertGreater(focal, 0.)
        self.assertLess(focal, 3000.)
        self.assertEqual(report["failed_frame_count"], 0)
        self.assertEqual(report["tie_reference_source"], "positive_gate_boundaries")
        self.assertFalse(report["selected_at_gate_boundary"])

    def test_no_positive_gate_boundary_reports_infeasible_hypothesis_without_raising(self):
        constants = np.zeros((1, 2, 3))
        constants[:, :, 2] = 11.
        slopes = np.zeros_like(constants)
        slopes[:, :, 2] = .001
        focal, report = select_display_focal(constants, slopes, 1.)
        self.assertGreater(focal, 0.)
        self.assertEqual(report["failed_frame_count"], 1)
        self.assertEqual(report["frame_gate_failure_count"], 2)
        self.assertEqual(report["tie_reference_source"], "one_hand_length_depth_increment")

    def test_missing_display_samples_preserve_source_labels_and_adjustable_affine(self):
        samples, wrists = self.source()
        result = build_display_space(samples, wrists, 5, (1280, 720), 30., .18)
        self.assertTrue(result["display_only"])
        self.assertIsNone(result["source_camera_intrinsics"])
        self.assertIsNone(result["source_camera_distortion"])
        self.assertFalse(result["source_inter_hand_transform_known"])
        self.assertFalse(result["metric_scale_validated"])
        for side in samples:
            hands = [row["hands"][side] for row in result["frames"]]
            self.assertEqual([row["sample_kind"] for row in hands],
                ["endpoint_hold", "detected_model_observation", "interpolated", "detected_model_observation", "endpoint_hold"])
            self.assertEqual(hands[2]["source_observation_frames"], [1, 3])
            self.assertEqual(hands[2]["interpolation_alpha"], .5)
            self.assertIsNone(hands[2]["source_crop_camera_translation_model_units"])
            for row in hands:
                c = np.array(row["wrist_translation_constant_display_units"])
                s = np.array(row["wrist_translation_per_focal_px_display_units"])
                np.testing.assert_allclose(row["wrist_translation_display_units"], c + result["display_assumed_focal_px"] * s)
            for field in ("wrist_translation_constant_display_units", "wrist_translation_per_focal_px_display_units"):
                np.testing.assert_allclose(hands[2][field], (np.array(hands[1][field]) + hands[3][field]) / 2.)
        json.dumps(result, allow_nan=False)

    def test_invalid_crop_and_depth_cannot_silently_create_display_geometry(self):
        for field, value in (("crop_weak_perspective_camera", [0., 0., 0.]),
                             ("box_size_px", float("nan")), ("box_xyxy_px", [1., 2., 0., 4.])):
            hand = observation("right")
            hand["diagnostics"][field] = value
            with self.assertRaises(DisplayInputError):
                crop_display_affine(hand, "right", (1280, 720), [0., 0., 0.])
        for diagnostics in ({}, {"crop_weak_perspective_camera": [1., "invalid", 0.]},
                            {"crop_weak_perspective_camera": [1., 0., 0.], "box_size_px": []}):
            with self.assertRaises(DisplayInputError):
                crop_display_affine({"diagnostics": diagnostics}, "left", (1280, 720), [0., 0., 0.])


class DisplayExportIntegrationTests(unittest.TestCase):
    def test_export_binds_separate_sidecar_and_preserves_null_source_camera_and_globals(self):
        from cardcap import prepare_animation as prepare
        pipeline = Path(prepare.__file__).resolve().parents[1]
        mapping = json.loads((pipeline.parent / "Config/BoneMapping_UE5Mannequin.json").read_text(encoding="utf-8-sig"))

        class SyntheticGeometry:
            def __init__(self, *args):
                self.parents = np.array(mapping["mano_parents"])
                self.arrays = {}
                self.scale_factor = 1.

            def evaluate(self, matrices, side, return_vertices=False):
                count = len(matrices)
                joints16 = np.zeros((count, 16, 3))
                joints16[:, :, 1] = np.arange(16) * .006
                joints21 = np.zeros((count, 21, 3))
                joints21[:, :16] = joints16
                vertices = np.zeros((count, 778, 3))
                return (joints16, joints21, vertices) if return_vertices else (joints16, joints21)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "synthetic.mp4"
            video.write_bytes(b"Synthetic provenance fixture; not an encoded video")
            source = {"format_version": "cardcap.hand_observations/1.0", "meta": {
                "primary_view": "primary", "fps": 30., "views": [{"id": "primary", "resolution": [1280, 720],
                    "source_video": str(video), "source_sha256": prepare.sha256(video),
                    "camera": {"calibrated": False, "intrinsics": None, "distortion": None,
                               "metadata_report": {"status": "absent"}}}]},
                "frames": [{"frame": frame, "views": {"primary": {"hands": [observation(side)]}}}
                           for frame, side in enumerate(("left", "right"))]}
            observations = root / "observations.json"
            observations.write_text(json.dumps(source), encoding="utf-8")
            before = observations.read_bytes()
            real_hash = prepare.sha256
            scale = {"global_scale_factor_applied": 1., "meters_per_unit": None,
                     "neutral_hand_measurement": {"length_source_units": .18, "definition": "synthetic neutral fixture"},
                     "provenance": {"kind": "unobservable"}}
            with patch.object(prepare, "FixedShapeGeometry", SyntheticGeometry), \
                 patch.object(prepare, "export_research_hands", return_value={"manifest_path": "synthetic-manifest", "outputs": {}}), \
                 patch.object(prepare, "sha256", side_effect=lambda path: "a" * 64 if Path(path).name == "MANO_RIGHT.pkl" else real_hash(path)), \
                 patch("cardcap.solve.hand_scale_policy.resolve_hand_scale", return_value=scale), \
                 patch("cardcap.solve.joint_translation.solve_joint_translations") as solver:
                validation = prepare.export_cardcap(observations, root / "animation")
                solver.assert_not_called()
                # A real missing-crop input still exports its valid local poses.
                # Only the independent common display is marked unavailable.
                missing_source = deepcopy(source)
                for frame in missing_source["frames"]:
                    frame["views"]["primary"]["hands"][0]["diagnostics"] = {}
                missing_path = root / "missing_crop_observations.json"
                missing_path.write_text(json.dumps(missing_source), encoding="utf-8")
                missing_before = missing_path.read_bytes()
                unavailable = prepare.export_cardcap(missing_path, root / "local_only")
                self.assertEqual(unavailable["status"], "passed")
                self.assertEqual(unavailable["display_space_status"], "unavailable")
                self.assertIsNone(unavailable["display_space_config"])
                self.assertIsNone(unavailable["display_space_sha256"])
                self.assertIn("native crop camera", unavailable["display_space_unavailable_reason"])
                self.assertEqual(missing_path.read_bytes(), missing_before)
                diagnostic = json.loads((root / "local_only/display_space_unavailable.json").read_text())
                self.assertEqual(diagnostic["capture_sha256"], real_hash(root / "local_only/capture.cardcap.json"))
                self.assertTrue(diagnostic["local_pose_export_available"])
                self.assertFalse((root / "local_only/display_space.json").exists())
                local_capture = json.loads((root / "local_only/capture.cardcap.json").read_text())
                self.assertIsNone(local_capture["camera"]["intrinsics"])
                self.assertTrue(all(f["global_trans_cm"] is None for h in local_capture["hands"] for f in h["frames"]))
                # A programming defect is not disguised as a missing display input.
                with patch.object(prepare, "build_display_space", side_effect=RuntimeError("synthetic unexpected failure")):
                    with self.assertRaisesRegex(RuntimeError, "synthetic unexpected failure"):
                        prepare.export_cardcap(observations, root / "unexpected_failure")
            self.assertEqual(observations.read_bytes(), before)
            capture = json.loads((root / "animation/capture.cardcap.json").read_text(encoding="utf-8"))
            display_path = Path(validation["display_space_config"])
            display = json.loads(display_path.read_text(encoding="utf-8"))
            self.assertEqual(validation["display_space_status"], "available")
            self.assertEqual(display["capture_sha256"], real_hash(root / "animation/capture.cardcap.json"))
            self.assertEqual(validation["display_space_sha256"], real_hash(display_path))
            self.assertNotIn("display_assumed_focal_px", json.dumps(capture))
            self.assertIsNone(capture["camera"]["intrinsics"])
            self.assertIsNone(capture["camera"]["distortion"])
            self.assertIsNone(capture["scale"]["meters_per_unit"])
            self.assertFalse(capture["validity"]["inter_hand_transform"])
            for hand in capture["hands"]:
                for frame in hand["frames"]:
                    self.assertIsNone(frame["global_trans_cm"])
                    self.assertFalse(frame["validity"]["global_translation"])
                    native = frame["diagnostics"]["source_crop_camera_translation_model_units"]
                    if frame["sample_kind"] == "detected_model_observation":
                        self.assertEqual(native, observation(hand["side"])["diagnostics"]["crop_camera_translation_m"])
                    else:
                        self.assertIsNone(native)
                        self.assertEqual(frame["confidence"], 0.)
            self.assertEqual(hashlib.sha256(before).hexdigest(), display["source_observations_sha256"])


if __name__ == "__main__":
    unittest.main()
