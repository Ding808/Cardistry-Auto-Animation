"""Synthetic algebra/input/detector tests; no physical camera accuracy evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cardcap.calibrate_camera import (Chessboard, calibration_payload, collect_chessboards, fit_camera,
    prepare_view_split, run_calibration, validate_heldout_views, validate_planar_length)
from cardcap.ingest import VideoView, ViewSpec
from cardcap.cards.pose_pnp import _camera_from_json

BOARD = Chessboard(8, 5, 25.)
MATRIX = np.array([[850., 0, 480.], [0, 830., 360.], [0, 0, 1.]])
DISTORTION = np.array([-.03, .004, .0005, -.0003, 0.])


def views_fixture(count=12, offset=0, distortion=DISTORTION):
    result = []
    for i in range(offset, offset + count):
        rvec = np.array([-.32 + .11*(i % 5), -.38 + .16*(i % 6), -.12 + .045*i])
        tvec = np.array([-.20 + .055*(i % 5), -.12 + .045*(i % 4), .65 + .04*(i % 4)])
        pixels = cv2.projectPoints(BOARD.object_points(), rvec, tvec, MATRIX, distortion)[0].reshape(-1, 2)
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        result.append({"image_path": "synthetic algebra only", "image_sha256": digest, "decoded_gray_sha256": digest,
            "subset": "synthetic", "resolution": [960, 720], "detected": True, "corners_px": pixels.tolist(),
            "rejection_reasons": [], "fixture_rvec": rvec, "fixture_tvec": tvec})
    return result


def write_checkerboard(path, row):
    cell = 48
    texture = np.full(((BOARD.rows + 3)*cell, (BOARD.columns + 3)*cell), 255, np.uint8)
    for y in range(BOARD.rows + 1):
        for x in range(BOARD.columns + 1):
            if (x+y) % 2 == 0: texture[(y+1)*cell:(y+2)*cell, (x+1)*cell:(x+2)*cell] = 0
    outer = np.array([[-2, -2, 0], [BOARD.columns+1, -2, 0], [BOARD.columns+1, BOARD.rows+1, 0], [-2, BOARD.rows+1, 0]], np.float32) * (BOARD.square_mm/1000)
    target = cv2.projectPoints(outer, row["fixture_rvec"], row["fixture_tvec"], MATRIX, np.zeros(5))[0].reshape(4, 2)
    h, w = texture.shape
    homography = cv2.getPerspectiveTransform(np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32), target)
    image = cv2.warpPerspective(texture, homography, (960, 720), borderValue=200)
    ok, data = cv2.imencode(".png", image); assert ok
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data.tobytes())


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.views = views_fixture()
        cls.fit = fit_camera(cls.views, BOARD)

    def test_known_projection_algebra_recovers_intrinsics_and_reports_no_metric_accuracy(self):
        self.assertTrue(self.fit["accepted"], self.fit["rejection_reasons"])
        np.testing.assert_allclose(self.fit["matrix"], MATRIX, atol=.03)
        np.testing.assert_allclose(self.fit["distortion"], DISTORTION, atol=.001)
        self.assertLess(self.fit["training_rms_px"], .001)
        self.assertFalse(self.fit["ten_percent_scale_accuracy_verified"])
        scaled = fit_camera(self.views, Chessboard(8, 5, 50.))
        np.testing.assert_allclose(scaled["matrix"], self.fit["matrix"], atol=.005)
        np.testing.assert_allclose(np.array(scaled["training_views"][0]["tvec_m"]), 2*np.array(self.fit["training_views"][0]["tvec_m"]), atol=1e-5)

    def test_duplicates_and_split_leakage_are_not_new_views(self):
        repeated = deepcopy(self.views + [self.views[0]])
        selected = prepare_view_split(repeated, [], BOARD)
        self.assertEqual(len(selected), 12)
        self.assertIn("duplicate_or_near_duplicate_training_view", repeated[-1]["rejection_reasons"])
        with self.assertRaisesRegex(ValueError, "Held-out images overlap"):
            prepare_view_split(deepcopy(self.views), [deepcopy(self.views[0])], BOARD)
        with self.assertRaisesRegex(ValueError, "Need >="):
            fit_camera(self.views[:9], BOARD)

    def test_no_pose_diversity_is_rejected(self):
        # All board normals are equal even though the grid translates and scales.
        rows = deepcopy(self.views)
        for row in rows:
            row["corners_px"] = cv2.projectPoints(BOARD.object_points(), np.zeros(3), row["fixture_tvec"], MATRIX, np.zeros(5))[0].reshape(-1, 2).tolist()
        try:
            fit = fit_camera(rows, BOARD)
        except (ValueError, cv2.error):
            return  # Explicit singular/nonfinite estimator failure is acceptable.
        self.assertFalse(fit["accepted"])

    def test_heldout_fit_does_not_update_intrinsics_and_corner_sets_are_disjoint(self):
        original = deepcopy(self.fit)
        held = validate_heldout_views(views_fixture(3, offset=15), BOARD, self.fit)
        self.assertEqual(self.fit, original)
        self.assertFalse(set(held["pose_corner_indices"]) & set(held["validation_corner_indices"]))
        self.assertTrue(all(v["withheld_corners"]["rms_px"] < .01 for v in held["views"]))
        self.assertFalse(held["independent_metric_accuracy_measured"])
        with self.assertRaisesRegex(ValueError, "used to fit this calibration"):
            validate_heldout_views([self.views[0]], BOARD, self.fit)

    def test_independent_planar_length_contract_and_circular_length_rejection(self):
        rows = views_fixture(1, offset=15)
        held = validate_heldout_views(rows, BOARD, self.fit)["views"][0]
        # Segment at non-grid locations, withheld from every calibration fit.
        endpoints = np.array([[.037, .033, 0], [.123, .033, 0]], np.float32)
        pixels = cv2.projectPoints(endpoints, rows[0]["fixture_rvec"], rows[0]["fixture_tvec"], MATRIX, DISTORTION)[0].reshape(2, 2)
        evidence = {"description": "Synthetic independent-reference algebra ONLY", "sha256": "a"*64,
                    "independent_of_all_calibration_constraints": True, "endpoints_on_board_plane_verified": True}
        length = validate_planar_length(self.fit, held, endpoints_px=pixels, measured_length_mm=86., measurement_uncertainty_mm=.1, measurement_evidence=evidence)
        self.assertLess(length["relative_error"], .001)
        self.assertFalse(length["ten_percent_capture_scale_accuracy_verified"])
        with self.assertRaisesRegex(ValueError, "Independent measurement"):
            validate_planar_length(self.fit, held, endpoints_px=pixels, measured_length_mm=86., measurement_uncertainty_mm=.1,
                measurement_evidence=dict(evidence, independent_of_all_calibration_constraints=False))
        with self.assertRaisesRegex(ValueError, "overlaps pose-fitting"):
            validate_planar_length(self.fit, held, endpoints_px=np.array(held["pose_fit_corners_px"])[:2], measured_length_mm=25., measurement_uncertainty_mm=.1, measurement_evidence=evidence)

    def test_synthetic_detector_unicode_and_resolution_rejection(self):
        with tempfile.TemporaryDirectory(prefix="camera-fixture-") as directory:
            folder = Path(directory); first = folder / "棋盘.png"; second = folder / "blank.png"
            write_checkerboard(first, self.views[2])
            ok, encoded = cv2.imencode(".png", np.full((720, 960), 200, np.uint8)); self.assertTrue(ok); second.write_bytes(encoded.tobytes())
            records, size = collect_chessboards([first, second], BOARD, subset="synthetic_fixture")
            self.assertTrue(records[0]["detected"]); self.assertFalse(records[1]["detected"])
            self.assertEqual(size, [960, 720])
            with self.assertRaisesRegex(ValueError, "exactly one resolution"):
                collect_chessboards([first], BOARD, subset="synthetic_fixture", expected_resolution=[640, 480])

    def test_export_contract_compatibility_is_only_temporary_synthetic_fixture(self):
        with tempfile.TemporaryDirectory(prefix="camera-contract-fixture-") as directory:
            folder = Path(directory); calibration = folder / "synthetic-camera.json"
            calibration.write_text(json.dumps(calibration_payload(self.fit)), encoding="utf-8")
            # Use the real ingest parser, without opening/assigning any user video.
            view = VideoView.__new__(VideoView)
            view.spec = ViewSpec("synthetic_schema_test_only", folder / "not-a-user-video", calibration=calibration)
            view.width, view.height = 960, 720
            parsed = view._load_camera()
            self.assertEqual(parsed["image_space"], "undistorted_rectified_intrinsics")
            self.assertEqual(parsed["source_intrinsics"], calibration_payload(self.fit)["intrinsics"])
            self.assertEqual(parsed["intrinsics"]["fx"], parsed["intrinsics"]["fy"])
            self.assertIsNone(parsed["world_to_camera"])
            for name, dist in (("original", self.fit["distortion"]), ("undistorted", [0.]*5)):
                path = folder / f"synthetic-pose-{name}.json"
                path.write_text(json.dumps({"camera": {"matrix": self.fit["matrix"], "distortion": dist, "calibrated": False,
                    "provenance": "Synthetic schema test, not physical calibration"}}), encoding="utf-8")
                parsed_pose = _camera_from_json(path)
                self.assertFalse(parsed_pose.calibrated)
                np.testing.assert_allclose(parsed_pose.distortion, dist)


if __name__ == "__main__": unittest.main()
