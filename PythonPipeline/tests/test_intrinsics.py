"""Resolver regressions using synthetic pixels/metadata, not real accuracy evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cardcap.ingest import VideoView, ViewSpec, resolve_intrinsics
from test_camera_calibration import MATRIX, views_fixture, write_checkerboard
from test_video_metadata import atom, keyed_metadata


def write_video(path, images):
    height, width = images[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Synthetic video writer did not open")
    try:
        for image in images:
            writer.write(image)
    finally:
        writer.release()


class IntrinsicsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="cardcap-intrinsics-synthetic-")
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.warning_context = warnings.catch_warnings(record=True)
        self.recorded_warnings = self.warning_context.__enter__()
        self.addCleanup(self.warning_context.__exit__, None, None, None)
        warnings.simplefilter("always")

    def test_explicit_calibration_resamples_unequal_focals_and_preserves_source(self):
        y, x = np.indices((96, 128))
        image = np.stack(((x * 7) % 255, (y * 11) % 255, ((x + y) * 3) % 255), axis=-1).astype(np.uint8)
        video = self.folder / "source.mp4"
        write_video(video, [image])
        calibration = self.folder / "camera.json"
        k = {"fx": 110., "fy": 105., "cx": 64., "cy": 48.}
        lens = [.2, -.03, .001, -.002, 0.]
        calibration.write_text(json.dumps({"resolution": [128, 96], "intrinsics": k,
                                          "distortion": lens, "world_to_camera": None}), encoding="utf-8")
        original = cv2.VideoCapture(str(video))
        ok, raw = original.read(); original.release()
        self.assertTrue(ok)
        with patch("cardcap.ingest._adjacent_chessboard", side_effect=AssertionError("Explicit calibration must have priority")):
            view = VideoView(ViewSpec("synthetic", video, calibration=calibration))
        try:
            self.assertTrue(view.camera["calibrated"])
            self.assertFalse(view.camera["prior_based"])
            self.assertEqual(view.camera["intrinsics_source"], "explicit_calibration")
            self.assertEqual(view.camera["source_intrinsics"], k)
            self.assertEqual(view.camera["source_lens_distortion"], lens)
            self.assertEqual(view.camera["distortion"], [0.] * 5)
            self.assertEqual(view.camera["image_space"], "undistorted_rectified_intrinsics")
            matrix = np.array([[110., 0, 64.], [0, 105., 48.], [0, 0, 1.]])
            focal = np.sqrt(110. * 105.)
            effective = np.array([[focal, 0, 64.], [0, focal, 48.], [0, 0, 1.]])
            expected = cv2.undistort(raw, matrix, np.array(lens), None, effective)
            np.testing.assert_array_equal(view.read(0), expected)
            self.assertFalse(np.array_equal(expected, raw))
            self.assertEqual(view.read(0).shape, raw.shape)
        finally:
            view.close()

    def test_rectified_projection_matches_mapped_original_pixels_without_geometry_change(self):
        calibration = self.folder / "camera.json"
        source_k = {"fx": 850., "fy": 1000., "cx": 440., "cy": 340.}
        lens = [.03, -.002, .0003, -.0001, 0.]
        transform = np.eye(4); transform[:3, 3] = [.2, .1, .4]
        payload = {"resolution": [960, 720], "intrinsics": source_k,
                   "distortion": lens, "world_to_camera": transform.tolist()}
        calibration.write_text(json.dumps(payload), encoding="utf-8")
        camera = resolve_intrinsics(self.folder / "source.mp4", calibration, resolution=(960, 720))
        objects = np.array([[-.1, -.08, .1], [.13, -.06, .05], [.11, .10, -.02], [-.06, .09, .04]])
        original = objects.copy()
        rotation, translation = np.array([.1, -.2, .15]), np.array([.04, -.03, 1.2])
        source_matrix = np.array([[850., 0, 440.], [0, 1000., 340.], [0, 0, 1.]])
        f = np.sqrt(850. * 1000.)
        effective_matrix = np.array([[f, 0, 440.], [0, f, 340.], [0, 0, 1.]])
        raw_pixels = cv2.projectPoints(objects, rotation, translation, source_matrix, np.array(lens))[0]
        mapped = cv2.undistortPoints(raw_pixels, source_matrix, np.array(lens), P=effective_matrix)
        projected = cv2.projectPoints(objects, rotation, translation, effective_matrix, np.zeros(5))[0]
        np.testing.assert_allclose(mapped, projected, atol=1e-8, rtol=0)
        np.testing.assert_array_equal(objects, original)
        self.assertEqual(camera["world_to_camera"], transform.tolist())
        self.assertAlmostEqual(camera["intrinsics"]["fx"], f)
        self.assertEqual(camera["intrinsics"]["fx"], camera["intrinsics"]["fy"])
        self.assertEqual(camera["intrinsics"]["cx"], source_k["cx"])
        self.assertEqual(camera["intrinsics"]["cy"], source_k["cy"])
        # Equal-axis calibration keeps the established same-intrinsics branch.
        payload["intrinsics"]["fy"] = payload["intrinsics"]["fx"]
        calibration.write_text(json.dumps(payload), encoding="utf-8")
        same = resolve_intrinsics(self.folder / "source.mp4", calibration, resolution=(960, 720))
        self.assertEqual(same["image_space"], "undistorted_same_intrinsics")
        self.assertEqual(same["distortion"], lens)

    def test_adjacent_video_detects_grid_count_and_fits_without_square_size(self):
        images = []
        for index, row in enumerate(views_fixture(20, distortion=np.zeros(5))):
            path = self.folder / f"rendered_{index}.png"
            write_checkerboard(path, row)
            images.append(cv2.imread(str(path)))
        write_video(self.folder / "calib_synthetic.mp4", images)
        with patch("cardcap.ingest.read_video_metadata", side_effect=AssertionError("Accepted chessboard must have priority")):
            camera = resolve_intrinsics(self.folder / "source.mp4", resolution=(960, 720))
        self.assertEqual(camera["intrinsics_source"], "adjacent_chessboard")
        self.assertEqual(camera["provenance"]["grid_inner_corners"], [8, 5])
        self.assertGreaterEqual(len(camera["provenance"]["selected_frames"]), 10)
        self.assertTrue(camera["calibrated"])
        self.assertIsNone(camera["world_to_camera"])
        self.assertFalse(camera["provenance"]["extrinsics_exported"])
        self.assertFalse(camera["provenance"]["independent_accuracy_verified"])
        np.testing.assert_allclose([camera["source_intrinsics"]["fx"], camera["source_intrinsics"]["fy"]],
                                   [MATRIX[0, 0], MATRIX[1, 1]], rtol=.03)
        self.assertNotIn("tvec_m", json.dumps(camera))

    def test_camera_metadata_fields_are_preserved_but_never_upgraded_to_k(self):
        video = self.folder / "source.mp4"
        video.write_bytes(atom("moov", keyed_metadata([
            ("FocalLength", b"4.7 mm", 1),
            ("FocalLengthIn35mmFilm", b"28/1", 1),
            ("sensor_width_mm", b"7.2", 1),
        ])))
        with patch("subprocess.run", side_effect=AssertionError("No external metadata tool is permitted")):
            camera = resolve_intrinsics(video, resolution=(1280, 720))
        self.assertEqual(camera["intrinsics_source"], "unobservable")
        self.assertIsNone(camera["intrinsics"])
        self.assertIsNone(camera["distortion"])
        self.assertFalse(camera["calibrated"])
        self.assertFalse(camera["prior_based"])
        self.assertEqual(camera["metadata_report"]["status"], "fields_present_insufficient")
        self.assertEqual([r["value"] for r in camera["metadata_report"]["metadata_records"]], ["4.7 mm", "28/1", "7.2"])
        self.assertEqual(camera["image_space"], "original_distorted")
        self.assertTrue(any("present but" in w for w in camera["warnings"]))

    def test_valid_extreme_focals_are_not_replaced_by_width_policy(self):
        calibration = self.folder / "camera.json"
        source = {"resolution": [1280, 720], "intrinsics": {"fx": 250., "fy": 3200., "cx": 640., "cy": 360.},
                  "distortion": [.2, 0, 0, 0, 0], "world_to_camera": np.eye(4).tolist()}
        calibration.write_text(json.dumps(source), encoding="utf-8")
        camera = resolve_intrinsics(self.folder / "source.mp4", calibration, resolution=(1280, 720))
        self.assertEqual(camera["intrinsics_source"], "explicit_calibration")
        self.assertEqual(camera["source_intrinsics"], source["intrinsics"])
        self.assertAlmostEqual(camera["intrinsics"]["fx"], np.sqrt(250. * 3200.))
        self.assertTrue(camera["calibrated"])
        self.assertEqual(camera["world_to_camera"], source["world_to_camera"])
        self.assertNotIn("focal_policy", camera)
        self.assertNotIn("rejected_intrinsics", camera)
        # Removing an empirical range must not remove mathematical positivity.
        for bad in (0, -1, float("nan")):
            source["intrinsics"]["fx"] = bad
            calibration.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "positive|finite"):
                resolve_intrinsics(self.folder / "source.mp4", calibration, resolution=(1280, 720))
        source["intrinsics"]["fx"] = 250.
        source["resolution"] = [640, 360]
        calibration.write_text(json.dumps(source), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "resolution"):
            resolve_intrinsics(self.folder / "source.mp4", calibration, resolution=(1280, 720))

    def test_unreadable_metadata_is_distinct_from_absent_fields(self):
        with patch("subprocess.run", side_effect=AssertionError("Must not invoke ffprobe")):
            missing = resolve_intrinsics(self.folder / "missing.mp4", resolution=(1280, 720))
        self.assertEqual(missing["metadata_report"]["status"], "unreadable")
        self.assertIsNone(missing["intrinsics"])
        self.assertIsNone(missing["distortion"])
        self.assertTrue(any("could not be read" in w for w in missing["warnings"]))
        video = self.folder / "source.mp4"
        video.write_bytes(atom("moov", atom("udta", atom("test", b"opaque vendor data"))))
        absent = resolve_intrinsics(video, resolution=(1280, 720))
        self.assertEqual(absent["metadata_report"]["status"], "fields_absent")
        self.assertTrue(absent["metadata_report"]["unparsed_regions"])
        self.assertFalse(absent["metadata_report"]["all_possible_camera_metadata_locations_exhausted"])
        self.assertIsNone(absent["intrinsics"])
        self.assertTrue(any("parsed container metadata scopes" in w for w in absent["warnings"]))
        video.write_bytes(b"not an mp4")
        malformed = resolve_intrinsics(video, resolution=(1280, 720))
        self.assertEqual(malformed["metadata_report"]["status"], "unreadable")
        self.assertIsNone(malformed["intrinsics"])

    def test_unknown_camera_preserves_decoded_pixels_and_cache(self):
        y, x = np.indices((96, 128))
        image = np.stack(((x * 7) % 255, (y * 11) % 255, ((x + y) * 3) % 255), axis=-1).astype(np.uint8)
        video = self.folder / "source.mp4"
        write_video(video, [image, np.flip(image, axis=1).copy()])
        original = cv2.VideoCapture(str(video))
        expected = [original.read()[1], original.read()[1]]
        original.release()
        with patch("subprocess.run", side_effect=AssertionError("No external metadata reader")):
            view = VideoView(ViewSpec("synthetic", video))
        try:
            self.assertEqual(view.camera["intrinsics_source"], "unobservable")
            self.assertIsNone(view.camera["intrinsics"])
            self.assertIsNone(view.camera["distortion"])
            self.assertEqual(view.camera["image_space"], "original_distorted")
            np.testing.assert_array_equal(view.read(0), expected[0])
            cached = view.read(0)
            cached[:] = 0
            np.testing.assert_array_equal(view.read(0), expected[0])
            np.testing.assert_array_equal(view.read(1), expected[1])
        finally:
            view.close()


if __name__ == "__main__":
    unittest.main()
