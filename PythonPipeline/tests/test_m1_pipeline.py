"""M1 unit checks use explicit fixtures, never presented as real reconstruction."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from cardcap.export import atomic_json, frame_ranges, serialize_hand, summarize_frames
from cardcap.hand.base import HandEstimate
from cardcap.ingest import VideoView, ViewSpec, load_views, source_frame_for_time
from cardcap.viz import OverlayWriter, verify_video


def hand_fixture() -> HandEstimate:
    return HandEstimate("left", 0.9, np.zeros((21, 3)), np.zeros((21, 3)), "unit_fixture_not_a_model")


class ObservationTests(unittest.TestCase):
    def test_missing_mano_and_pose_confidence_stay_unavailable(self):
        result = serialize_hand(hand_fixture(), 640, 480)
        self.assertIsNone(result["mano"])
        self.assertIsNone(result["pose_confidence"])
        self.assertIsNone(result["joint_confidences"])

    def test_invalid_provider_arrays_cannot_export(self):
        for values in (np.zeros((20, 3)), np.full((21, 3), np.nan)):
            estimate = hand_fixture()
            estimate.world_landmarks_m = values
            with self.assertRaises(ValueError):
                serialize_hand(estimate, 640, 480)

    def test_nonfinite_output_rejected_before_replacing_valid_json(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "status.json"
            atomic_json(output, {"value": 1})
            with self.assertRaises(ValueError):
                atomic_json(output, {"value": float("nan")})
            self.assertEqual(json.loads(output.read_text()), {"value": 1})

    def test_actual_denominators_and_missing_side_ranges(self):
        record = serialize_hand(hand_fixture(), 640, 480)
        rows = [
            {"frame": 0, "views": {"one": {"available": True, "hands": [record], "blur_flag": False}}},
            {"frame": 1, "views": {"one": {"available": True, "hands": [], "blur_flag": False}}},
            {"frame": 2, "views": {"one": {"available": False, "hands": []}}},
        ]
        summary = summarize_frames(rows, ["one"], 0.8)["one"]
        self.assertEqual(summary["available_frames"], 2)
        self.assertEqual(summary["valid_frames_at_least_one_hand"], 1)
        self.assertEqual(summary["valid_frame_rate"], 0.5)
        self.assertEqual(summary["both_distinct_sides_frames"], 0)
        self.assertEqual(summary["review_frame_ranges"], [[0, 1]])
        self.assertIsNone(summary["pa_mpjpe_mm"])

    def test_ranges_preserve_discontinuities(self):
        self.assertEqual(frame_ranges([5, 1, 2, 2, 7, 8]), [[1, 2], [5, 5], [7, 8]])

    def test_duplicate_labels_are_not_two_anatomical_hands(self):
        record = serialize_hand(hand_fixture(), 640, 480)
        rows = [{"frame": 0, "views": {"one": {"available": True, "hands": [record, record], "blur_flag": False}}}]
        result = summarize_frames(rows, ["one"], 0.8)["one"]
        self.assertEqual(result["frames_with_two_observations"], 1)
        self.assertEqual(result["both_distinct_sides_frames"], 0)
        self.assertEqual(result["side_observations"]["left"], 2)
        self.assertEqual(result["side_unique_frames"]["left"], 1)
        self.assertEqual(result["ambiguous_handedness_ranges"], [[0, 0]])


class IngestTests(unittest.TestCase):
    def test_offsets_and_unequal_frame_rates(self):
        self.assertEqual(source_frame_for_time(1.0, 60, 0.25), 75)
        self.assertEqual(source_frame_for_time(0, 30, -0.2), -6)
        self.assertEqual(source_frame_for_time(1 / 30, 15, 0), 1)

    def test_manifest_has_distinct_views_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "views.json"
            data = {"primary_view": "front", "views": [
                {"id": "front", "video": "front.mp4"},
                {"id": "side", "video": "side.mp4", "offset_seconds": 0.25},
            ]}
            path.write_text(json.dumps(data))
            views, primary = load_views(None, path)
            self.assertEqual(primary, "front")
            self.assertEqual(views[1].video, Path(folder) / "side.mp4")
            self.assertEqual(views[1].offset_seconds, 0.25)
            data["views"][1]["id"] = "front"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_views(None, path)

    def test_video_roundtrip_seek_bounds_and_calibration_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.mp4"
            writer = OverlayWriter(path, 30, 64, 48)
            for level in (0, 50, 100, 150):
                writer.write(np.full((48, 64, 3), level, np.uint8))
            writer.close()
            report = verify_video(path, 4, 30)
            self.assertEqual(report["decoded_frames"], 4)
            view = VideoView(ViewSpec("unit_fixture", path))
            try:
                self.assertIsNone(view.read(-1))
                self.assertIsNone(view.read(4))
                third = view.read(2)
                self.assertTrue(np.array_equal(view.read(2), third))
                self.assertLess(abs(float(third.mean()) - 100), 8)
                self.assertLess(float(view.read(0).mean()), 8)
                self.assertFalse(view.camera["calibrated"])
            finally:
                view.close()
            calibration = Path(folder) / "camera.json"
            calibration.write_text(json.dumps({
                "resolution": [32, 24], "intrinsics": {"fx": 80, "fy": 80, "cx": 32, "cy": 24},
                "distortion": [0, 0, 0, 0, 0],
            }))
            with self.assertRaisesRegex(ValueError, "resolution"):
                VideoView(ViewSpec("invalid_calibration_fixture", path, calibration=calibration))


if __name__ == "__main__":
    unittest.main()
