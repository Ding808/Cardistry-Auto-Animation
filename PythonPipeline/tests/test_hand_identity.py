"""Detection-only identity edge cases; no network or invented missing poses."""
import unittest
import numpy as np

from cardcap.hand.base import HandEstimate
from cardcap.hand.identity import TemporalHandIdentityTracker


def hand(x, side, y=0.5, score=.9, size=.08):
    points = np.column_stack((x + np.linspace(-size, size, 21),
                             y + np.sin(np.linspace(0, np.pi * 2, 21)) * size,
                             np.zeros(21)))
    points[0, :2] = [x, y]
    return HandEstimate(side, score, points, np.zeros((21, 3)), "test_only")


class HandIdentityTests(unittest.TestCase):
    def test_single_hand_label_flip_keeps_track_side(self):
        tracker = TemporalHandIdentityTracker()
        initial = tracker.update([hand(.25, "left")], 0)[0]
        flipped = tracker.update([hand(.26, "right", score=.99)], 33)[0]
        self.assertEqual(flipped["track_id"], initial["track_id"])
        self.assertEqual(flipped["side"], "left")
        self.assertEqual(flipped["native_side"], "right")
        self.assertEqual(flipped["native_handedness_confidence"], .99)
        self.assertTrue(flipped["corrected"])
        self.assertFalse(flipped["ambiguous"])

    def test_two_hand_duplicates_resolved_by_geometry_and_order_independent(self):
        tracker = TemporalHandIdentityTracker()
        initial = tracker.update([hand(.2, "left"), hand(.8, "right")], 0)
        current = tracker.update([hand(.79, "right"), hand(.21, "right")], 33)
        self.assertEqual([item["side"] for item in current], ["right", "left"])
        self.assertEqual(current[0]["track_id"], initial[1]["track_id"])
        self.assertEqual(current[1]["track_id"], initial[0]["track_id"])
        self.assertTrue(current[1]["corrected"])

    def test_initial_conflicting_labels_are_not_declared_certain(self):
        result = TemporalHandIdentityTracker().update([hand(.2, "right"), hand(.8, "right", score=.8)], 0)
        self.assertEqual([item["side"] for item in result], ["right", "right"])
        self.assertTrue(result[1]["ambiguous"])
        self.assertTrue(result[1]["correction_suppressed"])
        self.assertFalse(result[1]["corrected"])

    def test_separate_second_hand_can_use_explicit_opposite_hand_inference(self):
        tracker = TemporalHandIdentityTracker()
        tracker.update([hand(.2, "right")], 0)
        result = tracker.update([hand(.2, "right"), hand(.8, "right")], 33)
        self.assertEqual(result[1]["side"], "left")
        self.assertTrue(result[1]["corrected"])
        self.assertTrue(result[1]["ambiguous"])
        self.assertTrue(result[1]["opposite_hand_exclusivity_inferred"])
        self.assertFalse(result[1]["correction_suppressed"])
        self.assertIsNone(result[1]["association_score"])

    def test_overlapping_native_conflict_does_not_force_opposite_side_to_enable_export(self):
        # Regression for the 452-frame input: an established hand can remain
        # usable while the second observation has no confirmed anatomical side.
        tracker = TemporalHandIdentityTracker()
        tracker.update([hand(.4, "right")], 0)
        result = tracker.update([hand(.4, "right"), hand(.52, "right", score=.99)], 33)
        self.assertEqual([item["side"] for item in result], ["right", "right"])
        self.assertFalse(result[0]["ambiguous"])
        self.assertTrue(result[1]["ambiguous"])
        self.assertEqual(result[1]["source"], "new_track_native_conflict")
        self.assertTrue(result[1]["correction_suppressed"])
        self.assertFalse(result[1]["opposite_hand_exclusivity_inferred"])
        # A later actual classifier agreement can establish the uncertain
        # left seed; duplication alone cannot establish it retroactively.
        next_result = tracker.update([hand(.4, "right"), hand(.53, "left")], 66)
        self.assertEqual(next_result[1]["track_id"], result[1]["track_id"])
        self.assertEqual(next_result[1]["side"], "left")
        self.assertFalse(next_result[1]["ambiguous"])

    def test_crossing_overlap_suppresses_arbitrary_side_correction(self):
        tracker = TemporalHandIdentityTracker()
        tracker.update([hand(.42, "left"), hand(.58, "right")], 0)
        overlap = tracker.update([hand(.5, "right"), hand(.5, "right")], 33)
        self.assertTrue(all(item["ambiguous"] for item in overlap))
        self.assertTrue(all(not item["corrected"] for item in overlap))
        self.assertEqual([item["side"] for item in overlap], ["right", "right"])

    def test_missing_hand_not_emitted_and_short_reentry_recovers(self):
        tracker = TemporalHandIdentityTracker()
        initial = tracker.update([hand(.2, "left"), hand(.8, "right")], 0)
        self.assertEqual(len(tracker.update([hand(.2, "left")], 33)), 1)
        self.assertEqual(tracker.update([], 66), [])
        current = tracker.update([hand(.2, "left"), hand(.8, "left")], 200)
        self.assertEqual(current[1]["track_id"], initial[1]["track_id"])
        self.assertEqual(current[1]["side"], "right")
        self.assertTrue(current[1]["reacquired"])

    def test_long_gap_creates_new_id_and_uses_native_seed(self):
        tracker = TemporalHandIdentityTracker(max_gap_ms=300)
        first = tracker.update([hand(.3, "left")], 0)[0]
        current = tracker.update([hand(.3, "right")], 400)[0]
        self.assertNotEqual(first["track_id"], current["track_id"])
        self.assertEqual(current["side"], "right")
        self.assertEqual(current["source"], "new_track_after_expiry")

    def test_implausible_jump_cannot_steal_track_identity(self):
        tracker = TemporalHandIdentityTracker()
        first = tracker.update([hand(.1, "left")], 0)[0]
        current = tracker.update([hand(.9, "left")], 33)[0]
        self.assertNotEqual(first["track_id"], current["track_id"])
        self.assertTrue(current["ambiguous"])
        self.assertEqual(current["source"], "new_track_after_spatial_rejection")

    def test_empty_timestamps_still_strict_and_no_input_mutation(self):
        tracker = TemporalHandIdentityTracker()
        original = hand(.3, "left")
        points = original.image_landmarks.copy()
        tracker.update([original], 0)
        tracker.update([], 1)
        with self.assertRaises(ValueError):
            tracker.update([], 1)
        self.assertEqual(original.side, "left")
        np.testing.assert_array_equal(original.image_landmarks, points)

    def test_more_than_two_hands_rejected(self):
        with self.assertRaises(ValueError):
            TemporalHandIdentityTracker().update([hand(.2, "left")] * 3, 0)

    def test_tracker_instances_do_not_share_identity(self):
        first, second = TemporalHandIdentityTracker(), TemporalHandIdentityTracker()
        first.update([hand(.3, "left")], 0)
        self.assertEqual(second.update([hand(.3, "right")], 0)[0]["side"], "right")
        self.assertEqual(first.update([hand(.3, "right")], 33)[0]["side"], "left")


if __name__ == "__main__":
    unittest.main()
