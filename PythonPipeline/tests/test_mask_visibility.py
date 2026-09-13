import unittest
import numpy as np

from cardcap.cards.visibility import assess_visible_masks


class VisibilityTests(unittest.TestCase):
    def test_swallowed_small_object_despite_low_global_iou(self):
        large = np.ones((100, 100), np.uint8)
        small = np.zeros_like(large)
        small[25:35, 25:35] = 1
        pair = assess_visible_masks({'a': large, 'b': small})['pairs'][0]
        self.assertLess(pair['mask_iou'], .02)
        self.assertEqual(pair['intersection_over_smaller_mask'], 1)
        self.assertEqual(pair['status'], 'identity_ambiguous_containment')
        self.assertFalse(pair['is_lifecycle_event'])

    def test_empty_masks_are_unknown_not_zero_overlap_proof(self):
        empty = np.zeros((100, 100), np.uint8)
        pair = assess_visible_masks({'a': empty, 'b': empty})['pairs'][0]
        self.assertIsNone(pair['mask_iou'])
        self.assertIsNone(pair['intersection_over_smaller_mask'])
        self.assertEqual(pair['status'], 'visibility_insufficient')

    def test_disjoint_masks_do_not_establish_rigid_membership(self):
        first = np.zeros((100, 100), np.uint8)
        second = np.zeros_like(first)
        first[:20, :20] = second[-20:, -20:] = 1
        result = assess_visible_masks({'a': first, 'b': second})
        self.assertEqual(result['pairs'][0]['status'], 'no_containment_flag')
        self.assertFalse(result['supported_rigid_membership'])


if __name__ == '__main__':
    unittest.main()
