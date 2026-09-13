"""Analytic image/geometry tests only; no fabricated video card counts."""
import unittest
import numpy as np

from cardcap.cards.thickness import estimate_side_strip


def provenance(supported=True):
    return {'source': 'analytic test image', 'source_sha256': '0' * 64,
            'selection_method': 'synthetic_test', 'measurement_definition': 'Rows cross the explicitly defined outer band; not real cards.',
            'sideband_supported': supported, 'boundaries_verified': supported}


def stripes(period=8):
    row = np.arange(96)
    values = np.rint(128 + 48 * np.cos(2 * np.pi * row / period)).astype(np.uint8)
    return np.repeat(values[:, None], 48, axis=1)


def measure(image, **kwargs):
    return estimate_side_strip(image, provenance=kwargs.pop('provenance', provenance()),
        thickness_axis='rows', band_bounds_px=(0, image.shape[0]), boundary_uncertainty_px=1, **kwargs)


class CardThicknessTests(unittest.TestCase):
    def test_period_is_texture_not_layer_count(self):
        result = measure(stripes())
        self.assertEqual(result['texture']['status'], 'periodic_texture_observed')
        self.assertAlmostEqual(result['texture']['period_px'], 8, delta=.15)
        self.assertAlmostEqual(result['texture']['visible_period_count'], 12, delta=.25)
        self.assertIsNone(result['conditional_layer_count'])
        self.assertFalse(result['texture']['period_is_physical_layer_spacing'])
        columns = estimate_side_strip(stripes().T, provenance=provenance(), thickness_axis='columns',
                                      band_bounds_px=(0, 96), boundary_uncertainty_px=1)
        self.assertAlmostEqual(columns['texture']['period_px'], result['texture']['period_px'])

    def test_uniform_and_too_few_cycles_are_unknown(self):
        flat = measure(np.full((96, 48), 128, np.uint8))
        self.assertEqual(flat['texture']['reason'], 'insufficient_texture_contrast')
        sparse = measure(stripes(period=48))
        self.assertEqual(sparse['texture']['status'], 'unknown')
        self.assertEqual(sparse['texture']['reason'], 'too_few_visible_cycles')

    def test_undersampling_and_small_band_are_unknown(self):
        result = measure(stripes(period=2))
        self.assertEqual(result['texture']['reason'], 'undersampled_texture_period')
        tiny = measure(stripes()[:8])
        self.assertEqual(tiny['texture']['reason'], 'insufficient_pixel_resolution_or_boundary_precision')

    def test_no_sideband_evidence_does_not_measure_roi_as_thickness(self):
        result = measure(stripes(), provenance=provenance(False))
        self.assertIsNone(result['sideband_thickness_px'])
        self.assertIsNone(result['texture']['period_px'])
        self.assertIsNone(result['conditional_layer_count'])

    def test_mapping_needs_separate_measured_layer_to_report_conditional_count(self):
        mapping = {'source': 'analytic low-reliability scale', 'definition': '0.0254 mm per row along thickness axis',
                   'calibrated': False, 'applies_to_thickness_axis': True}
        result = measure(stripes(), mm_per_pixel=.0254, mapping_provenance=mapping)
        self.assertAlmostEqual(result['sideband_thickness_mm'], 96 * .0254)
        self.assertIsNone(result['conditional_layer_count'])
        self.assertIsNone(result['measured_layer_thickness_mm'])
        layer = {'kind': 'user_measured', 'description': 'Synthetic measurement assertion; no real card',
                 'source_sha256': 'a' * 64, 'applies_to_this_stack': True, 'uncertainty': None}
        result = measure(stripes(), mm_per_pixel=.0254, mapping_provenance=mapping,
                         measured_layer_thickness_mm=.27, layer_thickness_provenance=layer)
        self.assertAlmostEqual(result['conditional_layer_count'], 96 * .0254 / .27)
        self.assertEqual(result['layer_thickness_provenance'], layer)
        self.assertEqual(result['count_reliability'], 'low')
        self.assertFalse(result['count_is_ground_truth'])
        self.assertNotAlmostEqual(result['conditional_layer_count'], result['texture']['visible_period_count'])

    def test_invalid_inputs_rejected(self):
        for kwargs in ({'mm_per_pixel': -1}, {'mm_per_pixel': float('nan')},
                       {'mm_per_pixel': .1}, {'measured_layer_thickness_mm': 0},
                       {'measured_layer_thickness_mm': .27},
                       {'layer_thickness_provenance': {'kind': 'user_measured'}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                measure(stripes(), **kwargs)
        with self.assertRaises(ValueError):
            measure(stripes().astype(float))
        with self.assertRaises(ValueError):
            estimate_side_strip(stripes(), provenance=provenance(), thickness_axis='rows',
                                band_bounds_px=(0, 100), boundary_uncertainty_px=1)


if __name__ == '__main__':
    unittest.main()
