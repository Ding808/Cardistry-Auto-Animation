"""Conditional focal-length diagnostics from a rectangular planar projection.

Zhang's two homography constraints are specialized to known principal point,
zero skew, square pixels and zero distortion. A card mask or a supplied
corner annotation does not verify those assumptions. Returned cameras ALWAYS
have calibrated=False and are not independent physical-scale evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .quad_fit import validate_quad

REFERENCE = 'https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf'


@dataclass(frozen=True)
class RectangleCameraConfig:
    max_normalized_orthogonality_error: float = 0.05
    max_relative_axis_length_error: float = 0.05
    minimum_perspective_information: float = 1e-10

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key}: expected positive finite number')


def _positive(value, name):
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name}: expected positive finite scalar')
    return float(value)


def _homography(objects, pixels):
    rows, values = [], []
    for (x, y), (u, v) in zip(objects, pixels):
        rows.extend(([x, y, 1, 0, 0, 0, -u*x, -u*y],
                     [0, 0, 0, x, y, 1, -v*x, -v*y]))
        values.extend((u, v))
    system = np.array(rows, dtype=np.float64)
    solution = np.linalg.solve(system, np.array(values))
    return np.r_[solution, 1.].reshape(3, 3), float(np.linalg.cond(system))


def estimate_rectangle_focal(corners_px, *, width: float, height: float,
                             resolution: tuple[int, int], principal_point_px=None,
                             config: RectangleCameraConfig | None = None) -> dict:
    """Estimate one focal under an EXPLICIT physical width/height edge order.

    Corner order corresponds to [-w/2,-h/2], [w/2,-h/2], [w/2,h/2],
    [-w/2,h/2]. Absolute object units cancel; only aspect ratio matters.
    This calculation cannot recover object dimensions or validate them.
    Frontoparallel and inconsistent configurations return rejected diagnostics.
    Invalid numeric/quad inputs raise ValueError rather than guessing corners.
    accepted_under_assumptions means geometric constraint consistency only;
    it does not establish focal reliability or identifiability under noise.
    """
    cfg = config or RectangleCameraConfig()
    width, height = _positive(width, 'width'), _positive(height, 'height')
    if len(resolution) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in resolution):
        raise ValueError('resolution: two positive integers required')
    pixels = validate_quad(np.asarray(corners_px, np.float64))
    principal = np.asarray(principal_point_px if principal_point_px is not None else np.asarray(resolution) / 2,
                           dtype=np.float64)
    if principal.shape != (2,) or not np.isfinite(principal).all():
        raise ValueError('principal point: finite pair required')
    image_scale = float(max(resolution))
    object_scale = max(width, height)
    objects = np.array([[-width/2, -height/2], [width/2, -height/2],
                        [width/2, height/2], [-width/2, height/2]]) / object_scale
    normalized_pixels = (pixels - principal) / image_scale
    result = {'accepted_under_assumptions': False, 'focal_px': None, 'camera': None,
              'acceptance_semantics': 'Geometric constraint consistency only; not focal reliability, identifiability or calibration',
              'focal_reliability': 'unknown', 'focal_identifiability': 'unknown',
              'physical_camera_calibrated': False, 'scale_anchor_eligible': False,
              'input_corners_px': pixels.tolist(), 'width_to_height_ratio': width / height,
              'principal_point_px': principal.tolist(), 'resolution': list(resolution),
              'assumptions': ['known principal point (image center if omitted)', 'zero skew',
                              'fx equals fy', 'zero lens distortion', 'correct single planar rectangle corners and edge assignment'],
              'independent_accuracy_measured': False, 'rejection_reasons': [], 'metrics': {},
              'method': 'specialized planar homography orthogonality/equal-axis constraints',
              'reference': REFERENCE, 'config': asdict(cfg)}
    try:
        matrix, condition = _homography(objects, normalized_pixels)
    except np.linalg.LinAlgError:
        result['rejection_reasons'].append('singular_corner_homography')
        return result
    a, b = matrix[:, 0], matrix[:, 1]
    # q = 1/(f/image_scale)^2. The two equations are A*q + B = 0.
    coefficients = np.array([np.dot(a[:2], b[:2]), np.dot(a[:2], a[:2]) - np.dot(b[:2], b[:2])])
    constants = np.array([a[2]*b[2], a[2]**2 - b[2]**2])
    denominator = float(coefficients @ coefficients)
    information = float(np.linalg.norm(constants) / max(float(a @ a + b @ b), 1e-30))
    result['metrics'].update(homography_normalized=matrix.tolist(), corner_system_condition_number=condition,
                             normalized_constraint_coefficients=coefficients.tolist(), constraint_constants=constants.tolist(),
                             perspective_information=information)
    if information < cfg.minimum_perspective_information or denominator <= 1e-24:
        result['rejection_reasons'].append('focal_unidentifiable_from_frontoparallel_or_degenerate_projection')
        return result
    q = -float(coefficients @ constants) / denominator
    result['metrics']['inverse_squared_normalized_focal'] = q
    if not np.isfinite(q) or q <= 0:
        result['rejection_reasons'].append('no_positive_focal_under_assumptions')
        return result
    normalized_focal = float(1 / np.sqrt(q))
    focal = normalized_focal * image_scale
    first = a * [1/normalized_focal, 1/normalized_focal, 1]
    second = b * [1/normalized_focal, 1/normalized_focal, 1]
    first_length, second_length = float(np.linalg.norm(first)), float(np.linalg.norm(second))
    orthogonality = float(abs(first @ second) / (first_length * second_length))
    relative_length = abs(first_length - second_length) / max(first_length, second_length)
    result['metrics'].update(normalized_orthogonality_error=orthogonality, relative_axis_length_error=relative_length)
    result['focal_px'] = focal
    if orthogonality > cfg.max_normalized_orthogonality_error:
        result['rejection_reasons'].append('rectangle_axis_orthogonality_inconsistent')
    if relative_length > cfg.max_relative_axis_length_error:
        result['rejection_reasons'].append('rectangle_axis_length_inconsistent')
    result['accepted_under_assumptions'] = not result['rejection_reasons']
    # Keep a rejected positive focal as a diagnostic, never as a usable camera.
    if result['accepted_under_assumptions']:
        result['camera'] = {'intrinsics': {'fx': focal, 'fy': focal, 'cx': float(principal[0]), 'cy': float(principal[1])},
                        'distortion': [0., 0., 0., 0., 0.], 'calibrated': False,
                        'provenance': 'Conditional rectangle estimate with assumed principal point/aspect/zero distortion; not independent camera calibration'}
    return result


def analyze_edge_assignments(corners_px, *, width: float, height: float,
                             resolution: tuple[int, int], principal_point_px=None,
                             corner_uncertainty_px: float = 2., perturbations: int = 128,
                             random_seed: int = 0, minimum_consistent_fraction: float = 0.9,
                             maximum_relative_focal_spread: float = 0.5) -> dict:
    """Preserve width/height edge ambiguity and deterministic perturbation spread.

    Perturbations are independent uniform +/- supplied pixel bounds, not a
    probabilistic noise model or confidence interval. They quantify sensitivity
    to the stated bound; systematic annotation/visibility error remains unknown.
    Instability is an engineering heuristic: too few consistent trials OR too
    much (max-min)/median focal spread. Invalid trials count in the denominator.
    Passing this check does not establish reliability or physical calibration.
    Neither branch is automatically selected or installed as the video camera.
    """
    corners = validate_quad(np.asarray(corners_px, dtype=np.float64))
    uncertainty = _positive(corner_uncertainty_px, 'corner_uncertainty_px')
    minimum_consistent_fraction = _positive(minimum_consistent_fraction, 'minimum_consistent_fraction')
    maximum_relative_focal_spread = _positive(maximum_relative_focal_spread, 'maximum_relative_focal_spread')
    if minimum_consistent_fraction > 1:
        raise ValueError('minimum_consistent_fraction: expected (0,1]')
    if isinstance(perturbations, bool) or not isinstance(perturbations, int) or not 1 <= perturbations <= 4096:
        raise ValueError('perturbations: integer in [1,4096] required')
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError('random_seed: nonnegative integer required')
    rng = np.random.default_rng(random_seed)
    perturbations_xy = rng.uniform(-uncertainty, uncertainty, (perturbations, 4, 2))
    branches = []
    for start in (0, 1):
        indices = [(start + i) % 4 for i in range(4)]
        base = estimate_rectangle_focal(corners[indices], width=width, height=height,
                                        resolution=resolution, principal_point_px=principal_point_px)
        all_positive, consistent = [], []
        invalid = 0
        for offset in perturbations_xy:
            try:
                sample = estimate_rectangle_focal((corners + offset)[indices], width=width, height=height,
                                                  resolution=resolution, principal_point_px=principal_point_px)
            except (ValueError, np.linalg.LinAlgError):
                invalid += 1
                continue
            if sample['focal_px'] is not None:
                all_positive.append(sample['focal_px'])
            if sample['accepted_under_assumptions']:
                consistent.append(sample['focal_px'])
        consistent_fraction = len(consistent) / perturbations
        relative_spread = ((max(consistent) - min(consistent)) / float(np.median(consistent))) if consistent else None
        instability_reasons = []
        if consistent_fraction < minimum_consistent_fraction:
            instability_reasons.append('consistent_trial_fraction_below_threshold')
        if relative_spread is not None and relative_spread > maximum_relative_focal_spread:
            instability_reasons.append('relative_focal_spread_above_threshold')
        unstable = bool(instability_reasons)
        branches.append({'corner_indices': indices, 'base': base,
            'focal_reliability': 'unstable' if unstable else 'unknown',
            'focal_identifiability': 'unstable_under_stated_perturbations' if unstable else 'unknown',
            'perturbation_sensitivity': {'trials': perturbations, 'invalid_quads': invalid,
                'positive_focal_count': len(all_positive), 'consistent_constraint_count': len(consistent),
                'consistent_trial_fraction': consistent_fraction,
                'consistent_focal_relative_spread': relative_spread,
                'positive_focal_min_max_px': [min(all_positive), max(all_positive)] if all_positive else None,
                'consistent_focal_min_median_max_px': [min(consistent), float(np.median(consistent)), max(consistent)] if consistent else None,
                'stability_status': 'unstable' if unstable else 'no_instability_flag',
                'instability_reasons': instability_reasons,
                'no_instability_flag_establishes_reliability': False,
                'is_statistical_confidence_interval': False}})
    return {'branches': branches, 'selected_branch': None, 'calibrated': False,
            'corner_uncertainty_bound_px': uncertainty, 'perturbation_distribution': 'independent bounded uniform offsets',
            'random_seed': random_seed, 'scale_accuracy_validated': False,
            'stability_heuristic': {'minimum_consistent_fraction': minimum_consistent_fraction,
                'maximum_relative_focal_spread': maximum_relative_focal_spread,
                'relative_spread_definition': '(max-min)/median of geometrically consistent focal samples',
                'consistent_fraction_denominator': 'all perturbation trials, including invalid or rejected trials',
                'rule': 'unstable if consistent fraction is below minimum OR relative focal spread exceeds maximum',
                'thresholds_are_engineering_choices': True, 'is_statistical_confidence_interval': False,
                'is_independent_calibration_validation': False},
            'ambiguity_note': 'Cyclic 180-degree and reversed face winding have equivalent focal constraints; two distinct width/height assignments retained.'}
