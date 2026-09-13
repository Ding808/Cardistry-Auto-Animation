"""Conditional measurements of an explicitly selected card-stack side strip.

This is not a side-strip detector or a card counter. Image periodicity is not
the number of physical card layers. Metric thickness needs an independent
mapping along this strip's thickness axis. Optional separately measured layer
thickness can supply a conditional count, never a visual count or ground truth.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math
import re

import cv2
import numpy as np


@dataclass(frozen=True)
class ThicknessConfig:
    minimum_axis_samples: int = 16
    minimum_lateral_samples: int = 8
    minimum_period_px: float = 3.0
    minimum_cycles: float = 3.0
    minimum_contrast: float = 0.03
    minimum_spectral_concentration: float = 0.6
    minimum_lag_correlation: float = 0.6
    minimum_lateral_correlation: float = 0.6

    def __post_init__(self):
        for name in ('minimum_axis_samples', 'minimum_lateral_samples'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 4:
                raise ValueError(f'{name}: expected integer >= 4')
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name}: expected positive finite value')
            if ('correlation' in name or name in ('minimum_contrast', 'minimum_spectral_concentration')) and value > 1:
                raise ValueError(f'{name}: expected (0,1]')
        if self.minimum_period_px < 3 or self.minimum_cycles < 3:
            raise ValueError('Need >= 3 pixels per visible period and >= 3 visible cycles')


def _positive(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f'{name}: expected finite {"nonnegative" if allow_zero else "positive"} number')
    return float(value)


def _correlation(first, second):
    first, second = first - first.mean(), second - second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.clip(np.dot(first, second) / denominator, -1, 1)) if denominator > 1e-12 else 0.0


def estimate_side_strip(strip: np.ndarray, *, provenance: dict, thickness_axis: str,
                        band_bounds_px: tuple[float, float], boundary_uncertainty_px: float,
                        mm_per_pixel: float | None = None, mapping_provenance: dict | None = None,
                        measured_layer_thickness_mm: float | None = None,
                        layer_thickness_provenance: dict | None = None,
                        config: ThicknessConfig | None = None) -> dict:
    """Measure a supplied uint8 gray/BGR strip; axes must already be aligned.

    ``band_bounds_px=(lower, upper)`` uses pixel-edge coordinates on rows or
    columns and must describe *outer sideband boundaries*, not arbitrary ROI
    borders. Uncertainty is per boundary. No rectification or detection occurs.
    Provenance must explicitly assert sideband_supported and boundaries_verified
    and provide source, source_sha256, selection_method, measurement_definition.
    False support retains unknowns. These flags are caller evidence contracts.

    Optional mapping requires source, definition, calibrated and
    applies_to_thickness_axis=True. Texture and mapped geometric measurements
    are independent: a uniform band can have measured outer width while its
    texture-period observation remains unknown. No metric mapping is inferred
    from texture, hands, a face-plane scale, or assumed card count.
    """
    config = config or ThicknessConfig()
    if not isinstance(config, ThicknessConfig):
        raise ValueError('config: expected ThicknessConfig')
    values = np.asarray(strip)
    if values.dtype != np.uint8 or values.ndim not in (2, 3) or not all(values.shape) or (values.ndim == 3 and values.shape[2] != 3):
        raise ValueError('strip: expected nonempty uint8 gray or BGR image')
    if thickness_axis not in ('rows', 'columns'):
        raise ValueError('thickness_axis: expected rows or columns')
    if not isinstance(provenance, dict):
        raise ValueError('provenance: expected explicit source and sideband evidence')
    for key in ('source', 'source_sha256', 'selection_method', 'measurement_definition'):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            raise ValueError(f'provenance.{key}: required nonempty string')
    if not re.fullmatch('[0-9a-fA-F]{64}', provenance['source_sha256']):
        raise ValueError('source_sha256: expected SHA256 hex digest')
    for key in ('sideband_supported', 'boundaries_verified'):
        if not isinstance(provenance.get(key), bool):
            raise ValueError(f'provenance.{key}: required explicit bool')
    if not isinstance(band_bounds_px, (tuple, list)) or len(band_bounds_px) != 2:
        raise ValueError('band_bounds_px: expected two pixel-edge coordinates')
    lower, upper = [_positive(x, 'band bound', allow_zero=True) for x in band_bounds_px]
    extent = values.shape[0 if thickness_axis == 'rows' else 1]
    if not 0 <= lower < upper <= extent:
        raise ValueError('band bounds must lie inside the image thickness axis')
    uncertainty = _positive(boundary_uncertainty_px, 'boundary_uncertainty_px', allow_zero=True)
    layer = None
    if measured_layer_thickness_mm is not None:
        layer = _positive(measured_layer_thickness_mm, 'measured_layer_thickness_mm')
        evidence = layer_thickness_provenance
        if not isinstance(evidence, dict) or evidence.get('kind') != 'user_measured':
            raise ValueError('Measured layer thickness requires explicit user_measured provenance')
        if any(not isinstance(evidence.get(key), str) or not evidence[key].strip() for key in ('description', 'source_sha256')):
            raise ValueError('Layer thickness provenance requires description and source_sha256')
        if not re.fullmatch('[0-9a-fA-F]{64}', evidence['source_sha256']):
            raise ValueError('Layer thickness source_sha256 must be SHA256')
        if evidence.get('applies_to_this_stack') is not True:
            raise ValueError('Layer measurement must explicitly apply to this stack')
    elif layer_thickness_provenance is not None:
        raise ValueError('Layer provenance supplied without measured layer thickness')
    if mm_per_pixel is not None:
        mm_per_pixel = _positive(mm_per_pixel, 'mm_per_pixel')
        if not isinstance(mapping_provenance, dict) or mapping_provenance.get('applies_to_thickness_axis') is not True or not isinstance(mapping_provenance.get('calibrated'), bool):
            raise ValueError('mapping requires explicit calibration status and applies_to_thickness_axis=True')
        for key in ('source', 'definition'):
            if not isinstance(mapping_provenance.get(key), str) or not mapping_provenance[key].strip():
                raise ValueError(f'mapping.{key}: required nonempty string')
    elif mapping_provenance is not None:
        raise ValueError('mapping provenance supplied without mm_per_pixel')

    result = {'format_version': 'cardcap.side_strip_thickness/1.0', 'provenance': deepcopy(provenance),
        'input_pixel_sha256': hashlib.sha256(values.tobytes(order='C')).hexdigest(),
        'input_shape': list(values.shape), 'input_dtype': 'uint8', 'thickness_axis': thickness_axis,
        'supplied_band_bounds_px': [lower, upper], 'boundary_uncertainty_px_per_edge': uncertainty,
        'config': asdict(config), 'sideband_thickness_px': None, 'sideband_thickness_mm': None,
        'mapping': deepcopy(mapping_provenance), 'mm_per_pixel': mm_per_pixel,
        'texture': {'status': 'unknown', 'reason': None, 'period_px': None, 'visible_period_count': None,
                    'period_is_physical_layer_spacing': False},
        'conditional_layer_count': None, 'conditional_layer_count_boundary_only_range': None,
        'count_reliability': 'unknown', 'measured_layer_thickness_mm': layer,
        'layer_thickness_provenance': deepcopy(layer_thickness_provenance),
        'count_is_ground_truth': False, 'automatic_sideband_detection': False,
        'warnings': ['Texture cycles may reflect printing, packet gaps, edges, compression or aliasing; they are not physical card layers.',
                     'Metric mapping and boundary support are caller assertions, not verified by this module.']}
    if not provenance['sideband_supported'] or not provenance['boundaries_verified']:
        result['texture']['reason'] = 'sideband_or_outer_boundary_evidence_insufficient'
        return result
    gray = values if values.ndim == 2 else cv2.cvtColor(values, cv2.COLOR_BGR2GRAY)
    aligned = gray if thickness_axis == 'rows' else gray.T
    band = aligned[math.ceil(lower):math.floor(upper)].astype(np.float64) / 255.0
    span = upper - lower
    if band.shape[0] < config.minimum_axis_samples or band.shape[1] < config.minimum_lateral_samples or span <= 4 * uncertainty:
        result['texture']['reason'] = 'insufficient_pixel_resolution_or_boundary_precision'
        return result
    result['sideband_thickness_px'] = span
    if mm_per_pixel is not None:
        result['sideband_thickness_mm'] = span * mm_per_pixel
        if not math.isfinite(result['sideband_thickness_mm']):
            raise ValueError('Metric mapping produces nonfinite thickness')
        if layer is not None:
            result['conditional_layer_count'] = span * mm_per_pixel / layer
            result['conditional_layer_count_boundary_only_range'] = [max(0, span - 2 * uncertainty) * mm_per_pixel / layer,
                                                                    (span + 2 * uncertainty) * mm_per_pixel / layer]
            if not all(math.isfinite(value) for value in (result['conditional_layer_count'], *result['conditional_layer_count_boundary_only_range'])):
                raise ValueError('Metric mapping/layer measurement produces nonfinite conditional count')
            result['count_reliability'] = 'conditional' if mapping_provenance['calibrated'] else 'low'
            result['warnings'].append('Conditional count assumes the separately measured layer applies uniformly and no inter-layer gaps; boundary-only range excludes mapping/layer/appearance uncertainty.')
        else:
            result['warnings'].append('No separately measured layer thickness; physical card count remains unknown despite the metric band width.')
    else:
        result['warnings'].append('No independent thickness-axis metric mapping: physical thickness and card count remain unknown.')

    profile = np.median(band, axis=1)
    axis = np.arange(len(profile), dtype=np.float64)
    trend = np.polyval(np.polyfit(axis, profile, 1), axis)
    signal = profile - trend
    contrast = float(np.percentile(signal, 95) - np.percentile(signal, 5))
    result['texture']['detrended_contrast_fraction'] = contrast
    if contrast < config.minimum_contrast:
        result['texture']['reason'] = 'insufficient_texture_contrast'
        return result
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
    spectrum[0] = 0
    peak = int(np.argmax(spectrum))
    if peak == 0 or spectrum.sum() <= 1e-15:
        result['texture']['reason'] = 'no_periodic_support'
        return result
    offset = 0.0
    if 1 < peak < len(spectrum) - 1:
        left, center, right = np.log(np.maximum(spectrum[peak-1:peak+2], 1e-30))
        denominator = left - 2 * center + right
        if abs(denominator) > 1e-12:
            offset = float(np.clip(.5 * (left - right) / denominator, -.5, .5))
    period = len(signal) / (peak + offset)
    if period < config.minimum_period_px:
        result['texture']['reason'] = 'undersampled_texture_period'
        return result
    if len(signal) / period < config.minimum_cycles:
        result['texture']['reason'] = 'too_few_visible_cycles'
        return result
    concentration = float(spectrum[max(1, peak-1):min(len(spectrum), peak+2)].sum() / spectrum.sum())
    lag = int(round(period))
    correlation = _correlation(signal[:-lag], signal[lag:])
    lateral = [_correlation(np.median(part, axis=1), profile) for part in np.array_split(band, 4, axis=1)]
    result['texture'].update(spectral_concentration=concentration, lag_correlation=correlation,
                             lateral_correlations=lateral)
    if concentration < config.minimum_spectral_concentration or correlation < config.minimum_lag_correlation or min(lateral) < config.minimum_lateral_correlation:
        result['texture']['reason'] = 'periodicity_or_lateral_consistency_insufficient'
        return result
    result['texture'].update(status='periodic_texture_observed', reason=None, period_px=float(period),
                             visible_period_count=float(span / period))
    return result
