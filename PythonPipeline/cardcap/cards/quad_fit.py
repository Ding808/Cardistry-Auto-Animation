"""Conservative perspective quadrilateral proposals from binary face masks.

Mask geometry cannot establish that four *physical card corners* are visible.
``visibility='full'`` is a caller assertion about a planar card face, supported
by an explicit evidence string; it is never inferred from rectangularity.
An enclosing quadrilateral is fitted to the convex hull, not a minAreaRect.
It encloses the hull for a searched set of edge orientations; it is not a
proof of the globally minimum-area enclosing quadrilateral.
"""

from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class QuadFitConfig:
    min_area_px2: float = 64.0
    min_edge_px: float = 4.0
    min_corner_sine: float = 0.03
    min_hull_solidity: float = 0.85
    min_quad_fill: float = 0.80
    max_secondary_component_fraction: float = 0.02
    max_hole_fraction: float = 0.01
    min_edge_support_fraction: float = 0.30
    edge_tolerance_px: float = 2.0
    border_margin_px: int = 1

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"quad config.{name}: expected finite nonnegative value")
        for name in ("min_corner_sine", "min_hull_solidity", "min_quad_fill",
                     "max_secondary_component_fraction", "max_hole_fraction", "min_edge_support_fraction"):
            if getattr(self, name) > 1:
                raise ValueError(f"quad config.{name}: expected [0, 1]")


@dataclass
class QuadFitResult:
    corners_px: np.ndarray | None
    accepted: bool
    visibility: str
    visibility_evidence: str | None
    full_face_verified: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    method: str = "convex_hull_support_line_enclosing_quad"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["corners_px"] = None if self.corners_px is None else self.corners_px.tolist()
        value["corner_order"] = "cyclic image order, positive signed area; smallest y then x first; no physical corner identity"
        value["full_face_verification_source"] = "caller visibility assertion plus geometry gates; not inferred from mask"
        return value


def validate_quad(corners: np.ndarray, *, min_area_px2: float = 1.0,
                  min_edge_px: float = 1e-3, min_corner_sine: float = 1e-4) -> np.ndarray:
    """Validate the supplied cyclic order without silently repairing correspondences."""
    points = np.asarray(corners, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("corners_px: expected finite array of shape (4, 2)")
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths < min_edge_px):
        raise ValueError("corners_px: repeated corner or edge too short")
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if not (np.all(cross > 0) or np.all(cross < 0)):
        raise ValueError("corners_px: nonconvex, self-intersecting or collinear cyclic order")
    if np.any(np.abs(cross) / (lengths * np.roll(lengths, -1)) < min_corner_sine):
        raise ValueError("corners_px: near-collinear corner")
    area = abs(float(cv2.contourArea(points.astype(np.float32))))
    if not np.isfinite(area) or area < min_area_px2:
        raise ValueError("corners_px: area too small")
    return np.ascontiguousarray(points)


def _canonical_order(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))]
    first = int(np.lexsort((points[:, 0], points[:, 1]))[0])
    return np.roll(points, -first, axis=0)


def _enclosing_support_quad(proposal: np.ndarray, hull: np.ndarray) -> np.ndarray | None:
    proposal = _canonical_order(proposal)
    edges = np.roll(proposal, -1, axis=0) - proposal
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths < 1e-8):
        return None
    normals = np.column_stack((edges[:, 1], -edges[:, 0])) / lengths[:, None]
    offsets = (hull @ normals.T).max(axis=0)
    vertices = []
    for index in range(4):
        matrix = np.stack((normals[(index - 1) % 4], normals[index]))
        if abs(float(np.linalg.det(matrix))) < 1e-5:
            return None
        vertices.append(np.linalg.solve(matrix, offsets[[(index - 1) % 4, index]]))
    return _canonical_order(np.asarray(vertices))


def _search_support_quads(hull: np.ndarray, approximations: list[np.ndarray],
                          cfg: QuadFitConfig) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Bounded direction search when polygon approximation never has 4 vertices.

    Candidate directions come from hull/coarse polygon edges, favoring long
    edges. At most 16 distinct directions gives at most 1820 four-line sets.
    Every offset is recomputed from the ORIGINAL hull. This is an approximate
    direction search, not a globally optimal enclosing-quadrilateral solver.
    """
    max_directions = 16
    merge_angle = np.deg2rad(2.0)
    directions = []
    for polygon in [hull, *approximations]:
        points = _canonical_order(polygon)
        edges = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(edges, axis=1)
        for edge, length in zip(edges, lengths):
            if length < 1e-8:
                continue
            normal = np.array([edge[1], -edge[0]]) / length
            angle = float(np.arctan2(normal[1], normal[0]) % (2 * np.pi))
            directions.append((float(length), angle, normal))
    directions.sort(key=lambda item: (-item[0], item[1]))
    selected = []
    for _, angle, normal in directions:
        if all(abs((angle - old_angle + np.pi) % (2 * np.pi) - np.pi) >= merge_angle
               for old_angle, _ in selected):
            selected.append((angle, normal))
        if len(selected) == max_directions:
            break
    selected.sort(key=lambda item: item[0])
    metadata = {"used": True, "maximum_directions": max_directions,
                "selected_direction_count": len(selected), "maximum_four_line_combinations": 1820,
                "direction_merge_degrees": 2.0, "tested_four_line_combinations": 0,
                "bounded_nondegenerate_enclosing_candidates": 0,
                "offset_source": "maximum projection of every original hull vertex",
                "globally_minimum_area_proven": False}
    if len(selected) < 4:
        return [], metadata
    angles = np.array([item[0] for item in selected])
    normals = np.array([item[1] for item in selected])
    offsets = np.max(hull @ normals.T, axis=0)
    tolerance = max(float(np.ptp(hull, axis=0).max()), 1.0) * 1e-7
    candidates = []
    for chosen in combinations(range(len(selected)), 4):
        metadata["tested_four_line_combinations"] += 1
        indices = list(chosen)
        selected_angles = angles[indices]
        gaps = np.diff(np.r_[selected_angles, selected_angles[0] + 2 * np.pi])
        # Outward normals must positively span the plane for a bounded polygon.
        if np.max(gaps) >= np.pi - 1e-5:
            continue
        current = normals[indices]
        current_offsets = offsets[indices]
        previous = np.roll(current, 1, axis=0)
        previous_offsets = np.roll(current_offsets, 1)
        determinant = previous[:, 0] * current[:, 1] - previous[:, 1] * current[:, 0]
        if np.any(np.abs(determinant) < 1e-5):
            continue
        vertices = np.column_stack((
            (previous_offsets * current[:, 1] - previous[:, 1] * current_offsets) / determinant,
            (previous[:, 0] * current_offsets - previous_offsets * current[:, 0]) / determinant))
        if not np.isfinite(vertices).all() or np.max(vertices @ current.T - current_offsets) > tolerance:
            continue
        try:
            validate_quad(vertices, min_area_px2=cfg.min_area_px2,
                          min_edge_px=cfg.min_edge_px, min_corner_sine=cfg.min_corner_sine)
        except ValueError:
            continue
        candidates.append(_canonical_order(vertices))
    metadata["bounded_nondegenerate_enclosing_candidates"] = len(candidates)
    return candidates, metadata


def fit_quad(mask: np.ndarray, *, visibility: str = "unknown",
             visibility_evidence: str | None = None,
             config: QuadFitConfig | None = None) -> QuadFitResult:
    """Fit a diagnostic card-face quad; unknown/partial masks never become anchors.

    Masks must be 2-D bool or binary values 0/1/255. Logits and probabilities
    require an explicit threshold in the segmentation caller. Invalid API
    inputs raise ValueError; valid but unsuitable masks return accepted=False.
    """
    cfg = config or QuadFitConfig()
    if visibility not in ("unknown", "partial", "full"):
        raise ValueError("visibility: expected unknown, partial or full")
    if visibility_evidence is not None and (not isinstance(visibility_evidence, str) or not visibility_evidence.strip()):
        raise ValueError("visibility_evidence: expected nonempty string or None")
    if visibility == "full" and not visibility_evidence:
        raise ValueError("visibility='full' requires explicit independent visibility evidence")
    array = np.asarray(mask)
    if array.ndim != 2 or min(array.shape) < 2 or array.dtype.kind not in "buif":
        raise ValueError("mask: expected two-dimensional numeric/bool binary mask")
    if not np.isfinite(array).all() or not np.isin(array, (0, 1, 255)).all():
        raise ValueError("mask: binary 0/1/255 required; threshold logits explicitly")
    binary = (array != 0).astype(np.uint8)
    result = QuadFitResult(None, False, visibility, visibility_evidence)
    result.warnings.append("mask shape is not evidence of a complete card face or physical scale")
    if visibility != "full":
        result.warnings.append(f"face visibility {visibility}; not eligible as a complete-face scale anchor")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    total = int(binary.sum())
    result.metrics.update(mask_area_pixels=total, component_count=max(count - 1, 0),
                          image_width=int(array.shape[1]), image_height=int(array.shape[0]))
    if count < 2 or total < cfg.min_area_px2:
        result.rejection_reasons.append("empty_or_too_small_mask")
        return result
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main_area = int(stats[label, cv2.CC_STAT_AREA])
    secondary = (total - main_area) / total
    result.metrics["secondary_component_fraction"] = secondary
    if secondary > cfg.max_secondary_component_fraction:
        result.rejection_reasons.append("multiple_material_mask_components")
    component = (labels == label).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    hull_cv = cv2.convexHull(contour)
    hull = hull_cv.reshape(-1, 2).astype(np.float64)
    hull_area = float(cv2.contourArea(hull_cv))
    contour_area = float(cv2.contourArea(contour))
    external_raster = np.zeros_like(binary)
    cv2.drawContours(external_raster, [contour], -1, 1, cv2.FILLED)
    hole_fraction = max(int(external_raster.sum()) - main_area, 0) / max(int(external_raster.sum()), 1)
    result.metrics["hole_fraction"] = hole_fraction
    if hole_fraction > cfg.max_hole_fraction:
        result.rejection_reasons.append("material_interior_hole_or_occlusion")
    if hull_area < cfg.min_area_px2:
        result.rejection_reasons.append("degenerate_hull")
        return result
    # Pixel occupancy also detects interior holes ignored by RETR_EXTERNAL.
    hull_raster = np.zeros_like(binary)
    cv2.fillConvexPoly(hull_raster, hull_cv, 1)
    solidity = main_area / max(int(hull_raster.sum()), 1)
    result.metrics.update(hull_area_px2=hull_area, contour_area_px2=contour_area,
                          hull_solidity=solidity)
    if solidity < cfg.min_hull_solidity:
        result.rejection_reasons.append("concave_occluded_or_holed_mask")
    perimeter = cv2.arcLength(hull_cv, True)
    proposals = []
    coarse_approximations = []
    for fraction in np.linspace(0.002, 0.05, 25):
        approx = cv2.approxPolyDP(hull_cv, float(fraction * perimeter), True)
        if 5 <= len(approx) <= 8:
            coarse_approximations.append(approx.reshape(-1, 2).astype(np.float64))
        if len(approx) != 4:
            continue
        candidate = _enclosing_support_quad(approx.reshape(4, 2).astype(np.float64), hull)
        if candidate is None:
            continue
        try:
            validate_quad(candidate, min_area_px2=cfg.min_area_px2,
                          min_edge_px=cfg.min_edge_px, min_corner_sine=cfg.min_corner_sine)
        except ValueError:
            continue
        proposals.append(candidate)
    result.metrics["support_line_fallback"] = {"used": False}
    if not proposals:
        proposals, search_metadata = _search_support_quads(hull, coarse_approximations, cfg)
        result.metrics["support_line_fallback"] = search_metadata
    if not proposals:
        result.rejection_reasons.append("no_nondegenerate_perspective_quad")
        return result
    corners = min(proposals, key=lambda value: abs(cv2.contourArea(value.astype(np.float32))))
    result.corners_px = corners
    area = abs(float(cv2.contourArea(corners.astype(np.float32))))
    fill = min(contour_area / area, 1.0)
    result.metrics.update(quad_area_px2=area, quad_fill_ratio=fill,
                          searched_orientation_proposals=len(proposals))
    if fill < cfg.min_quad_fill:
        result.rejection_reasons.append("poor_quad_fill")
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    points = contour.reshape(-1, 2).astype(np.float64)
    support = []
    for index in range(4):
        unit = edges[index] / lengths[index]
        normal = np.array([unit[1], -unit[0]])
        distance = np.abs((points - corners[index]) @ normal)
        along = ((points - corners[index]) @ unit) / lengths[index]
        supported = along[(distance <= cfg.edge_tolerance_px) & (along >= 0) & (along <= 1)]
        # Occupied bins test edge coverage rather than a few isolated vertices.
        bins = np.unique(np.minimum((supported * 20).astype(int), 19))
        support.append(len(bins) / 20.0)
    result.metrics.update(edge_lengths_px=lengths.tolist(), edge_support_fractions=support)
    if min(support) < cfg.min_edge_support_fraction:
        result.rejection_reasons.append("insufficient_support_on_an_edge")
    ys, xs = np.nonzero(component)
    margin = cfg.border_margin_px
    touches = bool(xs.min() <= margin or ys.min() <= margin or
                   xs.max() >= array.shape[1] - 1 - margin or ys.max() >= array.shape[0] - 1 - margin)
    result.metrics["touches_image_border"] = touches
    if touches:
        result.warnings.append("mask touches image border; complete face cannot be verified")
    result.accepted = not result.rejection_reasons
    result.full_face_verified = result.accepted and visibility == "full" and not touches
    return result
