"""Associate local image line segments to a coarse card quadrilateral.

This is bounded boundary refinement, not instance discovery or full-face proof.
LSD image lines can still be texture/occluder boundaries. Keep their provenance
and all rejected associations; independent original-image review remains useful.
"""
from dataclasses import asdict, dataclass
import cv2
import numpy as np

from .quad_fit import validate_quad


@dataclass(frozen=True)
class EdgeAssociationConfig:
    roi_margin_px: int = 40
    minimum_segment_length_px: float = 18.
    maximum_angle_degrees: float = 5.
    maximum_quad_edge_distance_px: float = 12.
    mask_neighborhood_radius_px: float = 6.
    minimum_near_mask_fraction: float = .6
    collinear_tolerance_px: float = 2.5
    minimum_edge_coverage_fraction: float = .15


def union_length(intervals):
    total = 0.
    previous = None
    for low, high in sorted(intervals):
        if high <= low:
            continue
        if previous is None:
            previous = [low, high]
        elif low <= previous[1]:
            previous[1] = max(previous[1], high)
        else:
            total += previous[1] - previous[0]
            previous = [low, high]
    return total + (0. if previous is None else previous[1] - previous[0])


def associate_image_edges(image_bgr, mask, corners_px, config=None):
    cfg = config or EdgeAssociationConfig()
    image, mask = np.asarray(image_bgr), np.asarray(mask)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2]:
        raise ValueError('Expected native uint8 BGR image and matching two-dimensional mask')
    quad = validate_quad(corners_px)
    height, width = mask.shape
    low = np.maximum(np.floor(quad.min(0)-cfg.roi_margin_px), [0, 0]).astype(int)
    high = np.minimum(np.ceil(quad.max(0)+cfg.roi_margin_px), [width, height]).astype(int)
    x0, y0 = low; x1, y1 = high
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Quad has no image ROI')
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    lines = [] if detected is None else detected.reshape(-1, 2, 2).astype(float) + low
    catalog, per_edge = [], [[] for _ in range(4)]
    for source_index, points in enumerate(lines):
        segment = points[1]-points[0]
        length = float(np.linalg.norm(segment))
        if length < cfg.minimum_segment_length_px:
            continue
        direction = segment/length
        record = {'line_id': source_index, 'points_px': points.tolist(), 'length_px': length, 'associations': []}
        for edge_index in range(4):
            a, b = quad[edge_index], quad[(edge_index+1) % 4]
            edge_length = float(np.linalg.norm(b-a))
            u = (b-a)/edge_length
            normal = np.array([-u[1], u[0]])
            angle = float(np.degrees(np.arccos(np.clip(abs(direction@u), -1, 1))))
            distances = np.abs((points-a)@normal)
            along = (points-a)@u
            clipped = [max(0., float(along.min())), min(edge_length, float(along.max()))]
            if angle > cfg.maximum_angle_degrees or float(distances.max()) > cfg.maximum_quad_edge_distance_px or clipped[1] <= clipped[0]:
                continue
            samples = np.linspace(points[0], points[1], 20)
            nearby = np.stack([samples+normal*offset for offset in np.linspace(-cfg.mask_neighborhood_radius_px, cfg.mask_neighborhood_radius_px, 5)], 1)
            nearby = np.rint(nearby).astype(int)
            nearby[..., 0] = np.clip(nearby[..., 0], 0, width-1)
            nearby[..., 1] = np.clip(nearby[..., 1], 0, height-1)
            fraction = float((mask[nearby[..., 1], nearby[..., 0]] != 0).any(1).mean())
            association = {'quad_edge_index': edge_index, 'angle_degrees': angle,
                'mean_quad_distance_px': float(distances.mean()), 'near_mask_fraction': fraction,
                'clipped_edge_interval_px': clipped, 'eligible': fraction >= cfg.minimum_near_mask_fraction}
            record['associations'].append(association)
            if association['eligible']:
                per_edge[edge_index].append((record, association))
        catalog.append(record)

    selected = []
    edge_groups = []
    for edge_index, eligible in enumerate(per_edge):
        edge_length = float(np.linalg.norm(quad[(edge_index+1) % 4]-quad[edge_index]))
        alternatives = []
        for seed, _ in eligible:
            p = np.asarray(seed['points_px'])
            u = (p[1]-p[0])/seed['length_px']
            normal = np.array([-u[1], u[0]])
            members = []
            for rec, assoc in eligible:
                q = np.asarray(rec['points_px'])
                direction = (q[1]-q[0])/rec['length_px']
                angle = float(np.degrees(np.arccos(np.clip(abs(direction@u), -1, 1))))
                if angle <= cfg.maximum_angle_degrees and np.max(np.abs((q-p[0])@normal)) <= cfg.collinear_tolerance_px:
                    members.append((rec, assoc))
            coverage = union_length([a['clipped_edge_interval_px'] for _, a in members])
            if not members or coverage/edge_length < cfg.minimum_edge_coverage_fraction:
                continue
            mean_distance = np.mean([a['mean_quad_distance_px'] for _, a in members])
            score = coverage/(1.+mean_distance/cfg.maximum_quad_edge_distance_px)
            alternatives.append({'seed_line_id': seed['line_id'], 'line_ids': [r['line_id'] for r, _ in members],
                'coverage_px': coverage, 'edge_coverage_fraction': coverage/edge_length, 'score': float(score)})
        best = max(alternatives, key=lambda a: a['score']) if alternatives else None
        edge_groups.append({'quad_edge_index': edge_index, 'eligible_line_count': len(eligible),
            'selected_cluster': best, 'cluster_alternatives': alternatives})
        if best is not None:
            for record, _ in eligible:
                if record['line_id'] in best['line_ids']:
                    selected.append({'quad_edge_index': edge_index, 'points_px': record['points_px'],
                        'line_id': record['line_id'], 'weight': record['length_px']/best['coverage_px']})
    return {'configuration': asdict(cfg), 'roi_xyxy': [int(x0), int(y0), int(x1), int(y1)],
        'image_line_catalog': catalog, 'edge_groups': edge_groups, 'selected_segments': selected,
        'selected_quad_edges': sorted({s['quad_edge_index'] for s in selected}),
        'full_face_verified': False, 'physical_edge_identity_verified': False,
        'method': 'OpenCV LSD, bounded quad association, mask proximity and greatest collinear coverage',
        'limitations': 'Automatic image segments may follow occluders, printing or thickness strips; this does not prove four visible physical card corners.'}
