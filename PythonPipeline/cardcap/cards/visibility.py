"""Visible-mask distinctness diagnostics, never physical packet membership."""
from itertools import combinations

import numpy as np


def assess_visible_masks(masks: dict[str, np.ndarray], *, containment_limit: float = 0.8,
                         minimum_area_px: int = 64) -> dict:
    """Detect when a small track is swallowed despite a low global mask IoU.

    A nonempty region below minimum_area_px is retained as insufficient support.
    Neither non-overlap nor containment proves a physical split/merge. These
    flags only determine where an upstream identity proposal needs review.
    """
    if not isinstance(masks, dict) or not masks:
        raise ValueError("masks: expected nonempty ID-to-binary-mask mapping")
    if isinstance(containment_limit, bool) or not np.isfinite(containment_limit) or not 0 < containment_limit <= 1:
        raise ValueError("containment_limit: expected (0,1]")
    if isinstance(minimum_area_px, bool) or not isinstance(minimum_area_px, int) or minimum_area_px < 1:
        raise ValueError("minimum_area_px: expected positive integer")
    binary, shape = {}, None
    for identity, mask in masks.items():
        if not isinstance(identity, str) or not identity:
            raise ValueError("mask IDs must be nonempty strings")
        values = np.asarray(mask)
        if values.ndim != 2 or not all(values.shape) or values.dtype.kind not in 'buif' or not np.isfinite(values).all() or not np.isin(values, [0, 1, 255]).all():
            raise ValueError("masks must be nonempty-size finite binary 2D arrays")
        if shape is not None and shape != values.shape:
            raise ValueError("masks must share image dimensions")
        shape = values.shape
        binary[identity] = values != 0
    areas = {identity: int(mask.sum()) for identity, mask in binary.items()}
    objects = {identity: {"area_px": area, "visible_mask_status": 'absent' if area == 0 else
                         ('insufficient_area' if area < minimum_area_px else 'observed_region'),
                         "physical_identity_verified": False} for identity, area in areas.items()}
    pairs = []
    for first, second in combinations(sorted(binary), 2):
        intersection = int(np.count_nonzero(binary[first] & binary[second]))
        union = areas[first] + areas[second] - intersection
        smaller = min(areas[first], areas[second])
        coverage = intersection / smaller if smaller else None
        ambiguous = coverage is not None and coverage >= containment_limit
        pairs.append({"object_ids": [first, second], "intersection_px": intersection,
                      "union_px": union, "mask_iou": intersection / union if union else None,
                      "intersection_over_smaller_mask": coverage,
                      "status": 'visibility_insufficient' if smaller < minimum_area_px else
                                ('identity_ambiguous_containment' if ambiguous else 'no_containment_flag'),
                      "containment_flag": ambiguous,
                      "physical_group_relation": 'unknown', "is_lifecycle_event": False})
    return {"objects": objects, "pairs": pairs, "containment_limit": containment_limit,
            "minimum_area_px": minimum_area_px, "supported_rigid_membership": False,
            "note": 'Pixel regions only: no containment flag does not establish distinct physical packets; disappearing masks do not establish merges.'}
