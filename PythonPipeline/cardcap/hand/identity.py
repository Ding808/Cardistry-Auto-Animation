"""Conservative two-hand identity association; detection-only, no pose filling.

Track sides are seeded by native handedness. Geometry and short-horizon motion
associate observations across time. Ambiguous geometric matches retain native
side. A distinct second hand can use an explicitly uncertain opposite-hand
bootstrap inference after an existing identity is established.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np

from .base import HandEstimate


@dataclass
class _Track:
    track_id: int
    side: str
    wrist: np.ndarray
    center: np.ndarray
    bbox: np.ndarray
    velocity: np.ndarray
    last_seen_ms: int
    uncertain_seed: bool = False


def _geometry(hand):
    points = np.asarray(hand.image_landmarks, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Identity association requires 21 finite image landmarks")
    minimum, maximum = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    return points[0, :2].copy(), (minimum + maximum) / 2, np.concatenate((minimum, maximum))


def _iou(first, second):
    overlap = np.maximum(np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2]), 0)
    intersection = float(np.prod(overlap))
    area1 = float(np.prod(np.maximum(first[2:] - first[:2], 0)))
    area2 = float(np.prod(np.maximum(second[2:] - second[:2], 0)))
    return intersection / max(area1 + area2 - intersection, 1e-12)


class TemporalHandIdentityTracker:
    """Per-video tracker for at most two hands, using normalized image geometry.

    Association scores are heuristic geometric agreement, not probabilities,
    handedness classifier scores, landmark confidence, or pose accuracy.
    Missing observations never produce outputs. A long gap creates new IDs.
    """
    def __init__(self, *, max_gap_ms=1200, ambiguity_margin=0.12):
        if isinstance(max_gap_ms, bool) or not isinstance(max_gap_ms, int) or max_gap_ms <= 0:
            raise ValueError("max_gap_ms must be a positive integer")
        if not math.isfinite(ambiguity_margin) or ambiguity_margin < 0:
            raise ValueError("ambiguity_margin must be finite and nonnegative")
        self.max_gap_ms = max_gap_ms
        self.ambiguity_margin = float(ambiguity_margin)
        self._tracks = {}
        self._next_id = 1
        self._last_timestamp = None

    def _cost(self, track, geometry, timestamp_ms):
        wrist, center, bbox = geometry
        gap = (timestamp_ms - track.last_seen_ms) / 1000
        movement = track.velocity * min(gap, 0.2)
        predicted_wrist = track.wrist + movement
        predicted_center = track.center + movement
        predicted_bbox = track.bbox + np.tile(movement, 2)
        old_diagonal = float(np.linalg.norm(track.bbox[2:] - track.bbox[:2]))
        diagonal = float(np.linalg.norm(bbox[2:] - bbox[:2]))
        scale = max((diagonal + old_diagonal) / 2, 0.05)
        wrist_distance = float(np.linalg.norm(wrist - predicted_wrist)) / scale
        center_distance = float(np.linalg.norm(center - predicted_center)) / scale
        size_ratio = max(diagonal, 0.01) / max(old_diagonal, 0.01)
        if wrist_distance > 1.6 or center_distance > 1.6 or not 0.3 <= size_ratio <= 3.3:
            return math.inf
        return (0.65 * wrist_distance + 0.25 * center_distance
                + 0.15 * (1 - _iou(bbox, predicted_bbox)) + 0.08 * abs(math.log(size_ratio)))

    def update(self, detections: list[HandEstimate], timestamp_ms: int) -> list[dict]:
        if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, (int, np.integer)) or timestamp_ms < 0:
            raise ValueError("timestamp_ms must be a nonnegative integer")
        timestamp_ms = int(timestamp_ms)
        if self._last_timestamp is not None and timestamp_ms <= self._last_timestamp:
            raise ValueError("Identity timestamps must strictly increase")
        if len(detections) > 2:
            raise ValueError("Temporal identity supports at most two detected hands")
        for detection in detections:
            if detection.side not in {"left", "right"}:
                raise ValueError("Identity association requires native left/right categories")
            if not math.isfinite(detection.handedness_confidence) or not 0 <= detection.handedness_confidence <= 1:
                raise ValueError("Native handedness confidence must be finite in [0,1]")
        geometries = [_geometry(item) for item in detections]
        self._last_timestamp = timestamp_ms
        expired = [key for key, track in self._tracks.items()
                   if timestamp_ms - track.last_seen_ms > self.max_gap_ms]
        for key in expired:
            del self._tracks[key]
        if not detections:
            return []
        keys = list(self._tracks)
        costs = [{key: self._cost(self._tracks[key], geometry, timestamp_ms) for key in keys}
                 for geometry in geometries]
        # Exhaust all partial one-to-one assignments. An unmatched observation
        # has a finite cost, so implausible jumps cannot steal a stale track.
        candidates = []
        for assignment in itertools.product([None] + keys, repeat=len(detections)):
            assigned = [key for key in assignment if key is not None]
            if len(assigned) != len(set(assigned)):
                continue
            cost = sum(1.25 if key is None else costs[index][key]
                       for index, key in enumerate(assignment))
            if math.isfinite(cost):
                candidates.append((cost, assignment))
        candidates.sort(key=lambda item: item[0])
        best_cost, assignment = candidates[0]
        proposed = {}
        reserved_sides = {self._tracks[key].side for key in assignment if key is not None}
        # Process matched observations first, then high native-confidence seeds.
        order = sorted(range(len(detections)), key=lambda index:
                       (assignment[index] is None, -float(detections[index].handedness_confidence)))
        for index in order:
            detected = detections[index]
            wrist, center, bbox = geometries[index]
            key = assignment[index]
            reasons = []
            exclusivity_inferred = False
            candidate_margin = min((cost - best_cost for cost, other in candidates[1:]
                                    if other[index] != key), default=math.inf)
            assignment_ambiguous = key is not None and candidate_margin < self.ambiguity_margin
            if key is None:
                available = {"left", "right"} - reserved_sides
                if detected.side in available:
                    side = detected.side
                    seed_uncertain = False
                else:
                    side = next(iter(available))
                    seed_uncertain = True
                    reasons.append("native_side_conflicts_with_other_observed_track")
                    # One historically established hand plus a clearly
                    # separate second region permits a one-person/two-hands
                    # exclusivity inference. This is explicitly weaker than
                    # a native classification, and not used at cold start or
                    # overlapping/ambiguous geometric associations.
                    for other_index, other_key in enumerate(assignment):
                        if other_key is None or other_index == index:
                            continue
                        other_track = self._tracks[other_key]
                        other_result = proposed.get(other_index)
                        other_wrist, _, other_bbox = geometries[other_index]
                        average_diagonal = max((np.linalg.norm(bbox[2:] - bbox[:2])
                                                + np.linalg.norm(other_bbox[2:] - other_bbox[:2])) / 2, .05)
                        if (other_result is not None and not other_result["ambiguous"]
                                and not other_track.uncertain_seed
                                and _iou(bbox, other_bbox) < .1
                                and np.linalg.norm(wrist - other_wrist) > .75 * average_diagonal):
                            exclusivity_inferred = True
                            reasons.append("single_person_opposite_hand_exclusivity_inference")
                # Replacing an unassociated same-side track is a new identity,
                # not a successful re-identification across an implausible jump.
                replaced = [old_key for old_key, track in self._tracks.items() if track.side == side]
                for old_key in replaced:
                    del self._tracks[old_key]
                track = _Track(self._next_id, side, wrist, center, bbox,
                               np.zeros(2), timestamp_ms, seed_uncertain)
                self._next_id += 1
                self._tracks[track.track_id] = track
                reserved_sides.add(side)
                source = "new_track_after_spatial_rejection" if replaced else "new_track_native_seed"
                if expired:
                    source = "new_track_after_expiry"
                if seed_uncertain:
                    source = "new_track_opposite_hand_inference" if exclusivity_inferred else "new_track_native_conflict"
                association_score = None
                gap_ms = None
                ambiguous = seed_uncertain or bool(replaced)
            else:
                track = self._tracks[key]
                gap_ms = timestamp_ms - track.last_seen_ms
                ambiguous = assignment_ambiguous
                if assignment_ambiguous:
                    reasons.append("competing_geometric_assignments_have_similar_cost")
                if track.uncertain_seed:
                    # A native agreement on an unambiguous later observation
                    # establishes an initially conflicted side seed.
                    if not assignment_ambiguous and detected.side == track.side:
                        track.uncertain_seed = False
                    else:
                        ambiguous = True
                        reasons.append("track_side_seed_not_yet_confirmed")
                source = "reacquired_geometry" if gap_ms > 100 else "continuous_geometry"
                association_score = float(math.exp(-costs[index][key]) * math.exp(-gap_ms / self.max_gap_ms))
                if ambiguous:
                    source = "ambiguous_geometry_native_retained"
                    # Freeze geometry through unresolved overlap. Committing
                    # an arbitrary crossing assignment would poison velocity.
                else:
                    velocity = (wrist - track.wrist) / max(gap_ms / 1000, 1e-3)
                    magnitude = float(np.linalg.norm(velocity))
                    if magnitude > 2.0:
                        velocity *= 2.0 / magnitude
                    track.velocity = 0.5 * track.velocity + 0.5 * velocity
                    track.wrist, track.center, track.bbox = wrist, center, bbox
                    track.last_seen_ms = timestamp_ms
            side_for_crop = track.side if exclusivity_inferred or not ambiguous else detected.side
            proposed[index] = {
                "side": side_for_crop, "native_side": detected.side,
                "native_handedness_confidence": float(detected.handedness_confidence),
                "track_id": track.track_id, "track_side": track.side,
                "corrected": side_for_crop != detected.side,
                "correction_suppressed": side_for_crop == detected.side and track.side != detected.side,
                "opposite_hand_exclusivity_inferred": exclusivity_inferred,
                "source": source, "association_score": association_score,
                "association_score_semantics": "geometric_temporal_agreement_not_probability_or_pose_confidence",
                "ambiguous": ambiguous, "ambiguity_reasons": reasons,
                "reacquired": gap_ms is not None and gap_ms > 100,
                "gap_since_associated_observation_ms": gap_ms,
                "assignment_margin": candidate_margin if math.isfinite(candidate_margin) else None,
                "expired_track_ids": expired,
                "timestamp_ms": timestamp_ms,
            }
        return [proposed[index] for index in range(len(detections))]
