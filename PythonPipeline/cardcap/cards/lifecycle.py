"""Packet identity graph from explicit, persistent rigid-group observations.

This module manages lifetimes; it does not infer rigid membership from masks.
A missing face, touching silhouettes, or a connected-component count is NOT a
rigid-group observation. Upstream geometry must supply supported membership,
or pass None for unknown. Membership atoms are persistent tracked parts, not
individual cards and not automatically equivalent to SAM2 object IDs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name}: expected integer >= {minimum}")
    return value


def _partition(groups: list[list[str]]) -> tuple[tuple[str, ...], ...]:
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups: expected nonempty list of nonempty membership groups")
    seen, result = set(), []
    for group in groups:
        if not isinstance(group, list) or not group:
            raise ValueError("group: expected nonempty list")
        for member in group:
            if not isinstance(member, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", member):
                raise ValueError("members must be nonempty safe identity strings")
            if member in seen:
                raise ValueError("membership groups must be disjoint, with no duplicate members")
            seen.add(member)
        result.append(tuple(sorted(group)))
    return tuple(sorted(result))


@dataclass(frozen=True)
class LifecycleConfig:
    confirmation_frames: int = 3
    id_prefix: str = "pkt"

    def __post_init__(self):
        _integer(self.confirmation_frames, "confirmation_frames", 1)
        if not isinstance(self.id_prefix, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", self.id_prefix):
            raise ValueError("id_prefix: expected safe nonempty string")


class PacketLifecycle:
    """Debounce complete supported partitions, preserving unknown intervals.

    Events use the first frame of the new persistently observed partition.
    Old nodes end at event_frame-1; new nodes begin at event_frame (inclusive
    lifetimes, no duplicated active interval). confirmed_at_frame records the
    later observation that established the configured persistence. This is
    an offline graph: consumers should export after finishing the timeline.
    """

    def __init__(self, config: LifecycleConfig | None = None):
        self.config = config or LifecycleConfig()
        self._nodes: list[dict] = []
        self._active: dict[tuple[str, ...], dict] = {}
        self._pending: tuple[tuple[str, ...], ...] | None = None
        self._pending_rows: list[dict] = []
        self._members: set[str] | None = None
        self._last_frame: int | None = None
        self._last_active_partition_observed_frame: int | None = None
        self._events = {"splits": [], "merges": []}
        self._diagnostics: list[dict] = []
        self._finished = False

    def _new_node(self, members, frame, confirmed_frame, evidence):
        node = {"id": f"{self.config.id_prefix}_{len(self._nodes):04d}",
                "birth_frame": frame, "death_frame": None,
                "membership_atoms": list(members), "confirmed_at_frame": confirmed_frame,
                "membership_evidence": deepcopy(evidence), "death_reason": None}
        self._nodes.append(node)
        self._active[members] = node
        return node

    def _clear_pending(self):
        self._pending = None
        self._pending_rows = []

    def observe(self, frame: int, groups: list[list[str]] | None, *, evidence: dict) -> None:
        """Consume one sequential frame; None suspends all topology inference.

        evidence requires kind, description, and supported_rigid_membership.
        Supported partitions must cover the complete established member set.
        Partial visibility must be sent as None; it must never remove atoms.
        An unknown frame clears pending confirmation, but preserves old nodes.
        """
        if self._finished:
            raise RuntimeError("lifecycle already finished")
        _integer(frame, "frame")
        if self._last_frame is not None and frame != self._last_frame + 1:
            raise ValueError("frames must be consecutive; pass unknown observations for gaps")
        if not isinstance(evidence, dict) or any(not isinstance(evidence.get(k), str) or not evidence[k].strip()
                                                for k in ("kind", "description")):
            raise ValueError("evidence requires nonempty kind and description")
        if not isinstance(evidence.get("supported_rigid_membership"), bool):
            raise ValueError("evidence.supported_rigid_membership requires explicit bool")
        partition = None
        if groups is not None:
            if evidence["supported_rigid_membership"] is not True:
                raise ValueError("groups require supported rigid-membership evidence; use None when unknown")
            partition = _partition(groups)
            members = {member for group in partition for member in group}
            if self._members is not None and members != self._members:
                raise ValueError("partition must conserve established membership atoms; visibility loss is not a death")
        elif evidence["supported_rigid_membership"]:
            raise ValueError("supported membership evidence requires a partition")
        # Validation precedes state mutation, so a rejected call is retryable.
        self._last_frame = frame
        if partition is None:
            self._clear_pending()
            self._diagnostics.append({"frame": frame, "state": "membership_unknown",
                                      "evidence": deepcopy(evidence), "inferred_event": False})
            return
        if partition == tuple(sorted(self._active)):
            self._last_active_partition_observed_frame = frame
            self._clear_pending()
            return
        if partition != self._pending:
            self._pending, self._pending_rows = partition, []
        self._pending_rows.append({"frame": frame, "evidence": deepcopy(evidence)})
        if len(self._pending_rows) < self.config.confirmation_frames:
            return
        birth = self._pending_rows[0]["frame"]
        support = deepcopy(self._pending_rows)
        if not self._active:
            self._members = {member for group in partition for member in group}
            for group in partition:
                self._new_node(group, birth, frame, support)
            self._diagnostics.append({"frame": birth, "confirmed_at_frame": frame,
                                      "state": "initial_membership_observed", "inferred_event": False})
            self._last_active_partition_observed_frame = frame
            self._clear_pending()
            return
        old = set(self._active)
        new = set(partition)
        changed_old, changed_new = old - new, new - old
        # Connected components of the membership-overlap bipartite graph.
        # Only one-to-many and many-to-one components establish events.
        components = []
        unseen = set(changed_old)
        while unseen:
            left = {min(unseen)}
            right: set[tuple[str, ...]] = set()
            while True:
                next_right = {b for b in changed_new if any(set(a) & set(b) for a in left)}
                next_left = {a for a in changed_old if any(set(a) & set(b) for b in next_right)}
                if left == next_left and right == next_right:
                    break
                left, right = next_left, next_right
            components.append((sorted(left), sorted(right)))
            unseen.difference_update(left)
        if any(not (len(left) == 1 < len(right) or len(right) == 1 < len(left))
               for left, right in components):
            self._diagnostics.append({"frame": birth, "confirmed_at_frame": frame,
                "state": "unresolved_many_to_many_repartition", "proposed_groups": [list(g) for g in partition],
                "evidence": support, "inferred_event": False})
            self._clear_pending()
            return
        for left, right in components:
            before = [self._active.pop(group) for group in left]
            event_type = "split" if len(left) == 1 else "merge"
            for node in before:
                node["death_frame"], node["death_reason"] = birth - 1, event_type
            after = [self._new_node(group, birth, frame, support) for group in right]
            event = {"frame": birth, "confirmed_at_frame": frame,
                     "frame_is_first_persistent_observation_not_ground_truth": True,
                     "transition_observation_bracket": {
                         "last_old_partition_observed_frame": self._last_active_partition_observed_frame,
                         "first_new_partition_observed_frame": birth,
                         "possible_first_new_state_frame_interval": [self._last_active_partition_observed_frame + 1, birth],
                         "semantics": "Observed endpoint bracket, conditional on supplied identity evidence; not an exact physical transition time",
                         "unobserved_intermediate_topologies_excluded": False,
                     },
                     "evidence": support}
            if event_type == "split":
                event.update(source=before[0]["id"], results=[node["id"] for node in after])
                self._events["splits"].append(event)
            else:
                event.update(sources=[node["id"] for node in before], result=after[0]["id"])
                self._events["merges"].append(event)
        self._last_active_partition_observed_frame = frame
        self._clear_pending()

    def finish(self) -> dict:
        if self._last_frame is None:
            raise ValueError("cannot finish an empty timeline")
        if not self._finished:
            for node in self._active.values():
                node["death_frame"] = self._last_frame
                node["death_reason"] = "end_of_observed_clip_not_physical_destruction"
            if self._pending_rows:
                self._diagnostics.append({"frame": self._pending_rows[0]["frame"],
                    "state": "unconfirmed_terminal_partition", "evidence": deepcopy(self._pending_rows),
                    "inferred_event": False})
            self._finished = True
        return deepcopy({"format_version": "cardcap.packet_lifecycle/1.0",
            "scope": "Identity and lifetime graph only; nodes lack geometry/contact fields required by final cardcap packets",
            "confirmation_frames": self.config.confirmation_frames,
            "last_frame": self._last_frame, "packet_nodes": self._nodes,
            "events": self._events, "diagnostics": self._diagnostics,
            "independent_event_accuracy_validated": False,
            "membership_detector_implemented_here": False})
