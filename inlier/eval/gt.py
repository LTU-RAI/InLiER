"""Ground truth, exclusion windows, and candidate filtering.

Ground truth used to be one inlined conjunction (``build_ground_truth``,
``evaluate_inlier_helipr.py`` :159): a database scan is a positive for a query
if their scan overlap is at least ``overlap_threshold`` *and* their XY pose
distance is at most ``max_pose_dist``.  That rule is right for the
cross-session protocol and wrong for every other one, so it is a policy here
rather than a hardcoded expression.

Exclusion
---------
Online protocols must not match a query against the frames it just came from.
The surrounding implementations each picked a different unit for that window --
one counts frames, one counts wall-clock seconds, one counts distance
travelled -- and they are not interchangeable: at a standstill a frame window
excludes almost no distance, and while driving fast a time window excludes far
more of the trajectory than intended.  :class:`Exclusion` supports all three
and resolves any of them to the same thing: an exclusive index cutoff.  That is
also exactly what an incremental matcher needs as a database bound, so the
protocol and the retrieval bound cannot disagree.

Candidate filtering
-------------------
:class:`RadiusFilter` restricts candidates by *true* pose distance before
descriptor matching.  That is a geometric oracle -- a deployed system has no
such prefilter -- so it makes results optimistic and is recorded in the results
JSON rather than applied silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, runtime_checkable

import numpy as np

GroundTruth = Dict[int, np.ndarray]


# ---------------------------------------------------------------------------
#  Ground-truth policies
# ---------------------------------------------------------------------------

@runtime_checkable
class GroundTruthPolicy(Protocol):
    """Maps a query index to the database indices that count as correct."""

    name: str

    def positives(self, q: int) -> np.ndarray: ...

    def describe(self) -> Dict[str, object]: ...


def _as_dict(policy: "GroundTruthPolicy", n_queries: int) -> GroundTruth:
    return {j: policy.positives(j) for j in range(n_queries)}


@dataclass
class OverlapAndDistance:
    """Overlap >= threshold **and** XY pose distance <= ``max_pose_dist``.

    The cross-session rule, verbatim from ``build_ground_truth``.  Note the
    overlap matrix is indexed ``[db, query]``, and that only the XY components
    of the positions are compared -- height is ignored, so a scan on a bridge
    above another counts as near.  Both are preserved.
    """

    overlap: np.ndarray            # (N_db, N_q)
    db_positions: np.ndarray       # (N_db, 3)
    q_positions: np.ndarray        # (N_q, 3)
    overlap_threshold: float = 0.3
    max_pose_dist: float = 25.0    # <= 0 disables the distance criterion

    name: str = "overlap_and_distance"

    def positives(self, q: int) -> np.ndarray:
        mask = self.overlap[:, q] >= self.overlap_threshold
        if self.max_pose_dist > 0.0:
            dists_xy = np.linalg.norm(
                self.db_positions[:, :2] - self.q_positions[q, :2], axis=1
            )
            mask = mask & (dists_xy <= self.max_pose_dist)
        return np.where(mask)[0]

    def describe(self):
        return {
            "policy": self.name,
            "overlap_threshold": float(self.overlap_threshold),
            "max_pose_dist": float(self.max_pose_dist),
        }


@dataclass
class DistanceOnly:
    """XY pose distance alone.

    For datasets with no overlap matrix -- a quick evaluation, or a sequence
    where building the matrix is not worth the compute.  Weaker than
    :class:`OverlapAndDistance`: two scans can be metres apart and still see
    nothing in common (opposite sides of a wall, opposite directions through a
    tunnel), and this policy calls those positives.
    """

    db_positions: np.ndarray
    q_positions: np.ndarray
    max_pose_dist: float = 25.0

    name: str = "distance_only"

    def positives(self, q: int) -> np.ndarray:
        d = np.linalg.norm(self.db_positions[:, :2] - self.q_positions[q, :2], axis=1)
        return np.where(d <= self.max_pose_dist)[0]

    def describe(self):
        return {"policy": self.name, "max_pose_dist": float(self.max_pose_dist)}


@dataclass
class Causal:
    """Past frames within ``max_pose_dist``, outside the exclusion window.

    The single-session loop-closure rule: query ``t`` may only be matched
    against frames at index ``< exclusion.cutoff(t)``.
    """

    positions: np.ndarray           # (N, 3)
    exclusion: "Exclusion"
    max_pose_dist: float = 10.0
    timestamps: Optional[np.ndarray] = None
    arc_length: Optional[np.ndarray] = None

    name: str = "causal_distance"

    def positives(self, q: int) -> np.ndarray:
        cutoff = self.exclusion.cutoff(q, self.timestamps, self.arc_length)
        if cutoff <= 0:
            return np.zeros(0, dtype=np.int64)
        past = np.arange(cutoff, dtype=np.int64)
        d = np.linalg.norm(self.positions[past, :2] - self.positions[q, :2], axis=1)
        return past[d <= self.max_pose_dist]

    def describe(self):
        return {
            "policy": self.name,
            "max_pose_dist": float(self.max_pose_dist),
            "exclusion": self.exclusion.describe(),
        }


def build(policy: GroundTruthPolicy, n_queries: int) -> GroundTruth:
    """Materialise a policy into the ``{query: db_indices}`` dict the metrics take."""
    return _as_dict(policy, n_queries)


# ---------------------------------------------------------------------------
#  Exclusion window
# ---------------------------------------------------------------------------

@dataclass
class Exclusion:
    """How much of the recent past a query may not match against.

    Exactly one unit must be set.  All three resolve to an exclusive index
    cutoff so the rest of the pipeline -- ground truth, candidate filtering,
    and the matcher's database bound -- cannot drift apart.
    """

    frames: Optional[int] = None
    seconds: Optional[float] = None
    metres: Optional[float] = None

    def __post_init__(self) -> None:
        given = [u for u in ("frames", "seconds", "metres") if getattr(self, u) is not None]
        if len(given) != 1:
            raise ValueError(
                "Exclusion takes exactly one of frames=, seconds=, metres=; "
                f"got {given or 'none'}"
            )
        value = getattr(self, given[0])
        if value < 0:
            raise ValueError(f"exclusion {given[0]}={value} must be >= 0")
        self._unit = given[0]

    @property
    def unit(self) -> str:
        return self._unit

    def cutoff(
        self,
        t: int,
        timestamps: Optional[np.ndarray] = None,
        arc_length: Optional[np.ndarray] = None,
    ) -> int:
        """Exclusive upper index: candidates are ``[0, cutoff)``.

        All three units use the same strict rule, so that the same physical
        window expressed three ways gives the same cutoff.  For frames a
        candidate ``i`` is allowed when ``t - i > frames``; for seconds and
        metres, correspondingly, when the gap is *strictly greater* than the
        window.  ``side="left"`` is what makes the boundary strict -- with
        ``"right"`` a candidate sitting exactly on the window edge would be
        admitted by the time/distance units but rejected by the frame unit.
        """
        if self.frames is not None:
            return max(0, t - int(self.frames))

        if self.seconds is not None:
            if timestamps is None:
                raise ValueError("Exclusion(seconds=...) needs timestamps")
            limit = float(timestamps[t]) - float(self.seconds)
            return int(np.searchsorted(timestamps[: t + 1], limit, side="left"))

        if arc_length is None:
            raise ValueError("Exclusion(metres=...) needs arc_length")
        limit = float(arc_length[t]) - float(self.metres)
        return int(np.searchsorted(arc_length[: t + 1], limit, side="left"))

    def describe(self) -> Dict[str, object]:
        return {"unit": self.unit, "value": getattr(self, self.unit)}


# ---------------------------------------------------------------------------
#  Candidate filters
# ---------------------------------------------------------------------------

@runtime_checkable
class CandidateFilter(Protocol):
    name: str

    def bound(self, q: int) -> Optional[int]:
        """Exclusive database index bound, or ``None`` for unbounded."""

    def allows(self, q: int, db: int) -> bool: ...

    def describe(self) -> Dict[str, object]: ...


@dataclass
class NoFilter:
    """Every database entry is a candidate (the cross-session default)."""

    name: str = "none"

    def bound(self, q: int) -> Optional[int]:
        return None

    def allows(self, q: int, db: int) -> bool:
        return True

    def describe(self):
        return {"filter": self.name}


@dataclass
class CausalFilter:
    """Only frames older than the exclusion window."""

    exclusion: Exclusion
    timestamps: Optional[np.ndarray] = None
    arc_length: Optional[np.ndarray] = None

    name: str = "causal"

    def bound(self, q: int) -> Optional[int]:
        return self.exclusion.cutoff(q, self.timestamps, self.arc_length)

    def allows(self, q: int, db: int) -> bool:
        return db < self.bound(q)

    def describe(self):
        return {"filter": self.name, "exclusion": self.exclusion.describe()}


@dataclass
class RadiusFilter:
    """Causal filter plus a true-pose radius prefilter.

    **This is a geometric oracle.**  Restricting candidates by ground-truth
    pose before descriptors are compared is information a deployed system does
    not have, and it inflates every metric: the hard far-away distractors are
    removed for free.  Reference implementations apply it silently; here it is
    reported in the results JSON so a number produced with it is never mistaken
    for one produced without.
    """

    positions: np.ndarray
    radius: float
    exclusion: Optional[Exclusion] = None
    timestamps: Optional[np.ndarray] = None
    arc_length: Optional[np.ndarray] = None

    name: str = "radius"
    uses_pose_oracle: bool = True

    def bound(self, q: int) -> Optional[int]:
        if self.exclusion is None:
            return None
        return self.exclusion.cutoff(q, self.timestamps, self.arc_length)

    def allows(self, q: int, db: int) -> bool:
        upper = self.bound(q)
        if upper is not None and db >= upper:
            return False
        d = float(np.linalg.norm(self.positions[db, :2] - self.positions[q, :2]))
        return d <= self.radius

    def allowed_mask(self, q: int, n_db: int) -> np.ndarray:
        upper = self.bound(q)
        idx = np.arange(n_db)
        mask = np.ones(n_db, dtype=bool) if upper is None else (idx < upper)
        d = np.linalg.norm(self.positions[:n_db, :2] - self.positions[q, :2], axis=1)
        return mask & (d <= self.radius)

    def describe(self):
        return {
            "filter": self.name,
            "radius_m": float(self.radius),
            "uses_pose_oracle": True,
            "exclusion": self.exclusion.describe() if self.exclusion else None,
        }
