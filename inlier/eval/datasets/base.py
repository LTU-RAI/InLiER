"""What a dataset has to provide, stated once.

``HeLiPR_Handler`` and ``Generic_Handler`` already agreed on a return shape --
both hand back ``{"poses", "point_clouds", "pose_timestamps", "pc_timestamps"}``
and ``Generic_Handler`` says so in a comment ("shaped like
HeLiPR_Handler.load_helipr"), padding the timestamp lists with zeros purely for
parity.  That agreement was the real dataset abstraction; it just was not
written down anywhere a third loader could find it.

``Sequence`` is that shape as a type, and ``SequenceSource`` is the protocol a
loader satisfies.  Adding a dataset means implementing ``load()`` and
registering it -- no evaluation code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence as Seq, runtime_checkable

import numpy as np


def arc_length(positions: np.ndarray) -> np.ndarray:
    """(N,) cumulative XY distance travelled along a trajectory.

    Free-standing because the online protocols have the positions but not the
    :class:`Sequence` -- the descriptor cache stores poses, not point clouds.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    step = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(step)))


@dataclass
class Sequence:
    """One loaded sequence: submaps with a global pose each.

    ``poses[i]`` is the global SE(3) pose of ``point_clouds[i]``, whose points
    are in that keyframe's own frame.  The two lists are always the same length
    -- loaders raise rather than return a mismatch, because a silent offset
    between poses and scans misaligns everything downstream.
    """

    poses: List[np.ndarray]                 # (4,4) float64, global
    point_clouds: List[np.ndarray]          # (N_i, 3) float32, keyframe frame
    pose_timestamps: List[float] = field(default_factory=list)
    pc_timestamps: List[float] = field(default_factory=list)
    #: free-form provenance, echoed into the results JSON
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.poses) != len(self.point_clouds):
            raise ValueError(
                f"pose/scan count mismatch: {len(self.poses)} poses vs "
                f"{len(self.point_clouds)} scans"
            )
        n = len(self.poses)
        if not self.pose_timestamps:
            self.pose_timestamps = [0.0] * n
        if not self.pc_timestamps:
            self.pc_timestamps = [0.0] * n

    def __len__(self) -> int:
        return len(self.poses)

    @property
    def positions(self) -> np.ndarray:
        """(N, 3) translations.  Only the XY columns are ever used for GT."""
        if not self.poses:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray([p[:3, 3] for p in self.poses], dtype=np.float64)

    def arc_length(self) -> np.ndarray:
        """(N,) cumulative distance travelled, for distance-based exclusion."""
        return arc_length(self.positions)

    def transformed(self, T: Optional[np.ndarray]) -> "Sequence":
        """Copy with ``T`` applied to every pose (DB world frame -> Q world frame).

        Needed whenever the database and query sessions were mapped
        independently: their poses start at unrelated origins, so comparing
        them directly -- which both the overlap GT and the ``max_pose_dist``
        filter do -- would be meaningless.
        """
        if T is None:
            return self
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"transform must be 4x4, got {T.shape}")
        return Sequence(
            poses=[T @ p for p in self.poses],
            point_clouds=self.point_clouds,
            pose_timestamps=list(self.pose_timestamps),
            pc_timestamps=list(self.pc_timestamps),
            meta={**self.meta, "transform_applied": True},
        )

    @classmethod
    def from_handler_dict(cls, data: Dict[str, Any], **meta: Any) -> "Sequence":
        """Adapt the dict both legacy handlers return."""
        return cls(
            poses=list(data["poses"]),
            point_clouds=list(data["point_clouds"]),
            pose_timestamps=list(data.get("pose_timestamps") or []),
            pc_timestamps=list(data.get("pc_timestamps") or []),
            meta=dict(meta),
        )


@runtime_checkable
class SequenceSource(Protocol):
    """A loader that can produce a :class:`Sequence`."""

    #: short name used by ``--dataset-type`` and written into results
    name: str

    def load(self, **kwargs: Any) -> Sequence:
        ...

    def describe(self) -> Dict[str, Any]:
        """Provenance for the results JSON (paths, sensors, accumulation)."""
        ...

    @classmethod
    def from_describe(cls, described: Dict[str, Any], *, root: Any = None,
                      verbose: bool = False) -> "SequenceSource":
        """Rebuild a source from a :meth:`describe` block in a results JSON.

        The inverse of :meth:`describe`, so a tool that reads a finished run
        (``inlier play``) can reload its scans without being told again how the
        sequence was assembled.
        """
        ...


def load_transform(path: Any) -> np.ndarray:
    """Read a 4x4 transform from a text file."""
    T = np.loadtxt(str(path), dtype=np.float64)
    if T.shape == (12,):
        T = np.vstack([T.reshape(3, 4), [0.0, 0.0, 0.0, 1.0]])
    elif T.shape == (3, 4):
        T = np.vstack([T, [0.0, 0.0, 0.0, 1.0]])
    if T.shape != (4, 4):
        raise ValueError(f"{path}: expected a 4x4 (or 3x4 / 12-value) transform, got {T.shape}")
    return T
