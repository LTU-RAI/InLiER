"""Registration accuracy: how good is the pose, not just the retrieval.

This is the metric family the retrieval-only baselines cannot report.  They
measure the XY distance between a query and its matched database pose -- a
proxy for "did we retrieve the right place" -- because they have no
registration step at all.  InLiER estimates a full 6-DoF transform, so it can
be scored on the transform.

Ported from ``evaluate_inlier_helipr.py`` (:595 ``_rotation_angle``,
:602 ``_pose_errors_from_T``, :618 / :676 the TP aggregations), plus
registration recall, which is the standard headline number for a registration
method and was not computed anywhere.

Frames
------
``VerifyOutput.T_sensor`` maps query sensor -> DB sensor.  The ground-truth
relative transform must run the same direction, so it is
``inv(T_db) @ T_q``, built explicitly from ``R_db.T`` rather than a matrix
inverse -- ``T_db`` is a rigid transform, so transposing the rotation is both
exact and cheaper than a general inverse.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: A registration counts as successful below both thresholds.  5 degrees /
#: 2 metres is the convention used for outdoor LiDAR registration recall.
DEFAULT_RRE_DEG = 5.0
DEFAULT_RTE_M = 2.0


def rotation_angle(R: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix, in degrees."""
    cos_val = (np.trace(R) - 1.0) / 2.0
    return float(math.degrees(math.acos(float(np.clip(cos_val, -1.0, 1.0)))))


def pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> Tuple[float, float]:
    """``(translation_error_m, rotation_error_deg)`` between two SE(3)."""
    dt = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
    dr = rotation_angle(T_est[:3, :3].T @ T_gt[:3, :3])
    return dt, dr


def relative_pose(T_q: np.ndarray, T_db: np.ndarray) -> np.ndarray:
    """Ground-truth query-sensor -> DB-sensor transform, ``inv(T_db) @ T_q``."""
    R_db, t_db = T_db[:3, :3], T_db[:3, 3]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_db.T @ T_q[:3, :3]
    T[:3, 3] = R_db.T @ (T_q[:3, 3] - t_db)
    return T


def _summarise(t_errors: List[float], r_errors: List[float],
               rre_deg: float, rte_m: float) -> Dict[str, float]:
    t = np.asarray(t_errors)
    r = np.asarray(r_errors)
    success = (r <= rre_deg) & (t <= rte_m)
    return {
        "n_pairs": int(t.size),
        "translation_mean_m": float(np.mean(t)),
        "translation_std_m": float(np.std(t)),
        "translation_median_m": float(np.median(t)),
        "rotation_mean_deg": float(np.mean(r)),
        "rotation_std_deg": float(np.std(r)),
        "rotation_median_deg": float(np.median(r)),
        # Registration recall: the fraction of matches whose pose is actually
        # usable.  A low mean error with a low registration recall means a few
        # gross failures are hiding behind good median behaviour.
        "registration_recall": float(np.mean(success)),
        "rre_threshold_deg": float(rre_deg),
        "rte_threshold_m": float(rte_m),
    }


def errors_for_pairs(
    pairs: Sequence[Tuple[int, int]],
    transforms: Dict[Tuple[int, int], np.ndarray],
    q_poses: np.ndarray,
    db_poses: np.ndarray,
    rre_deg: float = DEFAULT_RRE_DEG,
    rte_m: float = DEFAULT_RTE_M,
) -> Optional[Dict[str, float]]:
    """Aggregate pose error over ``(query, db)`` pairs with an estimated transform.

    Returns ``None`` when no pair has an estimate -- the caller reports that as
    "not measured" rather than as a zero error.
    """
    t_errors: List[float] = []
    r_errors: List[float] = []
    for j, d in pairs:
        T_est = transforms.get((j, d))
        if T_est is None:
            continue
        dt, dr = pose_error(T_est, relative_pose(q_poses[j], db_poses[d]))
        t_errors.append(dt)
        r_errors.append(dr)
    if not t_errors:
        return None
    return _summarise(t_errors, r_errors, rre_deg, rte_m)


def errors_from_verify(
    pairs: Sequence[Tuple[int, int]],
    verify_outputs: Dict[Tuple[int, int], Any],
    q_poses: np.ndarray,
    db_poses: np.ndarray,
    **kwargs,
) -> Optional[Dict[str, float]]:
    """Pose error of the verification stage, over pairs it verified successfully."""
    transforms = {
        key: out.T_sensor
        for key, out in verify_outputs.items()
        if out is not None and out.success
    }
    return errors_for_pairs(pairs, transforms, q_poses, db_poses, **kwargs)


def errors_from_gicp(
    pairs: Sequence[Tuple[int, int]],
    gicp_outputs: Dict[Tuple[int, int], Any],
    q_poses: np.ndarray,
    db_poses: np.ndarray,
    **kwargs,
) -> Optional[Dict[str, float]]:
    """Pose error after GICP refinement."""
    transforms = {
        key: out.T_sensor
        for key, out in gicp_outputs.items()
        if out is not None and out.success
    }
    return errors_for_pairs(pairs, transforms, q_poses, db_poses, **kwargs)


def match_distances(
    pairs: Sequence[Tuple[int, int]],
    q_positions: np.ndarray,
    db_positions: np.ndarray,
) -> Optional[Dict[str, float]]:
    """XY distance between matched poses -- the retrieval-only proxy metric.

    Kept so InLiER can be tabulated against methods that report only this.
    """
    if not pairs:
        return None
    d = np.asarray([
        float(np.linalg.norm(q_positions[j][:2] - db_positions[i][:2]))
        for j, i in pairs
    ])
    return {
        "n_pairs": int(d.size),
        "mean_m": float(np.mean(d)),
        "std_m": float(np.std(d)),
        "median_m": float(np.median(d)),
        "min_m": float(np.min(d)),
        "max_m": float(np.max(d)),
    }
