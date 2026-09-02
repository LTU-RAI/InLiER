"""GICP refinement of verified pairs.

Verification returns a pose from token correspondences; GICP refines it against
the actual point clouds.  Ported from the refinement block in
``run_evaluation`` (``evaluate_inlier_helipr.py`` :1388).

Raw clouds are loaded on demand and only for the scans that are actually
needed.  Refining a few hundred pairs out of a 2700-scan sequence would
otherwise mean holding every cloud in memory to use a handful of them.
"""

from __future__ import annotations

import gc
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from inlier.eval.encode import voxel_downsample
from inlier.eval.pipeline import minimal_keypoints


def _collect_raw(
    indices: Set[int],
    load_clouds: Callable[[], Sequence[np.ndarray]],
    voxel_size: float,
    desc: str,
    verbose: bool,
) -> Dict[int, np.ndarray]:
    if not indices:
        return {}
    clouds = load_clouds()
    iterator: Any = sorted(indices)
    if verbose:
        try:
            import tqdm
            iterator = tqdm.tqdm(iterator, desc=desc)
        except ImportError:
            pass
    out: Dict[int, np.ndarray] = {}
    for idx in iterator:
        pts = np.asarray(clouds[idx], dtype=np.float32)
        pts = pts[np.any(pts != 0, axis=1)]
        if voxel_size > 0:
            pts = voxel_downsample(pts, voxel_size)
        out[idx] = pts
    del clouds
    gc.collect()
    return out


def refine_pairs(
    pairs: Sequence[Tuple[int, int]],
    verify_outputs: Dict[Tuple[int, int], Any],
    q_encoded,
    db_encoded,
    gicp_cfg,
    voxel_size: float,
    load_q_clouds: Optional[Callable[[], Sequence[np.ndarray]]] = None,
    load_db_clouds: Optional[Callable[[], Sequence[np.ndarray]]] = None,
    verbose: bool = True,
) -> Tuple[Dict[Tuple[int, int], Any], float, int]:
    """Refine every pair that verification succeeded on.

    Returns ``(outputs, elapsed_seconds, n_converged)``.
    """
    from inlier.core.InLiER_Matcher import InLiER_Matcher

    usable = [(j, d) for j, d in pairs
              if verify_outputs.get((j, d)) is not None
              and verify_outputs[(j, d)].success]
    if not usable:
        return {}, 0.0, 0

    raw_q: Dict[int, np.ndarray] = {}
    raw_db: Dict[int, np.ndarray] = {}
    if gicp_cfg.use_raw_clouds:
        if load_q_clouds is None or load_db_clouds is None:
            raise ValueError(
                "gicp.use_raw_clouds is set but no cloud loader was provided; "
                "pass loaders or set gicp.use_raw_clouds: false to refine on keypoints"
            )
        raw_db = _collect_raw({d for _, d in usable}, load_db_clouds, voxel_size,
                              "  Loading DB raw clouds", verbose)
        raw_q = _collect_raw({j for j, _ in usable}, load_q_clouds, voxel_size,
                             "  Loading Q raw clouds", verbose)

    iterator: Any = usable
    if verbose:
        try:
            import tqdm
            iterator = tqdm.tqdm(usable, desc="  GICP refinement")
        except ImportError:
            pass

    outputs: Dict[Tuple[int, int], Any] = {}
    converged = 0
    start = time.time()
    for j, d in iterator:
        vout = verify_outputs[(j, d)]
        q_kp = minimal_keypoints(q_encoded.kp_aligned[j], q_encoded.kp_sensor[j],
                                 q_encoded.T_grounds[j])
        db_kp = minimal_keypoints(db_encoded.kp_aligned[d], db_encoded.kp_sensor[d],
                                  db_encoded.T_grounds[d])
        out = InLiER_Matcher.refine_gicp(
            vout, q_kp, db_kp,
            query_raw=raw_q.get(j) if gicp_cfg.use_raw_clouds else None,
            db_raw=raw_db.get(d) if gicp_cfg.use_raw_clouds else None,
            config=gicp_cfg, verbose=False,
        )
        outputs[(j, d)] = out
        converged += bool(out.success)
    elapsed = time.time() - start

    del raw_q, raw_db
    gc.collect()
    return outputs, elapsed, converged
