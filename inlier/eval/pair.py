"""Match two encoded scans against each other -- the ``inlier match`` core.

``inlier encode`` writes tokens and keypoints to an ``.npz``; this runs the
same pipeline stages the protocols run, on exactly two of them.  It is a
diagnostic, not a protocol: there is no database, no ground truth and no
metric, only "what does each stage say about this pair".

Everything here goes through :mod:`inlier.eval.pipeline`, so a stage cannot
score differently when asked one pair at a time than it does inside an
evaluation.  The database is a one-entry matcher, which is the degenerate case
of the real thing rather than a reimplementation of it.

Stages run in ``mode="eval"``: the score thresholds are relaxed so every stage
reports its number instead of pruning the single candidate away.  Being told
"stage 2 scored 0.11" is the point of the tool; being told "no candidate
survived" would not be.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from inlier.core.Dataclasses import InLiER_Tokens

#: Keys ``inlier encode`` always writes, and this needs.
REQUIRED_KEYS = ("token_id", "kp_sensor", "kp_aligned", "T_ground",
                 "N_h", "N_r", "N_s", "N_a")


@dataclass
class EncodedScan:
    """One ``inlier encode`` ``.npz``, loaded."""

    path: Path
    token_id: np.ndarray
    kp_sensor: np.ndarray
    kp_aligned: np.ndarray
    T_ground: np.ndarray
    radices: Tuple[int, int, int, int]      # (N_h, N_r, N_s, N_a)
    voxel_size: Optional[float]
    provenance: Dict[str, Any]              # whatever else the file carried

    @property
    def tokens(self) -> InLiER_Tokens:
        return InLiER_Tokens(token_id=self.token_id)

    @property
    def label(self) -> str:
        """A short human name: the submap it came from, else the filename."""
        index = self.provenance.get("submap_index")
        if index is not None:
            dataset = self.provenance.get("dataset")
            stem = Path(dataset).name if dataset else self.path.stem
            return f"{stem} submap {int(index)}"
        source = self.provenance.get("source")
        return Path(source).name if source else self.path.name


def load_encoded(path) -> EncodedScan:
    """Read an ``inlier encode`` ``.npz``.

    The radices are read from the file rather than from the active config:
    the mixed-radix packing is not invertible without the numbers it was
    packed with, so a token array and a config that disagree would silently
    unpack into a different descriptor.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such encoding: {path}")
    with np.load(path, allow_pickle=True) as data:
        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(
                f"{path} is missing {', '.join(missing)}; it does not look "
                f"like an `inlier encode` output.")
        held = {k: data[k] for k in data.files}

    voxel = held.get("voxel_size")
    try:
        voxel = None if voxel is None else float(voxel)
    except (TypeError, ValueError):       # saved as None -> 0-d object array
        voxel = None

    provenance = {k: _unwrap(v) for k, v in held.items()
                  if k not in REQUIRED_KEYS and k != "voxel_size"}
    return EncodedScan(
        path=path,
        token_id=held["token_id"],
        kp_sensor=np.asarray(held["kp_sensor"], dtype=np.float64),
        kp_aligned=np.asarray(held["kp_aligned"], dtype=np.float64),
        T_ground=np.asarray(held["T_ground"], dtype=np.float64),
        radices=tuple(int(held[k]) for k in ("N_h", "N_r", "N_s", "N_a")),
        voxel_size=voxel,
        provenance=provenance,
    )


def _unwrap(value):
    """0-d arrays back to scalars, so provenance reads like a dict."""
    arr = np.asarray(value)
    return arr.item() if arr.ndim == 0 else arr


def check_compatible(query: EncodedScan, db: EncodedScan, cfg) -> None:
    """Refuse pairs, or a config, that cannot be compared.

    Two token arrays packed with different radices index different descriptor
    spaces, so every stage score between them would be arithmetic on unrelated
    numbers -- a plausible-looking value with no meaning.  Same for scoring a
    file with a config whose encoder settings differ from the ones it was
    written with.
    """
    names = ("N_h", "N_r", "N_s", "N_a")
    if query.radices != db.radices:
        raise ValueError(
            f"the two encodings use different token radices, so their "
            f"descriptors are not comparable:\n"
            f"  {query.path.name}: "
            + ", ".join(f"{n}={v}" for n, v in zip(names, query.radices))
            + f"\n  {db.path.name}: "
            + ", ".join(f"{n}={v}" for n, v in zip(names, db.radices))
            + "\nRe-encode both with the same config.")

    active = (cfg.N_h, cfg.N_r, cfg.N_s, cfg.N_a)
    if active != query.radices:
        raise ValueError(
            f"the active config does not match how these files were encoded:\n"
            f"  files:  " + ", ".join(f"{n}={v}" for n, v in zip(names, query.radices))
            + "\n  config: " + ", ".join(f"{n}={v}" for n, v in zip(names, active))
            + "\nPass the config they were encoded with via -c/--config.")


@dataclass
class PairResult:
    """What each stage said about one pair."""

    mint: float
    beam: Optional[float] = None
    beam_shift: Optional[int] = None
    rerank: Optional[float] = None
    rerank_shift: Optional[int] = None
    verify: Optional[Any] = None            # VerifyOutput, or None if skipped
    gicp: Optional[Any] = None              # VerifyOutput after refinement
    gicp_on: str = ""                       # "raw clouds" / "keypoints" / ""

    @property
    def shift(self) -> int:
        """The azimuth shift verification was handed."""
        if self.rerank_shift is not None:
            return int(self.rerank_shift)
        return int(self.beam_shift or 0)

    @property
    def pose(self) -> Optional[np.ndarray]:
        """Best available query->DB transform, refined if GICP converged."""
        for out in (self.gicp, self.verify):
            if out is not None and out.success:
                return np.asarray(out.T_sensor, dtype=np.float64)
        return None

    def as_dict(self) -> Dict[str, Any]:
        """JSON-shaped summary, for ``--json``."""
        out: Dict[str, Any] = {"stage1_mint": round(float(self.mint), 6)}
        if self.beam is not None:
            out["stage2_beam"] = round(float(self.beam), 6)
            out["stage2_azimuth_shift"] = int(self.beam_shift or 0)
        if self.rerank is not None:
            out["rerank"] = round(float(self.rerank), 6)
            out["rerank_azimuth_shift"] = int(self.rerank_shift or 0)
        if self.verify is not None:
            v = self.verify
            out["verify"] = {
                "success": bool(v.success),
                "keypoint_inlier_ratio": round(float(v.keypoint_inlier_ratio), 6),
                "ransac_inlier_ratio": round(float(v.ransac_inlier_ratio), 6),
                "n_correspondences": int(v.n_correspondences),
                "n_ransac_inliers": int(v.n_ransac_inliers),
                "n_keypoint_inliers": int(v.n_keypoint_inliers),
                "n_total_keypoints": int(v.n_total_keypoints),
                "inlier_rmse": round(float(v.inlier_rmse), 6),
                "yaw_deg": round(float(np.degrees(v.yaw)), 4),
                "translation_m": [round(float(x), 6) for x in (v.tx, v.ty, v.tz)],
            }
        if self.gicp is not None:
            out["gicp"] = {"success": bool(self.gicp.success),
                           "converged": bool(self.gicp.converged),
                           "refined_on": self.gicp_on,
                           "n_iterations": int(self.gicp.n_iterations),
                           "n_inliers": int(self.gicp.n_inliers),
                           "final_error": round(float(self.gicp.final_error), 6)}
        pose = self.pose
        if pose is not None:
            out["T_query_to_db"] = [[round(float(x), 8) for x in row] for row in pose]
        return out


def match_pair(
    resolved,
    query: EncodedScan,
    db: EncodedScan,
    q_points: Optional[np.ndarray] = None,
    db_points: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> PairResult:
    """Run every enabled stage on one pair, in pipeline order.

    ``q_points``/``db_points`` are only used by GICP: without them refinement
    falls back to the keypoints, which needs no clouds and still reports a
    pose.  The stage scores themselves never look at a point cloud.
    """
    from inlier.eval.pipeline import build_matcher, minimal_keypoints

    matcher = build_matcher(resolved, [db.tokens], verbose=False)

    out = matcher.shortlist(query.tokens, topk=1, verbose=False)
    # A one-entry database always returns that entry, so the score is defined
    # even when it is a terrible match -- which is exactly what to show.
    result = PairResult(mint=float(out.scores[0]) if len(out.scores) else 0.0)

    if not resolved.skip_stage2:
        b = matcher.beam_score(query.tokens, [0], topk=1, verbose=False)
        if len(b.scores):
            result.beam = float(b.scores[0])
            result.beam_shift = int(b.best_shifts[0])

    if resolved.run_rerank and result.beam is not None:
        r = matcher.rerank(query.tokens, [0], [result.beam_shift],
                           topk=1, verbose=False)
        if len(r.scores):
            result.rerank = float(r.scores[0])
            result.rerank_shift = int(r.best_shifts[0])

    if resolved.skip_verify:
        return result

    q_kp = minimal_keypoints(query.kp_aligned, query.kp_sensor, query.T_ground)
    db_kp = minimal_keypoints(db.kp_aligned, db.kp_sensor, db.T_ground)
    result.verify = matcher.verify(
        query.tokens, q_kp, db.tokens, db_kp,
        azimuth_shift=result.shift, config=resolved.verify, verbose=False)

    if resolved.skip_gicp or not result.verify.success:
        return result

    _refine(resolved, result, q_kp, db_kp, q_points, db_points, verbose)
    return result


def _refine(resolved, result, q_kp, db_kp, q_points, db_points, verbose) -> None:
    """GICP on the raw clouds when we have them, on the keypoints otherwise."""
    import dataclasses

    from inlier.core.InLiER_Matcher import InLiER_Matcher

    cfg = resolved.gicp
    have_clouds = q_points is not None and db_points is not None
    if cfg.use_raw_clouds and not have_clouds:
        # The clouds are not in the .npz; refusing to refine would drop the
        # pose panel entirely, and keypoint GICP is the documented fallback.
        cfg = dataclasses.replace(cfg, use_raw_clouds=False)

    result.gicp_on = "raw clouds" if cfg.use_raw_clouds else "keypoints"
    result.gicp = InLiER_Matcher.refine_gicp(
        result.verify, q_kp, db_kp,
        query_raw=q_points if cfg.use_raw_clouds else None,
        db_raw=db_points if cfg.use_raw_clouds else None,
        config=cfg, verbose=verbose)
