"""``inlier run`` -- loop closures on data with no ground truth.

Everything else in :mod:`inlier.eval` answers "how good is this?", which needs
labels.  A deployment has none.  It has scans, an odometry estimate that
drifts, and one question: which frames close a loop, and what is the 6-DoF
constraint between them.

This drives the same stages the protocols drive -- MINT, BEAM, rerank, verify,
GICP, all imported from :mod:`inlier.eval.pipeline` so a closure found here is
scored exactly as one found inside an evaluation -- and stops before the
metrics.  It is deliberately *not* in :mod:`inlier.eval.protocols`: a protocol
fixes both which database entries a query may match *and* what counts as a
correct match, and this supplies only the first.  Filing it as a sibling of
``cross_session`` would invite comparing its output against an evaluation
table, which is the one reading that must never happen.

Two rules, and the rest is detail.

**The threshold cannot come from the data.**  ``inlier eval`` picks an
operating point by sweeping against labels; there is nothing here to sweep.
So ``threshold`` is required and has no default -- a default would be someone
else's operating point silently applied to your robot.

**Poses are odometry, and are never evidence.**  They accumulate submaps, they
may bound the search, and they fill two clearly-named diagnostic columns.  No
pose ever accepts or rejects a closure: that decision is the verification score
against the threshold, and nothing else.  The record says
``pose_source: "odometry"`` and ``ground_truth: null`` so a reader cannot
mistake the file for an evaluation.
"""

from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from inlier.eval import artifacts, gt as gtmod, pose
from inlier.eval.datasets.base import arc_length as arc_length_of
from inlier.eval.encode import encode_sequence
from inlier.eval.protocols.base import RunResult

#: Warn before writing score matrices bigger than this, in bytes.  They grow
#: with the square of the frame count, and a quarter-gigabyte of NaN should not
#: be a surprise.
SCORE_MATRIX_WARN_BYTES = 200 * 1024 * 1024


@dataclass
class DeploySpec:
    """One ``inlier run`` invocation.

    ``db_source`` is ``None`` for a single session, which streams causally and
    matches its own past.  Given one, it is a **fixed prior map**: encoded and
    finalized before the query starts, never added to.
    """

    resolved: Any
    source: Any                         # the query session
    threshold: float
    output_dir: Path
    db_source: Any = None               # None -> single-session
    exclusion: Optional[Any] = None     # single-session only
    search_radius: float = 0.0
    cache_dir: Optional[Path] = None
    config_path: Optional[Path] = None
    top_k: int = 20
    score_matrices: bool = True
    verbose: bool = True
    tag: str = "run"

    @property
    def cross_session(self) -> bool:
        return self.db_source is not None


def _log(spec: DeploySpec, message: str) -> None:
    if spec.verbose:
        print(message)


def validate(spec: DeploySpec) -> None:
    """Refuse the impossible before anything expensive happens.

    Both of these would otherwise surface after the encode -- forty minutes
    into a long session, having already written a cache.
    """
    r = spec.resolved
    if r.skip_verify:
        raise ValueError(
            "inlier run emits 6-DoF relative poses, and the verification "
            "stage is what produces them; verify.skip is set, so there would "
            "be nothing to emit. Unset it, or use `inlier eval` if you only "
            "want retrieval.")
    if not spec.threshold > 0.0:
        raise ValueError(
            f"--threshold must be > 0, got {spec.threshold}. A failed "
            f"verification scores exactly 0.0, so a threshold of 0 would "
            f"emit those failures as closures carrying an identity "
            f"transform.")
    if spec.cross_session and spec.exclusion is not None:
        raise ValueError(
            "an exclusion window withholds a query's own recent past, which "
            "only exists in a single-session run. The prior map is a "
            "different session; nothing in it needs excluding.")


# ---------------------------------------------------------------------------
#  Accepting closures -- the only decision this module makes
# ---------------------------------------------------------------------------

def accept(
    sims: Dict[int, Dict[int, float]],
    verify_outputs: Dict[Tuple[int, int], Any],
    threshold: float,
) -> List[Tuple[int, int, float, int]]:
    """``(query, db, score, rank)`` for every closure worth reporting.

    Every candidate at or above the threshold, not merely the best per query:
    a back-end can weigh or discard them, and dropping them here destroys
    information it cannot recover.  ``rank`` is the position within one
    query's accepted set, so a consumer wanting only the best can filter on
    ``rank == 0``.

    Success is required explicitly rather than inferred from the threshold
    being above zero -- the two are separate facts, and relying on the second
    to imply the first is how a failed verification ends up in the output.
    """
    out: List[Tuple[int, int, float, int]] = []
    for q in sorted(sims):
        passing = [
            (d, float(score)) for d, score in sims[q].items()
            if score >= threshold
            and (v := verify_outputs.get((q, d))) is not None and v.success
        ]
        passing.sort(key=lambda ds: (-ds[1], ds[0]))
        out.extend((q, d, score, rank)
                   for rank, (d, score) in enumerate(passing))
    return out


# ---------------------------------------------------------------------------
#  Rows
# ---------------------------------------------------------------------------

def _yaw_deg(T: np.ndarray) -> float:
    """Yaw of *this* transform, for eyeballing.

    Derived from the emitted rotation rather than taken from
    ``VerifyOutput.yaw``: that field is the *ground-aligned* yaw, a different
    frame from the ``T_sensor`` these columns carry, and it goes stale the
    moment GICP refines the pose.
    """
    return math.degrees(math.atan2(float(T[1, 0]), float(T[0, 0])))


def _odometry_diagnostics(T_est, q_pose, db_pose) -> Dict[str, Any]:
    """How far apart odometry puts the pair, and how much it disagrees.

    Both are diagnostics, never evidence.  A large ``odom_xy_distance_m``
    means large accumulated drift, *not* a wrong closure -- a correct closure
    across heavy drift is precisely the far-apart one.  The disagreement is
    the residual a back-end will have to absorb.
    """
    if q_pose is None or db_pose is None:
        return {"odom_xy_distance_m": "", "odom_disagreement_m": "",
                "odom_disagreement_deg": ""}
    d = float(np.linalg.norm(np.asarray(q_pose)[:2, 3] - np.asarray(db_pose)[:2, 3]))
    t_err, r_err = pose.pose_error(T_est, pose.relative_pose(q_pose, db_pose))
    return {"odom_xy_distance_m": round(d, 4),
            "odom_disagreement_m": round(float(t_err), 4),
            "odom_disagreement_deg": round(float(r_err), 4)}


def closure_rows(
    accepted: Sequence[Tuple[int, int, float, int]],
    verify_outputs: Dict[Tuple[int, int], Any],
    gicp_outputs: Dict[Tuple[int, int], Any],
    stage: str,
    q_poses,
    db_poses,
    q_stamps: Sequence[float],
    db_stamps: Sequence[float],
    q_stride: int,
    db_stride: int,
    odom_comparable: bool,
) -> List[Dict[str, Any]]:
    """One row per accepted closure, carrying the pose actually being offered.

    Where GICP refined a pair, that is the pose in the row -- the unrefined
    verify pose stays available in ``per_pair_verify_*.csv``, so refinement
    does not destroy the only record of what verification alone produced.
    """
    rows: List[Dict[str, Any]] = []
    for q, d, score, rank in accepted:
        v = verify_outputs[(q, d)]
        g = gicp_outputs.get((q, d))
        refined = g is not None and g.success
        T = np.asarray((g if refined else v).T_sensor, dtype=np.float64)

        row: Dict[str, Any] = {
            "query_idx": q, "db_idx": d, "rank": rank,
            "score": round(float(score), 6), "stage": stage,
            "q_stamp": (round(float(q_stamps[q]), 6) if q_stamps else ""),
            "db_stamp": (round(float(db_stamps[d]), 6) if db_stamps else ""),
            "q_scan_idx": q * q_stride,
            "db_scan_idx": d * db_stride,
            "yaw_deg": round(_yaw_deg(T), 4),
            "n_correspondences": int(v.n_correspondences),
            "n_ransac_inliers": int(v.n_ransac_inliers),
            "n_keypoint_inliers": int(v.n_keypoint_inliers),
            "n_total_keypoints": int(v.n_total_keypoints),
            "ransac_inlier_ratio": round(float(v.ransac_inlier_ratio), 6),
            "inlier_rmse": round(float(v.inlier_rmse), 6),
            "refined": int(refined),
        }
        row["tx"], row["ty"], row["tz"] = (float(T[0, 3]), float(T[1, 3]),
                                           float(T[2, 3]))
        for i in range(3):
            for j in range(3):
                row[f"r{i}{j}"] = float(T[i, j])

        # Empty, not 0, when GICP did not produce this row's pose: a 0 reads
        # as "converged in 0 iterations with 0 error".
        if g is not None:
            row.update({"gicp_converged": int(bool(g.converged)),
                        "gicp_error": round(float(g.final_error), 6),
                        "gicp_inliers": int(g.n_inliers),
                        "gicp_iterations": int(g.n_iterations)})
        else:
            row.update({"gicp_converged": "", "gicp_error": "",
                        "gicp_inliers": "", "gicp_iterations": ""})

        row.update(_odometry_diagnostics(
            T,
            q_poses[q] if odom_comparable else None,
            db_poses[d] if odom_comparable else None))
        rows.append(row)
    return rows


def _ranks(sims: Optional[Dict[int, Dict[int, float]]]) -> Dict[int, Dict[int, int]]:
    """Per-query rank of every candidate a stage scored, best first."""
    if not sims:
        return {}
    out: Dict[int, Dict[int, int]] = {}
    for q, per_db in sims.items():
        order = sorted(per_db, key=lambda d: (-per_db[d], d))
        out[q] = {d: rank for rank, d in enumerate(order)}
    return out


def score_rows(stage_sims: Dict[str, Any], shifts, top_k: int) -> List[Dict[str, Any]]:
    """The top candidates per query, with what each stage scored them.

    Deliberately decision-free: no threshold is applied and no acceptance is
    recorded.  A blank cell means *that stage never scored this pair* -- the
    funnel narrows at every step -- which is not the same as a score of 0.0,
    a value the stages can genuinely return.
    """
    ranks = {name: _ranks(sims) for name, sims in stage_sims.items()}
    rows: List[Dict[str, Any]] = []
    queries = sorted({q for sims in stage_sims.values() if sims for q in sims})

    for q in queries:
        mint = (stage_sims.get("mint") or {}).get(q, {})
        keep = sorted(mint, key=lambda d: (-mint[d], d))[:top_k]
        # A later stage can only score what MINT shortlisted, so MINT's order
        # is the right spine; anything else it scored is a bug, not a row.
        for d in keep:
            row: Dict[str, Any] = {"query_idx": q, "db_idx": d}
            for name in ("mint", "beam", "rerank", "verify"):
                sims = stage_sims.get(name) or {}
                value = sims.get(q, {}).get(d)
                row[name] = "" if value is None else round(float(value), 6)
                rank = ranks.get(name, {}).get(q, {}).get(d)
                row[f"rank_{name}"] = "" if rank is None else rank
            shift = (shifts or {}).get(q, {}).get(d)
            row["beam_shift"] = "" if shift is None else int(shift)
            rows.append(row)
    return rows


def score_matrices(stage_sims: Dict[str, Any], shifts,
                   n_q: int, n_db: int) -> Dict[str, np.ndarray]:
    """Dense per-stage score matrices, ``NaN`` where a stage did not score.

    ``NaN`` rather than ``0.0`` is the whole point.  The stages are a funnel --
    MINT scores the causal database, BEAM only MINT's shortlist, verify only
    the top-V -- so the later matrices are structurally sparse, and ``0.0`` is
    a real score a stage returns (a failed verification scores exactly that).
    Collapsing the two would destroy the only distinction these arrays carry.

    In single-session mode the result is lower-triangular by construction: a
    frame may only match its own past, so the diagonal, everything above it,
    and everything inside the exclusion window are ``NaN`` -- never looked at,
    not scored low.
    """
    out: Dict[str, np.ndarray] = {}
    for name, sims in stage_sims.items():
        if not sims:
            continue
        m = np.full((n_q, n_db), np.nan, dtype=np.float32)
        for q, per_db in sims.items():
            for d, score in per_db.items():
                m[q, d] = score
        out[name] = m
    if shifts:
        m = np.full((n_q, n_db), np.nan, dtype=np.float32)
        for q, per_db in shifts.items():
            for d, shift in per_db.items():
                m[q, d] = shift
        out["beam_shift"] = m
    return out


def write_score_matrices(path: Path, matrices: Dict[str, np.ndarray],
                         n_q: int, n_db: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
                        query_index=np.arange(n_q, dtype=np.int32),
                        db_index=np.arange(n_db, dtype=np.int32),
                        **matrices)
    return path


# ---------------------------------------------------------------------------
#  The run
# ---------------------------------------------------------------------------

def run(spec: DeploySpec) -> RunResult:
    from inlier import InLiER
    from inlier.eval.pipeline import (beam_stage, build_matcher,
                                      online_shortlist_stage, rerank_stage,
                                      shortlist_stage, verify_stage)
    from inlier.eval.refine import refine_pairs

    validate(spec)

    r = spec.resolved
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = InLiER(r.inlier)
    n_steps = 5 if spec.cross_session else 6

    # One loader for the whole run.  The exclusion may want timestamps and
    # GICP may want raw clouds; loading the sequence once for both beats
    # cross_session's separate thunks, which read it twice.
    loaded: Dict[str, Any] = {}

    def sequence(which: str):
        if which not in loaded:
            src = spec.db_source if which == "db" else spec.source
            loaded[which] = src.load()
        return loaded[which]

    # --- encode ------------------------------------------------------------
    _log(spec, f"\n[1/{n_steps}] Encoding ...")
    t0 = time.time()
    q_enc = encode_sequence(
        encoder, load=lambda: sequence("q"),
        tag=f"{spec.source.tag}_Undistorted", voxel_size=r.voxel_size,
        cache_dir=spec.cache_dir, verbose=spec.verbose,
        desc="  Encoding query session")
    if spec.cross_session:
        db_enc = encode_sequence(
            encoder, load=lambda: sequence("db"),
            tag=f"{spec.db_source.tag}_Undistorted", voxel_size=r.voxel_size,
            cache_dir=spec.cache_dir, verbose=spec.verbose,
            desc="  Encoding prior map")
    else:
        db_enc = q_enc
    encode_time = time.time() - t0

    n_q, n_db = len(q_enc.tokens), len(db_enc.tokens)

    # --- candidate scope ---------------------------------------------------
    bounds: List[int] = []
    allowed = None
    if spec.cross_session:
        cand_filter: Any = gtmod.NoFilter()
        _log(spec, f"\n[2/{n_steps}] Prior map: {n_db} frames, all searchable "
                   f"from the first query.")
    else:
        _log(spec, f"\n[2/{n_steps}] Candidate scope ...")
        timestamps, arc = _exclusion_inputs(spec, q_enc, sequence)
        bounds = [spec.exclusion.cutoff(t, timestamps, arc) for t in range(n_q)]
        if spec.search_radius > 0.0:
            # Not an oracle: these are the odometry positions the running
            # system already has, and bounding the search with them is what a
            # SLAM front-end does.  `pose_source` records that in the JSON.
            cand_filter = gtmod.RadiusFilter(
                positions=q_enc.positions, radius=spec.search_radius,
                exclusion=spec.exclusion, timestamps=timestamps,
                arc_length=arc, pose_source="odometry")
            allowed = cand_filter.allowed_mask
            _log(spec, f"  search radius {spec.search_radius:g} m over the "
                       f"odometry poses (not an oracle -- it is the same "
                       f"drifted estimate the system has)")
        else:
            cand_filter = gtmod.CausalFilter(
                exclusion=spec.exclusion, timestamps=timestamps, arc_length=arc)
        _log(spec, f"  {n_q} frames, {sum(1 for b in bounds if b > 0)} query "
                   f"the past")

    # --- retrieval ---------------------------------------------------------
    _log(spec, f"\n[3/{n_steps}] Retrieval ...")
    latency_ms = np.zeros(n_q, dtype=np.float64)
    if spec.cross_session:
        matcher = build_matcher(r, list(db_enc.tokens), verbose=spec.verbose)
        t0 = time.time()
        ranked_s1, sims_s1 = shortlist_stage(matcher, list(q_enc.tokens), n_db,
                                             verbose=spec.verbose)
        t_s1 = time.time() - t0
        effective_topk = (max(1, round(n_db * r.shortlist.topk_pct))
                          if r.shortlist.topk_pct is not None else r.shortlist.topk)
    else:
        matcher = build_matcher(r, [], verbose=False)
        ranked_s1, sims_s1, latency_ms = online_shortlist_stage(
            matcher, list(q_enc.tokens), bounds, verbose=spec.verbose,
            allowed=allowed)
        t_s1 = float(latency_ms.sum()) / 1e3
        effective_topk = (max(1, round(n_q * r.shortlist.topk_pct))
                          if r.shortlist.topk_pct is not None else r.shortlist.topk)

    sims_s2 = shifts_s2 = ranked_s2 = None
    t_s2 = 0.0
    if not r.skip_stage2:
        t0 = time.time()
        ranked_s2, sims_s2, shifts_s2 = beam_stage(
            matcher, list(q_enc.tokens), ranked_s1, effective_topk,
            verbose=spec.verbose)
        t_s2 = time.time() - t0

    sims_rr = shifts_rr = ranked_rr = None
    t_rr = 0.0
    if r.run_rerank and ranked_s2 is not None:
        topk_rr = (max(1, round(effective_topk * r.beam.topk_pct))
                   if r.beam.topk_pct is not None else effective_topk)
        t0 = time.time()
        ranked_rr, sims_rr, shifts_rr = rerank_stage(
            matcher, list(q_enc.tokens), ranked_s2, shifts_s2, topk_rr,
            verbose=spec.verbose)
        t_rr = time.time() - t0

    # --- verify ------------------------------------------------------------
    _log(spec, f"\n[4/{n_steps}] Verification (top-{r.verify_topv}) ...")
    ranked_in = ranked_rr or ranked_s2 or ranked_s1
    shifts_in = shifts_rr or shifts_s2
    t0 = time.time()
    sims_ver, ver_rank, verify_outputs = verify_stage(
        matcher, list(q_enc.tokens), list(q_enc.kp_aligned),
        list(db_enc.tokens), list(db_enc.kp_aligned),
        ranked_in, shifts_in, r.verify, top_v=r.verify_topv,
        q_kp_sensor=list(q_enc.kp_sensor), db_kp_sensor=list(db_enc.kp_sensor),
        q_T_grounds=list(q_enc.T_grounds), db_T_grounds=list(db_enc.T_grounds),
        verbose=spec.verbose)
    t_verify = time.time() - t0

    # --- accept ------------------------------------------------------------
    accepted = accept(sims_ver, verify_outputs, spec.threshold)
    best_seen = max((s for per in sims_ver.values() for s in per.values()),
                    default=0.0)
    _log(spec, f"\n[5/{n_steps}] {len(accepted)} closure(s) at or above "
               f"threshold {spec.threshold:g}"
               + ("" if accepted else
                  f" -- the highest score seen was {best_seen:.6f}, so a "
                  f"threshold above that accepts nothing"))

    # --- refine ------------------------------------------------------------
    gicp_outputs: Dict[Tuple[int, int], Any] = {}
    t_gicp = 0.0
    n_converged = 0
    if accepted and not r.skip_gicp:
        _log(spec, f"\n[{n_steps}/{n_steps}] GICP refinement ...")
        pairs = [(q, d) for q, d, _, _ in accepted]
        gicp_outputs, t_gicp, n_converged = refine_pairs(
            pairs, verify_outputs, q_enc, db_enc, r.gicp, r.voxel_size,
            load_q_clouds=lambda: sequence("q").point_clouds,
            load_db_clouds=lambda: (sequence("db") if spec.cross_session
                                    else sequence("q")).point_clouds,
            verbose=spec.verbose)

    # --- artifacts ---------------------------------------------------------
    stage = ("Verify" if sims_ver else "Rerank" if sims_rr
             else "Stage-2" if sims_s2 else "Stage-1")
    q_stride = int(getattr(spec.source, "stride", 1) or 1)
    db_stride = int(getattr(spec.db_source or spec.source, "stride", 1) or 1)
    q_stamps = _stamps(loaded.get("q"), n_q)
    db_stamps = (_stamps(loaded.get("db"), n_db) if spec.cross_session
                 else q_stamps)
    # In cross-session mode the two sessions sit in unrelated world frames
    # unless a transform was applied, so an odometry distance between them is
    # not a number -- better absent than meaningless.
    odom_comparable = not spec.cross_session

    rows = closure_rows(accepted, verify_outputs, gicp_outputs, stage,
                        q_enc.poses, db_enc.poses, q_stamps, db_stamps,
                        q_stride, db_stride, odom_comparable)

    stage_sims = {"mint": sims_s1, "beam": sims_s2, "rerank": sims_rr,
                  "verify": sims_ver}
    written: Dict[str, Path] = {}
    tag = spec.tag
    written["closures"] = artifacts.write_closures(
        output_dir / f"closures_{tag}.csv", rows)
    written["scores"] = artifacts.write_scores(
        output_dir / f"scores_{tag}.csv",
        score_rows(stage_sims, shifts_s2, spec.top_k))
    written["verify"] = artifacts.write_per_pair_verify(
        output_dir / f"per_pair_verify_{tag}.csv", verify_outputs)
    written["ranked"] = artifacts.write_ranked(
        output_dir / f"ranked_{tag}.csv", ranked_s1, sims_s1, top_k=spec.top_k)

    matrices: Dict[str, np.ndarray] = {}
    if spec.score_matrices:
        matrices = score_matrices(stage_sims, shifts_s2, n_q, n_db)
        n_bytes = sum(m.nbytes for m in matrices.values())
        if n_bytes > SCORE_MATRIX_WARN_BYTES:
            _log(spec, f"  note: the score matrices are {n_bytes / 2**20:.0f} "
                       f"MiB ({n_q}x{n_db} per stage). Pass "
                       f"--no-score-matrices to skip them.")
        written["score_matrices"] = write_score_matrices(
            output_dir / f"scores_{tag}.npz", matrices, n_q, n_db)
        figure = _write_score_figure(
            output_dir / f"scores_{tag}.png", matrices, tag, n_q, n_db)
        if figure is not None:
            written["score_figure"] = figure

    results = _build_results(
        spec, r, cand_filter, n_q, n_db, bounds, latency_ms, accepted, rows,
        stage, best_seen, q_stamps, gicp_outputs, n_converged,
        encode_time, t_s1, t_s2, t_rr, t_verify, t_gicp,
        effective_topk, odom_comparable, tag)
    written["results"] = artifacts.write_results(
        output_dir / f"run_{tag}.json", results)

    written["trajectory"] = _write_trajectory(
        spec, output_dir / f"trajectory_{tag}.png", q_enc, db_enc, accepted, tag)

    if spec.config_path is not None and Path(spec.config_path).is_file():
        target = output_dir / f"config_{tag}.yaml"
        shutil.copy2(spec.config_path, target)
        written["config"] = target

    return RunResult(protocol="run", results=results, output_dir=output_dir,
                     artifacts=written)


def _stamps(seq, n: int) -> List[float]:
    """Timestamps if the loaded sequence carried real ones, else ``[]``.

    Loaders that have none pad with zeros (``Sequence.__post_init__``), and a
    column of zeros looks like data.  All-zero is therefore treated as absent.
    """
    if seq is None:
        return []
    stamps = list(getattr(seq, "pose_timestamps", []) or [])
    if len(stamps) != n or not any(stamps):
        return []
    return stamps


def _exclusion_inputs(spec: DeploySpec, enc, sequence):
    """``(timestamps, arc_length)`` for whichever unit the exclusion uses."""
    unit = spec.exclusion.unit
    if unit == "seconds":
        stamps = list(sequence("q").pose_timestamps or [])
        if len(stamps) != len(enc.tokens):
            raise ValueError(
                f"--exclusion seconds= needs one timestamp per frame; this "
                f"sequence reports {len(stamps)} for {len(enc.tokens)} "
                f"frames. Use frames= or metres= instead.")
        axis = np.asarray(stamps, dtype=np.float64)
        if axis.size and float(axis[-1] - axis[0]) <= 0.0:
            raise ValueError(
                f"this sequence carries no usable timestamps (they never "
                f"advance: first and last are both {axis[0]:g}), so every "
                f"seconds= cutoff would collapse to 0. Use frames= or "
                f"metres= instead.")
        return axis, None
    if unit == "metres":
        return None, arc_length_of(enc.positions)
    return None, None


def _write_score_figure(path: Path, matrices, tag: str, n_q: int, n_db: int):
    """The score matrices as a page of heatmaps.

    Optional in the sense that matplotlib is: the arrays are already on disk
    in ``scores_*.npz``, so a missing plotting stack costs the picture and not
    the data.
    """
    try:
        from inlier.viz import write_score_matrix_figure
    except ImportError:                     # pragma: no cover - no [eval] extra
        return None
    return write_score_matrix_figure(
        path, matrices, f"{tag}   {n_q} queries x {n_db} database frames")


def _write_trajectory(spec: DeploySpec, path: Path, q_enc, db_enc, accepted, tag):
    """The closures drawn on the trajectory, in one neutral colour.

    There is no true/false distinction to make here, and green would assert
    one.  The title names the frame as odometry: these positions are drifted,
    and this is the artifact people screenshot.
    """
    from inlier.viz.trajectory import CLOSURE_COLOR

    edges = [(q, d) for q, d, _, _ in accepted]
    title = (f"{tag}: {len(edges)} loop closure(s) at threshold "
             f"{spec.threshold:g}\nodometry frame -- no ground truth, so no "
             f"closure here is labelled correct")
    labels = ("Closure", "")
    colors = (CLOSURE_COLOR, CLOSURE_COLOR)

    if spec.cross_session:
        from inlier.viz import write_trajectory_plot

        return write_trajectory_plot(
            path, db_enc.positions, q_enc.positions, edges, [], title,
            edge_labels=labels, edge_colors=colors)

    from inlier.viz import write_time_trajectory_plot

    return write_time_trajectory_plot(
        path, q_enc.positions, edges, [], title,
        edge_labels=labels, edge_colors=colors)


def _build_results(spec, r, cand_filter, n_q, n_db, bounds, latency_ms,
                   accepted, rows, stage, best_seen, q_stamps, gicp_outputs,
                   n_converged, encode_time, t_s1, t_s2, t_rr, t_verify,
                   t_gicp, effective_topk, odom_comparable, tag) -> Dict[str, Any]:
    """The run record: provenance, cost, and deliberately no metrics.

    No ``stage1``/``confusion``/``loop_closure`` blocks exist, because every
    one of them needs labels.  An absent block is better than a zeroed one.
    """
    sessions: Dict[str, Any] = ({"db": spec.db_source.describe(),
                                 "query": spec.source.describe()}
                                if spec.cross_session
                                else {"session": spec.source.describe()})

    results: Dict[str, Any] = {
        **artifacts.provenance("run"),
        "mode": "cross_session" if spec.cross_session else "single_session",
        "config_mode": r.mode,
        # Explicitly null, not absent: a consumer testing for it gets an
        # unambiguous answer, and its presence rules out "an older schema".
        "ground_truth": None,
        "candidate_filter": cand_filter.describe(),
        **sessions,
        "config": {
            "threshold": spec.threshold,
            "exclusion": (spec.exclusion.describe() if spec.exclusion else None),
            "search_radius": spec.search_radius,
            "effective_stage1_topk": effective_topk,
            "voxel_size": r.voxel_size,
            "verify_topv": r.verify_topv,
            "skip_stage2": r.skip_stage2,
            "run_rerank": r.run_rerank,
            "skip_gicp": r.skip_gicp,
        },
        "dataset": {
            "n_query_frames": n_q,
            "n_db_frames": n_db,
            "n_queried": (sum(1 for b in bounds if b > 0) if bounds else n_q),
            "timestamps_available": bool(q_stamps),
        },
        "closures": {
            "n_closures": len(accepted),
            "n_queries_with_a_closure": len({q for q, _, _, _ in accepted}),
            "n_refined": sum(1 for row in rows if row["refined"]),
            "n_gicp_converged": n_converged,
            "stage": stage,
            "score_definition": "keypoint_inlier_ratio",
            "selection": ("every candidate with a successful verification "
                          "scoring >= threshold, up to verify.topv"),
            "highest_score_seen": round(float(best_seen), 6),
        },
        "poses": {
            "frame": "sensor",
            "convention": ("T maps query points into the DB frame: "
                           "p_db = T @ p_query"),
            "refined_pose_in_closures_csv": ("GICP where refined=1, verify "
                                             "otherwise; the unrefined pose "
                                             "is in per_pair_verify_*.csv"),
            # No covariance or information matrix is emitted: InLiER does
            # not estimate one, and a constant invented here would be a
            # guess wearing a measurement's clothes.  The per-closure quality
            # columns are the raw material for whatever weighting a back-end
            # chooses.
            "edge_weight": None,
        },
        "odometry": {
            "pose_source": "odometry",
            "used_for": ["submap accumulation", "search radius",
                         "diagnostic columns"],
            "never_used_for": "accepting or rejecting a closure",
            "diagnostics_available": odom_comparable,
            **({} if odom_comparable else {
                "diagnostics_unavailable_because": (
                    "the two sessions are in unrelated world frames")}),
        },
        "scores": {
            "not_scored": ("NaN in the .npz, blank in the .csv -- a stage "
                           "never scored that pair; not the same as 0.0, "
                           "which a stage can genuinely return"),
            "decisions_applied": None,
            "top_k": spec.top_k,
        },
        "timing": {
            "encode_s": round(encode_time, 3),
            "stage1_s": round(t_s1, 3),
            "stage2_s": round(t_s2, 3),
            "rerank_s": round(t_rr, 3),
            "verify_s": round(t_verify, 3),
            "gicp_s": round(t_gicp, 3),
        },
        "artifacts": {"tag": tag, "cache": f"{spec.source.tag}_Undistorted"},
    }
    if bounds:
        results["latency"] = {
            **artifacts.latency_block(latency_ms, bounds),
            "covers": ("retrieval and database insertion; verification and "
                       "GICP run batched after the stream"),
        }
    return results
