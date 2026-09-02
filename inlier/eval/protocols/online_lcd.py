"""Online loop closure detection: one session, a database that grows.

The SLAM protocol.  There is no second sequence and no overlap matrix: a
single trajectory streams past, every frame queries the frames that came
before it, and the recent past is excluded because matching your own last few
metres is trivially correct and tells you nothing.

Three things separate it from :mod:`~inlier.eval.protocols.cross_session`:

* **Causal.**  Frame ``t`` may only match frames at index ``< cutoff(t)``.  The
  same :class:`~inlier.eval.gt.Exclusion` computes that cutoff for the ground
  truth and for the matcher's database bound, so the two cannot drift apart --
  which is the failure that makes an online benchmark quietly optimistic.
* **Incremental.**  The database is built one frame at a time, so the reported
  per-frame latency is the cost a real system pays, not a query against a
  database that was already complete before the run started.
* **Scored as loop closure.**  F1max and max-recall-at-100%-precision, because
  in a pose graph a single false positive can corrupt the whole map.  Recall@N
  is still reported, but R@1 is the number that matters.

Ground truth is pose distance alone.  There is no overlap matrix for a
single session -- building one would mean an N x N comparison of a sequence
against itself -- so ``max_pose_dist`` is the whole positive rule and it is
looser than the cross-session default for that reason.

The search radius
-----------------
``search_radius`` narrows the database to frames within a given distance of
the query, which is what a SLAM front-end backed by a local map actually does.
It is off by default (``0`` = the whole causal past) because **it is a
geometric oracle**: the radius is measured against the query's ground-truth
pose, which a deployed system does not have, and it deletes exactly the
far-away distractors that make retrieval hard.  Every metric goes up.  A run
that uses it says so in ``candidate_filter.uses_pose_oracle``, so a number
produced with a radius can never be mistaken for one produced without.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from inlier.eval import artifacts, gt as gtmod, metrics, pose, thresholds
from inlier.eval.datasets.base import arc_length
from inlier.eval.encode import EncodedSequence, encode_sequence
from inlier.eval.protocols.base import RunResult
from inlier.eval.retrieval import RankedResults

#: Same grids as cross-session, so the two protocols' PR-AUCs are comparable.
N_VALUES = [1, 5, 10, 20, 50, 100]
K_PCTS = [1.0, 5.0, 10.0]
THR_STAGE1 = np.arange(0.0, 1.001, 0.001)
THR_LATER = np.arange(0.0, 1.001, 0.005)


@dataclass
class OnlineLCDSpec:
    """Everything one online loop-closure run needs."""

    resolved: Any                       # inlier.config.ResolvedConfig
    source: Any                         # SequenceSource -- one session
    exclusion: gtmod.Exclusion
    output_dir: Path
    max_pose_dist: float = 10.0
    #: Restrict candidates to database frames within this many metres of the
    #: query's *true* pose.  0 searches the whole causal past.  See the note in
    #: the module docstring: any value > 0 is a geometric oracle.
    search_radius: float = 0.0
    cache_dir: Optional[Path] = None
    threshold_policy: str = "max_precision"
    threshold_value: Optional[float] = None
    config_path: Optional[Path] = None
    verbose: bool = True
    tag: str = ""


def _log(spec: OnlineLCDSpec, message: str = "") -> None:
    if spec.verbose:
        print(message)


def _exclusion_inputs(spec: OnlineLCDSpec, enc: EncodedSequence
                      ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """``(timestamps, arc_length)`` for the unit this run's window is in.

    Only the unit actually in use is materialised.  ``metres`` comes free from
    the cached poses; ``seconds`` does not -- the descriptor cache stores no
    timestamps -- so that unit alone pays for a sequence load, and only when
    asked for.
    """
    unit = spec.exclusion.unit
    if unit == "seconds":
        _log(spec, "  exclusion is in seconds; loading the sequence for timestamps")
        ts = np.asarray(spec.source.load().pose_timestamps, dtype=np.float64)
        if ts.size != len(enc.tokens):
            raise ValueError(
                f"{ts.size} timestamps for {len(enc.tokens)} encoded scans")
        if not np.all(np.diff(ts) >= 0):
            raise ValueError(
                "pose timestamps are not non-decreasing; Exclusion(seconds=) "
                "resolves its cutoff by binary search and needs sorted input. "
                "Use --exclusion frames=N or metres=M instead.")
        return ts, None
    if unit == "metres":
        return None, arc_length(enc.positions)
    return None, None


def run(spec: OnlineLCDSpec) -> RunResult:
    from inlier import InLiER
    from inlier.eval.pipeline import (
        beam_stage, build_matcher, online_shortlist_stage, rerank_stage, verify_stage,
    )

    r = spec.resolved
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. encode ---------------------------------------------------------
    _log(spec, "\n[1/6] Encoding the session ...")
    encoder = InLiER(r.inlier)
    t0 = time.time()
    enc = encode_sequence(
        encoder, load=spec.source.load, tag=f"{spec.source.tag}_Undistorted",
        voxel_size=r.voxel_size, cache_dir=spec.cache_dir,
        verbose=spec.verbose, desc="  Encoding session",
    )
    encode_time = time.time() - t0
    n = len(enc.tokens)
    positions = enc.positions

    # --- 2. exclusion + causal ground truth --------------------------------
    _log(spec, "\n[2/6] Causal ground truth ...")
    timestamps, arc = _exclusion_inputs(spec, enc)
    bounds = [spec.exclusion.cutoff(t, timestamps, arc) for t in range(n)]

    # A radius smaller than the ground-truth distance would put real revisits
    # outside the searchable set: those queries could never be answered, and
    # the recall loss would look like a retrieval failure rather than the
    # misconfiguration it is.
    if 0.0 < spec.search_radius < spec.max_pose_dist:
        raise ValueError(
            f"search_radius={spec.search_radius} m is smaller than "
            f"max_pose_dist={spec.max_pose_dist} m, so some ground-truth "
            f"revisits fall outside the searchable database and can never be "
            f"found. Raise --search-radius to at least --max-pose-dist, or "
            f"lower --max-pose-dist.")

    if spec.search_radius > 0.0:
        cand_filter = gtmod.RadiusFilter(
            positions=positions, radius=spec.search_radius,
            exclusion=spec.exclusion, timestamps=timestamps, arc_length=arc)
        allowed = cand_filter.allowed_mask
        _log(spec, f"  search radius {spec.search_radius} m -- NOTE: this is a "
                   f"true-pose oracle and inflates every metric")
    else:
        cand_filter = gtmod.CausalFilter(
            exclusion=spec.exclusion, timestamps=timestamps, arc_length=arc)
        allowed = None

    policy = gtmod.Causal(
        positions=positions, exclusion=spec.exclusion,
        max_pose_dist=spec.max_pose_dist,
        timestamps=timestamps, arc_length=arc,
    )
    ground_truth = gtmod.build(policy, n)
    n_with_gt = sum(1 for v in ground_truth.values() if v.size > 0)
    if n_with_gt == 0:
        raise ValueError(
            f"no frame has a revisit at max_pose_dist={spec.max_pose_dist} m "
            f"outside the exclusion window "
            f"({spec.exclusion.unit}={getattr(spec.exclusion, spec.exclusion.unit)}). "
            f"Either this session has no loop, or the window is longer than "
            f"the loop -- widen --max-pose-dist or shorten --exclusion."
        )
    n_queried = sum(1 for b in bounds if b > 0)
    _log(spec, f"  {n}  frames, {n_queried} query the past, "
               f"{n_with_gt} have a revisit to find")

    # --- 3. streaming retrieval -------------------------------------------
    _log(spec, "\n[3/6] Streaming the session through the matcher ...")
    matcher = build_matcher(r, [], verbose=False)   # starts empty and grows
    ranked_s1, sims_s1, latency_ms = online_shortlist_stage(
        matcher, enc.tokens, bounds, verbose=spec.verbose, allowed=allowed)
    t_s1 = float(latency_ms.sum()) / 1e3

    # The database is a different size at every frame, so a percentage topk has
    # no single value.  Resolving it against the full session is what the same
    # sequence would get under cross-session, which keeps the two comparable.
    if r.shortlist.topk_pct is not None:
        effective_topk = max(1, round(n * r.shortlist.topk_pct))
    else:
        effective_topk = r.shortlist.topk

    ranked_s2 = sims_s2 = shifts_s2 = None
    t_s2 = 0.0
    if not r.skip_stage2:
        t0 = time.time()
        ranked_s2, sims_s2, shifts_s2 = beam_stage(
            matcher, enc.tokens, ranked_s1, effective_topk, verbose=spec.verbose)
        t_s2 = time.time() - t0

    ranked_comb = sims_comb = shifts_comb = None
    comb_input_lists = None
    t_comb = 0.0
    if r.run_rerank and ranked_s2 is not None:
        comb_input_lists, comb_input_shifts = ranked_s2, shifts_s2
        comb_input_topk = (max(1, round(effective_topk * r.beam.topk_pct))
                           if r.beam.topk_pct is not None else effective_topk)
        t0 = time.time()
        ranked_comb, sims_comb, shifts_comb = rerank_stage(
            matcher, enc.tokens, comb_input_lists, comb_input_shifts,
            comb_input_topk, verbose=spec.verbose)
        t_comb = time.time() - t0

    # --- 4. metrics per stage ---------------------------------------------
    _log(spec, "\n[4/6] Computing metrics ...")
    recalls_s1 = metrics.recall_at_n(ranked_s1, ground_truth, N_VALUES)
    kpct_s1 = metrics.recall_at_kpct(ranked_s1, ground_truth, n, K_PCTS)
    _, _, auc_s1 = metrics.pr_curve(sims_s1, ground_truth, THR_STAGE1)

    recalls_s2 = kpct_s2 = auc_s2 = None
    if ranked_s2 is not None:
        recalls_s2 = metrics.recall_at_n(ranked_s2, ground_truth, N_VALUES)
        kpct_s2 = metrics.recall_at_kpct(ranked_s2, ground_truth, n, K_PCTS,
                                         fallback_ranked_lists=ranked_s1)
        _, _, auc_s2 = metrics.pr_curve(sims_s2, ground_truth, THR_LATER)

    recalls_comb = kpct_comb = auc_comb = None
    if ranked_comb is not None:
        recalls_comb = metrics.recall_at_n(ranked_comb, ground_truth, N_VALUES)
        kpct_comb = metrics.recall_at_kpct(ranked_comb, ground_truth, n, K_PCTS,
                                           fallback_ranked_lists=comb_input_lists)
        _, _, auc_comb = metrics.pr_curve(sims_comb, ground_truth, THR_LATER)

    # --- 5. verification ---------------------------------------------------
    verify_ranked = ranked_comb if ranked_comb is not None else (
        ranked_s2 if ranked_s2 is not None else ranked_s1)
    verify_shifts = shifts_comb if shifts_comb is not None else shifts_s2

    sims_ver = ver_rank_order = None
    verify_outputs: Dict[Tuple[int, int], Any] = {}
    recalls_ver = kpct_ver = auc_ver = None
    t_verify = 0.0

    if not r.skip_verify:
        _log(spec, "\n[5/6] Geometric verification ...")
        t0 = time.time()
        # One session: the query and the database are the same encoded scans.
        sims_ver, ver_rank_order, verify_outputs = verify_stage(
            matcher, enc.tokens, enc.kp_aligned, enc.tokens, enc.kp_aligned,
            verify_ranked, verify_shifts, r.verify, top_v=r.verify_topv,
            q_kp_sensor=enc.kp_sensor, db_kp_sensor=enc.kp_sensor,
            q_T_grounds=enc.T_grounds, db_T_grounds=enc.T_grounds,
            verbose=spec.verbose,
        )
        t_verify = time.time() - t0

        _, _, auc_ver = metrics.pr_curve(sims_ver, ground_truth, THR_LATER,
                                         rank_order=ver_rank_order)
        ranked_ver: Dict[int, List[int]] = {}
        for j, q_sims in sims_ver.items():
            scored = sorted(q_sims, key=lambda d: q_sims[d], reverse=True)
            seen = set(scored)
            ranked_ver[j] = scored + [d for d in verify_ranked.get(j, []) if d not in seen]
        recalls_ver = metrics.recall_at_n(ranked_ver, ground_truth, N_VALUES)
        kpct_ver = metrics.recall_at_kpct(ranked_ver, ground_truth, n, K_PCTS,
                                          fallback_ranked_lists=verify_ranked)
    else:
        _log(spec, "\n[5/6] Verification skipped (verify.skip)")

    # --- 6. operating point, loop-closure metrics, artifacts ---------------
    _log(spec, "\n[6/6] Selecting the operating threshold ...")
    if sims_ver is not None:
        conf_sims, conf_stage, conf_rank = sims_ver, "Verify", ver_rank_order
    elif sims_comb is not None:
        conf_sims, conf_stage, conf_rank = sims_comb, "Rerank", None
    elif sims_s2 is not None:
        conf_sims, conf_stage, conf_rank = sims_s2, "Stage-2", None
    else:
        conf_sims, conf_stage, conf_rank = sims_s1, "Stage-1", None

    ranked_results = RankedResults.from_similarity_map(
        conf_sims, ground_truth, n, n, conf_rank, "confusion", stage=conf_stage)
    chosen = thresholds.select(ranked_results, spec.threshold_policy, spec.threshold_value)
    _log(spec, f"  policy={chosen.policy}  threshold={chosen.threshold:.4f}  "
               f"precision={chosen.precision:.4f}  recall={chosen.recall:.4f}")

    tp_edges, fp_edges, tp, fp, fn, tn = metrics.confusion(
        conf_sims, ground_truth, chosen.threshold, n, rank_order=conf_rank)

    prec_curve, rec_curve, _ = metrics.pr_curve(
        conf_sims, ground_truth, THR_LATER, rank_order=conf_rank)
    f1_max, f1_max_thr = metrics.f1_from_curve(prec_curve, rec_curve, THR_LATER)
    max_recall_100p = metrics.max_recall_at_full_precision(prec_curve, rec_curve)
    _log(spec, f"  F1max={f1_max:.4f}  max recall @ 100% precision="
               f"{max_recall_100p:.4f}")

    tp_dist = pose.match_distances(tp_edges, positions, positions) or {
        "mean_m": 0.0, "std_m": 0.0, "min_m": 0.0, "max_m": 0.0, "median_m": 0.0}
    tp_pose_verify = pose.errors_from_verify(tp_edges, verify_outputs,
                                             enc.poses, enc.poses)

    results = _build_results(
        spec, r, enc, n, n_queried, n_with_gt, effective_topk, policy,
        cand_filter, encode_time, t_s1, t_s2, t_comb, t_verify, latency_ms, bounds,
        recalls_s1, kpct_s1, auc_s1, recalls_s2, kpct_s2, auc_s2,
        recalls_comb, kpct_comb, auc_comb, recalls_ver, kpct_ver, auc_ver,
        conf_stage, chosen, tp, fp, fn, tn, tp_dist, tp_pose_verify,
        f1_max, f1_max_thr, max_recall_100p,
    )

    tag = spec.tag or "run"
    written: Dict[str, Path] = {}
    written["results"] = artifacts.write_results(output_dir / f"results_{tag}.json", results)
    written["candidates"] = artifacts.write_candidates(
        output_dir / f"candidates_{tag}.csv",
        artifacts.candidate_rows(conf_sims, conf_rank, ground_truth, n,
                                 positions, positions, chosen.threshold))
    written["ranked"] = artifacts.write_ranked(
        output_dir / f"ranked_{tag}.csv", verify_ranked,
        sims_ver if sims_ver is not None else conf_sims)
    if verify_outputs:
        written["verify"] = artifacts.write_per_pair_verify(
            output_dir / f"per_pair_verify_{tag}.csv", verify_outputs)
    if spec.config_path is not None and Path(spec.config_path).exists():
        dest = output_dir / Path(spec.config_path).name
        shutil.copy2(spec.config_path, dest)
        written["config"] = dest

    plot = _write_trajectory_plot(
        spec, output_dir / f"trajectory_{tag}.png", positions,
        tp_edges, fp_edges, conf_stage, chosen.threshold, tp, fp, fn, tn)
    if plot is not None:
        written["trajectory"] = plot

    _log(spec, f"\n  Results -> {written['results']}")
    return RunResult("online_lcd", results, output_dir, written)


# ---------------------------------------------------------------------------

def _latency_block(latency_ms: np.ndarray, bounds: List[int]) -> Dict[str, Any]:
    """Per-frame cost, over the frames that actually ran a query.

    Frames inside the opening exclusion window do nothing but an insertion, so
    averaging them in would report a latency no query ever experienced.
    """
    queried = latency_ms[[t for t, b in enumerate(bounds) if b > 0]]
    if queried.size == 0:
        return {"n_frames": 0}
    return {
        "n_frames": int(queried.size),
        "mean_ms": round(float(queried.mean()), 3),
        "median_ms": round(float(np.median(queried)), 3),
        "p95_ms": round(float(np.percentile(queried, 95)), 3),
        "max_ms": round(float(queried.max()), 3),
    }


def _build_results(spec, r, enc, n, n_queried, n_with_gt, effective_topk, gt_policy,
                   cand_filter, encode_time, t_s1, t_s2, t_comb, t_verify, latency_ms, bounds,
                   recalls_s1, kpct_s1, auc_s1, recalls_s2, kpct_s2, auc_s2,
                   recalls_comb, kpct_comb, auc_comb, recalls_ver, kpct_ver, auc_ver,
                   conf_stage, chosen, tp, fp, fn, tn, tp_dist, tp_pose_verify,
                   f1_max, f1_max_thr, max_recall_100p) -> Dict[str, Any]:
    """Assemble the results payload, in the shape every protocol shares."""
    precision, recall, f1 = metrics.prf1(tp, fp, fn)

    return {
        **artifacts.provenance(
            "online_lcd",
            threshold_policy=chosen.policy,
            ground_truth=gt_policy.describe(),
            candidate_filter=cand_filter.describe(),
            session=spec.source.describe(),
            config_mode=r.mode,
            artifacts={"tag": spec.tag,
                       "cache": f"{spec.source.tag}_Undistorted"},
        ),
        "config": {
            "sequence": spec.source.describe().get(
                "sequence", spec.source.describe().get("path", "")),
            "sensor": spec.source.describe().get("sensor", ""),
            "exclusion": spec.exclusion.describe(),
            "max_pose_dist": spec.max_pose_dist,
            "search_radius": spec.search_radius,
            "voxel_size": r.voxel_size,
            "inlier": {
                "cell_size": r.inlier.cell_size, "N_h": r.inlier.N_h,
                "z_min": r.inlier.z_min, "z_max": r.inlier.z_max,
                "N_r": r.inlier.N_r, "N_s": r.inlier.N_s, "N_a": r.inlier.N_a,
                "point_mode": r.inlier.point_mode,
            },
            "shortlist": {
                "topk": r.shortlist.topk, "topk_pct": r.shortlist.topk_pct,
                "effective_topk": effective_topk,
            },
        },
        "dataset": {
            "n_frames": n,
            "n_queried": n_queried,
            "n_with_ground_truth": n_with_gt,
        },
        "stage1": artifacts.stage_block(recalls_s1, kpct_s1, auc_s1),
        "stage2": artifacts.stage_block(recalls_s2, kpct_s2, auc_s2),
        "combined": artifacts.stage_block(recalls_comb, kpct_comb, auc_comb),
        "verify": artifacts.stage_block(recalls_ver, kpct_ver, auc_ver),
        # The loop-closure headline: a pose graph survives a missed closure and
        # not a false one, so the operating point is judged on precision.
        "loop_closure": {
            "f1_max": round(f1_max, 6),
            "f1_max_threshold": round(f1_max_thr, 6),
            "max_recall_at_full_precision": round(max_recall_100p, 6),
            "recall_at_1": (round(recalls_ver[1], 6) if recalls_ver
                            else round(recalls_s1[1], 6)),
        },
        "confusion": {
            "stage": conf_stage,
            "threshold": round(chosen.threshold, 6),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        },
        "latency": _latency_block(latency_ms, bounds),
        "timing": {
            "encode_s": round(encode_time, 3),
            "stage1_s": round(t_s1, 3),
            "stage2_s": round(t_s2, 3),
            "rerank_s": round(t_comb, 3),
            "verify_s": round(t_verify, 3),
        },
        "tp_match_distance": tp_dist,
        "tp_pose_error_verify": tp_pose_verify,
    }


def _write_trajectory_plot(spec, path, positions, tp_edges, fp_edges,
                           stage, threshold, tp, fp, fn, tn):
    """Save the trajectory figure, or explain why it was skipped."""
    try:
        from inlier.viz import write_time_trajectory_plot
    except ImportError as exc:      # matplotlib lives in the [eval] extra
        _log(spec, f"\n  Trajectory plot skipped ({exc}); "
                   'install it with pip install "inlier[eval]"')
        return None

    described = spec.source.describe()
    name = Path(str(described.get("sequence") or described.get("path", ""))).name
    title = (f"InLiER online-LCD {stage}  {name}\n"
             f"thr={threshold:.3f}  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    # One session: z is the frame index, so a closure edge spans the frames
    # between leaving a place and coming back to it.
    written = write_time_trajectory_plot(path, positions,
                                         tp_edges, fp_edges, title)
    _log(spec, f"  Trajectory plot -> {written}")
    return written
