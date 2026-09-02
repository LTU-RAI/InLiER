"""Cross-session place recognition: a full database, a full query sequence.

The protocol behind the published results.  The database is one complete
sequence, the query is another, the whole database is visible to every query,
and correctness comes from a precomputed scan-overlap matrix.  It is an offline
protocol: nothing about it is causal, and a query may match a database scan
recorded later in wall-clock time.

Ported from ``run_evaluation`` in ``evaluate_inlier_helipr.py``, which existed
in two ~90%-identical copies (the HeLiPR and generic drivers).  The only thing
that differs between those is where the scans come from, so the loader is now a
parameter and the protocol is written once.

Numbers must not move: ``tests/test_golden_run.py`` re-runs this against the
checked-in results of the Roundabout01(Ouster) <- Roundabout03(Aeva) pair and
requires exact agreement on every recall, PR-AUC, and confusion count.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from inlier.eval import artifacts, gt as gtmod, metrics, overlap as overlapmod, pose, thresholds
from inlier.eval.encode import EncodedSequence, encode_sequence
from inlier.eval.protocols.base import RunResult
from inlier.eval.retrieval import RankedResults

#: Recall@N cut-offs and the database-relative percentages, unchanged.
N_VALUES = [1, 5, 10, 20, 50, 100]
K_PCTS = [1.0, 5.0, 10.0]
#: Stage-1 sweeps at 0.001; the later stages at 0.005.  Kept per stage because
#: PR-AUC depends on the grid, so a finer sweep here would change the number.
THR_STAGE1 = np.arange(0.0, 1.001, 0.001)
THR_LATER = np.arange(0.0, 1.001, 0.005)


@dataclass
class CrossSessionSpec:
    """Everything one cross-session run needs."""

    resolved: Any                       # inlier.config.ResolvedConfig
    db_source: Any                      # SequenceSource
    q_source: Any
    overlap_path: Path
    output_dir: Path
    overlap_threshold: float = 0.3
    max_pose_dist: float = 25.0
    cache_dir: Optional[Path] = None
    threshold_policy: str = "max_precision"
    threshold_value: Optional[float] = None
    config_path: Optional[Path] = None
    db_transform: Optional[np.ndarray] = None
    strict_overlap_check: bool = True
    verbose: bool = True
    tag: str = ""


def _log(spec: CrossSessionSpec, message: str = "") -> None:
    if spec.verbose:
        print(message)


def _encode(spec: CrossSessionSpec, encoder, source, cache_tag: str, what: str
            ) -> EncodedSequence:
    return encode_sequence(
        encoder,
        load=source.load,
        tag=cache_tag,
        voxel_size=spec.resolved.voxel_size,
        cache_dir=spec.cache_dir,
        verbose=spec.verbose,
        desc=f"  Encoding {what}",
    )


def run(spec: CrossSessionSpec) -> RunResult:
    from inlier import InLiER
    from inlier.eval.pipeline import (
        beam_stage, build_matcher, rerank_stage, shortlist_stage, verify_stage,
    )

    r = spec.resolved
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. encode ---------------------------------------------------------
    _log(spec, "\n[1/6] Encoding sequences ...")
    encoder = InLiER(r.inlier)
    t0 = time.time()
    db = _encode(spec, encoder, spec.db_source, spec.db_source.tag + "_Undistorted", "database")
    q = _encode(spec, encoder, spec.q_source, spec.q_source.tag + "_Undistorted", "query")
    encode_time = time.time() - t0

    db_positions, db_poses = db.positions, db.poses
    q_positions, q_poses = q.positions, q.poses
    if spec.db_transform is not None:
        # DB and query mapped independently: put the DB poses in the query's
        # world frame before any distance or overlap comparison.
        T = np.asarray(spec.db_transform, dtype=np.float64)
        db_poses = np.einsum("ij,njk->nik", T, db_poses)
        db_positions = db_poses[:, :3, 3]

    n_db, n_q = len(db.tokens), len(q.tokens)

    # --- 2. ground truth ---------------------------------------------------
    _log(spec, "\n[2/6] Loading overlap ground truth ...")
    matrix = overlapmod.load(spec.overlap_path)
    overlapmod.check(
        spec.overlap_path, matrix,
        overlapmod.OverlapProvenance(
            n_db=getattr(spec.db_source, "n_scans", 1),
            n_q=getattr(spec.q_source, "n_scans", 1),
            stride_db=getattr(spec.db_source, "stride", 1),
            stride_q=getattr(spec.q_source, "stride", 1),
            shape=(n_db, n_q),
        ),
        strict=spec.strict_overlap_check,
    )

    policy = gtmod.OverlapAndDistance(
        overlap=matrix, db_positions=db_positions, q_positions=q_positions,
        overlap_threshold=spec.overlap_threshold, max_pose_dist=spec.max_pose_dist,
    )
    ground_truth = gtmod.build(policy, n_q)
    n_with_gt = sum(1 for v in ground_truth.values() if v.size > 0)
    if n_with_gt == 0:
        raise ValueError(
            f"no query has a ground-truth positive at overlap_threshold="
            f"{spec.overlap_threshold} and max_pose_dist={spec.max_pose_dist}. "
            f"Loosen the thresholds, or check the overlap matrix orientation "
            f"(rows are DB, columns are query)."
        )
    _log(spec, f"  {n_with_gt}/{n_q} queries have at least one positive")

    # --- 3. retrieval ------------------------------------------------------
    _log(spec, "\n[3/6] Building the matcher ...")
    matcher = build_matcher(r, db.tokens, verbose=spec.verbose)

    if r.shortlist.topk_pct is not None:
        effective_topk = max(1, round(n_db * r.shortlist.topk_pct))
    else:
        effective_topk = r.shortlist.topk

    t0 = time.time()
    ranked_s1, sims_s1 = shortlist_stage(matcher, q.tokens, n_db, verbose=spec.verbose)
    t_s1 = time.time() - t0

    ranked_s2 = sims_s2 = shifts_s2 = None
    t_s2 = 0.0
    if not r.skip_stage2:
        t0 = time.time()
        ranked_s2, sims_s2, shifts_s2 = beam_stage(
            matcher, q.tokens, ranked_s1, effective_topk, verbose=spec.verbose)
        t_s2 = time.time() - t0

    ranked_comb = sims_comb = shifts_comb = None
    t_comb = 0.0
    comb_input_lists = None
    if r.run_rerank and ranked_s2 is not None:
        comb_input_lists, comb_input_shifts = ranked_s2, shifts_s2
        comb_input_topk = (max(1, round(effective_topk * r.beam.topk_pct))
                           if r.beam.topk_pct is not None else effective_topk)
        t0 = time.time()
        ranked_comb, sims_comb, shifts_comb = rerank_stage(
            matcher, q.tokens, comb_input_lists, comb_input_shifts,
            comb_input_topk, verbose=spec.verbose)
        t_comb = time.time() - t0

    # --- 4. metrics per stage ---------------------------------------------
    _log(spec, "\n[4/6] Computing metrics ...")
    recalls_s1 = metrics.recall_at_n(ranked_s1, ground_truth, N_VALUES)
    kpct_s1 = metrics.recall_at_kpct(ranked_s1, ground_truth, n_db, K_PCTS)
    _, _, auc_s1 = metrics.pr_curve(sims_s1, ground_truth, THR_STAGE1)

    recalls_s2 = kpct_s2 = auc_s2 = None
    if ranked_s2 is not None:
        recalls_s2 = metrics.recall_at_n(ranked_s2, ground_truth, N_VALUES)
        # A later stage returns fewer candidates than K% of the database asks
        # for, so the previous stage backfills rather than the query being
        # scored against a truncated list.
        kpct_s2 = metrics.recall_at_kpct(ranked_s2, ground_truth, n_db, K_PCTS,
                                         fallback_ranked_lists=ranked_s1)
        _, _, auc_s2 = metrics.pr_curve(sims_s2, ground_truth, THR_LATER)

    recalls_comb = kpct_comb = auc_comb = None
    if ranked_comb is not None:
        recalls_comb = metrics.recall_at_n(ranked_comb, ground_truth, N_VALUES)
        kpct_comb = metrics.recall_at_kpct(ranked_comb, ground_truth, n_db, K_PCTS,
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
        sims_ver, ver_rank_order, verify_outputs = verify_stage(
            matcher, q.tokens, q.kp_aligned, db.tokens, db.kp_aligned,
            verify_ranked, verify_shifts, r.verify, top_v=r.verify_topv,
            q_kp_sensor=q.kp_sensor, db_kp_sensor=db.kp_sensor,
            q_T_grounds=q.T_grounds, db_T_grounds=db.T_grounds,
            verbose=spec.verbose,
        )
        t_verify = time.time() - t0

        _, _, auc_ver = metrics.pr_curve(sims_ver, ground_truth, THR_LATER,
                                         rank_order=ver_rank_order)
        # Recall@N ranks verified candidates by score, then falls through to
        # the retrieval order for the candidates verification never looked at.
        ranked_ver: Dict[int, List[int]] = {}
        for j, q_sims in sims_ver.items():
            scored = sorted(q_sims, key=lambda d: q_sims[d], reverse=True)
            seen = set(scored)
            ranked_ver[j] = scored + [d for d in verify_ranked.get(j, []) if d not in seen]
        recalls_ver = metrics.recall_at_n(ranked_ver, ground_truth, N_VALUES)
        kpct_ver = metrics.recall_at_kpct(ranked_ver, ground_truth, n_db, K_PCTS,
                                          fallback_ranked_lists=verify_ranked)
    else:
        _log(spec, "\n[5/6] Verification skipped (verify.skip)")

    # --- 6. operating threshold, confusion, artifacts ----------------------
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
        conf_sims, ground_truth, n_q, n_db, conf_rank, "confusion", stage=conf_stage)
    chosen = thresholds.select(ranked_results, spec.threshold_policy, spec.threshold_value)
    _log(spec, f"  policy={chosen.policy}  threshold={chosen.threshold:.4f}  "
               f"precision={chosen.precision:.4f}  recall={chosen.recall:.4f}")

    tp_edges, fp_edges, tp, fp, fn, tn = metrics.confusion(
        conf_sims, ground_truth, chosen.threshold, n_q, rank_order=conf_rank)

    # f1_max is reported whatever policy chose the operating point, so runs
    # selected on precision stay comparable with baselines selected on F1.
    prec_curve, rec_curve, _ = metrics.pr_curve(
        conf_sims, ground_truth, THR_LATER, rank_order=conf_rank)
    f1_max, f1_max_thr = metrics.f1_from_curve(prec_curve, rec_curve, THR_LATER)

    tp_dist = pose.match_distances(tp_edges, q_positions, db_positions) or {
        "mean_m": 0.0, "std_m": 0.0, "min_m": 0.0, "max_m": 0.0, "median_m": 0.0}
    tp_pose_verify = pose.errors_from_verify(tp_edges, verify_outputs, q_poses, db_poses)

    # --- GICP refinement on the true positives -----------------------------
    gicp_outputs: Dict[Tuple[int, int], Any] = {}
    tp_pose_gicp = None
    t_gicp = 0.0
    n_converged = 0
    if not r.skip_gicp and not r.skip_verify and tp_edges:
        from inlier.eval.refine import refine_pairs

        _log(spec, "\n  GICP refinement on TP pairs ...")
        gicp_outputs, t_gicp, n_converged = refine_pairs(
            tp_edges, verify_outputs, q, db, r.gicp, r.voxel_size,
            load_q_clouds=lambda: spec.q_source.load().point_clouds,
            load_db_clouds=lambda: spec.db_source.load().point_clouds,
            verbose=spec.verbose,
        )
        tp_pose_gicp = pose.errors_from_gicp(tp_edges, gicp_outputs, q_poses, db_poses)
        _log(spec, f"  GICP: {n_converged}/{len(tp_edges)} TP pairs converged "
                   f"in {t_gicp:.1f} s")

    results = _build_results(
        spec, r, db, q, matrix, n_db, n_q, n_with_gt, effective_topk,
        encode_time, t_s1, t_s2, t_comb, t_verify,
        recalls_s1, kpct_s1, auc_s1,
        recalls_s2, kpct_s2, auc_s2,
        recalls_comb, kpct_comb, auc_comb,
        recalls_ver, kpct_ver, auc_ver,
        conf_stage, chosen, tp, fp, fn, tn, tp_dist, tp_pose_verify,
        f1_max, f1_max_thr, policy,
        tp_pose_gicp, gicp_outputs, t_gicp, n_converged, len(tp_edges),
    )

    tag = spec.tag or "run"
    written: Dict[str, Path] = {}
    written["results"] = artifacts.write_results(output_dir / f"results_{tag}.json", results)
    written["candidates"] = artifacts.write_candidates(
        output_dir / f"candidates_{tag}.csv",
        _candidate_rows(conf_sims, conf_rank, ground_truth, matrix,
                        q_positions, db_positions, n_q, chosen.threshold),
    )
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
        spec, output_dir / f"trajectory_{tag}.png",
        db_positions, q_positions, tp_edges, fp_edges,
        conf_stage, chosen.threshold, tp, fp, fn, tn)
    if plot is not None:
        written["trajectory"] = plot

    _log(spec, f"\n  Results -> {written['results']}")
    return RunResult("cross_session", results, output_dir, written)


# ---------------------------------------------------------------------------

def _session_label(source) -> str:
    """``sequence/sensor`` for HeLiPR, the folder name for anything else."""
    described = source.describe()
    name = described.get("sequence") or described.get("path", "")
    name = Path(str(name)).name
    sensor = described.get("sensor", "")
    return f"{name}/{sensor}" if sensor else name


def _write_trajectory_plot(spec, path, db_positions, q_positions,
                           tp_edges, fp_edges, stage, threshold, tp, fp, fn, tn):
    """Save the trajectory figure, or explain why it was skipped.

    A missing matplotlib must not throw away a completed run: the figure is
    the last thing written, after a job that can take tens of minutes, and
    every number it illustrates is already safely in the JSON.
    """
    try:
        from inlier.viz import write_trajectory_plot
    except ImportError as exc:  # matplotlib lives in the [eval] extra
        _log(spec, f"\n  Trajectory plot skipped ({exc}); "
                   'install it with pip install "inlier[eval]"')
        return None

    title = (f"InLiER {stage}  {_session_label(spec.db_source)} → "
             f"{_session_label(spec.q_source)}\n"
             f"thr={threshold:.3f}  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    written = write_trajectory_plot(
        path, db_positions, q_positions, tp_edges, fp_edges, title)
    _log(spec, f"  Trajectory plot -> {written}")
    return written


def _candidate_rows(conf_sims, conf_rank, ground_truth, matrix,
                    q_positions, db_positions, n_q, threshold) -> List[Dict[str, Any]]:
    """One decision per query at the operating threshold.

    Ordering follows ``metrics.confusion`` exactly so the CSV cannot disagree
    with the confusion matrix it accompanies.
    """
    rows: List[Dict[str, Any]] = []
    for j in range(n_q):
        gt_set = set(ground_truth[j].tolist()) if ground_truth[j].size > 0 else set()
        q_sims = conf_sims.get(j, {})
        if conf_rank is not None and j in conf_rank:
            ordered = [d for d in conf_rank[j] if d in q_sims]
            if len(ordered) < len(q_sims):
                seen = set(ordered)
                ordered = ordered + sorted((d for d in q_sims if d not in seen),
                                           key=lambda d: q_sims[d], reverse=True)
            ranked_d = ordered
        else:
            ranked_d = sorted(q_sims, key=lambda d: q_sims[d], reverse=True)

        top1 = next((d for d in ranked_d if q_sims[d] >= threshold), None)
        if top1 is not None:
            match_type = ("TP" if top1 in gt_set else "FP") if gt_set else "FP"
            rows.append({
                "query_idx": j,
                "predicted_db_idx": top1,
                "score": round(float(q_sims[top1]), 6),
                "match_type": match_type,
                "overlap": round(float(matrix[top1, j]), 6),
                "xy_distance_m": round(float(np.linalg.norm(
                    q_positions[j, :2] - db_positions[top1, :2])), 3),
                "has_gt_positive": bool(gt_set),
            })
        else:
            rows.append({
                "query_idx": j, "predicted_db_idx": -1, "score": 0.0,
                "match_type": "FN" if gt_set else "TN",
                "overlap": 0.0, "xy_distance_m": 0.0,
                "has_gt_positive": bool(gt_set),
            })
    return rows


def _build_results(spec, r, db, q, matrix, n_db, n_q, n_with_gt, effective_topk,
                   encode_time, t_s1, t_s2, t_comb, t_verify,
                   recalls_s1, kpct_s1, auc_s1,
                   recalls_s2, kpct_s2, auc_s2,
                   recalls_comb, kpct_comb, auc_comb,
                   recalls_ver, kpct_ver, auc_ver,
                   conf_stage, chosen, tp, fp, fn, tn, tp_dist, tp_pose_verify,
                   f1_max, f1_max_thr, gt_policy,
                   tp_pose_gicp, gicp_outputs, t_gicp, n_converged, n_tp_pairs
                   ) -> Dict[str, Any]:
    """Assemble the results payload, preserving every v1 key path."""
    def _round_pose(block):
        if block is None:
            return None
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in block.items()}

    return {
        **artifacts.provenance(
            "cross_session",
            threshold_policy=chosen.policy,
            ground_truth=gt_policy.describe(),
            candidate_filter={"filter": "none"},
            db=spec.db_source.describe(),
            query=spec.q_source.describe(),
            config_mode=r.mode,
        ),
        "config": {
            "db_sequence": spec.db_source.describe().get("sequence",
                            spec.db_source.describe().get("path", "")),
            "db_sensor": spec.db_source.describe().get("sensor", ""),
            "q_sequence": spec.q_source.describe().get("sequence",
                           spec.q_source.describe().get("path", "")),
            "q_sensor": spec.q_source.describe().get("sensor", ""),
            "overlap_threshold": spec.overlap_threshold,
            "max_pose_dist": spec.max_pose_dist,
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
                "min_shared_rows": r.shortlist.min_shared_rows,
                "mint_mode": r.shortlist.mint_mode,
                "mint_scoring": r.shortlist.mint_scoring,
            },
            "beam": {
                "topk": r.beam.topk, "topk_pct": r.beam.topk_pct,
                "min_shared_bins": r.beam.min_shared_bins,
                "min_shared_az_cols": r.beam.min_shared_az_cols,
                "score_threshold": r.stage2_score_threshold_deploy,
            },
            "rerank": None if r.rerank is None else {
                "topk": r.rerank.topk, "scoring_mode": r.rerank.scoring_mode,
                "score_threshold": r.rerank_score_threshold_deploy,
            },
        },
        "dataset_info": {
            "n_db_scans": n_db, "n_q_scans": n_q, "n_queries_with_gt": n_with_gt,
        },
        "timing": {
            "encoding_s": round(encode_time, 2),
            "stage1_retrieval_s": round(t_s1, 2),
            "stage2_reranking_s": round(t_s2, 2),
            "combined_reranking_s": round(t_comb, 2),
            "ms_per_query_s1": round(t_s1 / max(1, n_q) * 1000, 2),
            "ms_per_query_s2": round(t_s2 / max(1, n_q) * 1000, 2),
            "ms_per_query_comb": round(t_comb / max(1, n_q) * 1000, 2),
            "verify_s": round(t_verify, 2),
            "ms_per_query_ver": round(t_verify / max(1, n_q) * 1000, 2),
            "gicp_s": round(t_gicp, 2),
        },
        "stage1": artifacts.stage_block(recalls_s1, kpct_s1, auc_s1),
        "stage2": artifacts.stage_block(recalls_s2, kpct_s2, auc_s2),
        "combined": artifacts.stage_block(recalls_comb, kpct_comb, auc_comb),
        "verify": artifacts.stage_block(
            recalls_ver, kpct_ver, auc_ver,
            config={
                "top_v": r.verify_topv,
                "ransac_iters": r.verify.ransac_iters,
                "inlier_dist_thresh": r.verify.inlier_dist_thresh,
                "min_correspondences": r.verify.min_correspondences,
                "min_ransac_inliers": r.verify.min_ransac_inliers,
                "min_keypoint_inliers": r.verify.min_keypoint_inliers,
                "spatial_tol": r.verify.spatial_tol,
            },
        ),
        "confusion": {
            "stage": conf_stage,
            "threshold": round(chosen.threshold, 4),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(chosen.precision, 4),
            "recall": round(chosen.recall, 4),
            "policy": chosen.policy,
            "f1_max": round(f1_max, 4),
            "f1_max_threshold": round(f1_max_thr, 4),
            "tp_match_distance": {
                "mean_m": round(tp_dist["mean_m"], 3),
                "std_m": round(tp_dist["std_m"], 3),
                "min_m": round(tp_dist["min_m"], 3),
                "max_m": round(tp_dist["max_m"], 3),
                "median_m": round(tp_dist["median_m"], 3),
            },
            "tp_pose_error_verify": _round_pose(tp_pose_verify),
            "tp_pose_error_gicp": _round_pose(tp_pose_gicp),
        },
        "gicp": None if not gicp_outputs else {
            "config": {
                "registration_type": r.gicp.registration_type,
                "max_correspondence_distance": r.gicp.max_correspondence_distance,
                "downsampling_resolution": r.gicp.downsampling_resolution,
                "max_iterations": r.gicp.max_iterations,
                "use_raw_clouds": r.gicp.use_raw_clouds,
            },
            "n_tp_pairs": n_tp_pairs,
            "n_converged": n_converged,
            "timing_s": round(t_gicp, 2),
        },
    }
