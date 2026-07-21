#!/usr/bin/env python3
"""
evaluate_inlier_helipr.py

InLiER place recognition evaluation on the HeLiPR dataset using
precomputed overlap matrices as ground truth.

Ground-truth positives require BOTH:
    overlap >= --overlap_threshold
    XY-plane distance <= --max_pose_dist   (when > 0)

Outputs
-------
* JSON results  - Recall@1/5/10/20/50/100, Recall@1%/5%/10%,
                  PR-AUC, F1-max for both Stage-1 and Stage-2.
* 3D trajectory plot with TP (green) / FP (red) edges at best-F1 threshold
                  (using Stage-2 results).

Usage example
-------------
python3 evaluation/evaluate_inlier_helipr.py \
    --config config/default.yaml \
    --dataset ~/Documents/datasets/HeLiPR/ \
    --db_sequence Roundabout01 --q_sequence Roundabout03 \
    --pair O-Aeva \
    --overlap_dir overlap_matrices/ \
    --output_dir results/HeLiPR \
    --overlap_threshold 0.2 --max_pose_dist 10.0 \
    --pr_threshold 0.3

"""

import argparse, csv, gc, hashlib, json
import yaml, math, shutil, time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib, tqdm
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# numpy 2.0 renamed np.trapz -> np.trapezoid; compute_pr_curve uses the new
# name.  Shim it for older numpy (1.26 has trapz).
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

from inlier.core.InLiER import InLiER
from inlier.core.InLiER_Matcher import InLiER_Matcher
from inlier.core.Dataclasses import (
    InLiER_Config, ShortlistConfig, BEAMScoreConfig,
    RerankConfig, VerifyConfig, GICPRefineConfig, GICPRefineOutput,
    InLiER_Tokens, InLiER_Keypoints,
)
from utils.HeLiPR_Handler import HeLiPR_Handler

import open3d as _o3d


# ---------------------------------------------------------------------------
#  Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

# ---------------------------------------------------------------------------
#  Sensor name mapping
# ---------------------------------------------------------------------------
SENSOR_MAP = {
    "O":        "Ouster",
    "Ouster":   "Ouster",
    "V":        "Velodyne",
    "Velodyne": "Velodyne",
    "Aeva":     "Aeva",
    "Avia":     "Avia",
}


def _fmt_num_for_tag(v: float) -> str:
    s = f"{float(v):.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _short_seq_tag(seq: str) -> str:
    letters = "".join(ch for ch in seq if ch.isalpha())
    digits = ""
    i = len(seq) - 1
    while i >= 0 and seq[i].isdigit():
        digits = seq[i] + digits
        i -= 1
    if letters and digits:
        return f"{letters[0].upper()}{digits}"
    return seq


def _build_experiment_subdir(
    db_sequence: str,
    db_sensor_tag: str,
    q_sequence: str,
    q_sensor_tag: str,
    voxel_size: float,
    cell_size: float,
    N_h: int,
    N_r: int,
    N_a: int,
    N_s: int,
) -> str:
    return (
        f"db{_short_seq_tag(db_sequence)}-{db_sensor_tag}-"
        f"q{_short_seq_tag(q_sequence)}-{q_sensor_tag}_"
        f"vs{_fmt_num_for_tag(voxel_size)}_"
        f"cs{_fmt_num_for_tag(cell_size)}_"
        f"nh{N_h}_"
        f"nr{N_r}_"
        f"na{N_a}_"
        f"ns{N_s}"
    )


def voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel grid downsampling.  Uses open3d if available, else numpy centroid."""
    if voxel_size <= 0 or pts.shape[0] == 0:
        return pts
    if _o3d is not None:
        pcd = _o3d.geometry.PointCloud()
        pcd.points = _o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd = pcd.voxel_down_sample(float(voxel_size))
        return np.asarray(pcd.points, dtype=np.float32)
    coords = np.floor(pts / voxel_size).astype(np.int32)
    _, inv = np.unique(coords, axis=0, return_inverse=True)
    return np.array([pts[inv == k].mean(axis=0) for k in range(inv.max() + 1)],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
#  Overlap matrix / ground truth
# ---------------------------------------------------------------------------

def load_overlap_matrix(overlap_dir: Path,
                        db_seq: str, db_sensor: str,
                        q_seq: str,  q_sensor: str) -> np.ndarray:
    filename = f"overlap_{db_seq}_{db_sensor}_{q_seq}_{q_sensor}.txt"
    filepath = overlap_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Overlap matrix not found: {filepath}\n"
            f"Run scripts/build_overlap_data.py first."
        )
    return np.loadtxt(filepath)


def build_ground_truth(overlap_matrix: np.ndarray,
                       db_positions:   np.ndarray,
                       q_positions:    np.ndarray,
                       overlap_threshold: float = 0.5,
                       max_pose_dist:     float = 100.0,
                       ) -> Dict[int, np.ndarray]:
    """For every query j, return the array of DB indices that are GT positives."""
    N_db, N_q = overlap_matrix.shape
    gt: Dict[int, np.ndarray] = {}
    for j in range(N_q):
        mask = overlap_matrix[:, j] >= overlap_threshold
        if max_pose_dist > 0.0:
            dists_xy = np.linalg.norm(
                db_positions[:, :2] - q_positions[j, :2], axis=1
            )
            mask = mask & (dists_xy <= max_pose_dist)
        gt[j] = np.where(mask)[0]
    return gt


# ---------------------------------------------------------------------------
#  Encoding helpers
# ---------------------------------------------------------------------------

def _empty_tokens() -> InLiER_Tokens:
    """Return an empty InLiER_Tokens (no tokens)."""
    return InLiER_Tokens(token_id=np.zeros(0, dtype=np.uint32))


# ---------------------------------------------------------------------------
#  Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(cfg: InLiER_Config, voxel_size: float) -> str:
    d = asdict(cfg)
    d["voxel_size"] = float(voxel_size)
    s = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _cache_path(cache_dir: Path, sequence: str, sensor: str, seq_type: str,
                cfg: InLiER_Config, voxel_size: float) -> Path:
    key = _cache_key(cfg, voxel_size)
    return cache_dir / f"desc_{sequence}_{sensor}_{seq_type}_{key}.npz"


def _tokens_from_cache(token_ids: np.ndarray) -> InLiER_Tokens:
    """Reconstruct InLiER_Tokens from cached token_id array."""
    return InLiER_Tokens(token_id=token_ids.astype(np.uint32, copy=False))


def encode_sequence(
    handler:    HeLiPR_Handler,
    encoder:    InLiER,
    sequence:   str,
    sensor:     str,
    seq_type:   str,            # "Database" or "Query"
    voxel_size: float = 0.0,
    cache_dir:  Optional[Path] = None,
) -> Tuple[List[InLiER_Tokens], List[np.ndarray], List[np.ndarray],
           List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Encode every scan in a sequence with InLiER.

    Cache format: concatenated token_ids + weights + keypoint coordinates
    with per-scan offsets, so variable-length scans are stored efficiently.

    Returns
    -------
    tokens_list   : list of InLiER_Tokens, one per scan
    kp_aligned    : list of (K_i, 3) float64 arrays – ground-aligned keypoint
                    coordinates per scan (for geometric verification)
    kp_sensor     : list of (K_i, 3) float64 arrays – sensor-frame keypoint
                    coordinates per scan (for GICP and pose error evaluation)
    T_grounds     : list of (4, 4) float64 arrays – per-scan ground alignment
                    transforms (sensor → ground-aligned)
    positions     : (N, 3) float64 array – pose translations
    poses         : (N, 4, 4) float64 array – full SE(3) poses
    """
    cfg = encoder.config

    #  Try loading from cache
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath = _cache_path(cache_dir, sequence, sensor, seq_type, cfg, voxel_size)
        if cpath.exists():
            print(f"  [cache] loading {cpath.name}")
            npz       = np.load(cpath)
            positions = npz["positions"]             # (K, 3)
            offsets   = npz["offsets"]              # (K+1,) int64
            all_tids  = npz["token_ids"]            # (total_K,) uint32
            n_scans   = len(offsets) - 1
            tokens_list = [
                _tokens_from_cache(all_tids[offsets[i]:offsets[i + 1]])
                for i in range(n_scans)
            ]
            # Load cached keypoint coordinates if available
            if "kp_aligned" in npz:
                all_kpa = npz["kp_aligned"]          # (total_K, 3) float64
                kp_aligned = [
                    all_kpa[offsets[i]:offsets[i + 1]]
                    for i in range(n_scans)
                ]
            else:
                # Legacy cache without keypoints — fill with dummy arrays
                print("  [cache] WARNING: cache missing keypoint coords; "
                      "delete cache and re-encode for verify support")
                kp_aligned = [
                    np.zeros((int(offsets[i + 1] - offsets[i]), 3), dtype=np.float64)
                    for i in range(n_scans)
                ]
            # Load sensor-frame keypoints
            if "kp_sensor" in npz:
                all_kps = npz["kp_sensor"]           # (total_K, 3) float64
                kp_sensor = [
                    all_kps[offsets[i]:offsets[i + 1]]
                    for i in range(n_scans)
                ]
            else:
                print("  [cache] WARNING: cache missing sensor-frame keypoints; "
                      "delete cache and re-encode for correct GICP/pose support")
                kp_sensor = kp_aligned   # fallback: assume T_ground ≈ I
            # Load per-scan T_ground
            if "T_grounds" in npz:
                all_Tg = npz["T_grounds"]            # (K, 4, 4) float64
                T_grounds = [all_Tg[i] for i in range(n_scans)]
            else:
                print("  [cache] WARNING: cache missing T_ground; "
                      "delete cache and re-encode for correct GICP/pose support")
                T_grounds = [np.eye(4, dtype=np.float64) for _ in range(n_scans)]
            # Load cached poses if available, otherwise reconstruct from positions
            if "poses" in npz:
                full_poses = npz["poses"]                # (K, 4, 4)
            else:
                print("  [cache] WARNING: cache missing full poses; "
                      "delete cache and re-encode for pose error support")
                K = n_scans
                full_poses = np.tile(np.eye(4), (K, 1, 1)).astype(np.float64)
                full_poses[:, :3, 3] = positions
            return tokens_list, kp_aligned, kp_sensor, T_grounds, positions, full_poses

    # Encode from scratch 
    data  = handler.load_helipr(sequence, sensor, type=seq_type)
    poses = data["poses"]
    pcs   = data["point_clouds"]
    K     = len(pcs)

    positions   = np.array([p[:3, 3] for p in poses], dtype=np.float64)
    full_poses  = np.array([p[:4, :4] for p in poses], dtype=np.float64)
    tokens_list: List[InLiER_Tokens] = []
    kp_aligned:  List[np.ndarray]    = []
    kp_sensor:   List[np.ndarray]    = []
    T_grounds:   List[np.ndarray]    = []

    for i in tqdm.tqdm(range(K), desc=f"  Encoding {sequence}/{sensor}"):
        pts = pcs[i].astype(np.float32)

        mask = np.any(pts != 0, axis=1)
        pts  = pts[mask]

        if voxel_size > 0:
            pts = voxel_downsample(pts, voxel_size)

        if pts.shape[0] < 10:
            tokens_list.append(_empty_tokens())
            kp_aligned.append(np.zeros((0, 3), dtype=np.float64))
            kp_sensor.append(np.zeros((0, 3), dtype=np.float64))
            T_grounds.append(np.eye(4, dtype=np.float64))
        else:
            kp, tok = encoder.encode(pts, verbose=False)
            tokens_list.append(tok)
            kp_aligned.append(np.asarray(kp.p_aligned, dtype=np.float64))
            kp_sensor.append(np.asarray(kp.p, dtype=np.float64))
            T_grounds.append(np.asarray(kp.T_ground, dtype=np.float64))

        if (i + 1) % 50 == 0:
            gc.collect()

    gc.collect()

    # Save to cache
    if cache_dir is not None:
        tid_list = [t.token_id for t in tokens_list]
        lengths  = np.array([len(t) for t in tid_list], dtype=np.int64)
        offsets  = np.concatenate([[0], lengths.cumsum()])
        np.savez_compressed(
            cpath,
            positions  = positions,
            poses      = full_poses,
            offsets    = offsets,
            token_ids  = np.concatenate(tid_list) if len(tid_list) else np.zeros(0, dtype=np.uint32),
            kp_aligned = np.concatenate(kp_aligned) if kp_aligned else np.zeros((0, 3), dtype=np.float64),
            kp_sensor  = np.concatenate(kp_sensor) if kp_sensor else np.zeros((0, 3), dtype=np.float64),
            T_grounds  = np.stack(T_grounds),       # (K, 4, 4)
        )
        print(f"  [cache] saved {cpath.name}")

    return tokens_list, kp_aligned, kp_sensor, T_grounds, positions, full_poses


# ---------------------------------------------------------------------------
#  Retrieval
# ---------------------------------------------------------------------------

def compute_ranked_lists(
    q_tokens: List[InLiER_Tokens],
    matcher:  InLiER_Matcher,
    N_db:     int,
) -> Tuple[Dict[int, List[int]], Dict[int, Dict[int, float]]]:
    """
    Stage-1: rank all database scans by descending HCC score.

    Returns
    -------
    ranked_lists   : {query_idx: [db_idx_rank1, db_idx_rank2, ...]}
    similarity_map : {query_idx: {db_idx: score}}
    """
    ranked_lists:   Dict[int, List[int]]        = {}
    similarity_map: Dict[int, Dict[int, float]] = {}

    for j in tqdm.tqdm(range(len(q_tokens)), desc="  Stage-1 (MINT) retrieval"):
        s1_out = matcher.shortlist(q_tokens[j], topk=N_db, verbose=False)
        ranked_lists[j]   = s1_out.ids
        similarity_map[j] = {s1_out.ids[k]: float(s1_out.scores[k])
                             for k in range(len(s1_out.ids))}

    return ranked_lists, similarity_map


def compute_beam_ranked_lists(
    matcher:         InLiER_Matcher,
    q_tokens:        List[InLiER_Tokens],
    ranked_lists_s1: Dict[int, List[int]],
    stage1_topk:     int,
) -> Tuple[Dict[int, List[int]], Dict[int, Dict[int, float]], Dict[int, Dict[int, int]]]:
    """
    Stage-2 (BEAM): rerank the Stage-1 shortlist with azimuth-shift scoring.

    For evaluation we score ALL shortlisted candidates (topk=len(shortlist))
    and return raw scores without pre-filtering (score_threshold=-2.0 baked
    into beam_cfg at construction).  This ensures the PR curve sweep and
    Recall@N see the full Stage-2 ordering rather than an incomplete list
    truncated by the deployment-time topk / threshold.

    No rank-filtering is applied in evaluation mode — all candidates are
    kept so the PR curve and Recall@N metrics see the full reranked list.

    Returns
    -------
    ranked_lists_s2   : {query_idx: [db_idx_rank1, ...]}
    similarity_map_s2 : {query_idx: {db_idx: score}}
    shifts_map_s2     : {query_idx: {db_idx: best_shift}}
    """
    ranked_lists_s2:   Dict[int, List[int]]        = {}
    similarity_map_s2: Dict[int, Dict[int, float]] = {}
    shifts_map_s2:     Dict[int, Dict[int, int]]   = {}

    for j in tqdm.tqdm(range(len(q_tokens)), desc="  Stage-2 (BEAM) reranking"):
        shortlist = ranked_lists_s1[j][:stage1_topk]
        s2_out = matcher.beam_score(q_tokens[j], shortlist, topk=len(shortlist), verbose=False)

        ranked_lists_s2[j]   = list(s2_out.ids)
        similarity_map_s2[j] = {sid: sc for sid, sc in zip(s2_out.ids, s2_out.scores)}
        shifts_map_s2[j]     = {sid: sh for sid, sh in zip(s2_out.ids, s2_out.best_shifts)}

    return ranked_lists_s2, similarity_map_s2, shifts_map_s2


def compute_rerank_ranked_lists(
    matcher:           InLiER_Matcher,
    q_tokens:          List[InLiER_Tokens],
    ranked_lists_prev: Dict[int, List[int]],
    shifts_map_prev:   Optional[Dict[int, Dict[int, int]]],
    input_topk:        int,
) -> Tuple[Dict[int, List[int]], Dict[int, Dict[int, float]], Dict[int, Dict[int, int]]]:
    """
    Rerank stage: rerank with 4D histogram / token correspondence scores.

    Uses the Stage-2 best shift to pre-align candidates; passes shift=0 if no
    prior shift is available.  Scores all shortlisted candidates with no
    pre-threshold so the PR curve sees the full ordering.

    No rank-filtering is applied in evaluation mode — all candidates are
    kept so the PR curve and Recall@N metrics see the full reranked list.

    Returns
    -------
    ranked_lists_comb   : {query_idx: [db_idx_rank1, ...]}
    similarity_map_comb : {query_idx: {db_idx: combined_score}}
    shifts_map_comb     : {query_idx: {db_idx: best_shift}}
    """
    ranked_lists_comb:   Dict[int, List[int]]        = {}
    similarity_map_comb: Dict[int, Dict[int, float]] = {}
    shifts_map_comb:     Dict[int, Dict[int, int]]   = {}

    for j in tqdm.tqdm(range(len(q_tokens)), desc="  Rerank"):
        shortlist = ranked_lists_prev[j][:input_topk]
        if not shortlist:
            ranked_lists_comb[j]   = []
            similarity_map_comb[j] = {}
            shifts_map_comb[j]     = {}
            continue

        shifts = (
            [shifts_map_prev[j].get(sid, 0) for sid in shortlist]
            if shifts_map_prev is not None
            else [0] * len(shortlist)
        )
        comb_out = matcher.rerank(q_tokens[j], shortlist, shifts,
                                  topk=len(shortlist), verbose=False)

        ranked_lists_comb[j]   = list(comb_out.ids)
        similarity_map_comb[j] = {sid: sc for sid, sc in zip(comb_out.ids, comb_out.scores)}
        shifts_map_comb[j]     = {sid: sh for sid, sh in zip(comb_out.ids, comb_out.best_shifts)}

    return ranked_lists_comb, similarity_map_comb, shifts_map_comb


def _minimal_keypoints(
    p_aligned: np.ndarray,
    p_sensor: Optional[np.ndarray] = None,
    T_ground: Optional[np.ndarray] = None,
) -> InLiER_Keypoints:
    """Construct a minimal InLiER_Keypoints for verification / GICP.

    When ``p_sensor`` and ``T_ground`` are provided, the keypoints carry
    the real frame information needed for correct sensor-frame output.
    When omitted (legacy path), ``T_ground = I`` and ``p = p_aligned``.
    """
    if p_sensor is not None and T_ground is not None:
        return InLiER_Keypoints(p=p_sensor, T_ground=T_ground)
    else:
        return InLiER_Keypoints(
            p=p_aligned,
            T_ground=np.eye(4, dtype=np.float64),
        )


def compute_verify_similarity_map(
    matcher:       InLiER_Matcher,
    q_tokens:      List[InLiER_Tokens],
    q_kp_aligned:  List[np.ndarray],
    db_tokens:     List[InLiER_Tokens],
    db_kp_aligned: List[np.ndarray],
    ranked_lists:  Dict[int, List[int]],
    shifts_map:    Optional[Dict[int, Dict[int, int]]],
    gt:            Dict[int, np.ndarray],
    verify_cfg:    VerifyConfig,
    top_v:         int = 1,
    q_kp_sensor:   Optional[List[np.ndarray]] = None,
    db_kp_sensor:  Optional[List[np.ndarray]] = None,
    q_T_grounds:   Optional[List[np.ndarray]] = None,
    db_T_grounds:  Optional[List[np.ndarray]] = None,
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, List[int]],
           Dict[Tuple[int, int], Any]]:
    """Run geometric verification on the top-V candidates per query.

    Returns
    -------
    similarity_map : {query_idx: {db_idx: keypoint_inlier_ratio}}
        Scores for all verified candidates.  Failed verify → 0.0.
    verify_rank_order : {query_idx: [db_idx, ...]}
        Candidates in retrieval rank order (used by compute_pr_curve
        to try candidates sequentially: if top-1 is below threshold,
        fall through to top-2, etc.).
    verify_outputs : {(query_idx, db_idx): VerifyOutput}
        Raw verification outputs for every verified pair (for pose
        error analysis).

    Every verified candidate always receives a score entry:
    successful verification → keypoint_inlier_ratio ∈ [0, 1],
    failed verification    → -1.0  (sentinel < 0).
    The sentinel makes verify-fails distinguishable from low-confidence
    successes.  Threshold sweeps run over [0, 1], so verify-fails never
    qualify as positive predictions — they fall through to FN (if GT
    exists for that query) or TN (otherwise), instead of inflating FP
    counts at threshold 0.0.

    All queries are verified (not just GT-positive ones) so that
    false positives on GT-negative queries are captured in the PR
    curve.
    """
    n_queries = len(q_tokens)
    similarity_map: Dict[int, Dict[int, float]] = {}
    verify_rank_order: Dict[int, List[int]] = {}
    verify_outputs: Dict[Tuple[int, int], Any] = {}

    # Do we have proper sensor-frame data?
    have_sensor_data = (q_kp_sensor is not None and db_kp_sensor is not None
                        and q_T_grounds is not None and db_T_grounds is not None)

    desc = f"  Verify (top-{top_v})" if top_v > 1 else "  Verify (top-1)"
    for j in tqdm.tqdm(range(n_queries), desc=desc):
        cands = ranked_lists.get(j, [])
        if not cands:
            similarity_map[j] = {}
            verify_rank_order[j] = []
            continue

        cands_to_verify = cands[:top_v]
        if have_sensor_data:
            q_kp = _minimal_keypoints(q_kp_aligned[j], q_kp_sensor[j], q_T_grounds[j])
        else:
            q_kp = _minimal_keypoints(q_kp_aligned[j])
        q_sims: Dict[int, float] = {}

        for db_id in cands_to_verify:
            shift = 0
            if shifts_map is not None and j in shifts_map:
                shift = shifts_map[j].get(db_id, 0)

            if have_sensor_data:
                db_kp = _minimal_keypoints(db_kp_aligned[db_id], db_kp_sensor[db_id], db_T_grounds[db_id])
            else:
                db_kp = _minimal_keypoints(db_kp_aligned[db_id])

            vout = matcher.verify(
                q_tokens[j], q_kp,
                db_tokens[db_id], db_kp,
                azimuth_shift=shift,
                config=verify_cfg,
                verbose=False,
            )
            q_sims[db_id] = vout.keypoint_inlier_ratio if vout.success else 0.0
            verify_outputs[(j, db_id)] = vout

        similarity_map[j] = q_sims
        verify_rank_order[j] = cands_to_verify

    return similarity_map, verify_rank_order, verify_outputs


# ---------------------------------------------------------------------------
#  Pose error helpers
# ---------------------------------------------------------------------------

def _rotation_angle(R: np.ndarray) -> float:
    """Geodesic rotation angle (degrees) from a 3x3 rotation matrix."""
    cos_val = (np.trace(R) - 1.0) / 2.0
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_val)))


def _pose_errors_from_T(
    T_est: np.ndarray,
    T_gt:  np.ndarray,
) -> Tuple[float, float]:
    """Compute translation error (m) and rotation error (deg) between two SE(3).

    Both transforms map query → DB in the same coordinate frame.
    """
    # Translation error
    dt = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
    # Rotation error: angle of R_est^T @ R_gt
    R_err = T_est[:3, :3].T @ T_gt[:3, :3]
    dr = _rotation_angle(R_err)
    return dt, dr


def _compute_tp_pose_errors(
    tp_edges:       List[Tuple[int, int]],
    verify_outputs: Dict[Tuple[int, int], Any],
    q_poses:        np.ndarray,
    db_poses:       np.ndarray,
) -> Optional[Dict[str, float]]:
    """Compute translational and rotational errors for TP pairs.

    For each TP pair where a successful verify output exists, compares
    ``T_sensor`` against the GT relative transform.

    ``T_sensor`` is the estimated query→DB transform in the original
    sensor frame (computed via ``T_ground_db⁻¹ @ T_aligned @ T_ground_q``).
    The GT relative transform ``inv(T_q) @ T_db`` is also in the
    sensor/world frame, so the comparison is consistent.

    Returns None if no TP pairs have verify outputs, otherwise a dict with
    mean/median/std for both translation and rotation errors.
    """
    t_errors = []
    r_errors = []

    for j, d in tp_edges:
        vout = verify_outputs.get((j, d))
        if vout is None or not vout.success:
            continue

        # GT relative transform: T_rel = inv(T_db) @ T_q  (query_sensor → DB_sensor)
        # T_q maps query_sensor → world, T_db maps DB_sensor → world.
        # T_sensor from verify maps query_sensor → DB_sensor, so GT
        # must follow the same direction.
        T_q = q_poses[j]    # (4, 4)
        T_db = db_poses[d]  # (4, 4)
        R_db, t_db = T_db[:3, :3], T_db[:3, 3]
        T_gt = np.eye(4, dtype=np.float64)
        T_gt[:3, :3] = R_db.T @ T_q[:3, :3]
        T_gt[:3, 3]  = R_db.T @ (T_q[:3, 3] - t_db)

        dt, dr = _pose_errors_from_T(vout.T_sensor, T_gt)
        t_errors.append(dt)
        r_errors.append(dr)

    if not t_errors:
        return None

    t_arr = np.array(t_errors)
    r_arr = np.array(r_errors)
    return {
        "n_pairs":             len(t_errors),
        "translation_mean_m":  float(np.mean(t_arr)),
        "translation_std_m":   float(np.std(t_arr)),
        "translation_median_m": float(np.median(t_arr)),
        "rotation_mean_deg":   float(np.mean(r_arr)),
        "rotation_std_deg":    float(np.std(r_arr)),
        "rotation_median_deg": float(np.median(r_arr)),
    }


def _compute_tp_pose_errors_gicp(
    tp_edges:       List[Tuple[int, int]],
    gicp_outputs:   Dict[Tuple[int, int], GICPRefineOutput],
    q_poses:        np.ndarray,
    db_poses:       np.ndarray,
) -> Optional[Dict[str, float]]:
    """Compute pose errors for TP pairs after GICP refinement.

    Same as ``_compute_tp_pose_errors`` but uses the GICP-refined
    ``T_sensor`` instead of the verify-stage estimate.
    """
    t_errors = []
    r_errors = []

    for j, d in tp_edges:
        gout = gicp_outputs.get((j, d))
        if gout is None or not gout.success:
            continue

        # GT relative transform: T_rel = inv(T_db) @ T_q  (query_sensor → DB_sensor)
        T_q = q_poses[j]
        T_db = db_poses[d]
        R_db, t_db = T_db[:3, :3], T_db[:3, 3]
        T_gt = np.eye(4, dtype=np.float64)
        T_gt[:3, :3] = R_db.T @ T_q[:3, :3]
        T_gt[:3, 3]  = R_db.T @ (T_q[:3, 3] - t_db)

        dt, dr = _pose_errors_from_T(gout.T_sensor, T_gt)
        t_errors.append(dt)
        r_errors.append(dr)

    if not t_errors:
        return None

    t_arr = np.array(t_errors)
    r_arr = np.array(r_errors)
    return {
        "n_pairs":             len(t_errors),
        "translation_mean_m":  float(np.mean(t_arr)),
        "translation_std_m":   float(np.std(t_arr)),
        "translation_median_m": float(np.median(t_arr)),
        "rotation_mean_deg":   float(np.mean(r_arr)),
        "rotation_std_deg":    float(np.std(r_arr)),
        "rotation_median_deg": float(np.median(r_arr)),
    }


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------

def compute_recall_at_n(
    ranked_lists: Dict[int, List[int]],
    gt:           Dict[int, np.ndarray],
    n_values:     List[int],
) -> Dict[int, float]:
    """Fraction of queries with ≥1 GT positive in top-N retrieved."""
    valid = [j for j in ranked_lists if gt[j].size > 0]
    if not valid:
        return {n: 0.0 for n in n_values}
    recalls = {}
    for n in n_values:
        hits = sum(
            1 for j in valid
            if set(ranked_lists[j][:n]) & set(gt[j].tolist())
        )
        recalls[n] = hits / len(valid)
    return recalls


def compute_recall_at_kpct(
    ranked_lists:          Dict[int, List[int]],
    gt:                    Dict[int, np.ndarray],
    n_db:                  int,
    k_pcts:                List[float],
    fallback_ranked_lists: Optional[Dict[int, List[int]]] = None,
) -> Dict[float, float]:
    """Recall at K% of the database size.
    """
    results = {}
    for k in k_pcts:
        n = max(1, int(math.ceil(k / 100.0 * n_db)))
        if fallback_ranked_lists is None:
            r = compute_recall_at_n(ranked_lists, gt, [n])
            results[k] = r[n]
        else:
            valid = [j for j in ranked_lists if gt[j].size > 0]
            if not valid:
                results[k] = 0.0
                continue
            hits = 0
            for j in valid:
                candidates = ranked_lists[j]
                if len(candidates) < n and j in fallback_ranked_lists:
                    candidates = fallback_ranked_lists[j]
                if set(candidates[:n]) & set(gt[j].tolist()):
                    hits += 1
            results[k] = hits / len(valid)
    return results


def compute_pr_curve(
    similarity_map: Dict[int, Dict[int, float]],
    gt:             Dict[int, np.ndarray],
    thresholds:     np.ndarray,
    rank_order:     Optional[Dict[int, List[int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Sweep similarity thresholds and compute precision / recall.

    Parameters
    ----------
    rank_order : optional {query_idx: [db_idx, ...]}

    Returns precisions, recalls (arrays aligned with *thresholds*), and AUC.
    """
    valid_queries = [j for j in gt if gt[j].size > 0]
    if not valid_queries:
        z = np.zeros_like(thresholds)
        return z, z, 0.0

    n_thr = len(thresholds)
    tp = np.zeros(n_thr, dtype=np.float64)
    fp = np.zeros(n_thr, dtype=np.float64)
    fn = np.zeros(n_thr, dtype=np.float64)

    for j in valid_queries:
        gt_set   = set(gt[j].tolist())
        q_sims   = similarity_map.get(j, {})

        # Use retrieval rank order if provided, otherwise sort by score
        if rank_order is not None and j in rank_order:
            ranked_d = rank_order[j]
        else:
            ranked_d = sorted(q_sims, key=lambda d: q_sims[d], reverse=True)

        for ti, thr in enumerate(thresholds):
            top1 = next((d for d in ranked_d if q_sims.get(d, 0.0) >= thr), None)

            if top1 is not None:
                if top1 in gt_set:
                    tp[ti] += 1
                else:
                    fp[ti] += 1
            else:
                fn[ti] += 1

    precisions = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recalls    = np.where(tp + fn > 0, tp / (tp + fn), 0.0)

    order = np.argsort(recalls)
    auc   = float(np.trapezoid(precisions[order], recalls[order]))

    return precisions, recalls, auc


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_trajectory_3d(
    db_positions: np.ndarray,
    q_positions:  np.ndarray,
    tp_edges:     List[Tuple[int, int]],
    fp_edges:     List[Tuple[int, int]],
    title:        str,
    save_path:    Path,
    z_offset:     float = 10.0,
):
    """3D trajectory plot: DB at z=0, Query at z=z_offset."""
    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection="3d")

    db_z = np.zeros(len(db_positions))
    q_z  = np.full(len(q_positions), z_offset)

    ax.plot(db_positions[:, 0], db_positions[:, 1], db_z,
            linewidth=1.5, alpha=0.75, label="Database")
    ax.plot(q_positions[:, 0], q_positions[:, 1], q_z,
            linewidth=1.5, alpha=0.75, label="Query")

    for qi, di in fp_edges:
        ax.plot(
            [q_positions[qi, 0], db_positions[di, 0]],
            [q_positions[qi, 1], db_positions[di, 1]],
            [z_offset, 0.0],
            "r-", linewidth=1.0, alpha=0.35,
        )
    for qi, di in tp_edges:
        ax.plot(
            [q_positions[qi, 0], db_positions[di, 0]],
            [q_positions[qi, 1], db_positions[di, 1]],
            [z_offset, 0.0],
            "g-", linewidth=1.0, alpha=0.35,
        )

    if tp_edges:
        ax.plot([], [], [], "g-", linewidth=1.5, label=f"TP ({len(tp_edges)})")
    if fp_edges:
        ax.plot([], [], [], "r-", linewidth=1.5, label=f"FP ({len(fp_edges)})")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Session offset")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved trajectory plot → {save_path}")


def _build_tp_fp_edges(
    similarity_map: Dict[int, Dict[int, float]],
    gt:             Dict[int, np.ndarray],
    best_thr:       float,
    n_queries:      int,
    rank_order:     Optional[Dict[int, List[int]]] = None,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], int, int, int, int]:
    """Return (tp_edges, fp_edges, tp, fp, fn, tn) at the given threshold."""
    tp_edges: List[Tuple[int, int]] = []
    fp_edges: List[Tuple[int, int]] = []
    tp_count = fp_count = fn_count = tn_count = 0

    for j in range(n_queries):
        gt_set   = set(gt[j].tolist())
        q_sims   = similarity_map.get(j, {})
        if rank_order is not None and j in rank_order:
            ordered = [d for d in rank_order[j] if d in q_sims]
            seen = set(ordered)
            if len(ordered) < len(q_sims):
                tail = sorted((d for d in q_sims if d not in seen),
                              key=lambda d: q_sims[d], reverse=True)
                ranked_d = ordered + tail
            else:
                ranked_d = ordered
        else:
            ranked_d = sorted(q_sims, key=lambda d: q_sims[d], reverse=True)
        top1     = next((d for d in ranked_d if q_sims[d] >= best_thr), None)

        if gt_set:
            if top1 is not None:
                if top1 in gt_set:
                    tp_count += 1
                    tp_edges.append((j, top1))
                else:
                    fp_count += 1
                    fp_edges.append((j, top1))
            else:
                fn_count += 1
        else:
            if top1 is not None:
                fp_count += 1
            else:
                tn_count += 1

    return tp_edges, fp_edges, tp_count, fp_count, fn_count, tn_count


def _find_best_precision_threshold(
    similarity_map: Dict[int, Dict[int, float]],
    gt:             Dict[int, np.ndarray],
    n_queries:      int,
    rank_order:     Optional[Dict[int, List[int]]] = None,
) -> Tuple[float, float, float]:
    """Pick threshold that maximizes precision; ties broken by higher recall."""
    all_scores = [float(s) for q_sims in similarity_map.values() for s in q_sims.values() if s >= 0.0]
    if not all_scores:
        return 0.0, 0.0, 0.0

    uniq = np.unique(np.asarray(all_scores, dtype=np.float64))
    eps = 1e-9
    thresholds = np.concatenate(([uniq.min() - eps], uniq, [uniq.max() + eps]))

    best_thr = float(thresholds[0])
    best_prec = -1.0
    best_rec = -1.0

    for thr in thresholds:
        _, _, tp, fp, fn, _ = _build_tp_fp_edges(
            similarity_map, gt, float(thr), n_queries, rank_order=rank_order
        )
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)

        if (prec > best_prec) or (prec == best_prec and rec > best_rec):
            best_prec = float(prec)
            best_rec = float(rec)
            best_thr = float(thr)

    return best_thr, best_prec, best_rec


# ---------------------------------------------------------------------------
#  Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(args):
    dataset_path = Path(args.dataset).resolve()
    overlap_dir  = Path(args.overlap_dir)
    output_root  = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir    = Path(args.cache_dir) if args.cache_dir else None

    parts = args.pair.split("-")
    if len(parts) != 2:
        raise ValueError(f"--pair must be of the form DB-Q, e.g. 'O-Aeva', got '{args.pair}'")
    db_sensor = SENSOR_MAP.get(parts[0])
    q_sensor  = SENSOR_MAP.get(parts[1])
    if db_sensor is None or q_sensor is None:
        raise ValueError(f"Unknown sensor in pair '{args.pair}'.  "
                         f"Valid names: {list(SENSOR_MAP.keys())}")

    db_sensor_tag = parts[0].strip()
    q_sensor_tag  = parts[1].strip()

    exp_subdir = _build_experiment_subdir(
        db_sequence=args.db_sequence,
        db_sensor_tag=db_sensor_tag,
        q_sequence=args.q_sequence,
        q_sensor_tag=q_sensor_tag,
        voxel_size=args.voxel_size,
        cell_size=2 * args.voxel_size,
        N_h=args.N_h,
        N_r=args.N_r,
        N_a=args.N_a,
        N_s=args.N_s,
    )
    output_dir = output_root / exp_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_tag = (f"{args.db_sequence}_{db_sensor}_"
                f"{args.q_sequence}_{q_sensor}_"
                f"ov{args.overlap_threshold}_pd{args.max_pose_dist}m")

    print("=" * 70)
    print("  InLiER Place Recognition Evaluation  (HeLiPR dataset)")
    print(f"  DB: {args.db_sequence} / {db_sensor}")
    print(f"  Q:  {args.q_sequence} / {q_sensor}")
    print(f"  Overlap threshold : {args.overlap_threshold}")
    print(f"  Max XY pose dist  : {args.max_pose_dist} m")
    print(f"  Output folder     : {output_dir}")
    print("=" * 70)

    # 1. Build InLiER encoder 
    print("\n[1/5] Building InLiER encoder …")
    inlier_cfg = InLiER_Config(
        cell_size           = 2 * args.voxel_size,
        z_min               = args.z_min,
        z_max               = args.z_max,
        xy_max              = args.xy_max,
        N_h                 = args.N_h,
        ransac_dist_thresh  = 2 * args.voxel_size,
        point_mode          = args.point_mode,
        r_max               = args.r_max,
        N_r                 = args.N_r,
        N_a                 = args.N_a,
        shape_radius        = 3.0 * args.voxel_size,
        N_s                 = args.N_s,
    )
    encoder = InLiER(inlier_cfg)
    print(
        f"  voxel_size={args.voxel_size}  cell_size={inlier_cfg.cell_size}  "
        f"N_h={inlier_cfg.N_h}  z=[{inlier_cfg.z_min}, {inlier_cfg.z_max}]  "
        f"N_r={inlier_cfg.N_r}  N_s={inlier_cfg.N_s}  N_a={inlier_cfg.N_a}  "
        f"point_mode={inlier_cfg.point_mode}"
    )

    shortlist_cfg = ShortlistConfig(
        topk            = args.stage1_topk,
        topk_pct        = args.stage1_topk_pct,
        min_shared_rows = args.min_shared_rows,
        mint_mode       = args.mint_mode,
        mint_scoring    = args.mint_scoring,
    )
    if args.stage1_topk_pct is not None:
        print(f"  stage1_topk_pct={shortlist_cfg.topk_pct}  min_shared_rows={shortlist_cfg.min_shared_rows}  "
              f"mint_mode={shortlist_cfg.mint_mode}  mint_scoring={shortlist_cfg.mint_scoring}")
    else:
        print(f"  stage1_topk={shortlist_cfg.topk}  min_shared_rows={shortlist_cfg.min_shared_rows}  "
              f"mint_mode={shortlist_cfg.mint_mode}  mint_scoring={shortlist_cfg.mint_scoring}")

    # For eval: score_threshold=-2.0 so all candidates are returned for PR curve sweep
    beam_cfg = BEAMScoreConfig(
        topk               = args.stage2_topk,
        topk_pct           = args.stage2_topk_pct,
        min_shared_bins    = args.stage2_min_shared_bins,
        min_shared_az_cols = args.stage2_min_shared_az_cols,
        score_threshold    = -2.0,
    )
    s2_topk_str = (f"stage2_topk_pct={args.stage2_topk_pct}"
                   if args.stage2_topk_pct is not None
                   else f"stage2_topk={beam_cfg.topk}")
    print(
        f"  {s2_topk_str}  "
        f"min_shared_bins={beam_cfg.min_shared_bins}  "
        f"min_shared_az_cols={beam_cfg.min_shared_az_cols}  "
        f"score_threshold(eval)=-2.0  score_threshold(deploy)={args.stage2_score_threshold}"
    )

    rerank_cfg = None
    if args.run_combined:
        rerank_cfg = RerankConfig(
            topk          = args.combined_topk,
            scoring_mode  = args.combined_scoring_mode,
            score_threshold = -2.0,
        )
        print(
            f"  combined_topk={rerank_cfg.topk}  "
            f"scoring_mode={rerank_cfg.scoring_mode}  "
            f"score_threshold(eval)=-2.0  score_threshold(deploy)={args.combined_score_threshold}"
        )

    # 2. Load overlap matrix
    print("\n[2/5] Loading overlap matrix …")
    overlap = load_overlap_matrix(
        overlap_dir, args.db_sequence, db_sensor, args.q_sequence, q_sensor
    )
    print(f"  Shape: {overlap.shape}  (DB={overlap.shape[0]}, Q={overlap.shape[1]})")

    # 3. Encode sequences
    print("\n[3/5] Loading dataset and encoding scans …")

    for seq, label in [(args.db_sequence, "DB"), (args.q_sequence, "Q")]:
        seq_path = dataset_path / seq
        if not seq_path.exists():
            available = sorted(p.name for p in dataset_path.iterdir() if p.is_dir()) \
                        if dataset_path.exists() else []
            avail_str = "\n    ".join(available) if available else "(directory not found or empty)"
            raise FileNotFoundError(
                f"{label} sequence '{seq}' not found at: {seq_path}\n"
                f"  Available sequences in {dataset_path}:\n"
                f"    {avail_str}"
            )

    handler = HeLiPR_Handler(dataset_path, verbose=False)

    t0 = time.time()
    print(f"\n  Encoding database ({args.db_sequence}/{db_sensor}) …")
    db_tokens, db_kp_aligned, db_kp_sensor, db_T_grounds, db_positions, db_poses = encode_sequence(
        handler, encoder, args.db_sequence, db_sensor, seq_type="Undistorted",
        voxel_size=args.voxel_size, cache_dir=cache_dir,
    )

    print(f"\n  Encoding query ({args.q_sequence}/{q_sensor}) …")
    q_tokens, q_kp_aligned, q_kp_sensor, q_T_grounds, q_positions, q_poses = encode_sequence(
        handler, encoder, args.q_sequence, q_sensor, seq_type="Undistorted",
        voxel_size=args.voxel_size, cache_dir=cache_dir,
    )
    encode_time = time.time() - t0
    print(f"\n  Encoding done in {encode_time:.1f} s "
          f"(DB={len(db_tokens)}, Q={len(q_tokens)})")

    # Align overlap matrix dimensions
    if overlap.shape[0] != len(db_tokens) or overlap.shape[1] != len(q_tokens):
        print(f"\n  WARNING: overlap matrix {overlap.shape} ≠ scans "
              f"(DB={len(db_tokens)}, Q={len(q_tokens)}).  Truncating.")
        n_db = min(overlap.shape[0], len(db_tokens))
        n_q  = min(overlap.shape[1], len(q_tokens))
        overlap        = overlap[:n_db, :n_q]
        db_tokens      = db_tokens[:n_db]
        q_tokens       = q_tokens[:n_q]
        db_kp_aligned  = db_kp_aligned[:n_db]
        q_kp_aligned   = q_kp_aligned[:n_q]
        db_kp_sensor   = db_kp_sensor[:n_db]
        q_kp_sensor    = q_kp_sensor[:n_q]
        db_T_grounds   = db_T_grounds[:n_db]
        q_T_grounds    = q_T_grounds[:n_q]
        db_positions   = db_positions[:n_db]
        q_positions    = q_positions[:n_q]
        db_poses       = db_poses[:n_db]
        q_poses        = q_poses[:n_q]

    # 4. Ground truth 
    print("\n[4/5] Building ground truth …")
    gt = build_ground_truth(
        overlap, db_positions, q_positions,
        overlap_threshold=args.overlap_threshold,
        max_pose_dist=args.max_pose_dist,
    )
    n_with_gt = sum(1 for v in gt.values() if v.size > 0)
    print(f"  Queries with ≥1 GT positive: {n_with_gt} / {len(q_tokens)}")

    if n_with_gt == 0:
        print("\n  ERROR: No queries have ground-truth positives.  "
              "Lower --overlap_threshold or check the overlap matrix.")
        return

    # 5. Build matcher and run retrieval
    print("\n[5/5] Build matcher and run retrieval …")

    matcher = InLiER_Matcher(
        inlier_config   = inlier_cfg,
        shortlist_config = shortlist_cfg,
        beam_score_config = beam_cfg,
        rerank_config   = rerank_cfg if rerank_cfg is not None else RerankConfig(),
    )
    for i, tok in enumerate(db_tokens):
        matcher.add(i, tok)
    matcher.finalize()

    t0 = time.time()
    ranked_lists_s1, similarity_map_s1 = compute_ranked_lists(
        q_tokens, matcher, len(db_tokens),
    )
    t_s1 = time.time() - t0
    print(f"  Stage-1 ({shortlist_cfg.mint_mode}/{shortlist_cfg.mint_scoring}) "
          f"done in {t_s1:.1f} s "
          f"({t_s1 / max(1, len(q_tokens)) * 1000:.1f} ms/query)")

    # 6. Stage-2 (BEAM) reranking
    if args.stage1_topk_pct is not None:
        effective_stage1_topk = max(1, round(len(db_tokens) * args.stage1_topk_pct))
    else:
        effective_stage1_topk = args.stage1_topk

    ranked_lists_s2 = None
    similarity_map_s2 = None
    shifts_map_s2 = None
    t_s2 = 0.0
    if not args.skip_stage2:
        t0 = time.time()
        ranked_lists_s2, similarity_map_s2, shifts_map_s2 = compute_beam_ranked_lists(
            matcher, q_tokens, ranked_lists_s1,
            stage1_topk=effective_stage1_topk,
        )
        t_s2 = time.time() - t0
        print(f"  Stage-2 done in {t_s2:.1f} s "
              f"({t_s2 / max(1, len(q_tokens)) * 1000:.1f} ms/query)")
    else:
        print("\n  Stage-2 skipped.")

    # 7. Rerank stage
    ranked_lists_comb = None
    similarity_map_comb = None
    shifts_map_comb = None
    t_comb = 0.0
    if args.run_combined:
        if not args.skip_stage2:
            comb_input_lists  = ranked_lists_s2
            comb_input_shifts = shifts_map_s2
            if args.stage2_topk_pct is not None:
                comb_input_topk = max(1, round(effective_stage1_topk * args.stage2_topk_pct))
            else:
                comb_input_topk = args.stage2_topk
        else:
            comb_input_lists  = ranked_lists_s1
            comb_input_shifts = None
            comb_input_topk   = effective_stage1_topk

        t0 = time.time()
        ranked_lists_comb, similarity_map_comb, shifts_map_comb = compute_rerank_ranked_lists(
            matcher, q_tokens, comb_input_lists, comb_input_shifts, comb_input_topk,
        )
        t_comb = time.time() - t0
        print(f"  Rerank done in {t_comb:.1f} s "
              f"({t_comb / max(1, len(q_tokens)) * 1000:.1f} ms/query)")

    # 8. Metrics – Stage-1
    n_values_s1 = [1, 5, 10, 20, 50, 100]
    k_pcts      = [1.0, 5.0, 10.0]
    thr_s1      = np.arange(0.0, 1.001, 0.001)

    recalls_s1      = compute_recall_at_n(ranked_lists_s1, gt, n_values_s1)
    recalls_kpct_s1 = compute_recall_at_kpct(ranked_lists_s1, gt, len(db_tokens), k_pcts)
    prec_s1, rec_s1, auc_s1 = compute_pr_curve(similarity_map_s1, gt, thr_s1)

    # 9. Metrics – Stage-2
    prec_s2 = rec_s2 = auc_s2 = None
    if ranked_lists_s2 is not None:
        n_values_s2 = [1, 5, 10, 20, 50, 100]
        recalls_s2_n = compute_recall_at_n(ranked_lists_s2, gt, n_values_s2)

        recalls_kpct_s2 = compute_recall_at_kpct(
            ranked_lists_s2, gt, len(db_tokens), k_pcts,
            fallback_ranked_lists=ranked_lists_s1,
        )

        thr_s2 = np.arange(0.0, 1.001, 0.005)
        prec_s2, rec_s2, auc_s2 = compute_pr_curve(similarity_map_s2, gt, thr_s2)
    else:
        recalls_s2_n = recalls_kpct_s2 = None

    # 10. Metrics – Combined / Rerank
    recalls_comb_n = recalls_kpct_comb = prec_comb = rec_comb = auc_comb = None
    if ranked_lists_comb is not None:
        n_values_comb  = [1, 5, 10, 20, 50, 100]
        recalls_comb_n = compute_recall_at_n(ranked_lists_comb, gt, n_values_comb)
        recalls_kpct_comb = compute_recall_at_kpct(
            ranked_lists_comb, gt, len(db_tokens), k_pcts,
            fallback_ranked_lists=comb_input_lists,
        )

        thr_comb = np.arange(0.0, 1.001, 0.005)
        prec_comb, rec_comb, auc_comb = compute_pr_curve(
            similarity_map_comb, gt, thr_comb)

    # 11. Metrics – Verify (geometric verification on top-V)
    prec_ver = rec_ver = auc_ver = None
    similarity_map_ver = None
    verify_rank_order = None
    verify_outputs: Dict[Tuple[int, int], Any] = {}
    recalls_ver_n = recalls_kpct_ver = None
    n_values_ver = None
    t_verify = 0.0

    # Use the best available ranked list for verification input
    verify_ranked  = ranked_lists_comb if ranked_lists_comb is not None else (
                     ranked_lists_s2 if ranked_lists_s2 is not None else ranked_lists_s1)
    verify_shifts  = shifts_map_comb if shifts_map_comb is not None else (
                     shifts_map_s2 if shifts_map_s2 is not None else None)

    verify_cfg = VerifyConfig(
        ransac_iters=args.verify_ransac_iters,
        inlier_dist_thresh=args.verify_inlier_dist,
        min_correspondences=args.verify_min_correspondences,
        min_ransac_inliers=args.verify_min_ransac_inliers,
        min_keypoint_inliers=args.verify_min_keypoint_inliers,
        spatial_tol=args.verify_spatial_tol,
    )

    if not args.skip_verify:
        t0 = time.time()
        similarity_map_ver, verify_rank_order, verify_outputs = compute_verify_similarity_map(
            matcher, q_tokens, q_kp_aligned, db_tokens, db_kp_aligned,
            verify_ranked, verify_shifts, gt, verify_cfg,
            top_v=args.verify_topv,
            q_kp_sensor=q_kp_sensor,
            db_kp_sensor=db_kp_sensor,
            q_T_grounds=q_T_grounds,
            db_T_grounds=db_T_grounds,
        )
        t_verify = time.time() - t0
        print(f"  Verify done in {t_verify:.1f} s "
              f"({t_verify / max(1, len(q_tokens)) * 1000:.1f} ms/query)")

        thr_ver = np.arange(0.0, 1.001, 0.005)
        prec_ver, rec_ver, auc_ver = compute_pr_curve(
            similarity_map_ver, gt, thr_ver,
            rank_order=verify_rank_order)

        # Build score-sorted ranked lists for Recall@K.
        # Verified candidates (sorted by score) come first; for K > top_v,
        # fall back to the previous stage's ranked list.
        ranked_lists_ver: Dict[int, List[int]] = {}
        for j, q_sims in similarity_map_ver.items():
            scored = sorted(q_sims, key=lambda d: q_sims[d], reverse=True)
            verified_set = set(scored)
            fallback = verify_ranked.get(j, [])
            tail = [d for d in fallback if d not in verified_set]
            ranked_lists_ver[j] = scored + tail
        n_values_ver = [1, 5, 10, 20, 50, 100]
        recalls_ver_n = compute_recall_at_n(ranked_lists_ver, gt, n_values_ver)
        recalls_kpct_ver = compute_recall_at_kpct(
            ranked_lists_ver, gt, len(db_tokens), k_pcts,
            fallback_ranked_lists=verify_ranked,
        )

    # Confusion matrix at auto/manual threshold
    # Use the latest available similarity map for confusion / trajectory
    conf_rank_order = None
    if similarity_map_ver is not None:
        conf_sim_map = similarity_map_ver
        conf_stage   = "Verify"
        conf_rank_order = verify_rank_order
    elif similarity_map_comb is not None:
        conf_sim_map = similarity_map_comb
        conf_stage   = "Rerank"
    elif similarity_map_s2 is not None:
        conf_sim_map = similarity_map_s2
        conf_stage   = "Stage-2"
    else:
        conf_sim_map = similarity_map_s1
        conf_stage   = "Stage-1"

    if args.pr_threshold is None:
        pr_threshold, auto_prec, auto_rec = _find_best_precision_threshold(
            conf_sim_map, gt, len(q_tokens), rank_order=conf_rank_order
        )
        print(f"\n  Auto-selected threshold (max precision): {pr_threshold:.4f} "
              f"(precision={auto_prec:.4f}, recall={auto_rec:.4f})")
    else:
        pr_threshold = args.pr_threshold

    tp_edges, fp_edges, tp_count, fp_count, fn_count, tn_count = _build_tp_fp_edges(
        conf_sim_map, gt, pr_threshold, len(q_tokens), rank_order=conf_rank_order,
    )

    # TP match distance statistics
    if tp_edges:
        tp_dists = np.array([
            np.linalg.norm(q_positions[j, :2] - db_positions[d, :2])
            for j, d in tp_edges
        ])
        tp_dist_mean = float(np.mean(tp_dists))
        tp_dist_std  = float(np.std(tp_dists))
        tp_dist_min  = float(np.min(tp_dists))
        tp_dist_max  = float(np.max(tp_dists))
        tp_dist_median = float(np.median(tp_dists))
    else:
        tp_dist_mean = tp_dist_std = tp_dist_min = tp_dist_max = tp_dist_median = 0.0

    # TP pose error statistics (verify)
    tp_pose_errors = _compute_tp_pose_errors(
        tp_edges, verify_outputs, q_poses, db_poses,
    )

    # GICP refinement on TP pairs 
    gicp_outputs: Dict[Tuple[int, int], GICPRefineOutput] = {}
    tp_pose_errors_gicp = None
    t_gicp = 0.0

    if not args.skip_gicp and not args.skip_verify and tp_edges:
        gicp_cfg = GICPRefineConfig(
            registration_type=args.gicp_registration_type,
            max_correspondence_distance=args.gicp_max_corr_dist,
            downsampling_resolution=args.gicp_downsampling_resolution,
            voxel_resolution=args.gicp_voxel_resolution,
            num_threads=args.gicp_num_threads,
            use_raw_clouds=args.gicp_use_raw_clouds,
            max_iterations=args.gicp_max_iterations,
        )

        # When use_raw_clouds=True, load raw point clouds on-demand per pair.
        # Collect unique scan indices needed so we only load each once.
        raw_clouds_q:  Dict[int, np.ndarray] = {}
        raw_clouds_db: Dict[int, np.ndarray] = {}
        if gicp_cfg.use_raw_clouds:
            needed_q  = set(j for j, d in tp_edges
                           if verify_outputs.get((j, d), None) is not None
                           and verify_outputs[(j, d)].success)
            needed_db = set(d for j, d in tp_edges
                           if verify_outputs.get((j, d), None) is not None
                           and verify_outputs[(j, d)].success)
            # Lazy-load via handler (scans are in sensor/local frame)
            data_db = handler.load_helipr(args.db_sequence, db_sensor, type="Undistorted")
            data_q  = handler.load_helipr(args.q_sequence, q_sensor, type="Undistorted")
            for idx in tqdm.tqdm(sorted(needed_db), desc="  Loading DB raw clouds"):
                pts = np.asarray(data_db["point_clouds"][idx], dtype=np.float32)
                pts = pts[np.any(pts != 0, axis=1)]
                if args.voxel_size > 0:
                    pts = voxel_downsample(pts, args.voxel_size)
                raw_clouds_db[idx] = pts
            for idx in tqdm.tqdm(sorted(needed_q), desc="  Loading Q raw clouds"):
                pts = np.asarray(data_q["point_clouds"][idx], dtype=np.float32)
                pts = pts[np.any(pts != 0, axis=1)]
                if args.voxel_size > 0:
                    pts = voxel_downsample(pts, args.voxel_size)
                raw_clouds_q[idx] = pts
            del data_db, data_q
            gc.collect()

        t0 = time.time()
        n_gicp_ok = 0
        for j, d in tqdm.tqdm(tp_edges, desc="  GICP refinement (TP pairs)"):
            vout = verify_outputs.get((j, d))
            if vout is None or not vout.success:
                continue
            q_kp = _minimal_keypoints(q_kp_aligned[j], q_kp_sensor[j], q_T_grounds[j])
            db_kp = _minimal_keypoints(db_kp_aligned[d], db_kp_sensor[d], db_T_grounds[d])
            gout = InLiER_Matcher.refine_gicp(
                vout, q_kp, db_kp,
                query_raw=raw_clouds_q.get(j) if gicp_cfg.use_raw_clouds else None,
                db_raw=raw_clouds_db.get(d) if gicp_cfg.use_raw_clouds else None,
                config=gicp_cfg, verbose=False,
            )
            gicp_outputs[(j, d)] = gout
            if gout.success:
                n_gicp_ok += 1
        t_gicp = time.time() - t0
        print(f"  GICP done in {t_gicp:.1f} s  "
              f"({n_gicp_ok}/{len(tp_edges)} TP pairs converged)")

        tp_pose_errors_gicp = _compute_tp_pose_errors_gicp(
            tp_edges, gicp_outputs, q_poses, db_poses,
        )
        # Free raw clouds
        del raw_clouds_q, raw_clouds_db
        gc.collect()

    # Build per-candidate records for CSV
    candidate_records = []
    for j in range(len(q_tokens)):
        gt_set = set(gt[j].tolist()) if gt[j].size > 0 else set()
        q_sims = conf_sim_map.get(j, {})
        if conf_rank_order is not None and j in conf_rank_order:
            ordered = [d for d in conf_rank_order[j] if d in q_sims]
            seen = set(ordered)
            if len(ordered) < len(q_sims):
                tail = sorted((d for d in q_sims if d not in seen),
                              key=lambda d: q_sims[d], reverse=True)
                ranked_d = ordered + tail
            else:
                ranked_d = ordered
        else:
            ranked_d = sorted(q_sims, key=lambda d: q_sims[d], reverse=True)

        top1 = next((d for d in ranked_d if q_sims[d] >= pr_threshold), None)

        if top1 is not None:
            pred_db   = top1
            score     = float(q_sims[top1])
            xy_dist   = float(np.linalg.norm(q_positions[j, :2] - db_positions[top1, :2]))
            olap      = float(overlap[top1, j])
            if gt_set:
                match_type = "TP" if top1 in gt_set else "FP"
            else:
                match_type = "FP"
        else:
            pred_db    = -1
            score      = 0.0
            xy_dist    = 0.0
            olap       = 0.0
            match_type = "FN" if gt_set else "TN"

        candidate_records.append({
            "query_idx":       j,
            "predicted_db_idx": pred_db,
            "score":           round(score, 6),
            "match_type":      match_type,
            "overlap":         round(olap, 6),
            "xy_distance_m":   round(xy_dist, 3),
            "has_gt_positive":  bool(gt_set),
        })

    # Print summary 
    print("\n" + "─" * 60)
    print(f"  STAGE-1  MINT {shortlist_cfg.mint_mode}/{shortlist_cfg.mint_scoring}  (rotation-invariant)")
    print("  Recall@N:")
    for n in n_values_s1:
        print(f"    R@{n:<4d} = {recalls_s1[n]:.4f}")
    print("  Recall@K%:")
    for k in k_pcts:
        n_k = max(1, int(math.ceil(k / 100.0 * len(db_tokens))))
        print(f"    R@{k:.0f}%  = {recalls_kpct_s1[k]:.4f}  (top-{n_k})")
    print(f"\n  PR AUC  = {auc_s1:.4f}")

    if recalls_s2_n is not None:
        print("\n" + "─" * 60)
        print("  STAGE-2  (BEAM azimuth-shift reranking)")
        print("  Recall@N:")
        for n in n_values_s2:
            print(f"    R@{n:<4d} = {recalls_s2_n[n]:.4f}")
        print("  Recall@K%:")
        for k in k_pcts:
            n_k = max(1, int(math.ceil(k / 100.0 * len(db_tokens))))
            print(f"    R@{k:.0f}%  = {recalls_kpct_s2[k]:.4f}  (top-{n_k})")
        print(f"\n  PR AUC  = {auc_s2:.4f}")
        print("─" * 60)

    if recalls_comb_n is not None:
        print(f"\n  RERANK  ({rerank_cfg.scoring_mode if rerank_cfg else 'jaccard4d'})")
        print("  Recall@N:")
        for n in n_values_comb:
            print(f"    R@{n:<4d} = {recalls_comb_n[n]:.4f}")
        print("  Recall@K%:")
        for k in k_pcts:
            n_k = max(1, int(math.ceil(k / 100.0 * len(db_tokens))))
            print(f"    R@{k:.0f}%  = {recalls_kpct_comb[k]:.4f}  (top-{n_k})")
        print(f"\n  PR AUC  = {auc_comb:.4f}")
        print("─" * 60)

    if auc_ver is not None:
        print(f"\n  VERIFY  (RANSAC geometric verification, top-{args.verify_topv})")
        print("  Recall@N:")
        for n in n_values_ver:
            print(f"    R@{n:<4d} = {recalls_ver_n[n]:.4f}")
        print("  Recall@K%:")
        for k in k_pcts:
            n_k = max(1, int(math.ceil(k / 100.0 * len(db_tokens))))
            print(f"    R@{k:.0f}%  = {recalls_kpct_ver[k]:.4f}  (top-{n_k})")
        print(f"\n  PR AUC  = {auc_ver:.4f}")
        print("─" * 60)

    print(f"\n  Confusion @ threshold={pr_threshold:.3f}  ({conf_stage})")
    print(f"    TP={tp_count}  FP={fp_count}  FN={fn_count}  TN={tn_count}")
    prec_at_thr = tp_count / max(1, tp_count + fp_count)
    rec_at_thr  = tp_count / max(1, tp_count + fn_count)
    print(f"    Precision={prec_at_thr:.4f}  Recall={rec_at_thr:.4f}")
    if tp_edges:
        print(f"    TP match dist: mean={tp_dist_mean:.2f} m  "
              f"std={tp_dist_std:.2f} m  "
              f"median={tp_dist_median:.2f} m  "
              f"[{tp_dist_min:.2f}, {tp_dist_max:.2f}]")
    print("─" * 60)

    if tp_pose_errors is not None:
        print(f"\n  TP pose errors — Verify ({tp_pose_errors['n_pairs']} pairs)")
        print(f"    Translation: "
              f"mean={tp_pose_errors['translation_mean_m']:.3f} m  "
              f"median={tp_pose_errors['translation_median_m']:.3f} m  "
              f"std={tp_pose_errors['translation_std_m']:.3f} m")
        print(f"    Rotation:    "
              f"mean={tp_pose_errors['rotation_mean_deg']:.3f} deg  "
              f"median={tp_pose_errors['rotation_median_deg']:.3f} deg  "
              f"std={tp_pose_errors['rotation_std_deg']:.3f} deg")
        print("─" * 60)

    if tp_pose_errors_gicp is not None:
        print(f"\n  TP pose errors — GICP ({tp_pose_errors_gicp['n_pairs']} pairs)")
        print(f"    Translation: "
              f"mean={tp_pose_errors_gicp['translation_mean_m']:.3f} m  "
              f"median={tp_pose_errors_gicp['translation_median_m']:.3f} m  "
              f"std={tp_pose_errors_gicp['translation_std_m']:.3f} m")
        print(f"    Rotation:    "
              f"mean={tp_pose_errors_gicp['rotation_mean_deg']:.3f} deg  "
              f"median={tp_pose_errors_gicp['rotation_median_deg']:.3f} deg  "
              f"std={tp_pose_errors_gicp['rotation_std_deg']:.3f} deg")
        print("─" * 60)

    # ── Save results JSON ────────────────────────────────────────────────
    results = {
        "config": {
            "db_sequence":       args.db_sequence,
            "db_sensor":         db_sensor,
            "q_sequence":        args.q_sequence,
            "q_sensor":          q_sensor,
            "overlap_threshold": args.overlap_threshold,
            "max_pose_dist":     args.max_pose_dist,
            "voxel_size":        args.voxel_size,
            "inlier": {
                "cell_size":        inlier_cfg.cell_size,
                "N_h":              inlier_cfg.N_h,
                "z_min":            inlier_cfg.z_min,
                "z_max":            inlier_cfg.z_max,
                "N_r":              inlier_cfg.N_r,
                "N_s":              inlier_cfg.N_s,
                "N_a":              inlier_cfg.N_a,
                "point_mode":       inlier_cfg.point_mode,
            },
            "shortlist": {
                "topk":              shortlist_cfg.topk,
                "topk_pct":          shortlist_cfg.topk_pct,
                "effective_topk":    effective_stage1_topk,
                "min_shared_rows":   shortlist_cfg.min_shared_rows,
                "mint_mode":         shortlist_cfg.mint_mode,
                "mint_scoring":      shortlist_cfg.mint_scoring,
            },
            "beam": {
                "topk":              args.stage2_topk,
                "topk_pct":          args.stage2_topk_pct,
                "min_shared_bins":   beam_cfg.min_shared_bins,
                "min_shared_az_cols": beam_cfg.min_shared_az_cols,
                "score_threshold":   args.stage2_score_threshold,
            },
            "rerank": None if rerank_cfg is None else {
                "topk":              rerank_cfg.topk,
                "scoring_mode":      rerank_cfg.scoring_mode,
                "score_threshold":   args.combined_score_threshold,
            },
        },
        "dataset_info": {
            "n_db_scans":        len(db_tokens),
            "n_q_scans":         len(q_tokens),
            "n_queries_with_gt": n_with_gt,
        },
        "timing": {
            "encoding_s":          round(encode_time, 2),
            "stage1_retrieval_s":  round(t_s1, 2),
            "stage2_reranking_s":  round(t_s2, 2),
            "combined_reranking_s": round(t_comb, 2),
            "ms_per_query_s1":     round(t_s1   / max(1, len(q_tokens)) * 1000, 2),
            "ms_per_query_s2":     round(t_s2   / max(1, len(q_tokens)) * 1000, 2),
            "ms_per_query_comb":   round(t_comb / max(1, len(q_tokens)) * 1000, 2),
            "verify_s":            round(t_verify, 2),
            "ms_per_query_ver":    round(t_verify / max(1, len(q_tokens)) * 1000, 2),
            "gicp_s":              round(t_gicp, 2),
        },
        "stage1": {
            "recall_at_n":    {str(n): round(recalls_s1[n], 6) for n in n_values_s1},
            "recall_at_kpct": {f"{k:.0f}pct": round(recalls_kpct_s1[k], 6) for k in k_pcts},
            "pr_auc":         round(auc_s1, 6),
        },
        "stage2": None if recalls_s2_n is None else {
            "recall_at_n":    {str(n): round(recalls_s2_n[n], 6) for n in n_values_s2},
            "recall_at_kpct": {f"{k:.0f}pct": round(recalls_kpct_s2[k], 6) for k in k_pcts},
            "pr_auc":         round(auc_s2, 6),
        },
        "combined": None if recalls_comb_n is None else {
            "recall_at_n":    {str(n): round(recalls_comb_n[n], 6) for n in n_values_comb},
            "recall_at_kpct": {f"{k:.0f}pct": round(recalls_kpct_comb[k], 6) for k in k_pcts},
            "pr_auc":         round(auc_comb, 6),
        },
        "verify": None if auc_ver is None else {
            "recall_at_n":    {str(n): round(recalls_ver_n[n], 6) for n in n_values_ver},
            "recall_at_kpct": {f"{k:.0f}pct": round(recalls_kpct_ver[k], 6) for k in k_pcts},
            "pr_auc":         round(auc_ver, 6),
            "config": {
                "top_v":              args.verify_topv,
                "ransac_iters":       verify_cfg.ransac_iters,
                "inlier_dist_thresh": verify_cfg.inlier_dist_thresh,
                "min_correspondences":  verify_cfg.min_correspondences,
                "min_ransac_inliers":   verify_cfg.min_ransac_inliers,
                "min_keypoint_inliers": verify_cfg.min_keypoint_inliers,
                "spatial_tol":        verify_cfg.spatial_tol,
            },
        },
        "confusion": {
            "stage":     conf_stage,
            "threshold": round(pr_threshold, 4),
            "TP": tp_count, "FP": fp_count, "FN": fn_count, "TN": tn_count,
            "precision": round(prec_at_thr, 4),
            "recall":    round(rec_at_thr, 4),
            "tp_match_distance": {
                "mean_m":   round(tp_dist_mean, 3),
                "std_m":    round(tp_dist_std, 3),
                "min_m":    round(tp_dist_min, 3),
                "max_m":    round(tp_dist_max, 3),
                "median_m": round(tp_dist_median, 3),
            },
            "tp_pose_error_verify": None if tp_pose_errors is None else {
                "n_pairs":               tp_pose_errors["n_pairs"],
                "translation_mean_m":    round(tp_pose_errors["translation_mean_m"], 3),
                "translation_median_m":  round(tp_pose_errors["translation_median_m"], 3),
                "translation_std_m":     round(tp_pose_errors["translation_std_m"], 3),
                "rotation_mean_deg":     round(tp_pose_errors["rotation_mean_deg"], 3),
                "rotation_median_deg":   round(tp_pose_errors["rotation_median_deg"], 3),
                "rotation_std_deg":      round(tp_pose_errors["rotation_std_deg"], 3),
            },
            "tp_pose_error_gicp": None if tp_pose_errors_gicp is None else {
                "n_pairs":               tp_pose_errors_gicp["n_pairs"],
                "translation_mean_m":    round(tp_pose_errors_gicp["translation_mean_m"], 3),
                "translation_median_m":  round(tp_pose_errors_gicp["translation_median_m"], 3),
                "translation_std_m":     round(tp_pose_errors_gicp["translation_std_m"], 3),
                "rotation_mean_deg":     round(tp_pose_errors_gicp["rotation_mean_deg"], 3),
                "rotation_median_deg":   round(tp_pose_errors_gicp["rotation_median_deg"], 3),
                "rotation_std_deg":      round(tp_pose_errors_gicp["rotation_std_deg"], 3),
            },
        },
        "gicp": None if not gicp_outputs else {
            "config": {
                "registration_type":           args.gicp_registration_type,
                "max_correspondence_distance": args.gicp_max_corr_dist,
                "downsampling_resolution":     args.gicp_downsampling_resolution,
                "max_iterations":              args.gicp_max_iterations,
                "use_raw_clouds":              args.gicp_use_raw_clouds,
            },
            "n_tp_pairs":   len(tp_edges),
            "n_converged":  sum(1 for g in gicp_outputs.values() if g.success),
            "timing_s":     round(t_gicp, 2),
        },
    }
    results_path = output_dir / f"results_{pair_tag}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {results_path}")

    # Save per-candidate CSV
    csv_path = output_dir / f"candidates_{pair_tag}.csv"
    csv_fields = ["query_idx", "predicted_db_idx", "score", "match_type",
                  "overlap", "xy_distance_m", "has_gt_positive"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(candidate_records)
    print(f"  Candidates CSV  → {csv_path}")

    # Per-pair verify poses (for playback / qualitative visualisation).
    # T_sensor maps query -> DB in the sensor frame (p_db = T_sensor @ p_query);
    # exported losslessly as translation + row-major 3x3 rotation.
    verify_csv_path = output_dir / f"per_pair_verify_{pair_tag}.csv"
    verify_fields = ["query_idx", "db_idx", "success",
                     "tx", "ty", "tz",
                     "r00", "r01", "r02",
                     "r10", "r11", "r12",
                     "r20", "r21", "r22"]
    with open(verify_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=verify_fields)
        writer.writeheader()
        for (j, d), vout in sorted(verify_outputs.items()):
            T = np.asarray(vout.T_sensor, dtype=np.float64)
            R, t = T[:3, :3], T[:3, 3]
            writer.writerow({
                "query_idx": j, "db_idx": d,
                "success": int(bool(vout.success)),
                "tx": t[0], "ty": t[1], "tz": t[2],
                "r00": R[0, 0], "r01": R[0, 1], "r02": R[0, 2],
                "r10": R[1, 0], "r11": R[1, 1], "r12": R[1, 2],
                "r20": R[2, 0], "r21": R[2, 1], "r22": R[2, 2],
            })
    print(f"  Verify poses CSV → {verify_csv_path}")

    # Copy config YAML into the output folder
    config_dest = output_dir / Path(args.config_path).name
    shutil.copy2(args.config_path, config_dest)
    print(f"  Config copied → {config_dest}")

    # Plots
    plot_trajectory_3d(
        db_positions, q_positions,
        tp_edges, fp_edges,
        title=(
            f"InLiER {conf_stage}  {args.db_sequence}/{db_sensor} → "
            f"{args.q_sequence}/{q_sensor}\n"
            f"thr={pr_threshold:.3f}  "
            f"TP={tp_count}  FP={fp_count}  FN={fn_count}  TN={tn_count}"
        ),
        save_path=output_dir / f"trajectory_{pair_tag}.png",
    )

    print("\n  Done.")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate InLiER place recognition on HeLiPR with overlap GT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (encoder + matcher params)
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file with InLiER encoder/matcher parameters.")

    # Dataset / IO
    parser.add_argument("--dataset", type=str, required=True,
                        help="Root path to the HeLiPR dataset.")
    parser.add_argument("--db_sequence", type=str, required=True,
                        help="Database sequence, e.g. 'Roundabout01'.")
    parser.add_argument("--q_sequence", type=str, required=True,
                        help="Query sequence, e.g. 'Roundabout03'.")
    parser.add_argument("--pair", type=str, required=True,
                        help="Sensor pair DB-Q, e.g. 'O-O', 'O-Aeva', 'Aeva-Aeva'.")
    parser.add_argument("--overlap_dir", type=str, default="overlap_matrices",
                        help="Directory containing precomputed overlap .txt files.")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Directory to write results and plots.")

    # Ground-truth parameters
    parser.add_argument("--overlap_threshold", type=float, default=0.3,
                        help="Min overlap to consider a pair a GT positive.")
    parser.add_argument("--max_pose_dist", type=float, default=25.0,
                        help="Max XY pose distance (m) to count as GT positive "
                             "(0 = no limit).")

    # Caching
    parser.add_argument("--cache_dir", type=str, default="cache_inlier",
                        help="Directory for cached descriptors ('' to disable).")

    # Threshold for confusion matrix / trajectory plot
    parser.add_argument("--pr_threshold", type=float, default=None,
                        help="Score threshold for confusion matrix and trajectory plot. "
                             "If omitted, auto-selects the threshold that maximizes precision.")

    args = parser.parse_args()

    # Load encoder / matcher params from YAML
    cfg = load_config(args.config)
    args.config_path = str(Path(args.config).resolve())
    args.cfg = cfg

    # Preprocessing
    args.voxel_size = cfg.get("voxel_size", 0.5)

    # Encoder
    enc = cfg.get("encoder", {})
    args.N_h            = enc.get("N_h", 10)
    args.z_min          = enc.get("z_min", 0.0)
    args.z_max          = enc.get("z_max", 20.0)
    args.N_r            = enc.get("N_r", 20)
    args.N_s            = enc.get("N_s", 7)
    args.N_a            = enc.get("N_a", 60)
    args.r_max          = enc.get("r_max", 100.0)
    args.xy_max         = enc.get("xy_max", 100.0)
    args.point_mode     = enc.get("point_mode", "keypoints")

    # Stage-1
    s1 = cfg.get("stage1", {})
    args.stage1_topk     = s1.get("topk", 100)
    args.stage1_topk_pct = s1.get("topk_pct", None)
    args.min_shared_rows = s1.get("min_shared_rows", 3)
    args.mint_mode       = s1.get("mint_mode", "compact")
    args.mint_scoring    = s1.get("mint_scoring", "l1_intersection")

    # Stage-2
    s2 = cfg.get("stage2", {})
    args.skip_stage2              = s2.get("skip", False)
    args.stage2_topk              = s2.get("topk", 20)
    args.stage2_topk_pct          = s2.get("topk_pct", None)
    args.stage2_min_shared_bins   = s2.get("min_shared_bins", 0)
    args.stage2_min_shared_az_cols = s2.get("min_shared_az_cols", 3)
    args.stage2_score_threshold   = s2.get("score_threshold", 0.2)

    # Rerank
    rr = cfg.get("rerank", {})
    args.run_combined            = rr.get("run", True)
    args.combined_topk           = rr.get("topk", 10)
    args.combined_scoring_mode   = rr.get("scoring_mode", "jaccard4d")
    args.combined_score_threshold = rr.get("score_threshold", 0.0)

    # Verify
    ver = cfg.get("verify", {})
    args.skip_verify        = ver.get("skip", False)
    args.verify_topv        = ver.get("topv", 5)
    args.verify_ransac_iters = ver.get("ransac_iters", 500)
    args.verify_inlier_dist = ver.get("inlier_dist", 1.0)
    args.verify_min_correspondences  = ver.get("min_correspondences", 32)
    args.verify_min_ransac_inliers   = ver.get("min_ransac_inliers", 32)
    args.verify_min_keypoint_inliers = ver.get("min_keypoint_inliers", 3)
    args.verify_spatial_tol = ver.get("spatial_tol", 0)

    # GICP refinement
    gicp = cfg.get("gicp", {})
    args.skip_gicp                    = gicp.get("skip", True)
    args.gicp_registration_type       = gicp.get("registration_type", "GICP")
    args.gicp_max_corr_dist           = gicp.get("max_correspondence_distance", 1.0)
    args.gicp_downsampling_resolution = gicp.get("downsampling_resolution", 0.25)
    args.gicp_voxel_resolution        = gicp.get("voxel_resolution", 1.0)
    args.gicp_num_threads             = gicp.get("num_threads", 4)
    args.gicp_use_raw_clouds          = gicp.get("use_raw_clouds", True)
    args.gicp_max_iterations          = gicp.get("max_iterations", 64)

    run_evaluation(args)


if __name__ == "__main__":
    main()