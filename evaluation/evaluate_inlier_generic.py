#!/usr/bin/env python3
"""
evaluate_inlier_generic.py

InLiER place recognition evaluation on any folder-based dataset
(flat scans/ directory + poses_kitti.txt or poses_tum.txt, 1:1 with scans).

Identical retrieval/evaluation pipeline to evaluate_inlier_helipr.py; only
the data-loading layer differs. Each input is optionally built by accumulating
--n_db / --n_q consecutive scans into a submap, anchored at the first scan's
pose (keyframe), with configurable stride.

Usage example
-------------
python3 evaluation/evaluate_inlier_generic.py \
    --config config/default.yaml \
    --db_path /path/to/database \
    --q_path  /path/to/query \
    --transform /path/to/transform.txt \
    --overlap_file /path/to/overlap.txt \
    --overlap_threshold 0.2 --max_pose_dist 25.0 \
    --n_db 10 --n_q 10 --stride_db 1 --stride_q 1 \
    --output_dir results/
"""

import argparse, csv, gc, hashlib, json, math, shutil, time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tqdm,yaml

from inlier.core.InLiER import InLiER
from inlier.core.InLiER_Matcher import InLiER_Matcher
from inlier.core.Dataclasses import (
    InLiER_Config, ShortlistConfig, BEAMScoreConfig,
    RerankConfig, VerifyConfig, GICPRefineConfig, GICPRefineOutput,
    InLiER_Tokens, InLiER_Keypoints,
)
from utils.Generic_Handler import Generic_Handler


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


def _fmt_num_for_tag(v: float) -> str:
    s = f"{float(v):.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _build_experiment_subdir(
    db_tag: str,
    q_tag: str,
    n_db: int,
    stride_db: int,
    n_q: int,
    stride_q: int,
    voxel_size: float,
    cell_size: float,
    N_h: int,
    N_r: int,
    N_a: int,
    N_s: int,
) -> str:
    return (
        f"db{db_tag}-n{n_db}s{stride_db}-"
        f"q{q_tag}-n{n_q}s{stride_q}_"
        f"vs{_fmt_num_for_tag(voxel_size)}_"
        f"cs{_fmt_num_for_tag(cell_size)}_"
        f"nh{N_h}_"
        f"nr{N_r}_"
        f"na{N_a}_"
        f"ns{N_s}"
    )


def voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel grid downsampling (numpy centroid per cell).

    Open3d's voxel_down_sample silently collapses the whole cloud to a single
    voxel when any non-finite values are present in the input — this numpy
    implementation filters those points first and is therefore robust.
    """
    if voxel_size <= 0 or pts.shape[0] == 0:
        return pts
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return pts.astype(np.float32)
    coords = np.floor(pts / voxel_size).astype(np.int64)
    _, inv = np.unique(coords, axis=0, return_inverse=True)
    K = int(inv.max()) + 1
    sums = np.zeros((K, 3), dtype=np.float64)
    counts = np.zeros(K, dtype=np.int64)
    np.add.at(sums, inv, pts)
    np.add.at(counts, inv, 1)
    return (sums / counts[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
#  Overlap matrix / ground truth
# ---------------------------------------------------------------------------

def load_overlap_matrix(overlap_file: Path) -> np.ndarray:
    filepath = Path(overlap_file)
    if not filepath.exists():
        raise FileNotFoundError(f"Overlap matrix not found: {filepath}")
    return np.loadtxt(filepath)


# ---------------------------------------------------------------------------
#  DB-pose refinement via GICP (mirrors build_overlap_data._icp_refine_scan)
# ---------------------------------------------------------------------------

def refine_db_poses_gicp(
    handler:             Generic_Handler,
    db_path:             Path,
    n_db:                int,
    stride_db:           int,
    db_poses:            np.ndarray,            # (N_db, 4, 4) – in shared/Q frame
    q_path:              Path,
    n_q:                 int,
    stride_q:            int,
    q_poses:             np.ndarray,            # (N_q, 4, 4)
    voxel_size:          float,
    distance_threshold:  float,
    max_range:           float,
    icp_max_dist:        float,
    num_threads:         int = 4,
    max_iterations:      int = 50,
    cache_path:          Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-DB-submap GICP refinement against a merged local Q cloud.

    This mirrors the `_icp_refine_scan` step in build_overlap_data.py so the
    pose-derived GT transforms used downstream (for max_pose_dist filtering
    and TP pose-error metrics) are consistent with the overlap matrix.

    Returns (refined_db_poses, deltas) with `refined = delta @ original`.
    """
    import small_gicp                        # lazy — only needed when enabled
    from scipy.spatial import cKDTree

    if cache_path is not None and cache_path.exists():
        print(f"  [refine] loading cached deltas → {cache_path.name}")
        npz = np.load(cache_path)
        deltas = npz["deltas"]
        refined = np.einsum("nij,njk->nik", deltas, db_poses)
        return refined, deltas

    def _load_global_voxelized(dataset_dir, n, stride, keyframe_poses):
        """Load submaps, lift to global frame via keyframe pose, voxelize."""
        data = handler.load_generic(dataset_dir, n_scans=n, stride=stride)
        clouds: List[np.ndarray] = []
        for i, pts_local in enumerate(data["point_clouds"]):
            if pts_local.shape[0] == 0:
                clouds.append(pts_local.astype(np.float64))
                continue
            if max_range > 0:
                r = np.linalg.norm(pts_local, axis=1)
                pts_local = pts_local[r <= max_range]
            T = keyframe_poses[i]
            pts_global = (pts_local.astype(np.float64) @ T[:3, :3].T) + T[:3, 3]
            clouds.append(voxel_downsample(pts_global.astype(np.float32), voxel_size)
                          .astype(np.float64))
        return clouds

    print("  [refine] loading/voxelizing DB submaps in shared frame …")
    db_globals = _load_global_voxelized(db_path, n_db, stride_db, db_poses)
    print("  [refine] loading/voxelizing Q  submaps in shared frame …")
    q_globals  = _load_global_voxelized(q_path, n_q, stride_q, q_poses)

    N_db = db_poses.shape[0]
    db_pos = db_poses[:, :3, 3]
    q_pos  = q_poses[:, :3, 3]
    q_tree = cKDTree(q_pos)

    deltas = np.tile(np.eye(4, dtype=np.float64), (N_db, 1, 1))
    n_refined = 0

    for i in tqdm.tqdm(range(N_db), desc="  GICP refining DB poses"):
        if db_globals[i].shape[0] == 0:
            continue
        nearby = q_tree.query_ball_point(db_pos[i], distance_threshold)
        q_parts = [q_globals[j] for j in nearby if q_globals[j].shape[0] > 0]
        if not q_parts:
            continue
        q_merged = np.vstack(q_parts)

        result = small_gicp.align(
            q_merged,
            db_globals[i],
            init_T_target_source=np.eye(4),
            registration_type="GICP",
            downsampling_resolution=voxel_size,
            max_correspondence_distance=icp_max_dist,
            num_threads=num_threads,
            max_iterations=max_iterations,
        )
        deltas[i] = result.T_target_source
        n_refined += 1

    print(f"  [refine] {n_refined}/{N_db} DB submaps refined")

    refined = np.einsum("nij,njk->nik", deltas, db_poses)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, deltas=deltas)
        print(f"  [refine] cached deltas → {cache_path.name}")

    return refined, deltas


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
    handler:    Generic_Handler,
    encoder:    InLiER,
    dataset_dir: Path,
    dataset_tag: str,           # derived folder basename (used in cache filename)
    n_accum:    int,            # scans accumulated per submap
    stride:     int,            # step between successive submaps
    seq_type:   str,            # "Database" or "Query"
    voxel_size: float = 0.0,
    cache_dir:  Optional[Path] = None,
) -> Tuple[List[InLiER_Tokens], List[InLiER_Tokens], List[np.ndarray],
           List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Encode every scan in a sequence with InLiER.

    Cache format: concatenated token_ids + weights + keypoint coordinates
    with per-scan offsets, so variable-length scans are stored efficiently.

    Returns
    -------
    tokens_list    : list of InLiER_Tokens, one per scan (matches cfg.point_mode;
                     used for stage-1/2 retrieval)
    kp_tokens_list : list of InLiER_Tokens always built from keypoints (used by
                     verify so its token indices line up 1:1 with the keypoint
                     array). In keypoints mode this is the same list as
                     tokens_list (alias).
    kp_aligned     : list of (K_i, 3) float64 arrays – ground-aligned keypoint
                     coordinates per scan (for geometric verification)
    kp_sensor      : list of (K_i, 3) float64 arrays – sensor-frame keypoint
                     coordinates per scan (for GICP and pose error evaluation)
    T_grounds      : list of (4, 4) float64 arrays – per-scan ground alignment
                     transforms (sensor → ground-aligned)
    positions      : (N, 3) float64 array – pose translations
    poses          : (N, 4, 4) float64 array – full SE(3) poses
    """
    cfg = encoder.config
    dual_mode = cfg.point_mode.lower() == "all_points"

    # Try loading from cache
    sensor_tag = f"n{n_accum}s{stride}"
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath = _cache_path(cache_dir, dataset_tag, sensor_tag, seq_type, cfg, voxel_size)
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
            # Keypoint tokens (only stored when point_mode=all_points; in
            # keypoints mode they coincide with tokens_list).
            if dual_mode:
                if "kp_token_ids" in npz and "kp_offsets" in npz:
                    all_kp_tids = npz["kp_token_ids"]
                    kp_offsets  = npz["kp_offsets"]
                    kp_tokens_list = [
                        _tokens_from_cache(all_kp_tids[kp_offsets[i]:kp_offsets[i + 1]])
                        for i in range(n_scans)
                    ]
                else:
                    raise RuntimeError(
                        f"[cache] {cpath.name} was built without keypoint "
                        f"tokens but point_mode='all_points' needs them for "
                        f"verify. Delete the cache file and re-encode."
                    )
            else:
                kp_tokens_list = tokens_list
            # In dual_mode the per-scan keypoint counts differ from token
            # counts (tokens = all-points, kp = keypoints), so kp_aligned/
            # kp_sensor must be sliced with kp_offsets, not offsets.
            kp_slice_offsets = kp_offsets if dual_mode else offsets
            # Load cached keypoint coordinates if available
            if "kp_aligned" in npz:
                all_kpa = npz["kp_aligned"]          # (total_K, 3) float64
                kp_aligned = [
                    all_kpa[kp_slice_offsets[i]:kp_slice_offsets[i + 1]]
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
                    all_kps[kp_slice_offsets[i]:kp_slice_offsets[i + 1]]
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
            return tokens_list, kp_tokens_list, kp_aligned, kp_sensor, T_grounds, positions, full_poses

    # Encode from scratch
    data  = handler.load_generic(dataset_dir, n_scans=n_accum, stride=stride)
    poses = data["poses"]
    pcs   = data["point_clouds"]
    K     = len(pcs)

    positions   = np.array([p[:3, 3] for p in poses], dtype=np.float64)
    full_poses  = np.array([p[:4, :4] for p in poses], dtype=np.float64)
    tokens_list:    List[InLiER_Tokens] = []
    kp_tokens_list: List[InLiER_Tokens] = []
    kp_aligned:  List[np.ndarray]    = []
    kp_sensor:   List[np.ndarray]    = []
    T_grounds:   List[np.ndarray]    = []

    for i in tqdm.tqdm(range(K), desc=f"  Encoding {dataset_tag}/{sensor_tag}"):
        pts = pcs[i].astype(np.float32)

        mask = np.any(pts != 0, axis=1)
        pts  = pts[mask]

        if voxel_size > 0:
            pts = voxel_downsample(pts, voxel_size)

        if pts.shape[0] < 10:
            tokens_list.append(_empty_tokens())
            kp_tokens_list.append(_empty_tokens())
            kp_aligned.append(np.zeros((0, 3), dtype=np.float64))
            kp_sensor.append(np.zeros((0, 3), dtype=np.float64))
            T_grounds.append(np.eye(4, dtype=np.float64))
        else:
            kp, tok = encoder.encode(pts, verbose=False)
            tokens_list.append(tok)
            if dual_mode:
                # Build a second token set indexed 1:1 with keypoints, used
                # only by verify (so its correspondence indices are valid).
                kp_tok = encoder.tokenize_keypoints(
                    pts.astype(np.float64), kp, plane=None, verbose=False,
                )
                kp_tokens_list.append(kp_tok)
            else:
                kp_tokens_list.append(tok)
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
        save_kwargs = dict(
            positions  = positions,
            poses      = full_poses,
            offsets    = offsets,
            token_ids  = np.concatenate(tid_list) if len(tid_list) else np.zeros(0, dtype=np.uint32),
            kp_aligned = np.concatenate(kp_aligned) if kp_aligned else np.zeros((0, 3), dtype=np.float64),
            kp_sensor  = np.concatenate(kp_sensor) if kp_sensor else np.zeros((0, 3), dtype=np.float64),
            T_grounds  = np.stack(T_grounds),       # (K, 4, 4)
        )
        if dual_mode:
            kp_tid_list = [t.token_id for t in kp_tokens_list]
            kp_lengths  = np.array([len(t) for t in kp_tid_list], dtype=np.int64)
            kp_offsets_arr = np.concatenate([[0], kp_lengths.cumsum()])
            save_kwargs["kp_token_ids"] = (
                np.concatenate(kp_tid_list) if len(kp_tid_list)
                else np.zeros(0, dtype=np.uint32)
            )
            save_kwargs["kp_offsets"] = kp_offsets_arr
        np.savez_compressed(cpath, **save_kwargs)
        print(f"  [cache] saved {cpath.name}")

    return tokens_list, kp_tokens_list, kp_aligned, kp_sensor, T_grounds, positions, full_poses


# ---------------------------------------------------------------------------
#  Retrieval
# ---------------------------------------------------------------------------

def compute_ranked_lists(
    q_tokens: List[InLiER_Tokens],
    matcher:  InLiER_Matcher,
    N_db:     int,
    db_positions: Optional[np.ndarray] = None,
    q_positions:  Optional[np.ndarray] = None,
    local_radius: float = 0.0,
) -> Tuple[Dict[int, List[int]], Dict[int, Dict[int, float]]]:
    """
    Stage-1: rank all database scans by descending HCC score.

    If ``local_radius > 0`` and positions are provided, the shortlist is
    restricted to DB candidates within ``local_radius`` metres (XY) of the
    query — simulates online place recognition with a local search window.

    Returns
    -------
    ranked_lists   : {query_idx: [db_idx_rank1, db_idx_rank2, ...]}
    similarity_map : {query_idx: {db_idx: score}}
    """
    ranked_lists:   Dict[int, List[int]]        = {}
    similarity_map: Dict[int, Dict[int, float]] = {}

    use_local = (
        local_radius > 0.0
        and db_positions is not None
        and q_positions is not None
    )

    for j in tqdm.tqdm(range(len(q_tokens)), desc="  Stage-1 (MINT) retrieval"):
        s1_out = matcher.shortlist(q_tokens[j], topk=N_db, verbose=False)
        ids    = list(s1_out.ids)
        scores = [float(s) for s in s1_out.scores]

        if use_local:
            dxy = np.linalg.norm(
                db_positions[:, :2] - q_positions[j, :2], axis=1
            )
            in_window = dxy <= local_radius
            kept_ids, kept_scores = [], []
            for sid, sc in zip(ids, scores):
                if 0 <= sid < len(in_window) and in_window[sid]:
                    kept_ids.append(sid)
                    kept_scores.append(sc)
            ids, scores = kept_ids, kept_scores

        ranked_lists[j]   = ids
        similarity_map[j] = {ids[k]: scores[k] for k in range(len(ids))}

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
            # Always populate so the PR curve threshold sweep sees
            # every prediction.  Failed verify → score 0.0.
            q_sims[db_id] = vout.keypoint_inlier_ratio if vout.success else -1.0
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

    If *fallback_ranked_lists* is provided, any query whose current-stage
    candidate list is shorter than the required top-N uses the fallback list
    for that query instead.  This preserves recall that was already achieved
    by an earlier stage when a later reranking stage returns fewer candidates.
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
        When provided, candidates are tried in this retrieval rank order
        instead of sorted by score.  This implements the topV fallback:
        if top-1's score < threshold, try top-2, etc.

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
    auc   = float(np.trapz(precisions[order], recalls[order]))

    return precisions, recalls, auc


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_pr_curves_combined(
    s1_curves:       List[Tuple[str, np.ndarray, np.ndarray, float]],
    save_path:       Path,
    recalls_s2:      Optional[np.ndarray] = None,
    precisions_s2:   Optional[np.ndarray] = None,
    auc_s2:          Optional[float]      = None,
    recalls_comb:    Optional[np.ndarray] = None,
    precisions_comb: Optional[np.ndarray] = None,
    auc_comb:        Optional[float]      = None,
    recalls_ver:     Optional[np.ndarray] = None,
    precisions_ver:  Optional[np.ndarray] = None,
    auc_ver:         Optional[float]      = None,
):
    """Plot Stage-1 scoring variants and optional Stage-2/Combined/Verify PR curves.

    Parameters
    ----------
    s1_curves : list of (label, recalls, precisions, auc) for each MINT scoring.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    active = []
    for label, rec, prec, auc in s1_curves:
        order = np.argsort(rec)
        ax.plot(rec[order], prec[order], linewidth=2.0,
                label=f"{label}  AUC={auc:.4f}")
        active.append(label)

    if recalls_s2 is not None:
        order2 = np.argsort(recalls_s2)
        ax.plot(recalls_s2[order2], precisions_s2[order2],
                linewidth=2.0, linestyle="--", label=f"Stage-2  AUC={auc_s2:.4f}")
        active.append("Stage-2")

    if recalls_comb is not None:
        order_c = np.argsort(recalls_comb)
        ax.plot(recalls_comb[order_c], precisions_comb[order_c],
                linewidth=2.0, linestyle=":", label=f"Combined  AUC={auc_comb:.4f}")
        active.append("Combined")

    if recalls_ver is not None:
        order_v = np.argsort(recalls_ver)
        ax.plot(recalls_ver[order_v], precisions_ver[order_v],
                linewidth=2.0, linestyle="-.", label=f"Verify  AUC={auc_ver:.4f}")
        active.append("Verify")
    title_suffix = " vs ".join(active)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve  ({title_suffix})")
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved PR curve → {save_path}")


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
#  Diagnostics (D1–D5)
# ---------------------------------------------------------------------------

def _rank_of(ranked: List[int], target: int) -> Optional[int]:
    """1-based rank of `target` in `ranked`, or None if absent."""
    for i, x in enumerate(ranked):
        if x == target:
            return i + 1
    return None


def _gt_relative_yaw_deg(T_q: np.ndarray, T_db: np.ndarray) -> float:
    """Yaw of query→DB in DB frame, projected to z-axis (degrees, [-180, 180])."""
    R_rel = T_db[:3, :3].T @ T_q[:3, :3]
    yaw_rad = math.atan2(R_rel[1, 0], R_rel[0, 0])
    return float(math.degrees(yaw_rad))


def _wrap_180(x: float) -> float:
    return float(((x + 180.0) % 360.0) - 180.0)


def save_per_query_trace(
    output_path:        Path,
    gt:                 Dict[int, np.ndarray],
    ranked_lists_s1:    Dict[int, List[int]],
    similarity_map_s1:  Dict[int, Dict[int, float]],
    ranked_lists_s2:    Optional[Dict[int, List[int]]],
    similarity_map_s2:  Optional[Dict[int, Dict[int, float]]],
    ranked_lists_ver:   Optional[Dict[int, List[int]]],
    similarity_map_ver: Optional[Dict[int, Dict[int, float]]],
):
    """D1: one row per (query, gt_positive) with rank+score at each stage."""
    fields = [
        "query_idx", "gt_db_idx", "n_gt_positives",
        "rank_s1", "score_s1",
        "rank_s2", "score_s2",
        "rank_verify", "score_verify",
        "top1_s1", "top1_score_s1",
        "top1_s2", "top1_score_s2",
        "top1_verify", "top1_score_verify",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for j, gt_arr in gt.items():
            if gt_arr.size == 0:
                continue
            s1_list = ranked_lists_s1.get(j, [])
            s2_list = ranked_lists_s2.get(j, []) if ranked_lists_s2 is not None else []
            ver_list = ranked_lists_ver.get(j, []) if ranked_lists_ver is not None else []
            top1_s1  = s1_list[0] if s1_list else -1
            top1_s2  = s2_list[0] if s2_list else -1
            top1_ver = ver_list[0] if ver_list else -1
            for gt_d in gt_arr.tolist():
                row = {
                    "query_idx": j,
                    "gt_db_idx": int(gt_d),
                    "n_gt_positives": int(gt_arr.size),
                    "rank_s1":  _rank_of(s1_list, gt_d),
                    "score_s1": similarity_map_s1.get(j, {}).get(gt_d),
                    "rank_s2":  _rank_of(s2_list, gt_d) if ranked_lists_s2 is not None else None,
                    "score_s2": (similarity_map_s2.get(j, {}).get(gt_d)
                                 if similarity_map_s2 is not None else None),
                    "rank_verify":  _rank_of(ver_list, gt_d) if ranked_lists_ver is not None else None,
                    "score_verify": (similarity_map_ver.get(j, {}).get(gt_d)
                                     if similarity_map_ver is not None else None),
                    "top1_s1": int(top1_s1),
                    "top1_score_s1": (similarity_map_s1.get(j, {}).get(top1_s1)
                                      if top1_s1 >= 0 else None),
                    "top1_s2": int(top1_s2) if top1_s2 != -1 else None,
                    "top1_score_s2": (similarity_map_s2.get(j, {}).get(top1_s2)
                                      if (similarity_map_s2 is not None and top1_s2 != -1) else None),
                    "top1_verify": int(top1_ver) if top1_ver != -1 else None,
                    "top1_score_verify": (similarity_map_ver.get(j, {}).get(top1_ver)
                                          if (similarity_map_ver is not None and top1_ver != -1) else None),
                }
                w.writerow(row)
    print(f"  Per-query trace → {output_path}")


def save_per_pair_verify(
    output_path:    Path,
    verify_outputs: Dict[Tuple[int, int], Any],
    gt:             Dict[int, np.ndarray],
    shifts_map_s2:  Optional[Dict[int, Dict[int, int]]],
    N_a:            int,
    q_poses:        np.ndarray,
    db_poses:       np.ndarray,
):
    """D2 + D5: per-pair verify diagnostics with optional yaw-error against GT."""
    fields = [
        "query_idx", "db_idx", "is_gt_positive",
        "success",
        "n_correspondences", "n_ransac_inliers", "n_keypoint_inliers",
        "n_total_keypoints", "ransac_inlier_ratio", "keypoint_inlier_ratio",
        "inlier_rmse",
        "yaw_aligned_deg", "tx", "ty", "tz",
        "beam_shift", "beam_shift_yaw_deg",
        "gt_yaw_deg", "verify_yaw_err_deg", "beam_yaw_err_deg",
    ]
    deg_per_shift = 360.0 / float(max(1, N_a))
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (j, d), vout in verify_outputs.items():
            gt_arr = gt.get(j, np.zeros(0, dtype=np.int64))
            is_gt = bool(int(d) in set(gt_arr.tolist())) if gt_arr.size else False
            shift = None
            if shifts_map_s2 is not None and j in shifts_map_s2:
                shift = shifts_map_s2[j].get(d)
            beam_yaw = (shift * deg_per_shift) if shift is not None else None
            gt_yaw  = _gt_relative_yaw_deg(q_poses[j], db_poses[d]) if is_gt else None
            ver_yaw = math.degrees(float(vout.yaw)) if vout.success else None
            w.writerow({
                "query_idx": j,
                "db_idx": int(d),
                "is_gt_positive": is_gt,
                "success": bool(vout.success),
                "n_correspondences":    int(vout.n_correspondences),
                "n_ransac_inliers":     int(vout.n_ransac_inliers),
                "n_keypoint_inliers":   int(vout.n_keypoint_inliers),
                "n_total_keypoints":    int(vout.n_total_keypoints),
                "ransac_inlier_ratio":  round(float(vout.ransac_inlier_ratio), 6),
                "keypoint_inlier_ratio": round(float(vout.keypoint_inlier_ratio), 6),
                "inlier_rmse":          round(float(vout.inlier_rmse), 6),
                "yaw_aligned_deg":      round(ver_yaw, 4) if ver_yaw is not None else None,
                "tx": round(float(vout.tx), 4),
                "ty": round(float(vout.ty), 4),
                "tz": round(float(vout.tz), 4),
                "beam_shift": shift,
                "beam_shift_yaw_deg": round(beam_yaw, 4) if beam_yaw is not None else None,
                "gt_yaw_deg":         round(gt_yaw, 4) if gt_yaw is not None else None,
                "verify_yaw_err_deg": (round(_wrap_180(ver_yaw - gt_yaw), 4)
                                       if (ver_yaw is not None and gt_yaw is not None) else None),
                "beam_yaw_err_deg":   (round(_wrap_180(beam_yaw - gt_yaw), 4)
                                       if (beam_yaw is not None and gt_yaw is not None) else None),
            })
    print(f"  Per-pair verify → {output_path}")


def save_score_distributions(
    output_path:    Path,
    similarity_map: Dict[int, Dict[int, float]],
    gt:             Dict[int, np.ndarray],
    rank_order:     Optional[Dict[int, List[int]]],
    n_queries:      int,
    pr_threshold:   float,
):
    """D3: dump TP/FP/FN/TN top-1 scores + all GT-positive vs GT-negative pair scores."""
    top1_scores = {"TP": [], "FP": [], "FN": [], "TN": []}
    pair_scores_gt_pos: List[float] = []
    pair_scores_gt_neg: List[float] = []
    for j in range(n_queries):
        gt_set = set(gt.get(j, np.zeros(0)).tolist())
        q_sims = similarity_map.get(j, {})
        if rank_order is not None and j in rank_order:
            ordered = [d for d in rank_order[j] if d in q_sims]
            seen = set(ordered)
            tail = sorted((d for d in q_sims if d not in seen),
                          key=lambda d: q_sims[d], reverse=True)
            ranked_d = ordered + tail
        else:
            ranked_d = sorted(q_sims, key=lambda d: q_sims[d], reverse=True)

        top1 = next((d for d in ranked_d if q_sims[d] >= pr_threshold), None)
        top1_score = float(q_sims[top1]) if top1 is not None else 0.0
        if gt_set:
            if top1 is None:
                top1_scores["FN"].append(top1_score)
            elif top1 in gt_set:
                top1_scores["TP"].append(top1_score)
            else:
                top1_scores["FP"].append(top1_score)
        else:
            if top1 is None:
                top1_scores["TN"].append(top1_score)
            else:
                top1_scores["FP"].append(top1_score)

        for d, sc in q_sims.items():
            if sc < 0.0:
                continue
            (pair_scores_gt_pos if d in gt_set else pair_scores_gt_neg).append(float(sc))

    def _stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"n": 0}
        a = np.asarray(arr, dtype=np.float64)
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)),
        }

    payload = {
        "threshold_used": float(pr_threshold),
        "top1_scores_by_match_type": {k: v for k, v in top1_scores.items()},
        "top1_stats_by_match_type":  {k: _stats(v) for k, v in top1_scores.items()},
        "all_pair_scores_gt_positive": pair_scores_gt_pos,
        "all_pair_scores_gt_negative": pair_scores_gt_neg,
        "all_pair_stats_gt_positive":  _stats(pair_scores_gt_pos),
        "all_pair_stats_gt_negative":  _stats(pair_scores_gt_neg),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Score distributions → {output_path}")


def save_threshold_sweep(
    output_path:    Path,
    similarity_map: Dict[int, Dict[int, float]],
    gt:             Dict[int, np.ndarray],
    n_queries:      int,
    rank_order:     Optional[Dict[int, List[int]]],
) -> Dict[str, float]:
    """D4: sweep threshold, write CSV, return best-F1 operating point."""
    thresholds = np.arange(0.0, 1.001, 0.01)
    rows = []
    best = {"threshold": 0.0, "f1": -1.0, "precision": 0.0, "recall": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for thr in thresholds:
        _, _, tp, fp, fn, tn = _build_tp_fp_edges(
            similarity_map, gt, float(thr), n_queries, rank_order=rank_order
        )
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        rows.append({
            "threshold": round(float(thr), 4),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(prec, 6),
            "recall":    round(rec, 6),
            "f1":        round(f1, 6),
        })
        if f1 > best["f1"]:
            best = {"threshold": float(thr), "f1": float(f1),
                    "precision": float(prec), "recall": float(rec),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "TP", "FP", "FN", "TN",
                                          "precision", "recall", "f1"])
        w.writeheader()
        w.writerows(rows)
    print(f"  Threshold sweep → {output_path}  "
          f"(best F1={best['f1']:.4f} @ thr={best['threshold']:.3f})")
    return best


# ---------------------------------------------------------------------------
#  Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(args):
    db_path = Path(args.db_path).resolve()
    q_path  = Path(args.q_path).resolve()
    overlap_file = Path(args.overlap_file).resolve()
    output_root  = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir    = Path(args.cache_dir) if args.cache_dir else None

    db_tag = db_path.name
    q_tag  = q_path.name

    stride_db = args.stride_db if args.stride_db is not None else args.n_db
    stride_q  = args.stride_q  if args.stride_q  is not None else args.n_q

    # Inter-sequence transform (DB world → Q world).
    # Matches build_overlap_data.py: auto-detect <db_path>/transform.txt, or
    # accept an explicit --transform path, or --no_transform to disable.
    inter_transform: Optional[np.ndarray] = None
    transform_source = "none (shared frame)"
    if not args.no_transform:
        if args.transform is not None:
            tp = Path(args.transform)
        else:
            tp = db_path / "transform.txt"
            if not tp.exists():
                tp = None
        if tp is not None:
            if not tp.exists():
                raise FileNotFoundError(f"--transform file not found: {tp}")
            inter_transform = np.loadtxt(tp)
            if inter_transform.shape != (4, 4):
                raise ValueError(f"transform must be 4x4, got {inter_transform.shape}")
            transform_source = str(tp)

    exp_subdir = _build_experiment_subdir(
        db_tag=db_tag,
        q_tag=q_tag,
        n_db=args.n_db,
        stride_db=stride_db,
        n_q=args.n_q,
        stride_q=stride_q,
        voxel_size=args.voxel_size,
        cell_size=2 * args.voxel_size,
        N_h=args.N_h,
        N_r=args.N_r,
        N_a=args.N_a,
        N_s=args.N_s,
    )
    output_dir = output_root / exp_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_tag = (f"{db_tag}_n{args.n_db}s{stride_db}_"
                f"{q_tag}_n{args.n_q}s{stride_q}_"
                f"ov{args.overlap_threshold}_pd{args.max_pose_dist}m")
    if float(getattr(args, "local_radius", 0.0)) > 0.0:
        pair_tag += f"_lr{args.local_radius}m"

    print("=" * 70)
    print("  InLiER Place Recognition Evaluation  (Generic dataset)")
    print(f"  DB: {db_path}  (n={args.n_db}, stride={stride_db})")
    print(f"  Q : {q_path}   (n={args.n_q}, stride={stride_q})")
    print(f"  Overlap file     : {overlap_file}")
    print(f"  Inter-seq transform: {transform_source}")
    print(f"  Overlap threshold: {args.overlap_threshold}")
    print(f"  Max XY pose dist : {args.max_pose_dist} m")
    print(f"  Output folder    : {output_dir}")
    print("=" * 70)

    # 1. Build InLiER encoder
    print("\n[1/5] Building InLiER encoder …")
    min_intens = 0.5 * (args.z_max - args.z_min)/args.N_h
    inlier_cfg = InLiER_Config(
        cell_size           = 2 * args.voxel_size,
        z_min               = args.z_min,
        z_max               = args.z_max,
        xy_max              = args.xy_max,
        N_h                 = args.N_h,
        window              = args.window,
        max_kp_per_slice    = 256,
        max_kp_total        = 1280,
        ransac_iters        = 250,
        ransac_dist_thresh  = 2 * args.voxel_size,
        ransac_min_inliers  = 200,
        point_mode          = args.point_mode,
        r_max               = args.r_max,
        N_r                 = args.N_r,
        N_a                 = args.N_a,
        shape_radius        = 3.0 * args.voxel_size,
        shape_min_neighbors = 8,
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
    overlap = load_overlap_matrix(overlap_file)
    print(f"  Shape: {overlap.shape}  (DB={overlap.shape[0]}, Q={overlap.shape[1]})")

    # 3. Encode sequences
    print("\n[3/5] Loading dataset and encoding scans …")

    for label, path in [("DB", db_path), ("Q", q_path)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} path not found: {path}")
        if not (path / "scans").exists():
            raise FileNotFoundError(
                f"{label} dataset at {path} is missing a 'scans/' subdirectory."
            )
        if not ((path / "poses_kitti.txt").exists() or (path / "poses_tum.txt").exists()):
            raise FileNotFoundError(
                f"{label} dataset at {path} is missing poses_kitti.txt or poses_tum.txt."
            )

    handler = Generic_Handler(verbose=False)

    t0 = time.time()
    print(f"\n  Encoding database ({db_tag}, n={args.n_db}, stride={stride_db}) …")
    (db_tokens, db_kp_tokens, db_kp_aligned, db_kp_sensor,
     db_T_grounds, db_positions, db_poses) = encode_sequence(
        handler, encoder, db_path, db_tag,
        n_accum=args.n_db, stride=stride_db, seq_type="Database",
        voxel_size=args.voxel_size, cache_dir=cache_dir,
    )

    print(f"\n  Encoding query ({q_tag}, n={args.n_q}, stride={stride_q}) …")
    (q_tokens, q_kp_tokens, q_kp_aligned, q_kp_sensor,
     q_T_grounds, q_positions, q_poses) = encode_sequence(
        handler, encoder, q_path, q_tag,
        n_accum=args.n_q, stride=stride_q, seq_type="Query",
        voxel_size=args.voxel_size, cache_dir=cache_dir,
    )
    encode_time = time.time() - t0
    print(f"\n  Encoding done in {encode_time:.1f} s "
          f"(DB={len(db_tokens)}, Q={len(q_tokens)})")

    # Apply inter-sequence transform to DB keyframe poses so DB/Q positions
    # live in the same frame as the overlap matrix was built in. Submap points
    # are in the keyframe's local frame, so they are invariant under this
    # left-multiplication and do not need re-encoding.
    if inter_transform is not None:
        db_poses = np.einsum("ij,njk->nik", inter_transform, db_poses)
        db_positions = db_poses[:, :3, 3].copy()

    # Optional: per-DB-submap GICP refinement (matches build_overlap --icp).
    # Only affects pose-derived quantities: max_pose_dist filter and TP
    # pose-error metrics. Retrieval is frame-invariant and unaffected.
    if args.refine_db_poses:
        print("\n  Refining DB keyframe poses via GICP "
              "(matches build_overlap_data.py --icp) …")
        rkey_parts = [
            db_tag, f"n{args.n_db}s{stride_db}",
            q_tag, f"n{args.n_q}s{stride_q}",
            f"vs{args.refine_voxel_size}",
            f"dt{args.refine_distance_threshold}",
            f"mr{args.refine_max_range}",
            f"im{args.refine_icp_max_dist}",
        ]
        rkey = "_".join(rkey_parts)
        refine_cache = (cache_dir / f"refine_deltas_{rkey}.npz") if cache_dir else None
        db_poses, _deltas = refine_db_poses_gicp(
            handler=handler,
            db_path=db_path, n_db=args.n_db, stride_db=stride_db, db_poses=db_poses,
            q_path=q_path,  n_q=args.n_q,  stride_q=stride_q,  q_poses=q_poses,
            voxel_size=args.refine_voxel_size,
            distance_threshold=args.refine_distance_threshold,
            max_range=args.refine_max_range,
            icp_max_dist=args.refine_icp_max_dist,
            cache_path=refine_cache,
        )
        db_positions = db_poses[:, :3, 3].copy()

    # Align overlap matrix dimensions
    if overlap.shape[0] != len(db_tokens) or overlap.shape[1] != len(q_tokens):
        print(f"\n  WARNING: overlap matrix {overlap.shape} ≠ scans "
              f"(DB={len(db_tokens)}, Q={len(q_tokens)}).  Truncating.")
        n_db = min(overlap.shape[0], len(db_tokens))
        n_q  = min(overlap.shape[1], len(q_tokens))
        overlap        = overlap[:n_db, :n_q]
        db_tokens      = db_tokens[:n_db]
        q_tokens       = q_tokens[:n_q]
        db_kp_tokens   = db_kp_tokens[:n_db]
        q_kp_tokens    = q_kp_tokens[:n_q]
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

    # ── 3b. Optional DB pruning by overlap with queries ──────────────────
    # Drop DB submaps that have no meaningful overlap with any query — they
    # cannot be a TP for any query, and only inflate FP counts and runtime.
    db_orig_idx = np.arange(len(db_tokens), dtype=np.int64)
    if args.db_overlap_filter > 0.0:
        thr = float(args.db_overlap_filter)
        db_max_ov = overlap.max(axis=1)  # per-DB best overlap across queries
        keep = db_max_ov > thr
        n_kept = int(keep.sum())
        n_total = len(db_tokens)
        print(f"\n  DB overlap filter: keeping {n_kept}/{n_total} DB submaps "
              f"(max overlap > {thr})")
        if n_kept == 0:
            print("\n  ERROR: db_overlap_filter removed every DB submap. "
                  "Lower the threshold or disable the filter.")
            return
        keep_idx = np.where(keep)[0]
        overlap        = overlap[keep_idx, :]
        db_tokens      = [db_tokens[i] for i in keep_idx]
        db_kp_tokens   = [db_kp_tokens[i] for i in keep_idx]
        db_kp_aligned  = [db_kp_aligned[i] for i in keep_idx]
        db_kp_sensor   = [db_kp_sensor[i] for i in keep_idx]
        db_T_grounds   = [db_T_grounds[i] for i in keep_idx]
        db_positions   = db_positions[keep_idx]
        db_poses       = db_poses[keep_idx]
        db_orig_idx    = db_orig_idx[keep_idx]

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
        db_positions=db_positions,
        q_positions=q_positions,
        local_radius=float(getattr(args, "local_radius", 0.0)),
    )
    t_s1 = time.time() - t0
    print(f"  Stage-1 ({shortlist_cfg.mint_mode}/{shortlist_cfg.mint_scoring}) "
          f"done in {t_s1:.1f} s "
          f"({t_s1 / max(1, len(q_tokens)) * 1000:.1f} ms/query)")

    # Stage-2 (BEAM) reranking 
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

    # Rerank stage
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

    # Metrics – Stage-1
    n_values_s1 = [1, 5, 10, 20, 50, 100]
    k_pcts      = [1.0, 5.0, 10.0]
    thr_s1      = np.arange(0.0, 1.001, 0.001)

    recalls_s1      = compute_recall_at_n(ranked_lists_s1, gt, n_values_s1)
    recalls_kpct_s1 = compute_recall_at_kpct(ranked_lists_s1, gt, len(db_tokens), k_pcts)
    prec_s1, rec_s1, auc_s1 = compute_pr_curve(similarity_map_s1, gt, thr_s1)

    # Metrics – Stage-2 
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

    # Metrics – Combined / Rerank
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

    # Metrics – Verify (geometric verification on top-V) 
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
            matcher, q_kp_tokens, q_kp_aligned, db_kp_tokens, db_kp_aligned,
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

    # ── Confusion matrix at auto/manual threshold ─────────────────────────
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

        # When use_raw_clouds=True, rebuild the submaps needed by GICP.
        # (In the generic pipeline the encoder input is a submap, not a raw
        # scan — we must feed GICP the same accumulated point cloud.)
        raw_clouds_q:  Dict[int, np.ndarray] = {}
        raw_clouds_db: Dict[int, np.ndarray] = {}
        if gicp_cfg.use_raw_clouds:
            needed_q  = set(j for j, d in tp_edges
                           if verify_outputs.get((j, d), None) is not None
                           and verify_outputs[(j, d)].success)
            needed_db = set(d for j, d in tp_edges
                           if verify_outputs.get((j, d), None) is not None
                           and verify_outputs[(j, d)].success)
            # Rebuild both full submap sequences (cheap relative to GICP),
            # then pull out the ones we need.
            data_db = handler.load_generic(db_path, n_scans=args.n_db, stride=stride_db)
            data_q  = handler.load_generic(q_path,  n_scans=args.n_q,  stride=stride_q)
            for idx in tqdm.tqdm(sorted(needed_db), desc="  Loading DB submaps"):
                pts = np.asarray(data_db["point_clouds"][idx], dtype=np.float32)
                pts = pts[np.any(pts != 0, axis=1)]
                if args.voxel_size > 0:
                    pts = voxel_downsample(pts, args.voxel_size)
                raw_clouds_db[idx] = pts
            for idx in tqdm.tqdm(sorted(needed_q), desc="  Loading Q submaps"):
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

    # Save results JSON 
    results = {
        "config": {
            "db_path":           str(db_path),
            "q_path":            str(q_path),
            "overlap_file":      str(overlap_file),
            "n_db":              args.n_db,
            "stride_db":         stride_db,
            "n_q":               args.n_q,
            "stride_q":          stride_q,
            "overlap_threshold": args.overlap_threshold,
            "max_pose_dist":     args.max_pose_dist,
            "local_radius":      float(getattr(args, "local_radius", 0.0)),
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
    # ── Per-sequence keypoint / storage statistics ───────────────────────
    def _kp_storage_stats(kp_list, tok_list):
        n_kp = np.array([int(kp.shape[0]) for kp in kp_list], dtype=np.int64)
        kp_bytes  = np.array([int(kp.nbytes) for kp in kp_list], dtype=np.int64)
        tok_bytes = np.array([int(t.token_id.nbytes) for t in tok_list], dtype=np.int64)
        kt_bytes  = kp_bytes + tok_bytes
        def _stats(a):
            if a.size == 0:
                return {"min": 0, "max": 0, "mean": 0.0}
            return {"min": int(a.min()), "max": int(a.max()),
                    "mean": round(float(a.mean()), 3)}
        return {
            "n_scans": int(len(kp_list)),
            "keypoints":                 _stats(n_kp),
            "token_bytes":               _stats(tok_bytes),
            "keypoint_plus_token_bytes": _stats(kt_bytes),
        }

    results["sequence_stats"] = {
        "database": _kp_storage_stats(db_kp_aligned, db_kp_tokens),
        "query":    _kp_storage_stats(q_kp_aligned,  q_kp_tokens),
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

    # Diagnostics (D1–D5) 
    print("\n  Writing diagnostics …")
    save_per_query_trace(
        output_dir / f"per_query_trace_{pair_tag}.csv",
        gt,
        ranked_lists_s1,  similarity_map_s1,
        ranked_lists_s2,  similarity_map_s2,
        ranked_lists_ver if 'ranked_lists_ver' in locals() else None,
        similarity_map_ver,
    )
    if verify_outputs:
        save_per_pair_verify(
            output_dir / f"per_pair_verify_{pair_tag}.csv",
            verify_outputs, gt, shifts_map_s2, args.N_a, q_poses, db_poses,
        )
    save_score_distributions(
        output_dir / f"score_distributions_{pair_tag}.json",
        conf_sim_map, gt, conf_rank_order, len(q_tokens), pr_threshold,
    )
    best_f1 = save_threshold_sweep(
        output_dir / f"threshold_sweep_{pair_tag}.csv",
        conf_sim_map, gt, len(q_tokens), conf_rank_order,
    )

    # Inject best-F1 operating point into results JSON.
    results["confusion"]["best_f1"] = {
        "threshold": round(best_f1["threshold"], 4),
        "f1":        round(best_f1["f1"], 6),
        "precision": round(best_f1["precision"], 6),
        "recall":    round(best_f1["recall"], 6),
        "TP": best_f1["tp"], "FP": best_f1["fp"],
        "FN": best_f1["fn"], "TN": best_f1["tn"],
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Copy config YAML into the output folder
    config_dest = output_dir / Path(args.config_path).name
    shutil.copy2(args.config_path, config_dest)
    print(f"  Config copied → {config_dest}")

    # Plots 
    s1_label = f"Stage-1 ({shortlist_cfg.mint_mode}/{shortlist_cfg.mint_scoring})"
    s1_curves = [(s1_label, rec_s1, prec_s1, auc_s1)]
    plot_pr_curves_combined(
        s1_curves,
        save_path=output_dir / f"pr_curve_{pair_tag}.png",
        recalls_s2=rec_s2,
        precisions_s2=prec_s2,
        auc_s2=auc_s2,
        recalls_comb=rec_comb,
        precisions_comb=prec_comb,
        auc_comb=auc_comb,
        recalls_ver=rec_ver,
        precisions_ver=prec_ver,
        auc_ver=auc_ver,
    )

    plot_trajectory_3d(
        db_positions, q_positions,
        tp_edges, fp_edges,
        title=(
            f"InLiER {conf_stage}  {db_tag} → {q_tag}\n"
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
        description="Evaluate InLiER place recognition on a generic folder-based dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (encoder + matcher params)
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file with InLiER encoder/matcher parameters.")

    # Dataset / IO
    parser.add_argument("--db_path", type=str, required=True,
                        help="Path to the database dataset folder "
                             "(must contain scans/ and poses_kitti.txt or poses_tum.txt).")
    parser.add_argument("--q_path", type=str, required=True,
                        help="Path to the query dataset folder (same layout as --db_path).")
    parser.add_argument("--overlap_file", type=str, required=True,
                        help="Path to the precomputed overlap matrix .txt "
                             "(shape must match submap counts after accumulation).")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Directory to write results and plots.")

    # Submap accumulation
    parser.add_argument("--n_db", type=int, default=1,
                        help="Number of consecutive DB scans accumulated into one submap.")
    parser.add_argument("--n_q", type=int, default=1,
                        help="Number of consecutive query scans accumulated into one submap.")
    parser.add_argument("--stride_db", type=int, default=None,
                        help="Step between DB submaps (default: n_db = non-overlapping).")
    parser.add_argument("--stride_q", type=int, default=None,
                        help="Step between query submaps (default: n_q = non-overlapping).")

    # Inter-sequence transform (DB world -> Q world)
    parser.add_argument("--transform", type=str, default=None,
                        help="Path to a 4x4 transform.txt mapping DB world frame "
                             "into Q world frame. Defaults to <db_path>/transform.txt "
                             "if it exists.")
    parser.add_argument("--no_transform", action="store_true",
                        help="Disable the inter-sequence transform even if "
                             "<db_path>/transform.txt exists (both sequences "
                             "already share a frame).")

    # Optional per-DB-submap GICP refinement (matches build_overlap_data --icp)
    parser.add_argument("--refine_db_poses", action="store_true",
                        help="Run per-DB-submap GICP against local Q clouds to "
                             "correct residual inter-session pose misalignment. "
                             "Use the SAME values used when building the overlap "
                             "matrix (voxel_size, distance_threshold, icp_max_dist). "
                             "Only affects pose-derived metrics (max_pose_dist filter, "
                             "TP pose errors); retrieval is unaffected.")
    parser.add_argument("--refine_voxel_size", type=float, default=0.5,
                        help="Voxel size for GICP refinement (match build_overlap).")
    parser.add_argument("--refine_distance_threshold", type=float, default=100.0,
                        help="Max pose distance (m) for GICP local-Q merge "
                             "(match build_overlap --distance_threshold).")
    parser.add_argument("--refine_icp_max_dist", type=float, default=1.5,
                        help="GICP max correspondence distance "
                             "(match build_overlap --icp_max_dist).")
    parser.add_argument("--refine_max_range", type=float, default=100.0,
                        help="Max point range from sensor origin "
                             "(match build_overlap --max_range).")

    # DB pruning (build the matcher index only from candidates that share
    # overlap with at least one query). Reduces the search space and stops
    # the eval from scoring DB submaps that can never be a true positive.
    parser.add_argument("--db_overlap_filter", type=float, default=0.0,
                        help="If > 0, drop DB submaps whose maximum overlap "
                             "across all queries is <= this threshold. The "
                             "matcher index, GT, and all downstream metrics "
                             "use the pruned DB. 0.0 disables the filter "
                             "(default; full DB is searched).")

    # Ground-truth parameters
    parser.add_argument("--overlap_threshold", type=float, default=0.3,
                        help="Min overlap to consider a pair a GT positive.")
    parser.add_argument("--max_pose_dist", type=float, default=25.0,
                        help="Max XY pose distance (m) to count as GT positive "
                             "(0 = no limit).")
    parser.add_argument("--local_radius", type=float, default=0.0,
                        help="Online place-recognition mode: only consider DB "
                             "submaps within this XY distance (m) of the query "
                             "during Stage-1 shortlist. 0 = global search "
                             "(default).")

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
    args.window         = enc.get("window", 3)

    # Stage-1
    s1 = cfg.get("stage1", {})
    args.stage1_topk     = s1.get("topk", 100)
    args.stage1_topk_pct = s1.get("topk_pct", None)
    args.min_shared_rows = s1.get("min_shared_rows", 3)
    args.mint_mode       = s1.get("mint_mode", "compact")
    args.mint_scoring    = s1.get("mint_scoring", "cosine")

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