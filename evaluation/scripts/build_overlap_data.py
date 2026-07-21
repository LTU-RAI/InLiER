#!/usr/bin/env python3
"""
build_overlap_data.py

Compute the HeLiOS overlap matrix between a database sequence and a query
sequence.  Produces an NxM text file where N = #database scans, M = #query scans.

Overlap formula follows HeLiOS (Jung et al., 2025, Eq. 3):

    Ô(P1, P2) = 2 * Σ 𝟙(NN(P1_i, P2) < τ) / (N1 + N2)
    O(P1, P2) = max(Ô(P1, P2), Ô(P2, P1)),  clamped to [0, 1]

Point clouds are voxelized with voxel size δ before overlap computation.
τ = 1.5 * δ.  Only scan pairs whose poses are within a distance threshold
(default 100 m) receive non-zero overlap; all others are set to 0.

Memory-efficient: poses are loaded first to build a distance matrix and
identify valid pairs, then scans are loaded/voxelized in blocks of DB
indices so only a small subset of point clouds lives in memory at once.

Usage — HeLiPR:
  python3 evaluation/scripts/build_overlap_data.py \
    --dataset_type helipr \
    --dataset ~/Documents/datasets/HeLiPR/ \
    --db_sequence Roundabout01 --q_sequence Roundabout03 \
    --output_dir overlap_matrices --pairs O-Aeva \
    --voxel_size 0.5 --distance_threshold 100 --block_size 200

Usage — custom (.pcd + poses_kitti.txt):
  python3 evaluation/scripts/build_overlap_data.py \
    --dataset_type custom \
    --db_path ~/Documents/datasets/campus_ouster \
    --q_path  ~/Documents/datasets/campus_robinw \
    --transform ~/Documents/datasets/campus_ouster/T_robinw_ouster.txt \
    --voxel_size 0.5 --distance_threshold 100 \
    --icp --icp_max_dist 1.5                                           
"""

import argparse
import gc
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import tqdm
import sys
import time
import open3d as o3d
import small_gicp

# Run directly (python3 evaluation/scripts/build_overlap_data.py), so sys.path[0]
# is this folder; put evaluation/ on the path to reach the shared handlers.
_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from utils.HeLiPR_Handler import HeLiPR_Handler


# ---------------------------------------------------------------------------
#  HeLiOS overlap helpers
# ---------------------------------------------------------------------------

def voxelize(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel-downsample using Open3D's voxel_down_sample."""
    if points.shape[0] == 0:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd_down.points)


def transform_to_global(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply a 4x4 pose to Nx3 points, returning Nx3 in the global frame."""
    pts_h = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float64)])
    return (pose @ pts_h.T).T[:, :3]


# ---------------------------------------------------------------------------
#  HeLiPR-specific loading
# ---------------------------------------------------------------------------

def _load_and_voxelize(handler, bin_files, sensor, scan_type, poses,
                       voxel_size, max_range):
    """Load N HeLiPR scans, transform each to global with its own pose,
    concatenate and voxelize once.  N=1 recovers the original single-scan
    behavior.
    """
    all_pts = []
    for bin_file, pose in zip(bin_files, poses):
        pts, _ = handler.load_scan_file(bin_file, sensor, type=scan_type)
        pts = pts.astype(np.float64)
        if max_range > 0 and pts.shape[0] > 0:
            r = np.linalg.norm(pts, axis=1)
            pts = pts[r <= max_range]
        if pts.shape[0] > 0:
            all_pts.append(transform_to_global(pts, pose))
    if not all_pts:
        return np.empty((0, 3)), None, 0
    combined = np.vstack(all_pts)
    pts_v = voxelize(combined, voxel_size)
    n_vox = pts_v.shape[0]
    tree = cKDTree(pts_v) if n_vox > 0 else None
    return pts_v, tree, n_vox


def _load_sequence_metadata(handler, sequence, sensor, scan_type):
    """Return per-scan positions, associated poses, and bin file list (HeLiPR)."""
    seq_path = handler.dataset_path / sequence
    all_poses, pose_ts = handler.load_poses(seq_path, sensor)
    bin_files = handler.list_scan_files(seq_path, sensor, type=scan_type)
    bin_ts = HeLiPR_Handler._bin_timestamps(bin_files)

    positions = np.zeros((len(bin_files), 3))
    scan_poses = []
    for k, ts in enumerate(bin_ts):
        idx = handler.get_closest_pose_index(ts, pose_ts)
        positions[k] = all_poses[idx][:3, 3]
        scan_poses.append(all_poses[idx])

    return positions, scan_poses, bin_files


# ---------------------------------------------------------------------------
#  Custom dataset loading (pcd + poses_kitti.txt)
# ---------------------------------------------------------------------------

def _load_kitti_poses(poses_file: Path) -> list:
    """Read KITTI-format poses: 12 values per line → list of 4x4 np.ndarray."""
    poses = []
    with open(poses_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 12:
                T = np.eye(4)
                T[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(T)
    return poses


def _load_custom_sequence_metadata(seq_path: Path, inter_transform: np.ndarray = None):
    """Load poses and scan list for a custom dataset directory.

    seq_path       – directory with scans/ subdirectory and poses_kitti.txt
    inter_transform – 4x4 matrix that maps this sequence's world frame into the
                      common reference frame (e.g. T_robinw_ouster for the DB).
                      Pass None for the sequence that defines the reference frame.

    Returns
    -------
    positions       : np.ndarray (K, 3)  – scan positions in reference frame
    effective_poses : list[np.ndarray]   – 4x4 poses in reference frame
    scan_files      : list[Path]
    """
    poses = _load_kitti_poses(seq_path / "poses_kitti.txt")
    scan_files = sorted((seq_path / "scans").glob("*.pcd"))

    if len(scan_files) != len(poses):
        raise ValueError(
            f"{seq_path}: {len(scan_files)} scans vs {len(poses)} poses"
        )

    if inter_transform is not None:
        effective_poses = [inter_transform @ p for p in poses]
    else:
        effective_poses = poses

    positions = np.array([p[:3, 3] for p in effective_poses])
    return positions, effective_poses, scan_files


def _load_and_voxelize_pcd(pcd_files, poses, voxel_size, max_range):
    """Load N .pcd scans, transform each to global with its own pose,
    concatenate and voxelize once.  N=1 recovers single-scan behavior.
    """
    all_pts = []
    for pcd_file, pose in zip(pcd_files, poses):
        cloud = o3d.io.read_point_cloud(str(pcd_file))
        pts = np.asarray(cloud.points, dtype=np.float64)
        if max_range > 0 and pts.shape[0] > 0:
            r = np.linalg.norm(pts, axis=1)
            pts = pts[r <= max_range]
        if pts.shape[0] > 0:
            all_pts.append(transform_to_global(pts, pose))
    if not all_pts:
        return np.empty((0, 3)), None, 0
    combined = np.vstack(all_pts)
    pts_v = voxelize(combined, voxel_size)
    n_vox = pts_v.shape[0]
    tree = cKDTree(pts_v) if n_vox > 0 else None
    return pts_v, tree, n_vox


# ---------------------------------------------------------------------------
#  Submap grouping
# ---------------------------------------------------------------------------

def _group_into_submaps(positions, poses, files, n, stride=None):
    """Group K consecutive scans into submaps of size n, stepping by `stride`.

    The first scan of each submap is the keyframe: its pose's translation
    defines the submap position used by the distance pre-filter.

    stride=None (default) means stride=n → non-overlapping submaps (original
    behaviour). stride<n produces overlapping submaps; stride>n leaves gaps.
    The final (trailing) window may be shorter than n when (K - start) < n.

    Returns
    -------
    kf_positions : np.ndarray (K_sub, 3)
    submap_poses : list[list[np.ndarray]]   – per-scan poses for each submap
    submap_files : list[list[Path]]
    """
    if stride is None:
        stride = n
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    if n <= 1 and stride == 1:
        submap_poses = [[p] for p in poses]
        submap_files = [[f] for f in files]
        return np.asarray(positions), submap_poses, submap_files

    K = len(files)
    kf_positions = []
    submap_poses = []
    submap_files = []
    for start in range(0, K, stride):
        end = min(start + n, K)
        kf_positions.append(positions[start])
        submap_poses.append(list(poses[start:end]))
        submap_files.append(list(files[start:end]))
    return np.asarray(kf_positions), submap_poses, submap_files


# ---------------------------------------------------------------------------
#  ICP scan-level refinement
# ---------------------------------------------------------------------------

def _icp_refine_scan(db_pts: np.ndarray, q_pts: np.ndarray,
                     max_dist: float, voxel_size: float,
                     num_threads: int = 4) -> np.ndarray:
    """GICP refinement of a single DB scan against a merged local Q cloud.

    Both inputs are Nx3 in the common global frame (already voxelized at
    voxel_size).  Returns 4x4 T_target_source that brings DB points into
    better alignment with the Q cloud.
    """
    result = small_gicp.align(
        q_pts.astype(np.float64),   # target
        db_pts.astype(np.float64),  # source
        init_T_target_source=np.eye(4),
        registration_type="GICP",
        downsampling_resolution=voxel_size,
        max_correspondence_distance=max_dist,
        num_threads=num_threads,
        max_iterations=50,
    )
    return result.T_target_source


# ---------------------------------------------------------------------------
#  Core overlap-matrix builder (block-based, dataset-agnostic)
# ---------------------------------------------------------------------------

def _compute_overlap_blocks(
    db_pos, db_poses, db_files,
    q_pos, q_poses, q_files,
    load_db_fn, load_q_fn,
    voxel_size, distance_threshold, max_range, block_size,
    use_icp=False, icp_max_dist=None,
):
    """Block-based HeLiOS overlap computation.

    db_files / q_files : list[list[Path]]   – per-submap scan file lists
    db_poses / q_poses : list[list[np.ndarray]] – per-submap scan poses
    db_pos / q_pos     : (K, 3) keyframe positions
    load_db_fn / load_q_fn : callable(files_list, poses_list, voxel_size, max_range)
                             -> (pts_v, tree, n_vox)

    When use_icp=True, each DB submap is refined with GICP against its merged
    local Q cloud (submaps within distance_threshold) before overlap is
    computed.  icp_max_dist defaults to 3 * voxel_size when None.
    """
    tau = 1.5 * voxel_size
    if icp_max_dist is None:
        icp_max_dist = 3.0 * voxel_size
    N, M = len(db_files), len(q_files)

    n_steps = 4 if use_icp else 3

    # Step 2: valid pairs via pose distance pre-filter
    print(f"  [Step 2/{n_steps}] Building valid-pair index from pose distances ...")
    q_pos_tree = cKDTree(q_pos)
    pairs_per_db = {}
    total_pairs = 0
    for i in range(N):
        js = q_pos_tree.query_ball_point(db_pos[i], distance_threshold)
        if js:
            pairs_per_db[i] = sorted(js)
            total_pairs += len(js)

    db_with_pairs = sorted(pairs_per_db.keys())
    print(f"    {total_pairs} valid pairs across {len(db_with_pairs)} DB scans "
          f"(out of {N * M} total)")

    if total_pairs == 0:
        print("    Nothing to compute – returning zero matrix.")
        return np.zeros((N, M), dtype=np.float32)

    # Step 3: process in blocks
    print(f"  [Step 3/{n_steps}] Computing overlaps in blocks ...\n")
    overlap_matrix = np.zeros((N, M), dtype=np.float32)
    computed = 0
    t0 = time.time()

    n_blocks = (len(db_with_pairs) + block_size - 1) // block_size

    for blk_idx in range(n_blocks):
        blk_start = blk_idx * block_size
        blk_end = min(blk_start + block_size, len(db_with_pairs))
        blk_db_indices = db_with_pairs[blk_start:blk_end]

        blk_q_set = set()
        blk_pairs = 0
        for i in blk_db_indices:
            blk_q_set.update(pairs_per_db[i])
            blk_pairs += len(pairs_per_db[i])
        blk_q_indices = sorted(blk_q_set)

        db_range_str = f"DB[{blk_db_indices[0]}..{blk_db_indices[-1]}]"
        print(f"    Block {blk_idx + 1}/{n_blocks}: {db_range_str}, "
              f"{blk_pairs} pairs, "
              f"loading {len(blk_db_indices)} DB + {len(blk_q_indices)} Q scans")

        db_loaded = {}
        for i in tqdm.tqdm(blk_db_indices, desc="      Loading DB scans", leave=False):
            db_loaded[i] = load_db_fn(db_files[i], db_poses[i], voxel_size, max_range)

        q_loaded = {}
        for j in tqdm.tqdm(blk_q_indices, desc="      Loading Q  scans", leave=False):
            q_loaded[j] = load_q_fn(q_files[j], q_poses[j], voxel_size, max_range)

        # Optional ICP refinement: align each DB scan to its local Q cloud
        if use_icp:
            for i in tqdm.tqdm(blk_db_indices, desc="      ICP refine", leave=False):
                pts_db, _, n_db = db_loaded[i]
                if n_db == 0:
                    continue
                q_parts = [q_loaded[j][0] for j in pairs_per_db[i]
                           if q_loaded[j][2] > 0]
                if not q_parts:
                    continue
                q_merged = np.vstack(q_parts)
                delta_T = _icp_refine_scan(pts_db, q_merged, icp_max_dist, voxel_size)
                pts_refined = (delta_T[:3, :3] @ pts_db.T).T + delta_T[:3, 3]
                tree_refined = cKDTree(pts_refined) if len(pts_refined) > 0 else None
                db_loaded[i] = (pts_refined, tree_refined, len(pts_refined))

        for i in tqdm.tqdm(blk_db_indices, desc="      Overlaps", leave=False):
            p1, tree1, n1 = db_loaded[i]
            if tree1 is None:
                continue
            for j in pairs_per_db[i]:
                p2, tree2, n2 = q_loaded[j]
                if tree2 is None:
                    continue

                denom = float(n1 + n2)
                dists12, _ = tree2.query(p1, k=1)
                o12 = 2.0 * np.sum(dists12 < tau) / denom
                dists21, _ = tree1.query(p2, k=1)
                o21 = 2.0 * np.sum(dists21 < tau) / denom
                overlap_matrix[i, j] = min(max(o12, o21), 1.0)
                computed += 1

        del db_loaded, q_loaded
        gc.collect()

    elapsed = time.time() - t0
    print(f"\n  Done: {computed} overlap pairs computed  ({elapsed:.1f} s)")
    return overlap_matrix


# ---------------------------------------------------------------------------
#  Public builders
# ---------------------------------------------------------------------------

SENSOR_MAP = {
    "O": "Ouster",
    "Ouster": "Ouster",
    "V": "Velodyne",
    "Velodyne": "Velodyne",
    "Aeva": "Aeva",
    "Avia": "Avia",
}

DEFAULT_PAIRS = ["O-O", "Aeva-Aeva", "O-Aeva", "O-Avia",
                 "Aeva-V", "Aeva-Avia", "O-V"]


def build_overlap_matrix(
    handler,
    db_sequence,
    q_sequence,
    db_sensor,
    q_sensor,
    voxel_size=0.04,
    distance_threshold=200.0,
    max_range=100.0,
    block_size=50,
    use_icp=False,
    icp_max_dist=None,
    n_db=1,
    n_q=1,
    stride_db=None,
    stride_q=None,
):
    """Build an NxM HeLiOS overlap matrix for a HeLiPR sensor pair.

    When n_db > 1 / n_q > 1, consecutive scans are accumulated into submaps
    (first scan's pose is the keyframe) and overlap is computed submap-to-submap.
    stride_db / stride_q control the step between successive submaps
    (default = n_db / n_q → non-overlapping).
    """
    tau = 1.5 * voxel_size
    n_steps = 4 if use_icp else 3
    eff_stride_db = stride_db if stride_db is not None else n_db
    eff_stride_q  = stride_q  if stride_q  is not None else n_q
    print(f"\n{'=' * 64}")
    print(f"  DB = {db_sequence} / {db_sensor}    Q = {q_sequence} / {q_sensor}")
    print(f"  voxel δ = {voxel_size} m   τ = {tau} m   "
          f"dist thresh = {distance_threshold} m   max range = {max_range} m")
    print(f"  block size = {block_size} DB submaps"
          + (f"   submap sizes: n_db={n_db}(stride {eff_stride_db}) "
             f"n_q={n_q}(stride {eff_stride_q})"
             if (n_db > 1 or n_q > 1 or eff_stride_db != n_db or eff_stride_q != n_q)
             else "")
          + (f"   ICP max dist = {icp_max_dist or 3*voxel_size:.3f} m" if use_icp else ""))
    print(f"{'=' * 64}")

    print(f"\n  [Step 1/{n_steps}] Loading poses & listing scans ...")
    db_pos_raw, db_poses_raw, db_bins_raw = _load_sequence_metadata(
        handler, db_sequence, db_sensor, scan_type="Undistorted")
    q_pos_raw, q_poses_raw, q_bins_raw = _load_sequence_metadata(
        handler, q_sequence, q_sensor, scan_type="Undistorted")

    db_pos, db_poses, db_bins = _group_into_submaps(
        db_pos_raw, db_poses_raw, db_bins_raw, n_db, stride=eff_stride_db)
    q_pos, q_poses, q_bins = _group_into_submaps(
        q_pos_raw, q_poses_raw, q_bins_raw, n_q, stride=eff_stride_q)

    N, M = len(db_bins), len(q_bins)
    print(f"    DB: {len(db_bins_raw)} scans → {N} submaps   "
          f"Q: {len(q_bins_raw)} scans → {M} submaps")

    def load_db(files, poses, vs, mr):
        return _load_and_voxelize(handler, files, db_sensor, "Undistorted", poses, vs, mr)

    def load_q(files, poses, vs, mr):
        return _load_and_voxelize(handler, files, q_sensor, "Undistorted", poses, vs, mr)

    return _compute_overlap_blocks(
        db_pos, db_poses, db_bins,
        q_pos, q_poses, q_bins,
        load_db, load_q,
        voxel_size, distance_threshold, max_range, block_size,
        use_icp=use_icp, icp_max_dist=icp_max_dist,
    )


def build_overlap_matrix_custom(
    db_path,
    q_path,
    inter_transform,
    voxel_size=0.04,
    distance_threshold=200.0,
    max_range=100.0,
    block_size=50,
    use_icp=False,
    icp_max_dist=None,
    n_db=1,
    n_q=1,
    stride_db=None,
    stride_q=None,
):
    """Build an NxM HeLiOS overlap matrix for a custom pcd+poses_kitti.txt dataset.

    inter_transform – 4x4 matrix T that maps the DB world frame into the Q world
                      frame (e.g. T_robinw_ouster when DB=ouster, Q=robinw).
                      Pass None if both sequences already share a common frame.
    n_db / n_q      – submap size per sequence (first scan's pose = keyframe).
    stride_db / stride_q – step between successive submaps (default = n_db/n_q).
    """
    db_path = Path(db_path)
    q_path = Path(q_path)
    tau = 1.5 * voxel_size
    n_steps = 4 if use_icp else 3
    eff_stride_db = stride_db if stride_db is not None else n_db
    eff_stride_q  = stride_q  if stride_q  is not None else n_q

    print(f"\n{'=' * 64}")
    print(f"  DB = {db_path.name}    Q = {q_path.name}")
    print(f"  voxel δ = {voxel_size} m   τ = {tau} m   "
          f"dist thresh = {distance_threshold} m   max range = {max_range} m")
    print(f"  block size = {block_size} DB submaps"
          + (f"   submap sizes: n_db={n_db}(stride {eff_stride_db}) "
             f"n_q={n_q}(stride {eff_stride_q})"
             if (n_db > 1 or n_q > 1 or eff_stride_db != n_db or eff_stride_q != n_q)
             else "")
          + (f"   ICP max dist = {icp_max_dist or 3*voxel_size:.3f} m" if use_icp else ""))
    if inter_transform is not None:
        print(f"  inter-sequence transform: applied to DB poses")
    print(f"{'=' * 64}")

    print(f"\n  [Step 1/{n_steps}] Loading poses & listing scans ...")
    db_pos_raw, db_poses_raw, db_files_raw = _load_custom_sequence_metadata(
        db_path, inter_transform)
    q_pos_raw, q_poses_raw, q_files_raw = _load_custom_sequence_metadata(
        q_path, None)

    db_pos, db_poses, db_files = _group_into_submaps(
        db_pos_raw, db_poses_raw, db_files_raw, n_db, stride=eff_stride_db)
    q_pos, q_poses, q_files = _group_into_submaps(
        q_pos_raw, q_poses_raw, q_files_raw, n_q, stride=eff_stride_q)

    N, M = len(db_files), len(q_files)
    print(f"    DB: {len(db_files_raw)} scans → {N} submaps   "
          f"Q: {len(q_files_raw)} scans → {M} submaps")

    return _compute_overlap_blocks(
        db_pos, db_poses, db_files,
        q_pos, q_poses, q_files,
        _load_and_voxelize_pcd, _load_and_voxelize_pcd,
        voxel_size, distance_threshold, max_range, block_size,
        use_icp=use_icp, icp_max_dist=icp_max_dist,
    )


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute HeLiOS overlap matrices."
    )
    parser.add_argument(
        "--dataset_type", type=str, default="helipr",
        choices=["helipr", "custom"],
        help="Dataset type: 'helipr' (default) or 'custom' (.pcd + poses_kitti.txt)."
    )

    # ---- HeLiPR-only args ----
    helipr = parser.add_argument_group("HeLiPR options (dataset_type=helipr)")
    helipr.add_argument(
        "--dataset", type=str,
        help="Root path to the HeLiPR dataset."
    )
    helipr.add_argument(
        "--db_sequence", type=str,
        help="Database sequence name, e.g. 'Roundabout01'."
    )
    helipr.add_argument(
        "--q_sequence", type=str,
        help="Query sequence name, e.g. 'Roundabout03'."
    )
    helipr.add_argument(
        "--pairs", type=str, nargs="+", default=DEFAULT_PAIRS,
        help="Sensor pairs to compute (DB-Q).  "
             "Options: O-O, Aeva-Aeva, O-V, O-Aeva, O-Avia.  "
             "Default: all."
    )

    # ---- custom-only args ----
    custom = parser.add_argument_group("Custom options (dataset_type=custom)")
    custom.add_argument(
        "--db_path", type=str,
        help="Path to the DB dataset directory (contains scans/ and poses_kitti.txt)."
    )
    custom.add_argument(
        "--q_path", type=str,
        help="Path to the Q dataset directory (contains scans/ and poses_kitti.txt)."
    )
    custom.add_argument(
        "--transform", type=str, default=None,
        help="Path to a 4x4 transform.txt that maps DB world frame → Q world frame. "
             "Defaults to <db_path>/transform.txt if it exists."
    )

    # ---- shared args ----
    parser.add_argument(
        "--output_dir", type=str, default="overlap_matrices",
        help="Directory to save output files (default: overlap_matrices)."
    )
    parser.add_argument(
        "--icp", action="store_true",
        help="Refine each DB submap's alignment with GICP (small_gicp) against "
             "its local Q cloud before computing overlap."
    )
    parser.add_argument(
        "--icp_max_dist", type=float, default=None,
        help="Max correspondence distance for GICP refinement (default: 3 * voxel_size)."
    )
    parser.add_argument(
        "--n_db", type=int, default=1,
        help="Scans per DB submap. First scan's pose is the keyframe. "
             "Default 1 (single-scan, no accumulation)."
    )
    parser.add_argument(
        "--n_q", type=int, default=1,
        help="Scans per Q submap. First scan's pose is the keyframe. "
             "Default 1 (single-scan, no accumulation)."
    )
    parser.add_argument(
        "--stride_db", type=int, default=None,
        help="Step between successive DB submaps. Default = n_db "
             "(non-overlapping). Set < n_db for overlapping, > n_db for gaps."
    )
    parser.add_argument(
        "--stride_q", type=int, default=None,
        help="Step between successive Q submaps. Default = n_q "
             "(non-overlapping)."
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.04,
        help="Voxel size δ in metres for overlap (default: 0.04)."
    )
    parser.add_argument(
        "--distance_threshold", type=float, default=100.0,
        help="Max pose distance (m) for non-zero overlap (default: 100.0)."
    )
    parser.add_argument(
        "--max_range", type=float, default=100.0,
        help="Max point range from sensor origin in metres (default: 100.0)."
    )
    parser.add_argument(
        "--block_size", type=int, default=50,
        help="Number of DB scans to process per block.  "
             "Lower = less peak RAM.  (default: 50)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- HeLiPR mode -------------------------------------------------------
    if args.dataset_type == "helipr":
        if not args.dataset or not args.db_sequence or not args.q_sequence:
            parser.error(
                "--dataset, --db_sequence, and --q_sequence are required "
                "for dataset_type=helipr"
            )

        handler = HeLiPR_Handler(Path(args.dataset), verbose=True)

        for pair_str in args.pairs:
            parts = pair_str.split("-")
            if len(parts) != 2:
                print(f"WARNING: invalid pair format '{pair_str}', skipping.  "
                      "Use DB-Q, e.g. O-Aeva.")
                continue

            db_sensor = SENSOR_MAP.get(parts[0])
            q_sensor = SENSOR_MAP.get(parts[1])
            if db_sensor is None or q_sensor is None:
                print(f"WARNING: unknown sensor in '{pair_str}', skipping.")
                continue

            overlap = build_overlap_matrix(
                handler,
                args.db_sequence,
                args.q_sequence,
                db_sensor,
                q_sensor,
                args.voxel_size,
                args.distance_threshold,
                args.max_range,
                args.block_size,
                use_icp=args.icp,
                icp_max_dist=args.icp_max_dist,
                n_db=args.n_db,
                n_q=args.n_q,
                stride_db=args.stride_db,
                stride_q=args.stride_q,
            )

            eff_sdb = args.stride_db if args.stride_db is not None else args.n_db
            eff_sq  = args.stride_q  if args.stride_q  is not None else args.n_q
            submap_suffix = ""
            if args.n_db > 1 or args.n_q > 1:
                submap_suffix = f"_Ndb{args.n_db}_Nq{args.n_q}"
            if eff_sdb != args.n_db or eff_sq != args.n_q:
                submap_suffix += f"_Sdb{eff_sdb}_Sq{eff_sq}"
            filename = (
                f"overlap_{args.db_sequence}_{db_sensor}_"
                f"{args.q_sequence}_{q_sensor}{submap_suffix}.txt"
            )
            filepath = output_dir / filename
            header_lines = [
                f"HeLiOS overlap matrix",
                f"Database: {args.db_sequence} / {db_sensor}  "
                f"({overlap.shape[0]} submaps, n_db={args.n_db}, stride_db={eff_sdb})",
                f"Query:    {args.q_sequence} / {q_sensor}  "
                f"({overlap.shape[1]} submaps, n_q={args.n_q}, stride_q={eff_sq})",
                f"Voxel size (delta): {args.voxel_size} m   "
                f"tau: {1.5 * args.voxel_size} m",
                f"Pose distance threshold: {args.distance_threshold} m",
                f"Max point range: {args.max_range} m",
                f"Rows = DB submap index, Columns = Q submap index "
                f"(keyframe = first scan of each submap)",
            ]
            np.savetxt(filepath, overlap, fmt="%.6f",
                       header="\n".join(header_lines))

            nonzero = np.count_nonzero(overlap)
            print(f"\n  Saved {filepath}  "
                  f"({overlap.shape[0]}x{overlap.shape[1]}, "
                  f"{nonzero} non-zero entries, "
                  f"max overlap = {overlap.max():.4f})\n")

    # ---- custom mode -------------------------------------------------------
    else:
        if not args.db_path or not args.q_path:
            parser.error(
                "--db_path and --q_path are required for dataset_type=custom"
            )

        db_path = Path(args.db_path)
        q_path = Path(args.q_path)

        # Resolve transform path
        transform_path = args.transform
        if transform_path is None:
            candidate = db_path / "transform.txt"
            if candidate.exists():
                transform_path = str(candidate)
                print(f"  Using transform: {transform_path}")
            else:
                print("  No --transform specified and no transform.txt found "
                      "in db_path; assuming both sequences share a common frame.")

        inter_transform = None
        if transform_path is not None:
            inter_transform = np.loadtxt(transform_path)
            if inter_transform.shape != (4, 4):
                raise ValueError(
                    f"transform.txt must be a 4x4 matrix, "
                    f"got shape {inter_transform.shape}"
                )

        overlap = build_overlap_matrix_custom(
            db_path,
            q_path,
            inter_transform,
            args.voxel_size,
            args.distance_threshold,
            args.max_range,
            args.block_size,
            use_icp=args.icp,
            icp_max_dist=args.icp_max_dist,
            n_db=args.n_db,
            n_q=args.n_q,
            stride_db=args.stride_db,
            stride_q=args.stride_q,
        )

        eff_sdb = args.stride_db if args.stride_db is not None else args.n_db
        eff_sq  = args.stride_q  if args.stride_q  is not None else args.n_q
        submap_suffix = ""
        if args.n_db > 1 or args.n_q > 1:
            submap_suffix = f"_Ndb{args.n_db}_Nq{args.n_q}"
        if eff_sdb != args.n_db or eff_sq != args.n_q:
            submap_suffix += f"_Sdb{eff_sdb}_Sq{eff_sq}"
        filename = f"overlap_{db_path.name}_{q_path.name}{submap_suffix}.txt"
        filepath = output_dir / filename
        icp_str = (f"GICP max dist {args.icp_max_dist or 3*args.voxel_size:.3f} m"
                   if args.icp else "none")
        header_lines = [
            f"HeLiOS overlap matrix",
            f"Database: {db_path.name}  ({overlap.shape[0]} submaps, n_db={args.n_db}, stride_db={eff_sdb})",
            f"Query:    {q_path.name}  ({overlap.shape[1]} submaps, n_q={args.n_q}, stride_q={eff_sq})",
            f"Voxel size (delta): {args.voxel_size} m   "
            f"tau: {1.5 * args.voxel_size} m",
            f"Pose distance threshold: {args.distance_threshold} m",
            f"Max point range: {args.max_range} m",
            f"Inter-sequence transform: {transform_path or 'none (shared frame)'}",
            f"ICP refinement: {icp_str}",
            f"Rows = DB submap index, Columns = Q submap index "
            f"(keyframe = first scan of each submap)",
        ]
        np.savetxt(filepath, overlap, fmt="%.6f",
                   header="\n".join(header_lines))

        nonzero = np.count_nonzero(overlap)
        print(f"\n  Saved {filepath}  "
              f"({overlap.shape[0]}x{overlap.shape[1]}, "
              f"{nonzero} non-zero entries, "
              f"max overlap = {overlap.max():.4f})\n")


if __name__ == "__main__":
    main()
