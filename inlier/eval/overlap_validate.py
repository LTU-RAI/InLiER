#!/usr/bin/env python3
"""
validate_overlap.py

Visual validation of a pre-computed overlap matrix.

Produces a figure with:
  (a) 3D-style trajectory plot – database on a lower z-plane, query on an
      upper plane; edges between scan pairs with overlap > 0.5 *and*
      pose-to-pose distance < threshold, colored by overlap value.
  (b) Summary statistics panel (number of non-zero entries, entries > 0.5,
      histogram of overlap values, etc.).
  (c) Top-down aligned view of two example scans with *high* overlap (≥ 0.5).
  (d) Top-down aligned view of two example scans with *low* overlap (> 0, < 0.5).

The scan visualisation uses the *exact same pipeline* as build_overlap_data.py:
  raw scan → range filter → transform to global via associated pose → voxelize.

Usage — HeLiPR:
    python validate_overlap.py \
        --dataset-type helipr \
        --dataset /path/to/HeLiPR \
        --db-sequence Roundabout01 \
        --q-sequence Roundabout03 \
        --pair O-Aeva \
        --overlap-dir overlap_matrices

Usage — generic (.pcd + poses_kitti.txt):
    python3 validate_overlap.py \
        --dataset-type generic \
        --db-path ~/Documents/datasets/campus_ouster \
        --q-path  ~/Documents/datasets/campus_robinw \
        --transform ~/Documents/datasets/campus_ouster/T_robinw_ouster.txt \
        --overlap-dir overlap_matrices
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import open3d as o3d
import sys

try:
    import small_gicp
    _HAS_GICP = True
except ImportError:
    _HAS_GICP = False

from inlier.eval.datasets.helipr import HeLiPR_Handler

SENSOR_MAP = {
    "O": "Ouster",
    "Ouster": "Ouster",
    "V": "Velodyne",
    "Velodyne": "Velodyne",
    "Aeva": "Aeva",
    "Avia": "Avia",
}


# ---------------------------------------------------------------------------
#  Shared geometry helpers
# ---------------------------------------------------------------------------

def voxelize(points, voxel_size):
    if points.shape[0] == 0:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return np.asarray(pcd.voxel_down_sample(voxel_size).points)


def transform_to_global(points, pose):
    pts_h = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float64)])
    return (pose @ pts_h.T).T[:, :3]


# ---------------------------------------------------------------------------
#  HeLiPR-specific loading
# ---------------------------------------------------------------------------

def load_associated_poses(handler, sequence, sensor, scan_type):
    """Load per-scan poses for a HeLiPR sequence via timestamp matching."""
    seq_path = handler.dataset_path / sequence
    bin_files = handler.list_scan_files(seq_path, sensor, type=scan_type)
    pc_timestamps = handler._bin_timestamps(bin_files)
    gt_poses, gt_timestamps = handler.load_poses(seq_path, sensor)

    associated_poses = []
    for pc_ts in pc_timestamps:
        closest_idx = handler.get_closest_pose_index(pc_ts, gt_timestamps)
        associated_poses.append(gt_poses[closest_idx])

    return associated_poses, bin_files


def load_scan_global_voxelized_helipr(handler, bin_files, sensor, poses,
                                      scan_type, voxel_size, max_range):
    """HeLiPR: accumulate N scans → range filter → global frame → voxelize."""
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
        return np.empty((0, 3))
    return voxelize(np.vstack(all_pts), voxel_size)


# ---------------------------------------------------------------------------
#  Generic dataset loading (.pcd + poses_kitti.txt)
# ---------------------------------------------------------------------------

def _load_kitti_poses(poses_file: Path) -> list:
    poses = []
    with open(poses_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 12:
                T = np.eye(4)
                T[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(T)
    return poses


def group_into_submaps(positions, poses, files, n, stride=None):
    """Group K consecutive scans into submaps of size n, stepping by `stride`.

    First scan's pose is the keyframe (mirrors build_overlap_data.py).
    stride=None defaults to n → non-overlapping submaps.
    """
    if stride is None:
        stride = n
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    if n <= 1 and stride == 1:
        return (np.asarray(positions),
                [[p] for p in poses],
                [[f] for f in files])
    K = len(files)
    kf_positions, submap_poses, submap_files = [], [], []
    for start in range(0, K, stride):
        end = min(start + n, K)
        kf_positions.append(positions[start])
        submap_poses.append(list(poses[start:end]))
        submap_files.append(list(files[start:end]))
    return np.asarray(kf_positions), submap_poses, submap_files


def load_generic_sequence(seq_path: Path, inter_transform=None):
    """Load poses and scan file list for a generic dataset.

    inter_transform – 4x4 that maps this sequence's frame into the reference
                      frame; None means this sequence is the reference.
    Returns positions (K,3), effective_poses (list of 4x4), scan_files.
    """
    poses = _load_kitti_poses(seq_path / "poses_kitti.txt")
    scan_files = sorted((seq_path / "scans").glob("*.pcd"))

    if len(scan_files) != len(poses):
        raise ValueError(
            f"{seq_path}: {len(scan_files)} scans vs {len(poses)} poses"
        )

    effective_poses = (
        [inter_transform @ p for p in poses] if inter_transform is not None else poses
    )
    positions = np.array([p[:3, 3] for p in effective_poses])
    return positions, effective_poses, scan_files


def _gicp_refine(db_pts, q_pts, voxel_size, max_dist):
    """GICP refinement of DB cloud onto Q cloud (same pipeline as build script).
    Returns 4x4 delta to apply to DB points."""
    if not _HAS_GICP or db_pts.shape[0] == 0 or q_pts.shape[0] == 0:
        return np.eye(4)
    result = small_gicp.align(
        q_pts.astype(np.float64),
        db_pts.astype(np.float64),
        init_T_target_source=np.eye(4),
        registration_type="GICP",
        downsampling_resolution=voxel_size,
        max_correspondence_distance=max_dist,
        num_threads=4,
        max_iterations=50,
    )
    return result.T_target_source


def load_scan_global_voxelized_pcd(pcd_files, poses, voxel_size, max_range):
    """Generic: accumulate N .pcd → range filter → global frame → voxelize."""
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
        return np.empty((0, 3))
    return voxelize(np.vstack(all_pts), voxel_size)


# ---------------------------------------------------------------------------
#  Overlap matrix loading
# ---------------------------------------------------------------------------

def load_overlap_matrix(overlap_dir, filename_stem):
    """Load an overlap matrix by its full filename stem."""
    filepath = Path(overlap_dir) / f"{filename_stem}.txt"
    if not filepath.exists():
        raise FileNotFoundError(f"Overlap file not found: {filepath}")
    return np.loadtxt(filepath)


# ---------------------------------------------------------------------------
#  Pair selection
# ---------------------------------------------------------------------------

def pick_example_pair(overlap, db_poses_xyz, q_poses_xyz, low, high,
                      dist_max=None, rng=None):
    """Pick a random pair whose overlap falls in [low, high).

    Pass a seeded `rng` (np.random.Generator) for a reproducible choice;
    with rng=None a fresh, unseeded generator is used so each run differs.
    """
    mask = (overlap >= low) & (overlap < high)
    if not mask.any():
        return None

    ii, jj = np.where(mask)
    dists = np.linalg.norm(db_poses_xyz[ii] - q_poses_xyz[jj], axis=1)

    if dist_max is not None:
        valid = dists < dist_max
        ii, jj = ii[valid], jj[valid]

    if len(ii) == 0:
        return None

    if rng is None:
        rng = np.random.default_rng()
    pick = rng.integers(len(ii))
    return int(ii[pick]), int(jj[pick])


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def _plot_scan_pair(ax, db_files, db_poses, q_files, q_poses,
                    db_label, q_label,
                    load_db_fn, load_q_fn,
                    voxel_size, max_range,
                    use_icp=False, icp_max_dist=None):
    """Load two submaps (lists of scans) and show them top-down in the DB
    keyframe's local frame. When use_icp=True, refines DB onto Q with GICP."""
    db_pts_g = load_db_fn(db_files, db_poses, voxel_size, max_range)
    q_pts_g = load_q_fn(q_files, q_poses, voxel_size, max_range)

    if use_icp and db_pts_g.shape[0] > 0 and q_pts_g.shape[0] > 0:
        max_d = icp_max_dist if icp_max_dist is not None else 3.0 * voxel_size
        delta = _gicp_refine(db_pts_g, q_pts_g, voxel_size, max_d)
        db_pts_g = (delta[:3, :3] @ db_pts_g.T).T + delta[:3, 3]

    db_kf_pose = db_poses[0]
    db_inv = np.linalg.inv(db_kf_pose)
    to_local = lambda pts: (
        (db_inv @ np.hstack([pts, np.ones((pts.shape[0], 1))]).T).T[:, :3]
    )

    db_local = to_local(db_pts_g)
    q_local = to_local(q_pts_g)

    ax.scatter(db_local[:, 0], db_local[:, 1], s=0.5, alpha=0.3,
               label=db_label, color="tab:blue")
    ax.scatter(q_local[:, 0], q_local[:, 1], s=0.5, alpha=0.3,
               label=q_label, color="tab:orange")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, markerscale=8)


def plot_validation(
    overlap,
    db_poses,
    q_poses,
    db_files,
    q_files,
    db_label,
    q_label,
    load_db_fn,
    load_q_fn,
    voxel_size=4.0,
    max_range=100.0,
    pose_dist_threshold=10.0,
    overlap_threshold=0.5,
    use_icp=False,
    icp_max_dist=None,
    rng=None,
):
    # db_poses/q_poses are lists of submaps; first pose of each submap is the keyframe
    db_xyz = np.array([submap[0][:3, 3] for submap in db_poses])
    q_xyz = np.array([submap[0][:3, 3] for submap in q_poses])

    N, M = overlap.shape

    ii, jj = np.where(overlap >= overlap_threshold)
    if len(ii) > 0:
        dists = np.linalg.norm(db_xyz[ii] - q_xyz[jj], axis=1)
        valid = dists < pose_dist_threshold
        ii, jj = ii[valid], jj[valid]
    edge_overlaps = overlap[ii, jj] if len(ii) > 0 else np.array([])

    total_pairs = N * M
    nonzero = np.count_nonzero(overlap)
    above_thr = int(np.sum(overlap >= overlap_threshold))
    above_thr_close = len(ii)
    max_ov = float(overlap.max()) if overlap.size else 0.0
    mean_nonzero = float(overlap[overlap > 0].mean()) if nonzero else 0.0

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        f"Overlap Validation:  DB = {db_label}   Q = {q_label}",
        fontsize=14, fontweight="bold",
    )

    # (a) 3D trajectory with overlap edges
    ax3d = fig.add_subplot(2, 3, (1, 4), projection="3d")
    z_db, z_q = 0.0, 20.0

    ax3d.plot(db_xyz[:, 0], db_xyz[:, 1], zs=z_db, zdir="z",
              color="tab:blue", lw=3.5, label=f"DB ({db_label})", alpha=1.0)
    ax3d.plot(q_xyz[:, 0], q_xyz[:, 1], zs=z_q, zdir="z",
              color="tab:orange", lw=3.5, label=f"Q ({q_label})", alpha=1.0)

    if len(ii) > 0:
        vmax = float(edge_overlaps.max())
        norm = mcolors.Normalize(vmin=overlap_threshold, vmax=vmax)
        cmap = plt.get_cmap("RdYlGn")
        segments, colors = [], []
        for idx in range(len(ii)):
            segments.append([[db_xyz[ii[idx], 0], db_xyz[ii[idx], 1], z_db],
                             [q_xyz[jj[idx], 0], q_xyz[jj[idx], 1], z_q]])
            colors.append(cmap(norm(edge_overlaps[idx])))
        lc = Line3DCollection(segments, colors=colors, linewidths=0.5, alpha=0.1)
        ax3d.add_collection3d(lc)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax3d, shrink=0.5, pad=0.08)
        cbar.set_label("Overlap")

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zticks([z_db, z_q])
    ax3d.set_zticklabels(["DB", "Q"])
    ax3d.legend(loc="upper left", fontsize=8)
    ax3d.set_title(
        f"Trajectories & Overlap Edges "
        f"(≥ {overlap_threshold}, dist < {pose_dist_threshold} m)"
    )

    # (b) Summary statistics
    ax_stats = fig.add_subplot(2, 3, 3)
    ax_stats.axis("off")
    stats_text = (
        f"Matrix size:  {N} × {M}  ({total_pairs:,} pairs)\n"
        f"Non-zero entries:  {nonzero:,}  "
        f"({100 * nonzero / max(total_pairs, 1):.2f}%)\n"
        f"Overlap ≥ {overlap_threshold}:  {above_thr:,}\n"
        f"Overlap ≥ {overlap_threshold} & dist < {pose_dist_threshold} m:  "
        f"{above_thr_close:,}\n"
        f"Max overlap:  {max_ov:.4f}\n"
        f"Mean non-zero overlap:  {mean_nonzero:.4f}\n"
    )
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=11, verticalalignment="top", fontfamily="monospace")
    ax_stats.set_title("Summary Statistics", fontsize=12)

    # (b') Overlap histogram
    ax_hist = fig.add_subplot(2, 3, 6)
    nonzero_vals = overlap[overlap > 0].flatten()
    if nonzero_vals.size > 0:
        ax_hist.hist(nonzero_vals, bins=50, color="steelblue", edgecolor="k", alpha=0.8)
    ax_hist.axvline(overlap_threshold, color="red", ls="--", lw=1.5,
                    label=f"{overlap_threshold} threshold")
    ax_hist.set_xlabel("Overlap value")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Distribution of Non-Zero Overlaps")
    ax_hist.legend()

    # (c) High-overlap example
    ax_high = fig.add_subplot(2, 3, 2)
    pair_high = pick_example_pair(overlap, db_xyz, q_xyz,
                                  overlap_threshold, 1.01,
                                  dist_max=pose_dist_threshold, rng=rng)
    if pair_high is not None:
        i_h, j_h = pair_high
        _plot_scan_pair(
            ax_high,
            db_files[i_h], db_poses[i_h],
            q_files[j_h], q_poses[j_h],
            db_label, q_label,
            load_db_fn, load_q_fn,
            voxel_size, max_range,
            use_icp=use_icp, icp_max_dist=icp_max_dist,
        )
        ax_high.set_title(
            f"High Overlap Example  (O={overlap[i_h, j_h]:.3f})\n"
            f"DB idx {i_h}  /  Q idx {j_h}",
            fontsize=10,
        )
    else:
        ax_high.text(0.5, 0.5, "No high-overlap pair found",
                     ha="center", va="center", transform=ax_high.transAxes)
        ax_high.set_title("High Overlap Example")

    # (d) Low-overlap example
    ax_low = fig.add_subplot(2, 3, 5)
    pair_low = pick_example_pair(overlap, db_xyz, q_xyz,
                                 0.01, overlap_threshold, dist_max=None, rng=rng)
    if pair_low is not None:
        i_l, j_l = pair_low
        _plot_scan_pair(
            ax_low,
            db_files[i_l], db_poses[i_l],
            q_files[j_l], q_poses[j_l],
            db_label, q_label,
            load_db_fn, load_q_fn,
            voxel_size, max_range,
            use_icp=use_icp, icp_max_dist=icp_max_dist,
        )
        ax_low.set_title(
            f"Low Overlap Example  (O={overlap[i_l, j_l]:.3f})\n"
            f"DB idx {i_l}  /  Q idx {j_l}",
            fontsize=10,
        )
    else:
        ax_low.text(0.5, 0.5, "No low-overlap pair found",
                    ha="center", va="center", transform=ax_low.transAxes)
        ax_low.set_title("Low Overlap Example")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate a pre-computed HeLiOS overlap matrix with visualizations."
    )
    parser.add_argument(
        "--dataset-type", type=str, default="helipr",
        choices=["helipr", "generic", "custom"], metavar="{helipr,generic}",
        help="Dataset type: 'helipr' (default) or 'generic' (.pcd + poses_kitti.txt).  "
             "'custom' is accepted as a deprecated alias for 'generic'."
    )

    helipr = parser.add_argument_group("HeLiPR options (--dataset-type helipr)")
    helipr.add_argument("--dataset", type=str,
                        help="Root path to the HeLiPR dataset.")
    helipr.add_argument("--db-sequence", type=str,
                        help="Database sequence name, e.g. 'Roundabout01'.")
    helipr.add_argument("--q-sequence", type=str,
                        help="Query sequence name, e.g. 'Roundabout03'.")
    helipr.add_argument("--pair", type=str,
                        help="Sensor pair DB-Q, e.g. 'O-O', 'O-Aeva'.")

    generic = parser.add_argument_group("Generic options (--dataset-type generic)")
    generic.add_argument("--db-path", type=str,
                        help="DB dataset directory (scans/ + poses_kitti.txt).")
    generic.add_argument("--q-path", type=str,
                        help="Q dataset directory (scans/ + poses_kitti.txt).")
    generic.add_argument(
        "--transform", type=str, default=None,
        help="4x4 transform.txt mapping DB frame → Q frame. "
             "Defaults to <db_path>/transform.txt if it exists."
    )

    parser.add_argument("--overlap-dir", type=str, default="overlap_matrices",
                        help="Directory containing overlap matrix files.")
    parser.add_argument("--voxel-size", type=float, default=1.0,
                        help="Voxel size used during overlap computation (default: 2.0).")
    parser.add_argument("--max-range", type=float, default=100.0,
                        help="Max point range used during overlap computation (default: 100.0).")
    parser.add_argument("--pose-dist-threshold", type=float, default=10.0,
                        help="Max pose distance (m) for a true-positive overlap edge (default: 50.0).")
    parser.add_argument("--overlap-threshold", type=float, default=0.2,
                        help="Overlap threshold separating high/low examples (default: 0.2).")
    parser.add_argument("--n-db", type=int, default=1,
                        help="Scans per DB submap (must match build_overlap_data.py). Default 1.")
    parser.add_argument("--n-q", type=int, default=1,
                        help="Scans per Q submap (must match build_overlap_data.py). Default 1.")
    parser.add_argument("--stride-db", type=int, default=None,
                        help="Step between successive DB submaps (must match "
                             "build_overlap_data.py). Default = n_db.")
    parser.add_argument("--stride-q", type=int, default=None,
                        help="Step between successive Q submaps (must match "
                             "build_overlap_data.py). Default = n_q.")
    parser.add_argument("--icp", action="store_true",
                        help="Refine visualized DB submap onto Q with GICP (small_gicp).")
    parser.add_argument("--icp-max-dist", type=float, default=None,
                        help="Max correspondence distance for GICP (default: 3 * voxel_size).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the random example-pair selection. "
                             "Omit for a different pair each run.")
    args = parser.parse_args(argv)
    if args.dataset_type == "custom":     # pre-0.3 spelling of "generic"
        args.dataset_type = "generic"

    rng = np.random.default_rng(args.seed)

    if args.icp and not _HAS_GICP:
        parser.error("--icp requires the 'small_gicp' package to be installed.")

    # ---- HeLiPR mode -------------------------------------------------------
    if args.dataset_type == "helipr":
        if not all([args.dataset, args.db_sequence, args.q_sequence, args.pair]):
            parser.error(
                "--dataset, --db-sequence, --q-sequence, and --pair are "
                "required for --dataset-type helipr"
            )

        parts = args.pair.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair format: '{args.pair}'. Expected e.g. O-Aeva.")
        db_sensor = SENSOR_MAP.get(parts[0])
        q_sensor = SENSOR_MAP.get(parts[1])
        if db_sensor is None or q_sensor is None:
            raise ValueError(f"Unknown sensor in pair '{args.pair}'.")

        handler = HeLiPR_Handler(Path(args.dataset), verbose=False)

        eff_sdb = args.stride_db if args.stride_db is not None else args.n_db
        eff_sq  = args.stride_q  if args.stride_q  is not None else args.n_q
        submap_suffix = ""
        if args.n_db > 1 or args.n_q > 1:
            submap_suffix = f"_Ndb{args.n_db}_Nq{args.n_q}"
        if eff_sdb != args.n_db or eff_sq != args.n_q:
            submap_suffix += f"_Sdb{eff_sdb}_Sq{eff_sq}"
        stem = (f"overlap_{args.db_sequence}_{db_sensor}_"
                f"{args.q_sequence}_{q_sensor}{submap_suffix}")
        overlap = load_overlap_matrix(args.overlap_dir, stem)
        print(f"Loaded overlap matrix: {overlap.shape[0]} × {overlap.shape[1]}")

        print("Associating DB poses ...")
        db_poses_raw, db_files_raw = load_associated_poses(
            handler, args.db_sequence, db_sensor, scan_type="Undistorted"
        )
        print("Associating Q poses ...")
        q_poses_raw, q_files_raw = load_associated_poses(
            handler, args.q_sequence, q_sensor, scan_type="Undistorted"
        )

        db_positions = np.array([p[:3, 3] for p in db_poses_raw])
        q_positions = np.array([p[:3, 3] for p in q_poses_raw])
        _, db_poses, db_files = group_into_submaps(
            db_positions, db_poses_raw, db_files_raw, args.n_db, stride=eff_sdb)
        _, q_poses, q_files = group_into_submaps(
            q_positions, q_poses_raw, q_files_raw, args.n_q, stride=eff_sq)

        N = min(overlap.shape[0], len(db_poses))
        M = min(overlap.shape[1], len(q_poses))
        overlap = overlap[:N, :M]
        db_poses, db_files = db_poses[:N], db_files[:N]
        q_poses, q_files = q_poses[:M], q_files[:M]

        def load_db_fn(files, poses, vs, mr):
            return load_scan_global_voxelized_helipr(
                handler, files, db_sensor, poses, "Database", vs, mr)

        def load_q_fn(files, poses, vs, mr):
            return load_scan_global_voxelized_helipr(
                handler, files, q_sensor, poses, "Query", vs, mr)

        db_label = f"{args.db_sequence}/{db_sensor}"
        q_label = f"{args.q_sequence}/{q_sensor}"

    # ---- generic mode -------------------------------------------------------
    else:
        if not args.db_path or not args.q_path:
            parser.error("--db-path and --q-path are required for --dataset-type generic")

        db_path = Path(args.db_path)
        q_path = Path(args.q_path)

        transform_path = args.transform
        if transform_path is None:
            candidate = db_path / "transform.txt"
            if candidate.exists():
                transform_path = str(candidate)
                print(f"  Using transform: {transform_path}")

        inter_transform = None
        if transform_path is not None:
            inter_transform = np.loadtxt(transform_path)
            if inter_transform.shape != (4, 4):
                raise ValueError(
                    f"transform.txt must be 4x4, got {inter_transform.shape}"
                )

        eff_sdb = args.stride_db if args.stride_db is not None else args.n_db
        eff_sq  = args.stride_q  if args.stride_q  is not None else args.n_q
        submap_suffix = ""
        if args.n_db > 1 or args.n_q > 1:
            submap_suffix = f"_Ndb{args.n_db}_Nq{args.n_q}"
        if eff_sdb != args.n_db or eff_sq != args.n_q:
            submap_suffix += f"_Sdb{eff_sdb}_Sq{eff_sq}"
        stem = f"overlap_{db_path.name}_{q_path.name}{submap_suffix}"
        overlap = load_overlap_matrix(args.overlap_dir, stem)
        print(f"Loaded overlap matrix: {overlap.shape[0]} × {overlap.shape[1]}")

        print("Loading DB poses ...")
        db_pos_raw, db_poses_raw, db_files_raw = load_generic_sequence(
            db_path, inter_transform)
        print("Loading Q poses ...")
        q_pos_raw, q_poses_raw, q_files_raw = load_generic_sequence(q_path, None)

        _, db_poses, db_files = group_into_submaps(
            db_pos_raw, db_poses_raw, db_files_raw, args.n_db, stride=eff_sdb)
        _, q_poses, q_files = group_into_submaps(
            q_pos_raw, q_poses_raw, q_files_raw, args.n_q, stride=eff_sq)

        N = min(overlap.shape[0], len(db_poses))
        M = min(overlap.shape[1], len(q_poses))
        overlap = overlap[:N, :M]
        db_poses, db_files = db_poses[:N], db_files[:N]
        q_poses, q_files = q_poses[:M], q_files[:M]

        load_db_fn = load_scan_global_voxelized_pcd
        load_q_fn = load_scan_global_voxelized_pcd

        db_label = db_path.name
        q_label = q_path.name

    plot_validation(
        overlap, db_poses, q_poses,
        db_files, q_files,
        db_label, q_label,
        load_db_fn, load_q_fn,
        voxel_size=args.voxel_size,
        max_range=args.max_range,
        pose_dist_threshold=args.pose_dist_threshold,
        overlap_threshold=args.overlap_threshold,
        use_icp=args.icp,
        icp_max_dist=args.icp_max_dist,
        rng=rng,
    )


if __name__ == "__main__":
    main()
