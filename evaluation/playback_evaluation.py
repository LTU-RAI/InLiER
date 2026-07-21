#!/usr/bin/env python3
"""
playback_evaluation.py

Animated replay of a HeLiPR DB<->Query place-recognition run.

  - DB map + DB trajectory drawn once (full prior).
  - Q trajectory, Q scan, Q keypoints reveal incrementally per keyframe.
  - TP/FP loop edges + matched DB keypoints appear at the query keyframe
    where the closure occurs.
  - Right panel: current keyframe with DB candidate aligned by the estimated
    verify pose, plus MINT / BEAM descriptor visualisations.
  - Controls: SPACE play/pause, LEFT/RIGHT step.

It consumes the artifacts written by ``evaluate_inlier_helipr.py``:
  - ``results_{pair_tag}.json``        (run identity + token grid)
  - ``candidates_{pair_tag}.csv``       (loop closures + TP/FP labels)
  - ``per_pair_verify_{pair_tag}.csv``  (per-pair estimated poses; optional)
  - ``desc_{seq}_{sensor}_{type}_*.npz`` descriptor caches (DB and Q)
and reloads the raw scans through ``HeLiPR_Handler``.

The sequences, sensors, GT thresholds, token grid and score threshold are read
from the results JSON in --output_dir, so playback always describes the run it
is rendering.  Only what the eval does not record is passed on the CLI: the
dataset root, the cache dir, and the scan subfolder (--db_type / --q_type).

The TP/FP labels are the eval's, not this script's: it renders the closures in
candidates_{pair_tag}.csv as labelled.  To see a different set, re-run
evaluate_inlier_helipr.py with a different --pr_threshold / GT definition.

Example
-------
python3 evaluation/playback_evaluation.py \
    --output_dir results/HeLiPR/Ouster01_Aeva03/dbR01-O-qR03-Aeva_vs0.5_cs1_nh10_nr20_na60_ns7 \
    --dataset ~/Documents/datasets/HeLiPR \
    --cache_dir cache_inlier \
    --record /path/to/output.mp4
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from inlier.core.InLiER import InLiER as _InLiER
from inlier.core.InLiER_Matcher import InLiER_Matcher as _Matcher
from inlier.core.Dataclasses import BEAMScoreConfig as _BEAMCfg
from utils.HeLiPR_Handler import HeLiPR_Handler


# ── Visualisation constants (tunable) ───────────────────────────────────────
Z_OFFSET   = 5.0
Z_SQUASH   = 0.75
DB_VOXEL_SIZE = 1.0
Q_VOXEL_SIZE  = 1.0
DB_MAP_STRIDE = 50      # keyframe stride when building the DB prior map
DB_MAP_POINT_SIZE = 0.5
Q_MAP_POINT_SIZE = 0.05
DB_MAP_ALPHA  = 0.05
Q_MAP_ALPHA   = 0.05
TRAJ_LW       = 3.0
TRAJ_Z_LIFT   = 0.35
LOOP_EDGE_ALPHA   = 0.6
LOOP_FP_EDGE_ALPHA = 0.7
LOOP_EDGE_LW  = 1.5

AXIS_LEN = 5.0   # length of pose-axis triad arms (world units)
AXIS_LW  = 3.0
ZOOM_FACTOR = 0.99  # <1 zooms in (fraction of full extent kept)

KP_SIZE  = 8.0
KP_ALPHA = 0.9

# Right-side panel ------------------------------------------------------------
PANEL_KP_SIZE      = 8.0
PANEL_KP_ALPHA     = 0.9
PANEL_POINT_SIZE   = 1.5
PANEL_POINT_ALPHA  = 0.3
PANEL_DB_POINT_ALPHA = 0.2
PANEL_HALF         = 60.0  # fixed half-extent (m) for the keyframe panel

# Typography ------------------------------------------------------------------
TITLE_FONTSIZE     = 16
LEGEND_FONTSIZE    = 12
PANEL_TITLE_FONTSIZE  = 16
PANEL_LEGEND_FONTSIZE = 12
SHOW_MAIN_TITLE    = True
SHOW_MAIN_LEGEND   = True
SHOW_PANEL_TITLE   = True
SHOW_PANEL_LEGEND  = True

DB_TRAJ_COLOR = "#1f77b4"
Q_TRAJ_COLOR  = "#ff7f0e"
DB_KP_COLOR   = "#1f77b4"
Q_KP_COLOR    = "#ff7f0e"
DB_MAP_COLOR  = "#808080"
Q_MAP_COLOR   = "#808080"
TP_COLOR = "#2ca02c"
FP_COLOR = "#d62728"

PLAY_INTERVAL_MS = 100

# Containers ffmpeg can infer from a filename suffix (for --record).
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


# ── Helpers ─────────────────────────────────────────────────────────────────
def voxel_downsample_np(pts: np.ndarray, voxel_size: float) -> np.ndarray:
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


def clean_scan(pts: np.ndarray) -> np.ndarray:
    """Drop zero-padding rows (HeLiPR scans are padded) and cast to float32."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.size == 0:
        return pts.reshape(0, 3)
    return pts[np.any(pts != 0, axis=1)]


def build_map(point_clouds, poses_world, voxel_size, stride,
              desc="Building map") -> np.ndarray:
    """Accumulate strided keyframe scans into a world-frame prior map."""
    chunks = []
    for i in tqdm(range(0, len(point_clouds), stride), desc=desc, unit="kf"):
        pts = clean_scan(point_clouds[i])
        if pts.size == 0:
            continue
        pts = voxel_downsample_np(pts, voxel_size)
        T = poses_world[i]
        pts = pts @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)
        chunks.append(pts)
    if not chunks:
        return np.empty((0, 3), dtype=np.float32)
    return voxel_downsample_np(np.concatenate(chunks, axis=0), voxel_size)


def transform_local_to_world(local_pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    if local_pts.size == 0:
        return local_pts
    R = T[:3, :3].astype(np.float32)
    t = T[:3, 3].astype(np.float32)
    return local_pts @ R.T + t


def transform_pts(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def find_cache(cache_dir: Path, sequence: str, sensor: str, seq_type: str) -> Path:
    """Locate the descriptor cache npz for a sequence/sensor/type."""
    pattern = str(cache_dir / f"desc_{sequence}_{sensor}_{seq_type}_*.npz")
    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not hits:
        raise FileNotFoundError(
            f"No descriptor cache matching {pattern}. Run "
            f"evaluate_inlier_helipr.py first (it writes the cache).")
    if len(hits) > 1:
        print(f"  [cache] {len(hits)} matches for {Path(pattern).name}; "
              f"using newest: {Path(hits[-1]).name}")
    return Path(hits[-1])


def load_results_json(output_dir: Path, explicit: Path | None) -> tuple[Path, dict]:
    """Locate and load the eval results JSON; return (path, the whole document).

    This is the single source of truth for the run's identity (sequences,
    sensors, GT thresholds), token grid and score threshold, so playback cannot
    disagree with the evaluation that produced the artifacts it renders.
    """
    if explicit is not None:
        path = explicit
        if not path.exists():
            raise FileNotFoundError(f"Results JSON not found: {path}")
    else:
        hits = sorted(glob.glob(str(output_dir / "results_*.json")),
                      key=os.path.getmtime)
        if not hits:
            raise FileNotFoundError(
                f"No results_*.json in {output_dir}. Run "
                f"evaluate_inlier_helipr.py first, or pass --results_json.")
        if len(hits) > 1:
            print(f"  [results] {len(hits)} results JSONs in {output_dir.name}; "
                  f"using newest: {Path(hits[-1]).name}  "
                  f"(pass --results_json to pin one)")
        path = Path(hits[-1])

    with open(path) as f:
        data = json.load(f) or {}
    cfg = data.get("config", {})
    missing = [k for k in ("db_sequence", "db_sensor", "q_sequence", "q_sensor")
               if k not in cfg]
    if missing:
        raise KeyError(f"{path.name} has no config/{{{','.join(missing)}}}; "
                       f"it predates this script's expectations.")
    return path, data


def score_threshold_from_results(data: dict) -> float:
    """The threshold the eval used when it picked each query's top-1.

    Every TP/FP row in the candidates CSV already satisfies it — the eval
    dropped anything below (those rows became FN/TN with score 0.0 and
    predicted_db_idx -1). Re-applying it here is a no-op that keeps the
    printed header honest; -inf when a JSON predates the field.
    """
    thr = data.get("confusion", {}).get("threshold")
    return float(thr) if thr is not None else float("-inf")


def token_grid_from_cfg(cfg: dict) -> tuple:
    """Read (NH, NR, NA, NS) from the results JSON's config/inlier block."""
    enc = cfg.get("inlier", {})
    return (int(enc.get("N_h", 10)), int(enc.get("N_r", 20)),
            int(enc.get("N_a", 60)), int(enc.get("N_s", 7)))


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Animated replay of a HeLiPR InLiER evaluation run.")
    # Eval outputs — the run's identity (sequences, sensors, GT thresholds,
    # token grid) is read from results_*.json here, never passed on the CLI.
    ap.add_argument("--output_dir", type=Path, required=True,
                    help="Eval output folder (results JSON + CSVs live here).")
    ap.add_argument("--results_json", type=Path, default=None,
                    help="Pin a specific results_*.json (default: newest in "
                         "--output_dir).")
    # Things the eval run does not record
    ap.add_argument("--dataset", type=Path,
                    default=Path("~/Documents/datasets/HeLiPR").expanduser(),
                    help="HeLiPR dataset root (contains the sequence folders).")
    ap.add_argument("--cache_dir", type=Path, default=Path("cache_inlier"))
    ap.add_argument("--db_type", default="Undistorted",
                    help="Scan subfolder / cache tag; must match the seq_type "
                         "evaluate_inlier_helipr.py encoded with.")
    ap.add_argument("--q_type", default="Undistorted",
                    help="Scan subfolder / cache tag; must match the seq_type "
                         "evaluate_inlier_helipr.py encoded with.")
    # Explicit overrides (skip auto-discovery)
    ap.add_argument("--candidates_csv", type=Path, default=None)
    ap.add_argument("--verify_csv", type=Path, default=None)
    ap.add_argument("--db_cache", type=Path, default=None)
    ap.add_argument("--q_cache", type=Path, default=None)
    # Display
    ap.add_argument("--score_col", default="score")
    # Recording
    ap.add_argument("--record", type=Path, default=None,
                    help="If set, render all keyframes to this MP4 and exit.")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    # Backend: Agg for headless recording, TkAgg for interactive playback.
    if args.record is not None:
        # Validate up front: loading scans and building the DB map costs ~40 s,
        # and ffmpeg only reports a bad output path once the first frame lands.
        if args.record.is_dir():
            raise ValueError(
                f"--record must be a video file, not a directory: {args.record}")
        if args.record.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(
                f"--record needs a video extension "
                f"({', '.join(sorted(VIDEO_SUFFIXES))}); ffmpeg infers the "
                f"container from it. Got: {args.record}")
        matplotlib.use("Agg", force=True)
    elif not os.environ.get("DISPLAY"):
        raise RuntimeError("No display found; pass --record <file.mp4> to "
                           "render headlessly.")
    else:
        matplotlib.use("TkAgg", force=True)

    SCORE_COL = args.score_col

    # ── Run identity: read from the eval's own results JSON ──────────────────
    results_json, results = load_results_json(args.output_dir, args.results_json)
    cfg = results["config"]
    db_sequence, db_sensor = cfg["db_sequence"], cfg["db_sensor"]
    q_sequence,  q_sensor  = cfg["q_sequence"],  cfg["q_sensor"]
    THRESHOLD = score_threshold_from_results(results)

    pair_tag = (f"{db_sequence}_{db_sensor}_"
                f"{q_sequence}_{q_sensor}_"
                f"ov{cfg['overlap_threshold']}_pd{cfg['max_pose_dist']}m")

    candidates_csv = args.candidates_csv or (
        args.output_dir / f"candidates_{pair_tag}.csv")
    verify_csv = args.verify_csv or (
        args.output_dir / f"per_pair_verify_{pair_tag}.csv")
    if not candidates_csv.exists():
        raise FileNotFoundError(f"Candidates CSV not found: {candidates_csv}")

    db_cache = args.db_cache or find_cache(
        args.cache_dir, db_sequence, db_sensor, args.db_type)
    q_cache = args.q_cache or find_cache(
        args.cache_dir, q_sequence, q_sensor, args.q_type)

    NH, NR, NA, NS = token_grid_from_cfg(cfg)
    print(f"Pair: {pair_tag}")
    print(f"  results    : {results_json.name}")
    print(f"  candidates : {candidates_csv}")
    print(f"  verify     : {verify_csv}{'' if verify_csv.exists() else '  (missing — panel shows GT frame only)'}")
    print(f"  DB cache   : {db_cache}")
    print(f"  Q  cache   : {q_cache}")
    print(f"  token grid : NH={NH} NR={NR} NA={NA} NS={NS}")

    # ── Descriptor caches ────────────────────────────────────────────────────
    db_npz = np.load(db_cache)
    q_npz  = np.load(q_cache)
    # Poses are already in the shared HeLiPR global frame (no inter-transform).
    db_poses_world = np.asarray(db_npz["poses"], dtype=np.float64)
    q_poses_world  = np.asarray(q_npz["poses"], dtype=np.float64)
    db_off, db_kp_sensor = db_npz["offsets"], db_npz["kp_sensor"]
    q_off,  q_kp_sensor  = q_npz["offsets"],  q_npz["kp_sensor"]
    db_tokens = db_npz["token_ids"]
    q_tokens  = q_npz["token_ids"]

    db_xy = db_poses_world[:, :2, 3]
    q_xy  = q_poses_world[:, :2, 3]
    N_q_kf = len(q_poses_world)

    # ── Raw scans via the handler (per-keyframe, sensor frame) ───────────────
    handler = HeLiPR_Handler(dataset_path=str(args.dataset), verbose=False)
    print("Loading DB scans…")
    db_data = handler.load_helipr(db_sequence, db_sensor, type=args.db_type)
    print("Loading Q scans…")
    q_data  = handler.load_helipr(q_sequence, q_sensor, type=args.q_type)
    db_clouds = db_data["point_clouds"]
    q_clouds  = q_data["point_clouds"]
    if len(db_clouds) < len(db_poses_world) or len(q_clouds) < N_q_kf:
        print(f"  [warn] scan count (DB {len(db_clouds)}, Q {len(q_clouds)}) < "
              f"cache keyframes (DB {len(db_poses_world)}, Q {N_q_kf}); "
              f"missing scans render empty.")

    # Per-submap world-frame keypoints (precomputed, cheap)
    def kp_world(idx, off, kp_sensor, poses_world):
        a, b = int(off[idx]), int(off[idx + 1])
        if b <= a:
            return np.empty((0, 3), dtype=np.float32)
        return transform_pts(kp_sensor[a:b].astype(np.float32),
                             poses_world[idx].astype(np.float32))

    # ── CSV → loop closures bucketed by query_idx ────────────────────────────
    df = pd.read_csv(candidates_csv)
    pred_pos = df[SCORE_COL] >= THRESHOLD
    valid = df[pred_pos & df["match_type"].isin(["TP", "FP"])]
    closures_by_q = {}  # qi -> list of (dbi, match_type, score)
    for _, row in valid.iterrows():
        qi, dbi, mt = int(row["query_idx"]), int(row["predicted_db_idx"]), row["match_type"]
        sc = float(row[SCORE_COL])
        if 0 <= qi < N_q_kf and 0 <= dbi < len(db_poses_world):
            closures_by_q.setdefault(qi, []).append((dbi, mt, sc))
    total_tp = int((valid["match_type"] == "TP").sum())
    total_fp = int((valid["match_type"] == "FP").sum())
    thr_str = f"{THRESHOLD:.3f} (from eval)" if np.isfinite(THRESHOLD) else "none recorded"
    print(f"Threshold={thr_str}  TP={total_tp}  FP={total_fp}  Q keyframes={N_q_kf}")

    # ── Verify CSV → (qi, dbi) -> T_sensor (query→DB). Successful entries only.
    verify_by_pair: dict[tuple[int, int], np.ndarray] = {}
    if verify_csv.exists():
        vdf = pd.read_csv(verify_csv)
        if "success" in vdf.columns:
            vdf = vdf[vdf["success"].astype(int) == 1]
        for _, r in vdf.iterrows():
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = np.array([[r["r00"], r["r01"], r["r02"]],
                                  [r["r10"], r["r11"], r["r12"]],
                                  [r["r20"], r["r21"], r["r22"]]], dtype=np.float64)
            T[:3, 3] = (r["tx"], r["ty"], r["tz"])
            verify_by_pair[(int(r["query_idx"]), int(r["db_idx"]))] = T

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    right_gs = gs[0, 1].subgridspec(
        4, 2, height_ratios=[2.0, 0.9, 0.25, 0.7], hspace=0.55, wspace=0.15)
    ax2 = fig.add_subplot(right_gs[0, :], projection="3d")
    ax_full_q = fig.add_subplot(right_gs[1, 0])
    ax_full_db = fig.add_subplot(right_gs[1, 1])
    ax_mint_q = fig.add_subplot(right_gs[2, 0])
    ax_mint_db = fig.add_subplot(right_gs[2, 1])
    ax_beam_q = fig.add_subplot(right_gs[3, 0])
    ax_beam_db = fig.add_subplot(right_gs[3, 1])
    for axd in (ax_full_q, ax_full_db, ax_mint_q, ax_mint_db,
                ax_beam_q, ax_beam_db):
        axd.set_xticks([]); axd.set_yticks([])
        for s in axd.spines.values():
            s.set_visible(False)
    ax2.set_facecolor("white")
    ax2.set_axis_off()
    for pane in (ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")
    ax2.grid(False)
    PANEL_ELEV, PANEL_AZIM = 30, 160
    ax2.view_init(elev=PANEL_ELEV, azim=PANEL_AZIM)
    if SHOW_PANEL_TITLE:
        ax2.set_title("Current keyframe", fontsize=PANEL_TITLE_FONTSIZE)
    ax.set_facecolor("white")
    ax.set_axis_off()
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")
    ax.grid(False)
    ax.view_init(elev=30, azim=-60)

    xr = float(np.ptp(np.r_[db_xy[:, 0], q_xy[:, 0]]))
    yr = float(np.ptp(np.r_[db_xy[:, 1], q_xy[:, 1]]))
    ax.set_box_aspect((xr, yr, max(xr, yr) * Z_SQUASH))

    db_traj_z = TRAJ_Z_LIFT
    q_traj_z  = Z_OFFSET + TRAJ_Z_LIFT  # trajectory above Q map

    # Zoom: tighten xy limits around the scene centre
    all_xy = np.r_[db_xy, q_xy]
    cx, cy = all_xy.mean(axis=0)
    half_x = (np.ptp(all_xy[:, 0]) / 2) * ZOOM_FACTOR
    half_y = (np.ptp(all_xy[:, 1]) / 2) * ZOOM_FACTOR
    ax.set_xlim(cx - half_x, cx + half_x)
    ax.set_ylim(cy - half_y, cy + half_y)
    ax.set_zlim(-1.0, Z_OFFSET + 2.0)

    # Compensate for Z-axis visual stretch from box_aspect: scale the Z arm
    # of the pose triad so it has the same on-screen length as the XY arms.
    box_x, _, box_z = ax.get_box_aspect()
    z_axis_scale = (box_x / (2 * half_x)) / (box_z / (Z_OFFSET + 3.0))

    # ── DB layer (drawn once) ────────────────────────────────────────────────
    db_map = build_map(db_clouds, db_poses_world, DB_VOXEL_SIZE, DB_MAP_STRIDE,
                       desc="Building DB prior map")
    print(f"  DB map: {len(db_map)} pts")
    ax.scatter(db_map[:, 0], db_map[:, 1], np.zeros(len(db_map)),
               c=DB_MAP_COLOR, s=DB_MAP_POINT_SIZE, alpha=DB_MAP_ALPHA, linewidths=0,
               depthshade=False)
    ax.plot(db_xy[:, 0], db_xy[:, 1], np.full(len(db_xy), db_traj_z),
            color=DB_TRAJ_COLOR, linewidth=TRAJ_LW, zorder=10)

    # ── Per-keyframe local scans (sensor frame). One scan per keyframe. ───────
    def get_q_local(i: int) -> np.ndarray:
        if not (0 <= i < len(q_clouds)):
            return np.empty((0, 3), dtype=np.float32)
        return voxel_downsample_np(clean_scan(q_clouds[i]), Q_VOXEL_SIZE)

    def get_db_local(i: int) -> np.ndarray:
        if not (0 <= i < len(db_clouds)):
            return np.empty((0, 3), dtype=np.float32)
        return voxel_downsample_np(clean_scan(db_clouds[i]), DB_VOXEL_SIZE)

    def get_q_world(i: int) -> np.ndarray:
        return transform_local_to_world(get_q_local(i), q_poses_world[i])

    # ── Q dynamic artists ────────────────────────────────────────────────────
    axis_colors = ("#e41a1c", "#4daf4a", "#377eb8")
    pose_axes = [
        ax.plot([0, 0], [0, 0], [0, 0], color=c, linewidth=AXIS_LW,
                zorder=25)[0] for c in axis_colors
    ]

    def update_pose_axes(idx: int):
        T = q_poses_world[idx]
        origin = np.array([T[0, 3], T[1, 3], q_traj_z])
        R = T[:3, :3]
        for k, line in enumerate(pose_axes):
            d = R[:, k] * AXIS_LEN
            tip = origin + np.array([d[0], d[1], d[2] * z_axis_scale])
            line.set_data_3d(
                [origin[0], tip[0]],
                [origin[1], tip[1]],
                [origin[2], tip[2]],
            )

    update_pose_axes(0)

    q_traj_line, = ax.plot([q_xy[0, 0]], [q_xy[0, 1]], [q_traj_z],
                           color=Q_TRAJ_COLOR, linewidth=TRAJ_LW, zorder=11)

    frame_artists: dict[int, list] = {i: [] for i in range(N_q_kf)}
    drawn_frames: set[int] = set()
    current_kp_artists: list = []

    def verify_T_db_to_q(qi: int, dbi: int):
        """Estimated DB->Q relative transform for the panel (inv of query->DB)."""
        T = verify_by_pair.get((qi, dbi))
        if T is None:
            return None
        return np.linalg.inv(T)

    title = ax.set_title("", fontsize=TITLE_FONTSIZE, pad=10)
    if not SHOW_MAIN_TITLE:
        title.set_visible(False)

    state = {"i": -1, "tp": 0, "fp": 0, "playing": False}

    def update_title():
        title.set_text(f"Q keyframe {state['i']+1}/{N_q_kf}    "
                       f"TP {state['tp']}    FP {state['fp']}    "
                       f"[{'PLAYING' if state['playing'] else 'PAUSED'}]")

    def reveal(i: int):
        if i in drawn_frames or not (0 <= i < N_q_kf):
            return
        artists = []
        seg = q_xy[: i + 1]
        q_traj_line.set_data_3d(seg[:, 0], seg[:, 1],
                                np.full(len(seg), q_traj_z))
        update_pose_axes(i)

        sub = get_q_world(i)
        if len(sub):
            sc = ax.scatter(sub[:, 0], sub[:, 1],
                            np.full(len(sub), Z_OFFSET),
                            c=Q_MAP_COLOR, s=Q_MAP_POINT_SIZE, alpha=Q_MAP_ALPHA,
                            linewidths=0, depthshade=False)
            artists.append(sc)

        for dbi, mt, _sc in closures_by_q.get(i, []):
            color = TP_COLOR if mt == "TP" else FP_COLOR
            alpha = LOOP_EDGE_ALPHA if mt == "TP" else LOOP_FP_EDGE_ALPHA
            ln, = ax.plot(
                [q_xy[i, 0], db_xy[dbi, 0]],
                [q_xy[i, 1], db_xy[dbi, 1]],
                [q_traj_z, db_traj_z],
                color=color, linewidth=LOOP_EDGE_LW, alpha=alpha, zorder=12,
            )
            artists.append(ln)
            if mt == "TP":
                state["tp"] += 1
            else:
                state["fp"] += 1

        frame_artists[i] = artists
        drawn_frames.add(i)

    def hide(i: int):
        if i not in drawn_frames:
            return
        for a in frame_artists[i]:
            try:
                a.remove()
            except Exception:
                pass
        frame_artists[i] = []
        drawn_frames.discard(i)
        for dbi, mt, _sc in closures_by_q.get(i, []):
            if mt == "TP":
                state["tp"] -= 1
            else:
                state["fp"] -= 1

    def refresh_keypoints(i: int):
        for a in current_kp_artists:
            try:
                a.remove()
            except Exception:
                pass
        current_kp_artists.clear()
        if not (0 <= i < N_q_kf):
            return
        qkp = kp_world(i, q_off, q_kp_sensor, q_poses_world)
        if len(qkp):
            sc = ax.scatter(qkp[:, 0], qkp[:, 1],
                            np.full(len(qkp), Z_OFFSET),
                            c=Q_KP_COLOR, s=KP_SIZE, alpha=KP_ALPHA,
                            linewidths=0, depthshade=False, zorder=15)
            current_kp_artists.append(sc)
        for dbi, _mt, _sc in closures_by_q.get(i, []):
            dkp = kp_world(dbi, db_off, db_kp_sensor, db_poses_world)
            if len(dkp):
                sc = ax.scatter(dkp[:, 0], dkp[:, 1],
                                np.zeros(len(dkp)),
                                c=DB_KP_COLOR, s=KP_SIZE, alpha=KP_ALPHA,
                                linewidths=0, depthshade=False, zorder=15)
                current_kp_artists.append(sc)

    def refresh_panel(i: int):
        ax2.clear()
        ax2.set_facecolor("white")
        ax2.set_axis_off()
        for pane in (ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("none")
        ax2.grid(False)
        ax2.view_init(elev=PANEL_ELEV, azim=PANEL_AZIM)
        if not (0 <= i < N_q_kf):
            if SHOW_PANEL_TITLE:
                ax2.set_title("Current keyframe", fontsize=PANEL_TITLE_FONTSIZE)
            return

        qsub = get_q_local(i)
        if len(qsub):
            ax2.scatter(qsub[:, 0], qsub[:, 1], qsub[:, 2],
                        s=PANEL_POINT_SIZE, c=Q_MAP_COLOR,
                        alpha=PANEL_POINT_ALPHA, linewidths=0,
                        depthshade=False)
        a, b = int(q_off[i]), int(q_off[i + 1])
        if b > a:
            qkp = q_kp_sensor[a:b]
            ax2.scatter(qkp[:, 0], qkp[:, 1], qkp[:, 2],
                        s=PANEL_KP_SIZE, c=Q_KP_COLOR,
                        alpha=PANEL_KP_ALPHA, linewidths=0, depthshade=False)

        L = max(2.0, Q_VOXEL_SIZE * 4)
        ax2.plot([0, L], [0, 0], [0, 0], color="#e41a1c", linewidth=2)
        ax2.plot([0, 0], [0, L], [0, 0], color="#4daf4a", linewidth=2)
        ax2.plot([0, 0], [0, 0], [0, L], color="#377eb8", linewidth=2)

        # DB candidates aligned via verify pose; compare to GT relative pose.
        T_q_inv = np.linalg.inv(q_poses_world[i])
        err_lines = []
        for dbi, mt, _sc in closures_by_q.get(i, []):
            T_est = verify_T_db_to_q(i, dbi)   # DB->Q, estimated
            color = TP_COLOR if mt == "TP" else FP_COLOR
            # GT DB->Q relative pose: inv(T_q) @ T_db
            T_gt = T_q_inv @ db_poses_world[dbi]
            if T_est is not None:
                R3, t3 = T_est[:3, :3], T_est[:3, 3]
                dsub = get_db_local(dbi)
                if len(dsub):
                    p = dsub @ R3.T + t3
                    ax2.scatter(p[:, 0], p[:, 1], p[:, 2],
                                s=PANEL_POINT_SIZE, c=color,
                                alpha=PANEL_DB_POINT_ALPHA, linewidths=0,
                                depthshade=False)
                da, db_ = int(db_off[dbi]), int(db_off[dbi + 1])
                if db_ > da:
                    dkp = db_kp_sensor[da:db_] @ R3.T + t3
                    ax2.scatter(dkp[:, 0], dkp[:, 1], dkp[:, 2],
                                s=PANEL_KP_SIZE, c=color, alpha=PANEL_KP_ALPHA,
                                linewidths=0, depthshade=False)
                t_err = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
                R_rel = T_est[:3, :3].T @ T_gt[:3, :3]
                cos_a = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
                r_err = float(np.degrees(np.arccos(cos_a)))
                err_lines.append(f"{mt} db{dbi}: Δt={t_err:.2f}m  Δr={r_err:.1f}°")
            else:
                # No estimated pose available: show DB scan in GT frame.
                dsub = get_db_local(dbi)
                if len(dsub):
                    p = dsub @ T_gt[:3, :3].T + T_gt[:3, 3]
                    ax2.scatter(p[:, 0], p[:, 1], p[:, 2],
                                s=PANEL_POINT_SIZE, c=color,
                                alpha=PANEL_DB_POINT_ALPHA, linewidths=0,
                                depthshade=False)

        ax2.set_xlim(-PANEL_HALF, PANEL_HALF)
        ax2.set_ylim(-PANEL_HALF, PANEL_HALF)
        ax2.set_zlim(-PANEL_HALF * 0.5, PANEL_HALF * 0.5)
        ax2.set_box_aspect((1, 1, 0.5))

        panel_legend = [
            Line2D([0], [0], marker="o", color="none", markeredgecolor="none",
                   markerfacecolor=Q_MAP_COLOR, markersize=6, label="cloud"),
            Line2D([0], [0], marker="o", color="none", markeredgecolor="none",
                   markerfacecolor=Q_KP_COLOR, markersize=6, label="Q kp"),
            Line2D([0], [0], marker="o", color="none", markeredgecolor="none",
                   markerfacecolor=TP_COLOR, markersize=6, label="TP DB kp"),
            Line2D([0], [0], marker="o", color="none", markeredgecolor="none",
                   markerfacecolor=FP_COLOR, markersize=6, label="FP DB kp"),
        ]
        if SHOW_PANEL_LEGEND:
            ax2.legend(handles=panel_legend, frameon=False,
                       fontsize=PANEL_LEGEND_FONTSIZE,
                       loc="upper right", bbox_to_anchor=(-0.05, 1.0))

        if SHOW_PANEL_TITLE:
            ptitle = f"Q kf {i + 1}/{N_q_kf}"
            if err_lines:
                ptitle += "\n" + "\n".join(err_lines)
            ax2.set_title(ptitle, fontsize=PANEL_TITLE_FONTSIZE)

    def _clear_axis(axd):
        axd.clear()
        axd.set_xticks([]); axd.set_yticks([])
        for s in axd.spines.values():
            s.set_visible(False)

    beam_cfg = _BEAMCfg()
    Rw = NR * NS
    V = NH * Rw

    def refresh_descriptors(i: int):
        for axd in (ax_full_q, ax_full_db, ax_mint_q, ax_mint_db,
                    ax_beam_q, ax_beam_db):
            _clear_axis(axd)

        if not (0 <= i < N_q_kf):
            return
        cls = closures_by_q.get(i, [])
        if not cls:
            return
        dbi, mt, sc = max(cls, key=lambda c: c[2])
        color = TP_COLOR if mt == "TP" else FP_COLOR

        qa, qb = int(q_off[i]), int(q_off[i + 1])
        da, db_ = int(db_off[dbi]), int(db_off[dbi + 1])
        q_tok = q_tokens[qa:qb]
        d_tok = db_tokens[da:db_]

        q_full = _Matcher._tokens_to_hist(q_tok, NA, V, NH, Rw)
        d_full = _Matcher._tokens_to_hist(d_tok, NA, V, NH, Rw)

        q_hb, q_rb, _q_sb, q_ab = _InLiER.unpack_token_ids(q_tok, NR, NS, NA)
        d_hb, d_rb, _d_sb, d_ab = _InLiER.unpack_token_ids(d_tok, NR, NS, NA)
        q_max_hb = int(q_hb.max()) if q_hb.size else -1
        d_max_hb = int(d_hb.max()) if d_hb.size else -1
        ceiling = min(q_max_hb, d_max_hb)

        q_pruned = q_full.copy()
        d_pruned = d_full.copy()
        if ceiling >= 0:
            q_pruned[ceiling + 1:] = 0
            d_pruned[ceiling + 1:] = 0
        q_compact = q_pruned.sum(axis=0, keepdims=True)
        d_compact = d_pruned.sum(axis=0, keepdims=True)

        vmax_full = max(q_full.max(), d_full.max(), 1.0)
        vmax_cmp  = max(q_compact.max(), d_compact.max(), 1.0)
        for axd, mat in ((ax_full_q, q_full), (ax_full_db, d_full)):
            axd.imshow(mat, aspect="auto", cmap="viridis",
                       vmin=0, vmax=vmax_full, interpolation="nearest",
                       origin="lower")
        for axd, mat in ((ax_mint_q, q_compact), (ax_mint_db, d_compact)):
            axd.imshow(mat, aspect="auto", cmap="viridis",
                       vmin=0, vmax=vmax_cmp, interpolation="nearest",
                       origin="lower")

        if ceiling >= 0 and ceiling + 1 < NH:
            for axd in (ax_full_q, ax_full_db):
                axd.add_patch(Rectangle(
                    (-0.5, ceiling + 0.5), Rw, NH - ceiling - 1,
                    linewidth=1.5, edgecolor="red",
                    facecolor="none", zorder=5))

        ax_full_q.set_title(
            r"$\mathcal{H}_{\mathrm{Q}}$",
            fontsize=PANEL_LEGEND_FONTSIZE, color=Q_KP_COLOR)
        ax_full_db.set_title(
            rf"$\mathcal{{H}}_{{\mathrm{{DB}}}}$  ceiling hb={ceiling}",
            fontsize=PANEL_LEGEND_FONTSIZE, color=color)
        ax_mint_q.set_title(r"$\mathcal{R}_{\mathrm{Q}}$",
                            fontsize=PANEL_LEGEND_FONTSIZE, color=Q_KP_COLOR)
        ax_mint_db.set_title(rf"$\mathcal{{R}}_{{\mathrm{{DB}}}}$ {mt}  $S_M=${sc:.2f}",
                             fontsize=PANEL_LEGEND_FONTSIZE, color=color)

        q_beam = _Matcher._build_beam(q_hb, q_rb, q_ab, NH, NR, NA)
        d_beam = _Matcher._build_beam(d_hb, d_rb, d_ab, NH, NR, NA)
        beam_score, beam_shift = _Matcher._score_beam_shifts(
            q_beam, q_max_hb, d_beam, d_max_hb, beam_cfg)

        def _popcount(mat):
            x = mat.astype(np.uint64)
            x = x - ((x >> 1) & 0x5555555555555555)
            x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
            x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
            return ((x * 0x0101010101010101) >> 56).astype(np.int32)
        q_vis = _popcount(q_beam)
        d_vis = _popcount(np.roll(d_beam, beam_shift, axis=1))
        bvmax = max(q_vis.max(), d_vis.max(), 1)
        ax_beam_q.imshow(q_vis, aspect="auto", cmap="inferno",
                         vmin=0, vmax=bvmax, interpolation="nearest",
                         origin="lower")
        ax_beam_db.imshow(d_vis, aspect="auto", cmap="inferno",
                          vmin=0, vmax=bvmax, interpolation="nearest",
                          origin="lower")
        ax_beam_q.set_title(r"$\mathcal{A}_{\mathrm{Q}}$", fontsize=PANEL_LEGEND_FONTSIZE,
                            color=Q_KP_COLOR)
        ax_beam_db.set_title(
            rf"$\mathcal{{A}}_{{\mathrm{{DB}}}}$  {mt} (shift={beam_shift})  $S_B=${beam_score:.2f}",
            fontsize=PANEL_LEGEND_FONTSIZE, color=color)

    def goto(target: int):
        target = max(-1, min(N_q_kf - 1, target))
        if target > state["i"]:
            for k in range(state["i"] + 1, target + 1):
                reveal(k)
        elif target < state["i"]:
            for k in range(state["i"], target, -1):
                hide(k)
            if target >= 0:
                seg = q_xy[: target + 1]
                q_traj_line.set_data_3d(seg[:, 0], seg[:, 1],
                                        np.full(len(seg), q_traj_z))
                update_pose_axes(target)
            else:
                q_traj_line.set_data_3d([q_xy[0, 0]], [q_xy[0, 1]], [q_traj_z])
                update_pose_axes(0)
        state["i"] = target
        refresh_keypoints(target)
        refresh_panel(target)
        refresh_descriptors(target)
        update_title()
        fig.canvas.draw_idle()

    state["i"] = -1
    update_title()

    def tick(_frame):
        if not state["playing"]:
            return ()
        if state["i"] >= N_q_kf - 1:
            state["playing"] = False
            update_title()
            return ()
        goto(state["i"] + 1)
        return ()

    anim = FuncAnimation(fig, tick, interval=PLAY_INTERVAL_MS,   # noqa: F841
                         blit=False, cache_frame_data=False)

    def on_key(event):
        if event.key == " ":
            state["playing"] = not state["playing"]
            update_title()
            fig.canvas.draw_idle()
        elif event.key == "right":
            state["playing"] = False
            goto(state["i"] + 1)
        elif event.key == "left":
            state["playing"] = False
            goto(state["i"] - 1)

    fig.canvas.mpl_connect("key_press_event", on_key)

    legend = [
        Line2D([0], [0], color=DB_TRAJ_COLOR, lw=TRAJ_LW,
               label=f"DB: {db_sensor}"),
        Line2D([0], [0], color=Q_TRAJ_COLOR,  lw=TRAJ_LW,
               label=f"Query: {q_sensor}"),
        Line2D([0], [0], color=TP_COLOR, lw=LOOP_EDGE_LW, label="TP"),
        Line2D([0], [0], color=FP_COLOR, lw=LOOP_EDGE_LW, label="FP"),
    ]
    if SHOW_MAIN_LEGEND:
        ax.legend(handles=legend, frameon=False,
                  fontsize=LEGEND_FONTSIZE, loc="upper left")

    plt.tight_layout()

    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        writer = FFMpegWriter(fps=args.fps, codec="libx264",
                              extra_args=["-pix_fmt", "yuv420p"])
        print(f"Recording {N_q_kf} frames → {args.record}")
        with writer.saving(fig, str(args.record), dpi=args.dpi):
            writer.grab_frame()  # initial frame (empty Q side)
            for k in tqdm(range(N_q_kf), desc="Rendering frames", unit="frame"):
                goto(k)
                writer.grab_frame()
        print(f"Saved {args.record}")
        return

    print("Controls: SPACE = play/pause, ← / → = step")
    plt.show()


if __name__ == "__main__":
    main()
