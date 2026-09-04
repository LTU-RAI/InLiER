#!/usr/bin/env python3
"""``inlier play`` -- animated replay of a finished evaluation run.

  - DB map + DB trajectory drawn once (full prior).
  - Q trajectory, Q scan, Q keypoints reveal incrementally per keyframe.
  - TP/FP loop edges + matched DB keypoints appear at the query keyframe
    where the closure occurs.
  - Right panel: current keyframe with DB candidate aligned by the estimated
    verify pose, plus MINT / BEAM descriptor visualisations.
  - Controls: SPACE play/pause, LEFT/RIGHT step.

Loader-agnostic: it replays a HeLiPR run and a generic (.pcd + poses) run the
same way, because it rebuilds the run's own ``SequenceSource`` from the
provenance the evaluation wrote rather than assuming one.  For a generic run
that means the submap accumulation comes back with it -- ``--n-scans`` and
``--stride`` are never retyped here, so a replay cannot window the sequence
differently from the run it is replaying.

It consumes the artifacts written by ``inlier eval cross-session``:
  - ``results_{tag}.json``            (run identity, token grid, file naming)
  - ``candidates_{tag}.csv``          (loop closures + TP/FP labels)
  - ``per_pair_verify_{tag}.csv``     (per-pair estimated poses; optional)
  - ``desc_{cache_tag}_*.npz``        descriptor caches (DB and Q)
and reloads the raw scans through the loader named in the results JSON.

Everything about the run's identity -- sequences, sensors or paths, submap
accumulation, GT thresholds, token grid, score threshold, and the tag every
filename is built from -- is read back out of the results JSON, so playback
always describes the run it is rendering.  Only the cache directory, and a
dataset root when the data has moved, are passed on the CLI.

``--run-dir`` is the run being replayed and is read only: the sole thing this
writes is ``--record``.

The TP/FP labels are the eval's, not this script's: it renders the closures in
``candidates_{tag}.csv`` as labelled.  To see a different set, re-run the
evaluation with a different --threshold / GT definition.

Example
-------
inlier play \
    --run-dir results/HeLiPR/dbR01-O-qR03-Aeva_vs0.5_cs1_nh10_nr20_na60_ns7 \
    --cache-dir cache_inlier \
    --record /path/to/output.mp4
"""

import argparse
import glob
import json
import os
from functools import lru_cache
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
from inlier.eval.datasets import source_from_describe


# ── Visualisation constants (tunable) ───────────────────────────────────────
Z_OFFSET   = 5.0
#: Downsampled submaps kept in memory; ~0.3 MB each at 1 m voxels.
LOCAL_SCAN_CACHE = 256
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
    """Voxel-grid mean of ``pts``, for display only.

    The encoder has its own downsampler (:func:`inlier.eval.encode.voxel_downsample`);
    this one exists so playback can thin a million-point submap before drawing
    it, and it is the hot path of the whole animation -- one accumulated submap
    is downsampled every time a frame is touched.

    Two things make it fast enough to sit in that loop.  ``np.unique(axis=0)``
    lexsorts a structured view of the (N, 3) coordinates; packing the three
    voxel indices into one integer first turns that into a plain 1-D sort.  The
    packing is positional and the coordinates are shifted non-negative, so its
    key order *is* the lexicographic row order -- the output is unchanged, not
    merely equivalent.  And ``np.add.at`` is an unbuffered scatter-add;
    ``np.bincount`` does the same reduction with a buffered one.
    """
    if voxel_size <= 0 or pts.shape[0] == 0:
        return pts
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return pts.astype(np.float32)

    coords = np.floor(pts / voxel_size).astype(np.int64)
    coords -= coords.min(axis=0)
    dims = [int(d) for d in coords.max(axis=0) + 1]
    if dims[0] * dims[1] * dims[2] < (1 << 62):
        keys = (coords[:, 0] * dims[1] + coords[:, 1]) * dims[2] + coords[:, 2]
        _, inv = np.unique(keys, return_inverse=True)
    else:
        # Extent too large to pack without overflowing int64; the slow path is
        # still correct, and a cloud this spread out is not the common case.
        _, inv = np.unique(coords, axis=0, return_inverse=True)

    inv = inv.ravel()
    K = int(inv.max()) + 1
    counts = np.bincount(inv, minlength=K)
    sums = np.stack(
        [np.bincount(inv, weights=pts[:, c], minlength=K) for c in range(3)],
        axis=1)
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


def session_label(described: dict) -> str:
    """How to name one session in the legend: the sensor, or the folder."""
    return described.get("sensor") or Path(described.get("path", "")).name


def load_clouds(described: dict, root: Path | None):
    """Per-keyframe point clouds for one session, via the run's own loader.

    One entry per *keyframe*, which for the generic loader means one per
    accumulated submap -- matching the descriptor cache index for index, which
    is what every panel here assumes.

    Verbose on purpose: the descriptor cache spares the *encoding*, not the
    reading, and a generic session re-accumulates every submap from its .pcd
    files.  That is minutes of silence otherwise.
    """
    source = source_from_describe(described, root=root, verbose=True)
    return source.load().point_clouds


def find_cache(cache_dir: Path, cache_tag: str) -> Path:
    """Locate the descriptor cache npz the run wrote under ``cache_tag``.

    The tag comes from the results JSON, so this cannot disagree with the name
    the evaluation actually used.  It differs per loader:
    ``Roundabout01_Ouster_Undistorted`` vs ``campus_ouster_n40s5_Undistorted``.
    """
    pattern = str(cache_dir / f"desc_{cache_tag}_*.npz")
    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not hits:
        raise FileNotFoundError(
            f"No descriptor cache matching {pattern}. Re-run `inlier eval` "
            f"with --cache-dir {cache_dir} (it writes the cache).")
    if len(hits) > 1:
        print(f"  [cache] {len(hits)} matches for {Path(pattern).name}; "
              f"using newest: {Path(hits[-1]).name}")
    return Path(hits[-1])


def load_results_json(run_dir: Path, explicit: Path | None) -> tuple[Path, dict]:
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
        hits = sorted(glob.glob(str(run_dir / "results_*.json")),
                      key=os.path.getmtime)
        if not hits:
            raise FileNotFoundError(
                f"No results_*.json in {run_dir}. Run "
                f"evaluate_inlier_helipr.py first, or pass --results-json.")
        if len(hits) > 1:
            print(f"  [results] {len(hits)} results JSONs in {run_dir.name}; "
                  f"using newest: {Path(hits[-1]).name}  "
                  f"(pass --results-json to pin one)")
        path = Path(hits[-1])

    with open(path) as f:
        data = json.load(f) or {}
    return path, data


def run_identity(data: dict, dataset_root) -> tuple[dict, dict, dict]:
    """``(db, query, artifacts)`` for this run, whichever schema wrote it.

    A single-session run records one ``session`` block instead of ``db`` and
    ``query``; it is returned for both, which is what makes the rest of the
    replay work unchanged.

    A current run records its loaders and its own file naming, so nothing has
    to be guessed.  Runs from 0.2.x recorded neither -- they were all HeLiPR,
    and playback rebuilt the names from ``config/db_sequence``.  The published
    results checked into this repository are still that shape, so the
    reconstruction stays as a fallback instead of being deleted; it needs
    ``--dataset``, which those runs also did not record.
    """
    if "session" in data and "artifacts" in data:
        # Single-session protocols (online-lcd): the database *is* the query
        # sequence, so both layers read the same loader and the same cache.
        session, art = data["session"], data["artifacts"]
        cache = art.get("cache")
        return session, session, {"tag": art.get("tag", ""),
                                  "db_cache": cache, "q_cache": cache,
                                  "db_transform": None}

    if all(k in data for k in ("db", "query", "artifacts")):
        return data["db"], data["query"], data["artifacts"]

    cfg = data.get("config", {})
    missing = [k for k in ("db_sequence", "db_sensor", "q_sequence", "q_sensor")
               if k not in cfg]
    if missing:
        raise KeyError(f"results JSON has no config/{{{','.join(missing)}}} and "
                       f"no db/query blocks; it predates every schema this "
                       f"understands.")
    if dataset_root is None:
        raise ValueError(
            "this run predates the recorded dataset path (InLiER 0.2.x); "
            "pass --dataset <root>, or re-run `inlier eval` to refresh it.")

    def _helipr(sequence, sensor):
        return {"dataset_type": "helipr", "dataset_path": str(dataset_root),
                "sequence": sequence, "sensor": sensor,
                "scan_type": "Undistorted"}

    tag = (f"{cfg['db_sequence']}_{cfg['db_sensor']}_"
           f"{cfg['q_sequence']}_{cfg['q_sensor']}_"
           f"ov{cfg['overlap_threshold']}_pd{cfg['max_pose_dist']}m")
    return (
        _helipr(cfg["db_sequence"], cfg["db_sensor"]),
        _helipr(cfg["q_sequence"], cfg["q_sensor"]),
        {"tag": tag,
         "db_cache": f"{cfg['db_sequence']}_{cfg['db_sensor']}_Undistorted",
         "q_cache": f"{cfg['q_sequence']}_{cfg['q_sensor']}_Undistorted",
         "db_transform": None},
    )


def score_threshold_from_results(data: dict) -> float:
    """The threshold the eval used when it picked each query's top-1.

    Every TP/FP row in the candidates CSV already satisfies it — the eval
    dropped anything below (those rows became FN/TN with score 0.0 and
    predicted_db_idx -1). Re-applying it here is a no-op that keeps the
    printed header honest; -inf when a JSON predates the field.
    """
    thr = data.get("confusion", {}).get("threshold")
    return float(thr) if thr is not None else float("-inf")


def frame_index_z(n_frames: int, lift: float = TRAJ_Z_LIFT,
                  span: float = Z_OFFSET) -> np.ndarray:
    """Per-frame z that carries the frame index, for single-session playback.

    Time goes on z: the trajectory climbs as the run proceeds, so the height a
    closure edge spans is how many frames the loop took to come back around.
    Raw indices would run to N and blow past the z-limits the box aspect is
    built around, so they are scaled onto ``[lift, lift + span]``.  Only the
    ordering and the proportions carry meaning -- the axis is not drawn.
    """
    if n_frames <= 0:
        return np.zeros(0, dtype=float)
    return lift + (np.arange(n_frames, dtype=float)
                   / max(n_frames - 1, 1)) * span


def token_grid_from_cfg(cfg: dict) -> tuple:
    """Read (NH, NR, NA, NS) from the results JSON's config/inlier block."""
    enc = cfg.get("inlier", {})
    return (int(enc.get("N_h", 10)), int(enc.get("N_r", 20)),
            int(enc.get("N_a", 60)), int(enc.get("N_s", 7)))


# ── Main ────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Animated replay of an InLiER evaluation run.")
    # Eval outputs — the run's identity (sequences, sensors, GT thresholds,
    # token grid) is read from results_*.json here, never passed on the CLI.
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="The finished run to replay: the folder `inlier eval` "
                         "wrote its results JSON and CSVs into. Read only; "
                         "playback writes nothing except --record.")
    ap.add_argument("--results-json", type=Path, default=None,
                    help="Pin a specific results_*.json (default: newest in "
                         "--run-dir).")
    # Things the eval run does not record
    ap.add_argument("--dataset", type=Path, default=None,
                    help="Dataset root, if it has moved since the run. "
                         "Defaults to the path recorded in the results JSON.")
    ap.add_argument("--cache-dir", type=Path, default=Path("cache_inlier"))
    # Explicit overrides (skip auto-discovery)
    ap.add_argument("--candidates-csv", type=Path, default=None)
    ap.add_argument("--verify-csv", type=Path, default=None)
    ap.add_argument("--db-cache", type=Path, default=None)
    ap.add_argument("--q-cache", type=Path, default=None)
    # Display
    ap.add_argument("--score-col", default="score")
    # Point-cloud density. These are the whole cost of a frame: an accumulated
    # submap is a million points, and everything drawn is a downsample of one.
    # Coarser voxels and a longer map stride trade detail for speed directly.
    ap.add_argument("--q-voxel-size", type=float, default=Q_VOXEL_SIZE,
                    help=f"voxel size (m) for the scans drawn per keyframe; "
                         f"larger is coarser and faster, 0 disables "
                         f"downsampling (default: {Q_VOXEL_SIZE})")
    ap.add_argument("--db-voxel-size", type=float, default=DB_VOXEL_SIZE,
                    help=f"voxel size (m) for database scans -- the prior map, "
                         f"and the matched frame in the keyframe panel "
                         f"(default: {DB_VOXEL_SIZE})")
    ap.add_argument("--db-map-stride", type=int, default=DB_MAP_STRIDE,
                    help=f"keyframe stride when building the database prior "
                         f"map; every Nth keyframe. Ignored by single-session "
                         f"runs, which have no prior map "
                         f"(default: {DB_MAP_STRIDE})")
    # Recording
    ap.add_argument("--record", type=Path, default=None,
                    help="If set, render all keyframes to this MP4 and exit.")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

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
    q_voxel, db_voxel = args.q_voxel_size, args.db_voxel_size
    db_map_stride = args.db_map_stride
    if db_map_stride < 1:
        raise ValueError(f"--db-map-stride must be >= 1, got {db_map_stride}")

    # ── Run identity: read from the eval's own results JSON ──────────────────
    results_json, results = load_results_json(args.run_dir, args.results_json)
    cfg = results["config"]
    db_described, q_described, written = run_identity(results, args.dataset)
    THRESHOLD = score_threshold_from_results(results)
    ## Shape, not a protocol whitelist: any single-session protocol writes one
    ## `session` block, and every one of them replays the same way.
    SINGLE_SESSION = "session" in results

    # The run recorded its own file naming, so nothing here reconstructs it.
    pair_tag = written["tag"]
    candidates_csv = args.candidates_csv or (
        args.run_dir / f"candidates_{pair_tag}.csv")
    verify_csv = args.verify_csv or (
        args.run_dir / f"per_pair_verify_{pair_tag}.csv")
    if not candidates_csv.exists():
        raise FileNotFoundError(f"Candidates CSV not found: {candidates_csv}")

    db_cache = args.db_cache or find_cache(args.cache_dir, written["db_cache"])
    q_cache = args.q_cache or find_cache(args.cache_dir, written["q_cache"])

    NH, NR, NA, NS = token_grid_from_cfg(cfg)
    print(f"{'Session' if SINGLE_SESSION else 'Pair'}: {pair_tag}")
    print(f"  results    : {results_json.name}")
    print(f"  candidates : {candidates_csv}")
    print(f"  verify     : {verify_csv}{'' if verify_csv.exists() else '  (missing — panel shows GT frame only)'}")
    if SINGLE_SESSION:
        print(f"  cache      : {q_cache}")
    else:
        print(f"  DB cache   : {db_cache}")
        print(f"  Q  cache   : {q_cache}")
    print(f"  token grid : NH={NH} NR={NR} NA={NA} NS={NS}")
    print(f"  voxels     : q={q_voxel} m  db={db_voxel} m"
          + ("" if SINGLE_SESSION else f"  map stride={db_map_stride}"))

    # ── Descriptor caches ────────────────────────────────────────────────────
    db_npz = np.load(db_cache)
    q_npz  = np.load(q_cache)
    db_poses_world = np.asarray(db_npz["poses"], dtype=np.float64)
    q_poses_world  = np.asarray(q_npz["poses"], dtype=np.float64)
    # HeLiPR sequences already share a global frame; two independently mapped
    # generic sequences do not.  The cache holds the DB poses as loaded, and
    # the protocol applied the DB->query transform afterwards -- so a replay
    # reading the cache has to apply the same one to see the same trajectories.
    db_transform = written.get("db_transform")
    if db_transform is not None:
        db_poses_world = np.asarray(db_transform, dtype=np.float64) @ db_poses_world
    db_off, db_kp_sensor = db_npz["offsets"], db_npz["kp_sensor"]
    q_off,  q_kp_sensor  = q_npz["offsets"],  q_npz["kp_sensor"]
    db_tokens = db_npz["token_ids"]
    q_tokens  = q_npz["token_ids"]

    db_xy = db_poses_world[:, :2, 3]
    q_xy  = q_poses_world[:, :2, 3]
    N_q_kf = len(q_poses_world)

    # ── Raw scans via the run's own loader (per-keyframe, sensor frame) ──────
    # Rebuilt from what the eval recorded, so a generic run replays with the
    # submap accumulation it was evaluated with and nothing has to be retyped.
    if SINGLE_SESSION:
        print("Loading session scans…")
        db_clouds = q_clouds = load_clouds(q_described, args.dataset)
    else:
        print("Loading DB scans…")
        db_clouds = load_clouds(db_described, args.dataset)
        print("Loading Q scans…")
        q_clouds = load_clouds(q_described, args.dataset)
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

    # Per-index z for every drawn element, so the two layouts differ only in
    # these three lookups and every draw call below stays shared.
    #
    # Cross-session stacks two sessions: DB map and trajectory on the floor,
    # query map and trajectory at Z_OFFSET, every edge crossing the gap.
    #
    # A single session has one trajectory and one map, so the two axes carry
    # different meanings instead: the map lies flat on the floor and builds up
    # as the session plays, while z along the *trajectory* is the frame index.
    # The curve climbs as the run proceeds and a closure edge joins two points
    # on it, its height being how long the loop took to come back around.  The
    # index is normalised into the existing z range so the zlim and box aspect
    # set above still hold.
    if SINGLE_SESSION:
        q_traj_zs = frame_index_z(N_q_kf)
        db_traj_zs = q_traj_zs          # the database *is* this trajectory
        q_map_z = 0.0                   # scans build the floor map
    else:
        q_traj_zs = np.full(N_q_kf, q_traj_z)
        db_traj_zs = np.full(len(db_poses_world), db_traj_z)
        q_map_z = Z_OFFSET
    traj_color = DB_TRAJ_COLOR if SINGLE_SESSION else Q_TRAJ_COLOR

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
    if SINGLE_SESSION:
        # No prior map to draw: the map is built as the session streams, which
        # is the upper layer.  Painting the finished map underneath would show
        # frames the matcher had not reached yet -- the exact future leak the
        # protocol exists to avoid.
        print("  Single session: the map accumulates as it plays.")
    else:
        db_map = build_map(db_clouds, db_poses_world, db_voxel, db_map_stride,
                           desc="Building DB prior map")
        print(f"  DB map: {len(db_map)} pts")
        ax.scatter(db_map[:, 0], db_map[:, 1], np.zeros(len(db_map)),
                   c=DB_MAP_COLOR, s=DB_MAP_POINT_SIZE, alpha=DB_MAP_ALPHA,
                   linewidths=0, depthshade=False)
        ax.plot(db_xy[:, 0], db_xy[:, 1], np.full(len(db_xy), db_traj_z),
                color=DB_TRAJ_COLOR, linewidth=TRAJ_LW, zorder=10)

    # ── Per-keyframe local scans (sensor frame). One scan per keyframe. ───────
    # Memoised: one submap is asked for two to four times per frame -- once to
    # place it in the world, again for the keyframe panel, and once per closure
    # for the matched DB frame -- and downsampling an accumulated submap is the
    # single most expensive thing the animation does.  Bounded so scrubbing a
    # long session cannot grow without limit.
    @lru_cache(maxsize=LOCAL_SCAN_CACHE)
    def get_q_local(i: int) -> np.ndarray:
        if not (0 <= i < len(q_clouds)):
            return np.empty((0, 3), dtype=np.float32)
        return voxel_downsample_np(clean_scan(q_clouds[i]), q_voxel)

    @lru_cache(maxsize=LOCAL_SCAN_CACHE)
    def get_db_local(i: int) -> np.ndarray:
        if not (0 <= i < len(db_clouds)):
            return np.empty((0, 3), dtype=np.float32)
        return voxel_downsample_np(clean_scan(db_clouds[i]), db_voxel)

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
        origin = np.array([T[0, 3], T[1, 3], q_traj_zs[idx]])
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

    q_traj_line, = ax.plot([q_xy[0, 0]], [q_xy[0, 1]], [q_traj_zs[0]],
                           color=traj_color, linewidth=TRAJ_LW, zorder=11)

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

    frame_word = "Frame" if SINGLE_SESSION else "Q keyframe"

    def update_title():
        title.set_text(f"{frame_word} {state['i']+1}/{N_q_kf}    "
                       f"TP {state['tp']}    FP {state['fp']}    "
                       f"[{'PLAYING' if state['playing'] else 'PAUSED'}]")

    def reveal(i: int):
        if i in drawn_frames or not (0 <= i < N_q_kf):
            return
        artists = []
        seg = q_xy[: i + 1]
        q_traj_line.set_data_3d(seg[:, 0], seg[:, 1], q_traj_zs[: i + 1])
        update_pose_axes(i)

        sub = get_q_world(i)
        if len(sub):
            sc = ax.scatter(sub[:, 0], sub[:, 1],
                            np.full(len(sub), q_map_z),
                            c=Q_MAP_COLOR, s=Q_MAP_POINT_SIZE, alpha=Q_MAP_ALPHA,
                            linewidths=0, depthshade=False)
            artists.append(sc)

        for dbi, mt, _sc in closures_by_q.get(i, []):
            color = TP_COLOR if mt == "TP" else FP_COLOR
            alpha = LOOP_EDGE_ALPHA if mt == "TP" else LOOP_FP_EDGE_ALPHA
            ln, = ax.plot(
                [q_xy[i, 0], db_xy[dbi, 0]],
                [q_xy[i, 1], db_xy[dbi, 1]],
                [q_traj_zs[i], db_traj_zs[dbi]],
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
                            np.full(len(qkp), q_map_z),
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

        L = max(2.0, q_voxel * 4)
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
            ptitle = f"{'Frame' if SINGLE_SESSION else 'Q kf'} {i + 1}/{N_q_kf}"
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
                                        q_traj_zs[: target + 1])
                update_pose_axes(target)
            else:
                q_traj_line.set_data_3d([q_xy[0, 0]], [q_xy[0, 1]],
                                        [q_traj_zs[0]])
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

    if SINGLE_SESSION:
        # One trajectory, so one entry: listing it twice as DB and Query would
        # name the same curve under two roles it does not have here.
        legend = [Line2D([0], [0], color=traj_color, lw=TRAJ_LW,
                         label=f"Session: {session_label(q_described)}")]
    else:
        legend = [
            Line2D([0], [0], color=DB_TRAJ_COLOR, lw=TRAJ_LW,
                   label=f"DB: {session_label(db_described)}"),
            Line2D([0], [0], color=Q_TRAJ_COLOR,  lw=TRAJ_LW,
                   label=f"Query: {session_label(q_described)}"),
        ]
    legend += [
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
