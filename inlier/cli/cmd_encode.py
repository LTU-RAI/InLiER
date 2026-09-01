"""``inlier encode`` -- run just the encoder and write out the tokens.

The descriptor was only ever reachable by running a full evaluation, which
needs poses and an overlap matrix.  This exposes the first stage on its own:
useful for inspecting a descriptor, for feeding InLiER tokens to something
else, and for making the cache format a documented artifact rather than an
internal detail of ``encode_sequence``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SCAN_SUFFIXES = (".pcd", ".ply", ".bin", ".npy")


def register(subparsers, parent) -> None:
    p = subparsers.add_parser(
        "encode", parents=[parent],
        help="encode scan(s) into keypoints and tokens",
        description="Run the InLiER encoder on a scan or a directory of scans "
                    "and write keypoints + mixed-radix tokens to an .npz.",
    )
    p.add_argument("input", type=str,
                   help="a scan file (.pcd/.ply/.bin/.npy) or a directory of them")
    p.add_argument("-o", "--output", type=str, required=True,
                   help="output .npz (single scan) or output directory (many)")
    p.add_argument("--voxel-size", "--voxel_size", dest="voxel_size",
                   type=float, default=None,
                   help="override the config's preprocessing voxel size (m)")
    p.add_argument("--stats", action="store_true",
                   help="print per-scan keypoint/token counts and the descriptor size")
    p.set_defaults(func=run)


def _load_points(path: Path):
    """Read a scan into (N, 3) float32."""
    import numpy as np

    suffix = path.suffix.lower()
    if suffix == ".npy":
        pts = np.load(path)
    elif suffix == ".bin":
        # HeLiPR undistorted scans: float32 xyz + padding, inferred from size.
        raw = np.fromfile(path, dtype=np.float32)
        for width in (3, 4, 5, 6):
            if raw.size % width == 0:
                pts = raw.reshape(-1, width)[:, :3]
                break
        else:
            raise ValueError(f"{path}: cannot infer point stride from {raw.size} floats")
    else:
        import open3d as o3d

        pts = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"{path}: expected (N, 3) points, got {pts.shape}")
    return pts[:, :3]


def _voxel_downsample(points, voxel_size: float):
    """Same grid-hash downsample the evaluation uses before encoding."""
    import numpy as np

    if voxel_size is None or voxel_size <= 0 or points.size == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(keep)]


def run(args: argparse.Namespace) -> int:
    import numpy as np

    from inlier import InLiER
    from inlier.cli._common import resolved_config

    resolved = resolved_config(args, mode="deploy")
    voxel_size = args.voxel_size if args.voxel_size is not None else resolved.voxel_size

    src = Path(args.input)
    if src.is_dir():
        scans = sorted(p for p in src.iterdir() if p.suffix.lower() in SCAN_SUFFIXES)
        if not scans:
            raise FileNotFoundError(
                f"no scans in {src} (looked for {', '.join(SCAN_SUFFIXES)})")
    else:
        scans = [src]

    out = Path(args.output)
    many = len(scans) > 1 or src.is_dir()
    if many:
        out.mkdir(parents=True, exist_ok=True)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)

    encoder = InLiER(resolved.inlier)
    quiet = getattr(args, "quiet", False)

    for scan in scans:
        points = _voxel_downsample(_load_points(scan), voxel_size)
        keypoints, tokens = encoder.encode(points, verbose=not quiet)

        target = (out / f"{scan.stem}.npz") if many else out
        np.savez_compressed(
            target,
            token_id=tokens.token_id,
            kp_sensor=keypoints.p,
            kp_aligned=keypoints.p_aligned,
            T_ground=keypoints.T_ground,
            # provenance: without these the tokens cannot be interpreted,
            # since the mixed-radix packing depends on N_r, N_s and N_a.
            N_h=resolved.inlier.N_h, N_r=resolved.inlier.N_r,
            N_s=resolved.inlier.N_s, N_a=resolved.inlier.N_a,
            voxel_size=voxel_size,
        )

        if args.stats or not quiet:
            n_bytes = tokens.token_id.nbytes
            print(f"{scan.name}: {len(points):>7,} pts -> {len(keypoints):>5} keypoints, "
                  f"{len(tokens):>5} tokens ({n_bytes / 1024:.2f} KiB)  -> {target}")

    if many and not quiet:
        print(f"\nencoded {len(scans)} scan(s) into {out}")
    return 0
