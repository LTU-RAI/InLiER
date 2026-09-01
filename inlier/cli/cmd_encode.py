"""``inlier encode`` -- run just the encoder and write out the tokens.

The descriptor was only ever reachable by running a full evaluation, which
needs poses and an overlap matrix.  This exposes the first stage on its own:
useful for inspecting a descriptor, for feeding InLiER tokens to something
else, and for making the cache format a documented artifact rather than an
internal detail of ``encode_sequence``.

``--viz`` renders the same scan as a figure -- cloud, keypoints, and the
three matrices the matcher scores on -- which is the quickest way to tell
whether a token grid suits a new sensor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inlier.cli._common import user_path

SCAN_SUFFIXES = (".pcd", ".ply", ".bin", ".npy")
VIZ_SUFFIXES = (".png", ".pdf", ".svg", ".jpg", ".jpeg")


def register(subparsers, parent) -> None:
    p = subparsers.add_parser(
        "encode", parents=[parent],
        help="encode scan(s) into keypoints and tokens",
        description="Run the InLiER encoder on a scan or a directory of scans "
                    "and write keypoints + mixed-radix tokens to an .npz.",
    )
    p.add_argument("input", type=str,
                   help="a scan file (.pcd/.ply/.bin/.npy) or a directory of them")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="output .npz (single scan) or output directory (many); "
                        "optional when only --viz is wanted")
    p.add_argument("--voxel-size", "--voxel_size", dest="voxel_size",
                   type=float, default=None,
                   help="override the config's preprocessing voxel size (m)")
    p.add_argument("--stats", action="store_true",
                   help="print per-scan keypoint/token counts and the descriptor size")

    viz = p.add_argument_group("visualization")
    viz.add_argument("--viz", action="store_true",
                     help="plot the scan, its keypoints and the MINT/BEAM "
                          "descriptors")
    viz.add_argument("--viz-save", "--viz_save", dest="viz_save",
                     type=user_path, default=None, metavar="PATH",
                     help="write the figure(s) to PATH instead of opening a "
                          "window (implies --viz): a file for one scan, a "
                          "directory for many")
    viz.add_argument("--viz-dpi", "--viz_dpi", dest="viz_dpi", type=int,
                     default=150, help="figure resolution when saving (150)")
    p.set_defaults(func=run)


def _load_points(path: Path):
    """Read a scan into (N, 3) float32, dropping non-finite points.

    Returns ``(points, n_dropped)``.  Sensors mark invalid returns as NaN and
    both PCD and the raw HeLiPR binaries carry them through, so a scan can
    arrive with holes in it.  Left in, they poison the voxel-grid cast and
    every axis limit downstream.
    """
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

    pts = pts[:, :3]
    finite = np.isfinite(pts).all(axis=1)
    n_dropped = int(pts.shape[0] - finite.sum())
    if n_dropped:
        pts = pts[finite]
    if pts.shape[0] == 0:
        raise ValueError(f"{path}: no finite points")
    return pts, n_dropped


def _viz_target(save: Path, scan: Path, many: bool) -> Path:
    """Where one scan's figure goes.

    A path with an image suffix names a file; anything else is a directory,
    which is also the only form that makes sense for a batch.
    """
    if many or save.suffix.lower() not in VIZ_SUFFIXES:
        save.mkdir(parents=True, exist_ok=True)
        return save / f"{scan.stem}.png"
    save.parent.mkdir(parents=True, exist_ok=True)
    return save


def run(args: argparse.Namespace) -> int:
    import numpy as np

    from inlier import InLiER
    from inlier.cli._common import resolved_config
    from inlier.eval.encode import voxel_downsample

    viz = args.viz or args.viz_save is not None
    if args.output is None and not viz:
        raise ValueError("nothing to do: pass -o/--output, --viz, or both.")

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

    many = len(scans) > 1 or src.is_dir()
    if viz and many and args.viz_save is None:
        raise ValueError(
            f"--viz on a directory would open a window per scan "
            f"({len(scans)} of them); pass --viz-save DIR to write the figures "
            f"instead, or point --viz at a single scan.")

    if viz:
        # Choose the backend before pyplot is imported anywhere: saving must
        # work over ssh and in CI, where there is no display to fall back on.
        import matplotlib

        if args.viz_save is not None:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        from inlier.viz import encode_figure

    out = Path(args.output) if args.output is not None else None
    if out is not None:
        if many:
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)

    encoder = InLiER(resolved.inlier)
    quiet = getattr(args, "quiet", False)

    for scan in scans:
        raw, n_dropped = _load_points(scan)
        if n_dropped and not quiet:
            print(f"{scan.name}: dropped {n_dropped:,} non-finite point(s)")
        points = raw if voxel_size is None else voxel_downsample(raw, voxel_size)
        keypoints, tokens = encoder.encode(points, verbose=not quiet)

        target = None
        if out is not None:
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
            written = f"  -> {target}" if target is not None else ""
            print(f"{scan.name}: {len(points):>7,} pts -> {len(keypoints):>5} keypoints, "
                  f"{len(tokens):>5} tokens ({n_bytes / 1024:.2f} KiB){written}")

        if viz:
            figure = encode_figure(points, keypoints, tokens, resolved.inlier,
                                   title=str(scan))
            if args.viz_save is not None:
                figure_path = _viz_target(args.viz_save, scan, many)
                figure.savefig(figure_path, dpi=args.viz_dpi,
                               bbox_inches="tight")
                plt.close(figure)
                if not quiet:
                    print(f"{scan.name}: figure -> {figure_path}")
            else:
                plt.show()
                plt.close(figure)

    if many and out is not None and not quiet:
        print(f"\nencoded {len(scans)} scan(s) into {out}")
    return 0
