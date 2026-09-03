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

from inlier.cli._common import add_generic_layout_flags, user_path

SCAN_SUFFIXES = (".pcd", ".ply", ".bin", ".npy")
VIZ_SUFFIXES = (".png", ".pdf", ".svg", ".jpg", ".jpeg")


def register(subparsers, parent) -> None:
    p = subparsers.add_parser(
        "encode", parents=[parent],
        help="encode scan(s) into keypoints and tokens",
        description="Run the InLiER encoder on a scan or a directory of scans "
                    "and write keypoints + mixed-radix tokens to an .npz.",
    )
    p.add_argument("input", type=str, nargs="?",
                   help="a scan file (.pcd/.ply/.bin/.npy) or a directory of "
                        "them; omit when using --dataset")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="output .npz (single scan) or output directory (many); "
                        "optional when only --viz is wanted")
    p.add_argument("--voxel-size", dest="voxel_size",
                   type=float, default=None,
                   help="override the config's preprocessing voxel size (m)")
    p.add_argument("--stats", action="store_true",
                   help="print per-scan keypoint/token counts and the descriptor size")

    data = p.add_argument_group(
        "dataset mode (submaps)",
        "Encode accumulated submaps rather than single scans, using the poses "
        "to merge each window into its keyframe's frame. This is what the "
        "evaluation encodes when --n-db/--n-q are above 1, so it is what to "
        "inspect if you want to see the descriptor the pipeline actually "
        "scores.")
    data.add_argument("--dataset", type=user_path, default=None, metavar="ROOT",
                      help="dataset root, instead of a scan path")
    data.add_argument("--dataset-type", dest="dataset_type",
                      choices=("generic", "helipr", "kitti"), default="generic",
                      help="loader for --dataset (default: generic)")
    data.add_argument("--sequence", type=str, default=None, metavar="XX",
                      help="KITTI sequence id, e.g. 00; not needed when "
                           "--dataset is itself a sequence directory")
    data.add_argument("--n-scans", dest="n_scans", type=int,
                      default=1, metavar="N",
                      help="scans per submap; must match the overlap ground "
                           "truth's --n-db/--n-q (default: 1)")
    data.add_argument("--stride", type=int, default=None, metavar="N",
                      help="step between submaps (default: --n-scans, "
                           "i.e. non-overlapping)")
    add_generic_layout_flags(data)
    which = data.add_mutually_exclusive_group()
    which.add_argument("--index", type=int, default=None, metavar="I",
                       help="which submap to encode (default: 0); "
                            "negative counts from the end")
    which.add_argument("--range", dest="submap_range", type=str, default=None,
                       metavar="A:B",
                       help="a half-open range of submaps, e.g. 0:10")

    viz = p.add_argument_group("visualization")
    viz.add_argument("--viz", action="store_true",
                     help="plot the scan, its keypoints and the MINT/BEAM "
                          "descriptors")
    viz.add_argument("--viz-save", dest="viz_save",
                     type=user_path, default=None, metavar="PATH",
                     help="write the figure(s) to PATH instead of opening a "
                          "window (implies --viz): a file for one scan, a "
                          "directory for many")
    viz.add_argument("--viz-dpi", dest="viz_dpi", type=int,
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


def _parse_range(spec: str) -> range:
    """``A:B`` -> ``range(A, B)``, half-open like a Python slice."""
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"--range wants A:B, got {spec!r}")
    try:
        start, stop = (int(x) if x.strip() else None for x in parts)
    except ValueError:
        raise ValueError(f"--range wants integers, got {spec!r}") from None
    start = 0 if start is None else start
    if stop is None:
        raise ValueError(f"--range needs an end, e.g. {start}:{start + 10}")
    if stop <= start:
        raise ValueError(f"--range is empty: {spec!r}")
    return range(start, stop)


def _submap_selection(args) -> list:
    if args.submap_range:
        return list(_parse_range(args.submap_range))
    return [0 if args.index is None else args.index]


def _submap_mode(args) -> bool:
    """Whether this invocation builds submaps rather than encoding one file."""
    return args.dataset is not None or getattr(args, "scans_dir", None) is not None


def _dataset_root(args) -> Path:
    """The sequence's identity, whichever loader named it."""
    if args.dataset_type == "kitti":
        return _kitti_handler(args, quiet=True)[1]
    return _generic_root(args)


def _kitti_handler(args, quiet: bool):
    """``(handler, sequence_dir)`` for a KITTI submap encode.

    The sequence directory is what the rest of this module treats as the
    dataset root, so the range-check message and the figure title read ``00``
    rather than the name of whatever folder the whole benchmark lives in.
    """
    from inlier.eval.datasets.kitti import (SCAN_SUBDIR, KITTI_Handler,
                                            normalise_sequence)

    root = Path(args.dataset)
    if not root.is_dir():
        raise FileNotFoundError(f"--dataset {root} is not a directory")
    if (root / SCAN_SUBDIR).is_dir():
        sequence = normalise_sequence(args.sequence or root.name)
    elif args.sequence:
        sequence = normalise_sequence(args.sequence)
    else:
        raise ValueError(
            "--dataset-type kitti needs --sequence (e.g. --sequence 00), "
            f"unless --dataset points straight at a sequence directory -- one "
            f"containing {SCAN_SUBDIR}/.")
    handler = KITTI_Handler(root, sequence, verbose=not quiet)
    return handler, handler.seq_dir


def _generic_root(args) -> Path:
    """The sequence's identity: ``--dataset``, or the scans folder's parent.

    Only ever used as a *name* -- the provenance record, the figure title, the
    range check's error message.  Where the data is actually read from is the
    handler's business, which is why an explicit ``--scans`` needs no dataset
    directory to sit under.
    """
    if args.dataset is not None:
        root = Path(args.dataset)
        if not root.is_dir():
            raise FileNotFoundError(f"--dataset {root} is not a directory")
        return root
    scans_dir = getattr(args, "scans_dir", None)
    if scans_dir is None:
        raise ValueError("no dataset: pass --dataset, or --scans with --poses")
    if getattr(args, "pose_file", None) is None:
        raise ValueError(
            "--scans without --dataset also needs --poses: there is nowhere "
            "else to look for the poses")
    return Path(scans_dir).parent


def _load_submaps(args, quiet: bool):
    """Build the requested submaps, reading only the scans they need.

    Goes through ``Generic_Handler`` rather than accumulating here, so the
    windows, the keyframe choice and the pose transform are the ones the
    overlap ground truth and the evaluation use.
    """
    if args.dataset_type == "helipr":
        # HeLiPRSource has no n_scans/stride, and cross_session reads them
        # with a default of 1 -- HeLiPR is evaluated scan-by-scan.  There is
        # no submap to accumulate, so point encode straight at the scan.
        raise NotImplementedError(
            "--dataset-type helipr has no submap accumulation: HeLiPR is "
            "evaluated scan by scan. Pass the .bin path directly, e.g. "
            "inlier encode <root>/<sequence>/Undistorted/<sensor>/<scan>.bin")

    from inlier.eval.datasets.generic import Generic_Handler
    from inlier.eval.submaps import submap_count

    if args.dataset_type == "kitti":
        handler, root = _kitti_handler(args, quiet)
    else:
        root = _generic_root(args)
        handler = Generic_Handler(verbose=not quiet,
                                  scans_dir=getattr(args, "scans_dir", None),
                                  pose_file=getattr(args, "pose_file", None))
    stride = args.n_scans if args.stride is None else args.stride
    total = submap_count(len(handler.list_scan_files(root)), args.n_scans, stride)

    # Resolve negatives here so the index that reaches the filenames, the
    # figure title and the .npz provenance is the real one.
    selection = []
    for index in _submap_selection(args):
        resolved = index + total if index < 0 else index
        if not 0 <= resolved < total:
            raise ValueError(
                f"submap {index} out of range: {root.name} has {total} submaps "
                f"at n_scans={args.n_scans}, stride={stride}")
        selection.append(resolved)

    if not quiet:
        print(f"{root.name}: {total} submaps at n_scans={args.n_scans} "
              f"stride={stride}; encoding {len(selection)}")

    data = handler.load_generic(root, n_scans=args.n_scans, stride=stride,
                                select=selection)
    return data["point_clouds"], data["poses"], selection, stride


def _viz_target(save: Path, stem: str, many: bool) -> Path:
    """Where one item's figure goes.

    A path with an image suffix names a file; anything else is a directory,
    which is also the only form that makes sense for a batch.
    """
    if many or save.suffix.lower() not in VIZ_SUFFIXES:
        save.mkdir(parents=True, exist_ok=True)
        return save / f"{stem}.png"
    save.parent.mkdir(parents=True, exist_ok=True)
    return save


def _file_items(args, quiet: bool):
    """Single scans, straight off disk: (label, stem, points, extra)."""
    src = Path(args.input)
    if src.is_dir():
        scans = sorted(p for p in src.iterdir() if p.suffix.lower() in SCAN_SUFFIXES)
        if not scans:
            raise FileNotFoundError(
                f"no scans in {src} (looked for {', '.join(SCAN_SUFFIXES)})")
    else:
        scans = [src]

    items = []
    for scan in scans:
        points, n_dropped = _load_points(scan)
        if n_dropped and not quiet:
            print(f"{scan.name}: dropped {n_dropped:,} non-finite point(s)")
        # `source` lets `inlier match` reload the cloud later; it is also the
        # only record of where a file-mode encoding came from.
        items.append((scan.name, scan.stem, points, {"source": str(scan)}))
    return items, len(scans) > 1 or src.is_dir()


def _submap_items(args, quiet: bool):
    """Accumulated submaps: same shape, plus the provenance they need."""
    clouds, poses, selection, stride = _load_submaps(args, quiet)
    items = []
    for index, cloud, pose in zip(selection, clouds, poses):
        items.append((
            f"submap {index}", f"submap_{index:05d}", cloud,
            {
                # A submap descriptor is only comparable to one built the same
                # way; these are the fields OverlapProvenance treats as
                # critical, so record them next to the tokens.
                "n_scans": args.n_scans,
                "stride": stride,
                "submap_index": index,
                # Which loader built it.  Without this the cloud cannot be
                # reloaded later -- `inlier match` would try KITTI's sequence
                # directory as a generic one and find no scans/ beside it,
                # and a KITTI cloud rebuilt with uncorrected poses would be
                # wrong rather than merely missing.
                "dataset_type": args.dataset_type,
                "keyframe_pose": pose,
                "dataset": str(_dataset_root(args)),
            },
        ))
    return items, len(items) > 1


def run(args: argparse.Namespace) -> int:
    import numpy as np

    from inlier import InLiER
    from inlier.cli._common import resolved_config
    from inlier.eval.encode import voxel_downsample

    viz = args.viz or args.viz_save is not None
    if args.output is None and not viz:
        raise ValueError("nothing to do: pass -o/--output, --viz, or both.")
    if (args.input is None) == (not _submap_mode(args)):
        raise ValueError(
            "pass either a scan path or a dataset (--dataset ROOT, or --scans "
            "with --poses), not both and not neither.")
    if args.input is not None:
        for flag, value in (("--n-scans", args.n_scans != 1),
                            ("--stride", args.stride is not None),
                            ("--index", args.index is not None),
                            ("--range", args.submap_range is not None)):
            if value:
                raise ValueError(
                    f"{flag} needs a dataset: submaps are built from poses, "
                    f"which a bare scan path does not carry.")

    resolved = resolved_config(args, mode="deploy")
    voxel_size = args.voxel_size if args.voxel_size is not None else resolved.voxel_size
    quiet = getattr(args, "quiet", False)

    if _submap_mode(args):
        items, many = _submap_items(args, quiet)
    else:
        items, many = _file_items(args, quiet)

    if viz and many and args.viz_save is None:
        raise ValueError(
            f"--viz would open a window per item ({len(items)} of them); pass "
            f"--viz-save DIR to write the figures instead, or narrow the "
            f"selection.")

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

    for label, stem, raw, extra in items:
        points = raw if voxel_size is None else voxel_downsample(raw, voxel_size)
        keypoints, tokens = encoder.encode(points, verbose=not quiet)

        target = None
        if out is not None:
            target = (out / f"{stem}.npz") if many else out
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
                **extra,
            )

        if args.stats or not quiet:
            n_bytes = tokens.token_id.nbytes
            written = f"  -> {target}" if target is not None else ""
            print(f"{label}: {len(points):>7,} pts -> {len(keypoints):>5} keypoints, "
                  f"{len(tokens):>5} tokens ({n_bytes / 1024:.2f} KiB){written}")

        if viz:
            figure = encode_figure(points, keypoints, tokens, resolved.inlier,
                                   title=_title(args, label, extra))
            if args.viz_save is not None:
                figure_path = _viz_target(args.viz_save, stem, many)
                figure.savefig(figure_path, dpi=args.viz_dpi,
                               bbox_inches="tight")
                plt.close(figure)
                if not quiet:
                    print(f"{label}: figure -> {figure_path}")
            else:
                plt.show()
                plt.close(figure)

    if many and out is not None and not quiet:
        print(f"\nencoded {len(items)} item(s) into {out}")
    return 0


def _title(args, label: str, extra: dict) -> str:
    # Keyed on the submap fields, not on `extra` being empty: file-mode
    # encodings now carry a `source` entry too.
    if "submap_index" not in extra:
        return str(Path(args.input) if Path(args.input).is_file()
                   else Path(args.input) / label)
    return (f"{_dataset_root(args).name}  {label}  "
            f"(n_scans={extra['n_scans']}, stride={extra['stride']})")
