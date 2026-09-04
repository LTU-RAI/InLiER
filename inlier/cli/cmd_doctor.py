"""``inlier doctor`` -- check that this install can actually run something.

Covers the failure modes that otherwise show up as a confusing result rather
than an error: the C++ extension silently falling back to the numpy reference
(a ~2x slowdown that only prints an import warning), a missing optional
dependency, a dataset laid out differently than its loader expects, and an
overlap matrix built with different submap parameters than the evaluation is
about to use.

``--dataset`` is checked against whichever layout ``--dataset-type`` names --
the same flag ``inlier eval`` takes, with the same default -- because the two
layouts share nothing: HeLiPR is per-sequence ``Undistorted/<sensor>/`` scan
directories, generic is one flat ``scans/`` plus a pose file.  Checking a
dataset against the wrong layout used to report it as empty, so a mismatch is
now called out by name.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from inlier.cli._common import add_generic_layout_flags

OK = "ok"
WARN = "warn"
FAIL = "FAIL"

CORE_DEPS = ("numpy", "yaml", "small_gicp")
EVAL_DEPS = ("scipy", "open3d", "tqdm", "matplotlib", "pandas")
VIZ_DEPS = ("pyridescence",)


def register(subparsers, parent) -> None:
    p = subparsers.add_parser(
        "doctor", parents=[parent],
        help="check backend, dependencies, and dataset layout",
        description="Diagnose an InLiER install and, with --dataset, the data "
                    "layout and ground-truth consistency.",
    )
    p.add_argument("--dataset", type=str, default=None,
                   help="dataset root to check, against --dataset-type's layout")
    p.add_argument("--dataset-type", dest="dataset_type",
                   choices=("helipr", "generic", "kitti"), default="helipr",
                   help="layout to check --dataset against (default: helipr)")
    p.add_argument("--sequence", type=str, default=None, metavar="XX",
                   help="KITTI sequence id to check, e.g. 00")
    add_generic_layout_flags(p)
    p.add_argument("--overlap-dir", dest="overlap_dir",
                   type=str, default="overlap_matrices",
                   help="directory of overlap matrices to check for sidecars")
    p.set_defaults(func=run)


def _row(status: str, label: str, detail: str = "") -> None:
    mark = {OK: " ok ", WARN: "warn", FAIL: "FAIL"}[status]
    print(f"  [{mark}] {label.ljust(22)} {detail}")


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def run(args: argparse.Namespace) -> int:
    import inlier

    failures = 0
    warnings = 0

    print("inlier doctor")
    print("=" * 62)

    print("\npackage")
    _row(OK, "version", inlier.__version__)
    _row(OK, "location", str(Path(inlier.__file__).parent))
    if inlier.__version__ == "0.0.0+unknown":
        _row(WARN, "metadata", "package not installed; version unavailable")
        warnings += 1

    print("\nbackend")
    from inlier.core.InLiER import _BACKEND

    if _BACKEND == "cpp":
        _row(OK, "core", "cpp (compiled extension loaded)")
    else:
        _row(WARN, "core", "python (numpy reference; ~2x slower end to end)")
        print("         reinstall with a C++17 compiler and CMake >= 3.16 to build it,")
        print("         or unset INLIER_FORCE_PYTHON if it is set deliberately.")
        warnings += 1

    import os
    if os.environ.get("INLIER_FORCE_PYTHON") == "1":
        _row(WARN, "INLIER_FORCE_PYTHON", "set to 1 -- forcing the numpy reference")
        warnings += 1

    try:
        import inlier._inlier_pybind  # noqa: F401
        _row(OK, "extension", "importable")
    except ImportError as exc:
        _row(WARN, "extension", f"not importable: {exc}")
        warnings += 1

    print("\ndependencies")
    for module in CORE_DEPS:
        if _have(module):
            _row(OK, module, "")
        else:
            _row(FAIL, module, "required -- pip install -e .")
            failures += 1
    for module in EVAL_DEPS:
        if _have(module):
            _row(OK, module, "")
        else:
            _row(WARN, module, 'evaluation only -- pip install -e ".[eval]"')
            warnings += 1
    for module in VIZ_DEPS:
        if _have(module):
            _row(OK, module, "")
        else:
            _row(WARN, module, '`inlier run --live` only -- pip install -e ".[viz]"')
            warnings += 1

    print("\nconfiguration")
    try:
        from inlier.cli._common import load_config
        from inlier.config import DEFAULT_CONFIG_PATH, resolve

        cfg = load_config(args)
        r = resolve(cfg)
        _row(OK, "defaults", str(DEFAULT_CONFIG_PATH))
        if args.config:
            _row(OK, "config", args.config)
        _row(OK, "resolves", f"voxel_size={r.voxel_size:g} cell_size={r.inlier.cell_size:g} "
                             f"N_h={r.inlier.N_h} N_r={r.inlier.N_r} "
                             f"N_a={r.inlier.N_a} N_s={r.inlier.N_s}")
        vocab = r.inlier.N_h * r.inlier.N_r * r.inlier.N_s * r.inlier.N_a
        _row(OK, "token vocabulary", f"{vocab:,} "
                                     f"({'uint32' if vocab <= 2**32 - 1 else 'uint64'})")
    except Exception as exc:
        _row(FAIL, "config", str(exc).splitlines()[0])
        failures += 1

    scans_dir = getattr(args, "scans_dir", None)
    pose_file = getattr(args, "pose_file", None)
    if args.dataset or scans_dir is not None:
        # --scans alone is enough to check: the root is only the heading.
        root = Path(args.dataset) if args.dataset else Path(scans_dir).parent
        failures += _check_dataset(root, args.dataset_type, scans_dir,
                                   pose_file, args.sequence)

    warnings += _check_overlap_sidecars(Path(args.overlap_dir))

    print("\n" + "=" * 62)
    if failures:
        print(f"{failures} failure(s), {warnings} warning(s)")
    elif warnings:
        print(f"no failures, {warnings} warning(s)")
    else:
        print("all checks passed")
    return 1 if failures else 0


LAYOUTS = {
    "helipr": "<root>/<sequence>/Undistorted/<sensor>/*.bin",
    "generic": "<root>/scans/*.{pcd,bin} + poses_kitti.txt or poses_tum.txt",
    "kitti": ("<root>/sequences/XX/{velodyne/*.bin, calib.txt, times.txt}"
              " + poses/XX.txt or sequences/XX/poses.txt"),
}


def _check_dataset(root: Path, dataset_type: str,
                   scans_dir: Path = None, pose_file: Path = None,
                   sequence: str = None) -> int:
    print(f"\ndataset  {root}")
    explicit = scans_dir is not None or pose_file is not None
    _row(OK, "layout",
         "generic -- explicit paths" if explicit
         else f"{dataset_type} -- {LAYOUTS[dataset_type]}")
    if not root.exists() and not explicit:
        _row(FAIL, "root", "does not exist")
        return 1
    if dataset_type == "kitti":
        return _check_kitti(root, sequence)
    if dataset_type == "generic" or explicit:
        return _check_generic(root, scans_dir, pose_file)
    return _check_helipr(root)


def _looks_generic(root: Path) -> bool:
    return (root / "scans").is_dir()


def _looks_kitti(root: Path) -> bool:
    return (root / "sequences").is_dir() or (root / "velodyne").is_dir()


def _looks_helipr(root: Path) -> bool:
    return any((seq / "Undistorted").is_dir() or (seq / "LiDAR").is_dir()
               for seq in root.iterdir() if seq.is_dir())


def _check_kitti(root: Path, sequence: str = None) -> int:
    """A KITTI odometry sequence, including whether the pose frame is sane.

    The frame check is the point of this one.  KITTI ships its poses in the
    camera frame, and a run against uncorrected poses fails in a way that
    looks like bad retrieval rather than bad geometry -- so the spans are
    reported here, where one command answers it.
    """
    import numpy as np

    from inlier.eval.datasets.kitti import (SCAN_SUBDIR, KITTI_Handler,
                                            normalise_sequence, read_calib_tr)

    if not _looks_kitti(root):
        _row(FAIL, "layout mismatch",
             f"{root} has neither sequences/ nor {SCAN_SUBDIR}/ -- this does "
             f"not look like a KITTI odometry tree")
        return 1

    if (root / SCAN_SUBDIR).is_dir():
        sequence = normalise_sequence(sequence or root.name)
    elif sequence is None:
        available = sorted(p.name for p in (root / "sequences").iterdir()
                           if p.is_dir()) if (root / "sequences").is_dir() else []
        _row(FAIL, "sequence", "--dataset-type kitti needs --sequence"
             + (f" (found: {', '.join(available)})" if available else ""))
        return 1
    else:
        sequence = normalise_sequence(sequence)

    failures = 0
    handler = KITTI_Handler(root, sequence, verbose=False)
    try:
        seq_dir = handler.seq_dir
    except FileNotFoundError as exc:
        _row(FAIL, "sequence", str(exc).splitlines()[0])
        return 1
    _row(OK, f"sequence {sequence}", str(seq_dir))

    scans = []
    try:
        scans = handler.list_scan_files(seq_dir)
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as exc:
        _row(FAIL, f"{SCAN_SUBDIR}/", str(exc).splitlines()[0])
        failures += 1
    else:
        _row(OK, f"{SCAN_SUBDIR}/", f"{len(scans):,} .bin files")

    try:
        calib = handler.calib_file
        Tr = read_calib_tr(calib)
    except (OSError, ValueError) as exc:
        _row(FAIL, "calib", str(exc).splitlines()[0])
        return failures + 1
    _row(OK, "calib", f"{calib}, Tr found")

    try:
        poses, stamps = handler.load_poses(seq_dir)
    except (OSError, ValueError) as exc:
        _row(FAIL, "poses", str(exc).splitlines()[0])
        return failures + 1
    pose_path, _ = handler._pose_file(seq_dir)
    _row(OK, "poses", f"{pose_path}, {len(poses):,} poses")

    if stamps:
        _row(OK, "times.txt", f"{len(stamps):,} timestamps "
                              f"({stamps[-1] - stamps[0]:.1f} s) "
                              f"-- --exclusion seconds= is available")
    else:
        _row(WARN, "times.txt", "missing or mismatched -- "
                                "--exclusion seconds= will not work")

    if scans and len(poses) != len(scans):
        _row(FAIL, "poses vs scans",
             f"{len(poses):,} poses but {len(scans):,} scans -- "
             f"load_generic requires they match")
        failures += 1

    # The frame check: KITTI drives are near-planar, so after the correction
    # the vertical span must be by far the smallest.  WARN, never FAIL -- a
    # short sequence straight up a hill could legitimately be tall.
    t = np.array([p[:3, 3] for p in poses]) if poses else np.zeros((0, 3))
    if len(t) > 1:
        spans = np.ptp(t, axis=0)
        detail = "  ".join(f"{a}={v:.1f}m" for a, v in zip("xyz", spans))
        if spans[2] <= 0.2 * max(spans[0], spans[1]):
            _row(OK, "pose frame", f"velodyne (z is vertical): {detail}")
        else:
            _row(WARN, "pose frame",
                 f"z is not clearly the vertical axis: {detail}. Expected a "
                 f"near-planar drive after the camera->velodyne correction; "
                 f"check that {calib.name} belongs to this sequence.")

    if scans:
        failures += _check_sample_scan(scans[0])
    return failures


def _check_helipr(root: Path) -> int:
    sequences = sorted(p for p in root.iterdir() if p.is_dir())
    if _looks_kitti(root) and not _looks_helipr(root):
        # Without this a KITTI tree passes the HeLiPR check: `sequences/` reads
        # as one sequence that merely lacks Undistorted/, which is a warning.
        _row(FAIL, "layout mismatch", "this looks like a KITTI odometry tree "
                                      "(it has sequences/) -- pass "
                                      "--dataset-type kitti --sequence XX")
        return 1
    if not sequences or (_looks_generic(root) and not _looks_helipr(root)):
        # The most likely reason a HeLiPR check finds nothing is that this is
        # not a HeLiPR tree.  Say that, rather than "no sequences found".
        if _looks_generic(root):
            _row(FAIL, "layout mismatch", "this looks like a generic dataset "
                                          "(it has scans/) -- pass "
                                          "--dataset-type generic")
        else:
            _row(FAIL, "sequences", "no sequence directories found")
        return 1
    _row(OK, "sequences", f"{len(sequences)} found: "
                          f"{', '.join(p.name for p in sequences[:4])}"
                          f"{' ...' if len(sequences) > 4 else ''}")

    failures = 0
    for seq in sequences[:4]:
        undistorted = seq / "Undistorted"
        raw = seq / "LiDAR"
        gt_dir = seq / "LiDAR_GT"
        if undistorted.exists():
            sensors = sorted(p.name for p in undistorted.iterdir() if p.is_dir())
            _row(OK, seq.name, f"Undistorted/ [{', '.join(sensors)}]")
        elif raw.exists():
            # Every evaluation script reads Undistorted/, never LiDAR/: the raw
            # HeLiPR scans are motion-distorted.
            _row(FAIL, seq.name, "has LiDAR/ but no Undistorted/ -- run the "
                                 "HeLiPR-Pointcloud-Toolbox first")
            failures += 1
        else:
            _row(WARN, seq.name, "neither Undistorted/ nor LiDAR/")
        if not gt_dir.exists():
            _row(WARN, "", f"{seq.name}: no LiDAR_GT/ (ground-truth poses)")
    return failures


def _check_generic(root: Path, scans_dir: Path = None,
                   pose_file: Path = None) -> int:
    """Flat scans/ + a pose file, per ``Generic_Handler``.

    ``scans_dir``/``pose_file`` mirror the loader's overrides, so a dataset
    that was never arranged into one tree can still be checked before a run.
    """
    from inlier.eval.datasets.generic import Generic_Handler

    explicit = scans_dir is not None or pose_file is not None
    if not explicit and not _looks_generic(root) and _looks_helipr(root):
        _row(FAIL, "layout mismatch", "this looks like a HeLiPR tree -- "
                                      "pass --dataset-type helipr")
        return 1
    if not explicit and not _looks_generic(root) and _looks_kitti(root):
        _row(FAIL, "layout mismatch", "this looks like a KITTI odometry tree "
                                      "-- pass --dataset-type kitti "
                                      "--sequence XX")
        return 1

    failures = 0
    handler = Generic_Handler(verbose=False, scans_dir=scans_dir,
                              pose_file=pose_file)

    label = str(scans_dir) if scans_dir is not None else "scans/"
    try:
        scans = handler.list_scan_files(root)
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as exc:
        _row(FAIL, label, str(exc))
        scans = []
        failures += 1
    else:
        _row(OK, label, f"{len(scans):,} {scans[0].suffix} files")

    n_poses = None
    if pose_file is not None:
        try:
            poses, _ = handler.load_poses(root)
        except (OSError, ValueError) as exc:
            _row(FAIL, "poses", str(exc).splitlines()[0])
            failures += 1
        else:
            n_poses = len(poses)
            kind = handler._sniff_pose_format(pose_file)
            _row(OK, "poses", f"{pose_file} ({kind}), {n_poses:,} poses")
    else:
        for name, kind in (("poses_kitti.txt", "kitti"), ("poses_tum.txt", "tum")):
            candidate = root / name
            if not candidate.exists():
                continue
            n_poses = sum(1 for line in candidate.read_text().splitlines()
                          if line.strip() and not line.startswith("#"))
            _row(OK, "poses", f"{name} ({kind}), {n_poses:,} poses")
            break
        else:
            _row(FAIL, "poses", "no poses_kitti.txt or poses_tum.txt")
            failures += 1

    # Generic_Handler.load_generic raises on this, well after the submap build
    # has started; it is cheap to catch here instead.
    if n_poses is not None and scans and n_poses != len(scans):
        _row(FAIL, "poses vs scans", f"{n_poses:,} poses but {len(scans):,} "
                                     f"scans -- load_generic requires they match")
        failures += 1

    if scans:
        failures += _check_sample_scan(scans[0])
    return failures


def _check_sample_scan(path: Path) -> int:
    """Read one scan: catches an unreadable file, and NaN-heavy sensors.

    Goes through the loader rather than open3d directly, so a ``.bin`` gets
    read the same way -- including the point-stride inference, which is the
    part most worth failing here rather than mid-run.
    """
    try:
        import numpy as np

        from inlier.eval.datasets.generic import Generic_Handler
    except ImportError:
        return 0

    handler = Generic_Handler(verbose=False)
    try:
        # The finite filter is the loader's; count against the raw points so
        # the NaN warning below still means something.
        points = handler.load_scan_file(path)
        if path.suffix.lower() == ".bin":
            raw = np.fromfile(path, dtype=np.float32)
            cols = handler._bin_cols(path, int(raw.size)) if raw.size else 4
            points = raw.reshape(-1, cols)[:, :3] if raw.size else points
        else:
            import open3d as o3d
            points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    except (ImportError, OSError, ValueError) as exc:
        _row(FAIL, "sample scan", f"{path.name}: {str(exc).splitlines()[0]}")
        return 1

    if points.size == 0:
        _row(FAIL, "sample scan", f"{path.name}: no points read")
        return 1

    n_bad = int((~np.isfinite(points).all(axis=1)).sum())
    detail = f"{path.name}: {len(points):,} points"
    if n_bad:
        # Dropped by the loader now, but a scan that is mostly NaN means the
        # submaps are far thinner than the file sizes suggest.
        _row(WARN, "sample scan",
             f"{detail}, {n_bad:,} non-finite ({n_bad / len(points):.0%}, dropped)")
    else:
        _row(OK, "sample scan", detail)
    return 0


def _check_overlap_sidecars(overlap_dir: Path) -> int:
    if not overlap_dir.exists():
        return 0
    matrices = sorted(overlap_dir.glob("overlap_*.txt"))
    if not matrices:
        return 0

    print(f"\noverlap ground truth  {overlap_dir}")
    missing = [m for m in matrices if not m.with_suffix(".json").exists()]
    _row(OK, "matrices", f"{len(matrices)} found")
    if missing:
        # Without the sidecar there is nothing to check the submap parameters
        # against, which is the mismatch docs/custom-data.md warns about in prose.
        _row(WARN, "sidecars", f"{len(missing)} matrix/matrices predate the "
                               f"provenance sidecar; parameters cannot be verified")
        for m in missing[:3]:
            print(f"         {m.name}")
        return 1
    _row(OK, "sidecars", "all matrices carry build parameters")
    return 0
