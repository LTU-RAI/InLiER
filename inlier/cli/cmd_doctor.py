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

OK = "ok"
WARN = "warn"
FAIL = "FAIL"

CORE_DEPS = ("numpy", "yaml", "small_gicp")
EVAL_DEPS = ("scipy", "open3d", "tqdm", "matplotlib", "pandas")


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
                   choices=("helipr", "generic"), default="helipr",
                   help="layout to check --dataset against (default: helipr)")
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

    if args.dataset:
        failures += _check_dataset(Path(args.dataset), args.dataset_type)

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
    "generic": "<root>/scans/*.pcd + poses_kitti.txt or poses_tum.txt",
}


def _check_dataset(root: Path, dataset_type: str) -> int:
    print(f"\ndataset  {root}")
    _row(OK, "layout", f"{dataset_type} -- {LAYOUTS[dataset_type]}")
    if not root.exists():
        _row(FAIL, "root", "does not exist")
        return 1
    if dataset_type == "generic":
        return _check_generic(root)
    return _check_helipr(root)


def _looks_generic(root: Path) -> bool:
    return (root / "scans").is_dir()


def _looks_helipr(root: Path) -> bool:
    return any((seq / "Undistorted").is_dir() or (seq / "LiDAR").is_dir()
               for seq in root.iterdir() if seq.is_dir())


def _check_helipr(root: Path) -> int:
    sequences = sorted(p for p in root.iterdir() if p.is_dir())
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


def _check_generic(root: Path) -> int:
    """Flat scans/ + a pose file, per ``Generic_Handler``."""
    if not _looks_generic(root) and _looks_helipr(root):
        _row(FAIL, "layout mismatch", "this looks like a HeLiPR tree -- "
                                      "pass --dataset-type helipr")
        return 1

    failures = 0
    scans_dir = root / "scans"
    if not scans_dir.is_dir():
        _row(FAIL, "scans/", "not found")
        return 1

    scans = sorted(scans_dir.glob("*.pcd"))
    if not scans:
        _row(FAIL, "scans/", "no .pcd files")
        failures += 1
    else:
        _row(OK, "scans/", f"{len(scans):,} .pcd files")

    n_poses = None
    for name, kind in (("poses_kitti.txt", "kitti"), ("poses_tum.txt", "tum")):
        pose_file = root / name
        if not pose_file.exists():
            continue
        n_poses = sum(1 for line in pose_file.read_text().splitlines()
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
    """Read one scan: catches an unreadable file, and NaN-heavy sensors."""
    try:
        import numpy as np
        import open3d as o3d
    except ImportError:
        return 0

    points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
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
