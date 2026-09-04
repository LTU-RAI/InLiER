#!/usr/bin/env python3
"""
Generic_Handler.py
    - Handler for generic folder-based datasets (flat scans/ directory + KITTI or TUM pose file).
    - Supports scan accumulation into submaps with a configurable window size and stride.

Expected dataset layout:
    /path/to/dataset/
        scans/
            000000.pcd             # .bin is also read (KITTI velodyne float32)
            000001.pcd
            ...
        poses_kitti.txt            # preferred: 12 floats per line (row-major 3x4)
        poses_tum.txt              # alternative: "#timestamp x y z qx qy qz qw"

That layout is a convenience, not a requirement.  ``scans_dir`` and
``pose_file`` name the two paths directly, for the common case of a dataset
whose scans and poses were never arranged into one tree::

    Generic_Handler(scans_dir="/data/seq/velodyne", pose_file="/data/gt/odom.txt")

An explicitly named pose file is *sniffed* rather than trusted to be named
sensibly: 12 fields on the first data line is KITTI, 8 is TUM.  Nothing else
is a valid line in either format, so the two cannot be confused.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: Scan formats ``list_scan_files`` will pick up, lowercase.
SCAN_SUFFIXES = (".pcd", ".bin")

#: Floats per point in a ``.bin`` scan when the file size does not settle it.
#: KITTI's velodyne dumps are ``x y z intensity``, and everything that copied
#: the format kept the four; a file that is not a multiple of four floats is
#: something else and gets inferred instead.
DEFAULT_BIN_COLS = 4


class Generic_Handler:
    """Reader for a folder of scans plus a pose file.

    ``scans_dir`` and ``pose_file`` override the conventional layout when the
    data does not sit in one directory.  Either may be given alone: naming the
    poses while letting the scans default to ``<dataset>/scans`` is a perfectly
    ordinary case.

    ``bin_cols`` forces the point stride of ``.bin`` scans.  Leave it ``None``
    unless a file is inferred wrongly -- see :meth:`_bin_cols`.
    """

    #: Prefix for this handler's progress lines; subclasses say their own name.
    _LOG_NAME = "Generic_Handler"

    def __init__(self, verbose: bool = True, scans_dir=None, pose_file=None,
                 bin_cols: Optional[int] = None):
        self.verbose = verbose
        self.scans_dir = Path(scans_dir) if scans_dir is not None else None
        self.pose_file = Path(pose_file) if pose_file is not None else None
        if bin_cols is not None and bin_cols < 3:
            raise ValueError(f"bin_cols must be >= 3 (x, y, z), got {bin_cols}")
        self.bin_cols = bin_cols

    # ------------------------------------------------------------------
    # Pose loading
    # ------------------------------------------------------------------
    @staticmethod
    def _sniff_pose_format(path: Path) -> str:
        """``"kitti"`` or ``"tum"``, from the first line that carries data.

        An explicitly named pose file cannot be identified by its name, and
        need not be: a KITTI line is 12 numbers and a TUM line is 8, so the
        field count settles it.  Guessing from the filename would quietly
        mis-read ``odometry.txt``.
        """
        with open(path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                n = len(line.split())
                if n == 12:
                    return "kitti"
                if n == 8:
                    return "tum"
                raise ValueError(
                    f"{path}: first data line has {n} fields, which is neither "
                    f"KITTI (12: row-major 3x4) nor TUM (8: timestamp x y z "
                    f"qx qy qz qw).\n  {line[:100]}")
        raise ValueError(f"{path} contains no pose lines (only blanks/comments).")

    def _pose_file(self, dataset_dir: Path) -> Tuple[Path, str]:
        """``(path, format)`` for the poses, honouring an explicit override."""
        if self.pose_file is not None:
            if not self.pose_file.exists():
                raise FileNotFoundError(f"Pose file not found: {self.pose_file}")
            if self.pose_file.is_dir():
                raise IsADirectoryError(
                    f"Pose file is a directory: {self.pose_file} -- name the "
                    f"file itself, not the folder holding it.")
            return self.pose_file, self._sniff_pose_format(self.pose_file)

        kitti = dataset_dir / "poses_kitti.txt"
        tum = dataset_dir / "poses_tum.txt"
        if kitti.exists():
            return kitti, "kitti"
        if tum.exists():
            return tum, "tum"
        raise FileNotFoundError(
            f"No pose file found in {dataset_dir} "
            f"(expected poses_kitti.txt or poses_tum.txt). "
            f"Name one explicitly instead if the poses live elsewhere."
        )

    def _timestamps_beside(self, dataset_dir: Path, n_poses: int) -> List[float]:
        """TUM timestamps for a dataset whose poses came from KITTI.

        KITTI is pose-only, but a dataset shipping both files has the times
        right there in ``poses_tum.txt`` column 0.  Discarding them would force
        ``--exclusion`` onto ``frames=`` or ``metres=`` for no reason.

        Which file supplies the *poses* is deliberately left alone: switching
        the preference would change the pose values existing generic results
        were produced with.  Only the timestamps are borrowed.  A count
        mismatch means the two files describe different runs, so they are
        dropped rather than misaligned onto the poses.
        """
        tum = dataset_dir / "poses_tum.txt"
        if not tum.exists():
            return []
        stamps: List[float] = []
        with open(tum, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    return []
                try:
                    stamps.append(float(parts[0]))
                except ValueError:
                    return []
        if len(stamps) != n_poses:
            if self.verbose:
                print(f"  [{self._LOG_NAME}] {tum.name} has {len(stamps)} lines "
                      f"for {n_poses} poses; ignoring its timestamps")
            return []
        return stamps

    def load_poses(self, dataset_dir: Path) -> Tuple[List[np.ndarray], List[float]]:
        """``(poses, timestamps)`` in dataset order.

        Timestamps come from TUM column 0 and are ``[]`` when the dataset has
        none.  The two-value return matches ``HeLiPR_Handler.load_poses``,
        which has always returned both.
        """
        dataset_dir = Path(dataset_dir)
        pose_path, fmt = self._pose_file(dataset_dir)
        poses: List[np.ndarray] = []
        stamps: List[float] = []
        with open(pose_path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if fmt == "kitti":
                    if len(parts) != 12:
                        raise ValueError(
                            f"KITTI pose line has {len(parts)} values (expected 12): {line[:80]}"
                        )
                    vals = np.array(parts, dtype=np.float64)
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :4] = vals.reshape(3, 4)
                else:  # tum
                    if len(parts) < 8:
                        raise ValueError(
                            f"TUM pose line has {len(parts)} values (expected 8): {line[:80]}"
                        )
                    stamps.append(float(parts[0]))
                    tx, ty, tz = map(float, parts[1:4])
                    qx, qy, qz, qw = map(float, parts[4:8])
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :3] = self._quaternion_to_rotation_matrix(qx, qy, qz, qw)
                    T[:3, 3] = [tx, ty, tz]
                poses.append(T)
        if fmt == "kitti":
            # Beside the *pose file*, which is the dataset dir in the
            # conventional layout and the override's folder otherwise.
            stamps = self._timestamps_beside(pose_path.parent, len(poses))
        if self.verbose:
            where = "" if not stamps else (
                " with timestamps" if fmt == "tum" else " + poses_tum.txt timestamps")
            print(f"  [{self._LOG_NAME}] loaded {len(poses)} poses ({fmt}) "
                  f"from {pose_path.name}{where}")
        return poses, stamps

    # ------------------------------------------------------------------
    # Scan discovery / loading
    # ------------------------------------------------------------------
    def scan_dir(self, dataset_dir: Path) -> Path:
        """Where the scans are: the override, else ``<dataset_dir>/scans``."""
        return self.scans_dir if self.scans_dir is not None else Path(dataset_dir) / "scans"

    def list_scan_files(self, dataset_dir: Path) -> List[Path]:
        """Scan paths in name order.

        ``.pcd`` and ``.bin`` are both read, but a directory holding *both* is
        rejected rather than merged: the usual cause is the same sequence
        exported twice, and quietly returning 2N files for N poses would fail
        much later as a pose/scan count mismatch.
        """
        scans_dir = self.scan_dir(dataset_dir)
        if not scans_dir.exists():
            raise FileNotFoundError(f"Scan directory not found: {scans_dir}")
        if not scans_dir.is_dir():
            raise NotADirectoryError(
                f"Scan path is not a directory: {scans_dir} -- name the folder "
                f"holding the scans, not one scan.")

        by_suffix = {suf: sorted(p for p in scans_dir.iterdir()
                                 if p.suffix.lower() == suf)
                     for suf in SCAN_SUFFIXES}
        present = {suf: files for suf, files in by_suffix.items() if files}
        if not present:
            raise FileNotFoundError(
                f"No {' or '.join(SCAN_SUFFIXES)} files under {scans_dir}")
        if len(present) > 1:
            counts = ", ".join(f"{len(f)} {suf}" for suf, f in present.items())
            raise RuntimeError(
                f"{scans_dir} holds more than one scan format ({counts}). "
                f"Point at a directory with just one, or split them.")
        return next(iter(present.values()))

    def _bin_cols(self, path: Path, n_floats: int) -> int:
        """Floats per point in a ``.bin`` scan.

        ``bin_cols`` wins if it was set.  Otherwise the answer is 4: KITTI's
        velodyne dumps are ``x y z intensity`` and the format was copied
        wholesale.  Deliberately *not* inferred when 4 divides evenly -- a
        point count divisible by 12 is equally consistent with 3 and with 6,
        and silently picking one would scramble the coordinates instead of
        failing.  Only a file that 4 cannot explain gets inferred, and only
        when exactly one width fits.
        """
        if self.bin_cols is not None:
            if n_floats % self.bin_cols:
                raise ValueError(
                    f"{path}: {n_floats} float32 values is not a multiple of "
                    f"bin_cols={self.bin_cols}")
            return self.bin_cols
        if n_floats % DEFAULT_BIN_COLS == 0:
            return DEFAULT_BIN_COLS
        fits = [c for c in (3, 5, 6) if n_floats % c == 0]
        if len(fits) == 1:
            if self.verbose:
                print(f"  [{self._LOG_NAME}] {path.name}: {n_floats} floats is "
                      f"not x,y,z,intensity; reading {fits[0]} per point")
            return fits[0]
        raise ValueError(
            f"{path}: {n_floats} float32 values fits "
            f"{fits or 'no'} floats per point, so the layout is ambiguous. "
            f"Pass bin_cols=N to say how wide a point is.")

    def _load_bin(self, path: Path) -> np.ndarray:
        """KITTI-style raw float32 dump -> (N, 3).

        The file is a flat little-endian float32 array with no header, so the
        only thing to get right is how many values make up a point.
        """
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        cols = self._bin_cols(path, int(raw.size))
        return raw.reshape(-1, cols)[:, :3]

    def load_scan_file(self, scan_path: Path) -> np.ndarray:
        """Read one scan and return (N, 3) float32 points.

        ``.pcd`` goes through open3d; ``.bin`` is read directly.  Non-finite
        points are dropped here rather than downstream: sensors write NaN for
        invalid returns, and the points are then accumulated into submaps and
        handed to GICP as raw clouds, neither of which tolerates them.
        """
        scan_path = Path(scan_path)
        suffix = scan_path.suffix.lower()
        if suffix == ".bin":
            pts = self._load_bin(scan_path)
        elif suffix == ".pcd":
            import open3d as o3d  # lazy import
            pcd = o3d.io.read_point_cloud(str(scan_path))
            pts = np.asarray(pcd.points, dtype=np.float32)
        else:
            raise ValueError(
                f"Unsupported scan format {suffix!r}: {scan_path} "
                f"(expected one of {', '.join(SCAN_SUFFIXES)})")
        return pts[np.isfinite(pts).all(axis=1)]

    # ------------------------------------------------------------------
    # Main entry point: submap accumulation
    # ------------------------------------------------------------------
    def load_generic(
        self,
        dataset_dir: Path,
        n_scans: int = 1,
        stride: Optional[int] = None,
        select: Optional[Sequence[int]] = None,
    ) -> dict:
        """Load the dataset and accumulate scans into submaps.

        For window ``w = [i, i+1, ..., i+n_scans-1]`` the keyframe is scan i.
        Every scan in the window is transformed into the keyframe's pose frame
        via ``inv(T_i) @ T_k`` and concatenated. The returned pose for that
        submap is ``T_i`` (the global pose of the keyframe).

        Args:
            dataset_dir: path containing scans/ and a pose file.  Ignored for
                         whichever of the two the handler was given an
                         explicit ``scans_dir``/``pose_file`` for.
            n_scans:     number of consecutive scans per submap (>= 1).
            stride:      step between consecutive submaps. Defaults to n_scans
                         (non-overlapping submaps).
            select:      submap indices to build, or None for all of them.
                         Only the scans those submaps need are read, which is
                         what makes inspecting one submap out of hundreds
                         cheap. The grouping is unaffected: indices always
                         refer to positions in the full submap sequence.

        Returns:
            dict with keys shaped like HeLiPR_Handler.load_helipr:
                "poses": list[np.ndarray(4,4) float64]  (length M)
                "point_clouds": list[np.ndarray(N_i, 3) float32]  (length M)
                "pose_timestamps": list[float] (TUM column 0 when the dataset
                                   has it, else zeros -- KITTI carries none)
                "pc_timestamps":   list[float] (same; one keyframe per submap)
        """
        dataset_dir = Path(dataset_dir)
        if n_scans < 1:
            raise ValueError(f"n_scans must be >= 1, got {n_scans}")
        if stride is None:
            stride = n_scans
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        from inlier.eval.submaps import build_submap, submap_windows

        poses, stamps = self.load_poses(dataset_dir)
        scan_files = self.list_scan_files(dataset_dir)
        if len(poses) != len(scan_files):
            # Name both paths: with explicit overrides they need not share a
            # parent, and "in <dataset>" would point at neither.
            pose_path, _ = self._pose_file(dataset_dir)
            raise RuntimeError(
                f"Pose/scan count mismatch: {len(poses)} poses in {pose_path} "
                f"vs {len(scan_files)} scans in {self.scan_dir(dataset_dir)}."
            )

        # The window rule is shared with build_overlap_data via
        # submap_windows, so submap counts cannot drift from the overlap
        # matrix's dimensions.
        windows = submap_windows(len(poses), n_scans, stride)
        if select is not None:
            total = len(windows)
            chosen = []
            for i in select:
                if not -total <= i < total:
                    raise IndexError(
                        f"submap index {i} out of range: {dataset_dir.name} has "
                        f"{total} submaps at n_scans={n_scans}, stride={stride}")
                chosen.append(windows[i])
            windows = chosen

        submap_poses: List[np.ndarray] = []
        submap_points: List[np.ndarray] = []
        submap_stamps: List[float] = []

        import tqdm  # lazy
        iterator = tqdm.tqdm(
            windows, disable=not self.verbose,
            desc=f"  Building submaps (n={n_scans}, stride={stride})")
        for window in iterator:
            s = window[0]
            submap = build_submap(self.load_scan_file, scan_files, poses, window)
            if submap is None:
                continue
            submap_points.append(submap)
            submap_poses.append(poses[s])
            ## keyframe-aligned: a window whose scans were all empty is
            ## skipped above, so appending here keeps all three in step
            submap_stamps.append(stamps[s] if stamps else 0.0)

        if self.verbose:
            scope = ("" if select is None
                     else f" (selected out of {len(submap_windows(len(poses), n_scans, stride))})")
            print(
                f"  [{self._LOG_NAME}] built {len(submap_points)} submap(s)"
                f"{scope} (n_scans={n_scans}, stride={stride}) from "
                f"{len(poses)} scans in {self.scan_dir(dataset_dir)}"
            )

        return {
            "poses": submap_poses,
            "point_clouds": submap_points,
            "pose_timestamps": submap_stamps,
            "pc_timestamps": list(submap_stamps),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        return np.array([
            [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw),     2 * (qx*qz + qy*qw)],
            [2 * (qx*qy + qz*qw),     1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
            [2 * (qx*qz - qy*qw),     2 * (qy*qz + qx*qw),     1 - 2 * (qx**2 + qy**2)],
        ], dtype=np.float64)


# if __name__ == "__main__":
#     handler = Generic_Handler(verbose=True)
#     data = handler.load_generic(
#         Path("/home/niksta/Documents/datasets/campus_ouster"),
#         n_scans=1,
#         stride=1,
#     )
#     print(f"poses: {len(data['poses'])}  first pose translation: {data['poses'][0][:3, 3]}")
#     print(f"first submap shape: {data['point_clouds'][0].shape}")


# ---------------------------------------------------------------------------
#  SequenceSource adapter
# ---------------------------------------------------------------------------

class GenericSource:
    """:class:`~inlier.eval.datasets.base.SequenceSource` over ``Generic_Handler``.

    ``n_scans``/``stride`` control submap accumulation and **must** match what
    the overlap ground truth was built with -- the matrix is indexed by submap,
    so a mismatch silently misaligns GT against retrieval.  docs/custom-data.md warns
    about this in prose; :mod:`inlier.eval.overlap` now records the values in a
    sidecar and checks them instead.
    """

    name = "generic"

    def __init__(
        self,
        path,
        n_scans: int = 1,
        stride: Optional[int] = None,
        transform=None,
        verbose: bool = True,
        scans_dir=None,
        pose_file=None,
        bin_cols: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        self.n_scans = int(n_scans)
        self.stride = int(stride) if stride is not None else int(n_scans)
        self.transform = transform
        self.verbose = verbose
        self.scans_dir = Path(scans_dir) if scans_dir is not None else None
        self.pose_file = Path(pose_file) if pose_file is not None else None
        self.bin_cols = bin_cols
        self._handler = Generic_Handler(verbose=verbose, scans_dir=scans_dir,
                                        pose_file=pose_file, bin_cols=bin_cols)

    @classmethod
    def from_paths(cls, scans_dir, pose_file, *args, path=None, **kw):
        """Build from the two explicit paths, with no dataset directory.

        ``path`` still exists downstream as the sequence's *identity* -- it
        names the cache entry, the run directory and the tag -- so it defaults
        to the scans directory's parent, which for ``/data/seq05/velodyne`` is
        the sequence.  Pass it when that guess would be unhelpful.
        """
        scans_dir = Path(scans_dir)
        return cls(path if path is not None else scans_dir.parent,
                   *args, scans_dir=scans_dir, pose_file=pose_file, **kw)

    def load(self, **_):
        from inlier.eval.datasets.base import Sequence

        data = self._handler.load_generic(self.path, n_scans=self.n_scans, stride=self.stride)
        seq = Sequence.from_handler_dict(data, **self.describe())
        return seq.transformed(self.transform)

    def describe(self):
        d = {
            "dataset_type": self.name,
            "path": str(self.path),
            "n_scans": self.n_scans,
            "stride": self.stride,
            "transform": self.transform is not None,
        }
        # Only recorded when they were actually used, so a run against the
        # conventional layout keeps the description it has always had.
        if self.scans_dir is not None:
            d["scans_dir"] = str(self.scans_dir)
        if self.pose_file is not None:
            d["pose_file"] = str(self.pose_file)
        if self.bin_cols is not None:
            d["bin_cols"] = self.bin_cols
        return d

    @classmethod
    def from_describe(cls, described, *, root=None, verbose=False):
        """Rebuild the source from what :meth:`describe` wrote into a run.

        The submap accumulation (``n_scans``/``stride``) comes back with it, so
        a replay cannot silently re-window the sequence differently from the
        run it is replaying.  So do explicit ``scans_dir``/``pose_file`` paths,
        which is what lets a replay find the scans of a run that never had a
        conventional dataset directory.  ``transform`` is not restored here:
        the protocol applies it to the poses, not the loader (see
        ``cross_session.run``).
        """
        return cls(root or described["path"],
                   described.get("n_scans", 1), described.get("stride"),
                   verbose=verbose,
                   scans_dir=described.get("scans_dir"),
                   pose_file=described.get("pose_file"),
                   bin_cols=described.get("bin_cols"))

    @property
    def tag(self) -> str:
        return f"{self.path.name}_n{self.n_scans}s{self.stride}"
