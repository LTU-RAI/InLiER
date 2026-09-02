#!/usr/bin/env python3
"""
Generic_Handler.py
    - Handler for generic folder-based datasets (flat scans/ directory + KITTI or TUM pose file).
    - Supports scan accumulation into submaps with a configurable window size and stride.

Expected dataset layout:
    /path/to/dataset/
        scans/
            000000.pcd
            000001.pcd
            ...
        poses_kitti.txt            # preferred: 12 floats per line (row-major 3x4)
        poses_tum.txt              # alternative: "#timestamp x y z qx qy qz qw"
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


class Generic_Handler:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Pose loading
    # ------------------------------------------------------------------
    def _pose_file(self, dataset_dir: Path) -> Tuple[Path, str]:
        kitti = dataset_dir / "poses_kitti.txt"
        tum = dataset_dir / "poses_tum.txt"
        if kitti.exists():
            return kitti, "kitti"
        if tum.exists():
            return tum, "tum"
        raise FileNotFoundError(
            f"No pose file found in {dataset_dir} "
            f"(expected poses_kitti.txt or poses_tum.txt)."
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
                print(f"  [Generic_Handler] {tum.name} has {len(stamps)} lines "
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
            stamps = self._timestamps_beside(dataset_dir, len(poses))
        if self.verbose:
            where = "" if not stamps else (
                " with timestamps" if fmt == "tum" else " + poses_tum.txt timestamps")
            print(f"  [Generic_Handler] loaded {len(poses)} poses ({fmt}) "
                  f"from {pose_path.name}{where}")
        return poses, stamps

    # ------------------------------------------------------------------
    # Scan discovery / loading
    # ------------------------------------------------------------------
    def list_scan_files(self, dataset_dir: Path) -> List[Path]:
        dataset_dir = Path(dataset_dir)
        scans_dir = dataset_dir / "scans"
        if not scans_dir.exists():
            raise FileNotFoundError(f"Scan directory not found: {scans_dir}")
        files = sorted(scans_dir.glob("*.pcd"))
        if not files:
            raise FileNotFoundError(f"No .pcd files under {scans_dir}")
        return files

    def load_scan_file(self, pcd_path: Path) -> np.ndarray:
        """Read a single .pcd file and return (N, 3) float32 points.

        Non-finite points are dropped here rather than downstream: sensors
        write NaN for invalid returns, and the points are then accumulated
        into submaps and handed to GICP as raw clouds, neither of which
        tolerates them.
        """
        import open3d as o3d  # lazy import
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        pts = np.asarray(pcd.points, dtype=np.float32)
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
            dataset_dir: path containing scans/ and a pose file.
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

        from inlier.eval.submaps import submap_windows

        poses, stamps = self.load_poses(dataset_dir)
        scan_files = self.list_scan_files(dataset_dir)
        if len(poses) != len(scan_files):
            raise RuntimeError(
                f"Pose/scan count mismatch in {dataset_dir}: "
                f"{len(poses)} poses vs {len(scan_files)} scans."
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
            ref_pose = poses[s]
            ref_inv = np.linalg.inv(ref_pose)
            window_pts: List[np.ndarray] = []
            for k in window:
                pts = self.load_scan_file(scan_files[k])
                if pts.size == 0:
                    continue
                if k == s:
                    window_pts.append(pts.astype(np.float32, copy=False))
                else:
                    T_rel = ref_inv @ poses[k]  # keyframe <- scan k
                    R = T_rel[:3, :3].astype(np.float32)
                    t = T_rel[:3, 3].astype(np.float32)
                    pts_local = (pts @ R.T) + t
                    window_pts.append(pts_local.astype(np.float32, copy=False))
            if not window_pts:
                continue
            submap = np.vstack(window_pts)
            submap_points.append(submap)
            submap_poses.append(ref_pose)
            ## keyframe-aligned: a window whose scans were all empty is
            ## skipped above, so appending here keeps all three in step
            submap_stamps.append(stamps[s] if stamps else 0.0)

        if self.verbose:
            scope = ("" if select is None
                     else f" (selected out of {len(submap_windows(len(poses), n_scans, stride))})")
            print(
                f"  [Generic_Handler] built {len(submap_points)} submap(s)"
                f"{scope} (n_scans={n_scans}, stride={stride}) from "
                f"{len(poses)} scans in {dataset_dir.name}"
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
    ) -> None:
        self.path = Path(path)
        self.n_scans = int(n_scans)
        self.stride = int(stride) if stride is not None else int(n_scans)
        self.transform = transform
        self.verbose = verbose
        self._handler = Generic_Handler(verbose=verbose)

    def load(self, **_):
        from inlier.eval.datasets.base import Sequence

        data = self._handler.load_generic(self.path, n_scans=self.n_scans, stride=self.stride)
        seq = Sequence.from_handler_dict(data, **self.describe())
        return seq.transformed(self.transform)

    def describe(self):
        return {
            "dataset_type": self.name,
            "path": str(self.path),
            "n_scans": self.n_scans,
            "stride": self.stride,
            "transform": self.transform is not None,
        }

    @classmethod
    def from_describe(cls, described, *, root=None, verbose=False):
        """Rebuild the source from what :meth:`describe` wrote into a run.

        The submap accumulation (``n_scans``/``stride``) comes back with it, so
        a replay cannot silently re-window the sequence differently from the
        run it is replaying.  ``transform`` is not restored here: the protocol
        applies it to the poses, not the loader (see ``cross_session.run``).
        """
        return cls(root or described["path"],
                   described.get("n_scans", 1), described.get("stride"),
                   verbose=verbose)

    @property
    def tag(self) -> str:
        return f"{self.path.name}_n{self.n_scans}s{self.stride}"
