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
from typing import List, Optional, Tuple

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

    def load_poses(self, dataset_dir: Path) -> List[np.ndarray]:
        """Return a list of (4,4) float64 SE(3) matrices in dataset order."""
        dataset_dir = Path(dataset_dir)
        pose_path, fmt = self._pose_file(dataset_dir)
        poses: List[np.ndarray] = []
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
                    tx, ty, tz = map(float, parts[1:4])
                    qx, qy, qz, qw = map(float, parts[4:8])
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :3] = self._quaternion_to_rotation_matrix(qx, qy, qz, qw)
                    T[:3, 3] = [tx, ty, tz]
                poses.append(T)
        if self.verbose:
            print(f"  [Generic_Handler] loaded {len(poses)} poses ({fmt}) from {pose_path.name}")
        return poses

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
        """Read a single .pcd file and return (N, 3) float32 points."""
        import open3d as o3d  # lazy import
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        pts = np.asarray(pcd.points, dtype=np.float32)
        return pts

    # ------------------------------------------------------------------
    # Main entry point: submap accumulation
    # ------------------------------------------------------------------
    def load_generic(
        self,
        dataset_dir: Path,
        n_scans: int = 1,
        stride: Optional[int] = None,
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

        Returns:
            dict with keys shaped like HeLiPR_Handler.load_helipr:
                "poses": list[np.ndarray(4,4) float64]  (length M)
                "point_clouds": list[np.ndarray(N_i, 3) float32]  (length M)
                "pose_timestamps": list[float] (zeros, present for API parity)
                "pc_timestamps":   list[float] (zeros, present for API parity)
        """
        dataset_dir = Path(dataset_dir)
        if n_scans < 1:
            raise ValueError(f"n_scans must be >= 1, got {n_scans}")
        if stride is None:
            stride = n_scans
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        poses = self.load_poses(dataset_dir)
        scan_files = self.list_scan_files(dataset_dir)
        if len(poses) != len(scan_files):
            raise RuntimeError(
                f"Pose/scan count mismatch in {dataset_dir}: "
                f"{len(poses)} poses vs {len(scan_files)} scans."
            )

        N = len(poses)
        # Match build_overlap_data.py._group_into_submaps: iterate starts at
        # every `stride`-th index up to N-1 and keep a short trailing window
        # when N is not divisible by stride. This ensures submap counts align
        # with the overlap matrix dimensions.
        starts = list(range(0, N, stride))

        submap_poses: List[np.ndarray] = []
        submap_points: List[np.ndarray] = []

        import tqdm  # lazy
        iterator = tqdm.tqdm(starts, desc=f"  Building submaps (n={n_scans}, stride={stride})")
        for s in iterator:
            ref_pose = poses[s]
            ref_inv = np.linalg.inv(ref_pose)
            window_pts: List[np.ndarray] = []
            window_end = min(s + n_scans, N)
            for k in range(s, window_end):
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

        if self.verbose:
            print(
                f"  [Generic_Handler] built {len(submap_points)} submaps "
                f"(n_scans={n_scans}, stride={stride}) from {N} scans in {dataset_dir.name}"
            )

        zeros = [0.0] * len(submap_points)
        return {
            "poses": submap_poses,
            "point_clouds": submap_points,
            "pose_timestamps": zeros,
            "pc_timestamps": zeros,
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
    so a mismatch silently misaligns GT against retrieval.  README.md warns
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

    @property
    def tag(self) -> str:
        return f"{self.path.name}_n{self.n_scans}s{self.stride}"
