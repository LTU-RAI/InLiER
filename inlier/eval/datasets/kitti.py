"""KITTI odometry, with the pose frame put right.

KITTI is the one dataset where reading the files correctly is not enough.  Its
ground-truth poses are expressed in the **left rectified camera** frame, while
the scans in ``velodyne/*.bin`` are in the **velodyne** frame.  Loading both
verbatim -- which is what ``--dataset-type generic`` does -- produces a sequence
whose poses and points live in different coordinate systems, and nothing
downstream can tell.

Concretely, on sequence 00 the raw pose translations span x=565 m, y=15 m,
z=498 m: the *vertical* axis is y.  Everything InLiER does with a position is
XY-only -- ``max_pose_dist``, the search radius, ``arc_length`` -- so ground
truth ends up measured across a 565x15 m sliver of a facade rather than the
498x565 m ground plane.  And with ``n_scans > 1`` the submaps are corrupted
outright, because accumulation applies a camera-frame relative transform
``inv(T_i) @ T_k`` to velodyne points.

``calib.txt`` carries the fix.  Its ``Tr`` maps velodyne points into the
rectified camera frame, and ``poses`` maps camera *i* into camera 0, so::

    T_i = inv(Tr) @ P_i @ Tr        # velodyne_i -> velodyne_0

is the pose of scan *i* in a velodyne-frame world.  After it, z is vertical
(9.4 m of elevation change on sequence 00) and the ground plane is xy.  The
correction is read from the dataset's own calibration, so there is nothing to
configure and nothing to configure wrongly.

Two layouts are in the wild and both are supported:

``KITTI odometry`` (the official devkit)
    ``<root>/sequences/XX/{velodyne/, calib.txt, times.txt}`` with the poses
    kept apart in ``<root>/poses/XX.txt`` -- and only for sequences 00-10,
    which are the ones with published ground truth.

``SemanticKITTI``
    the same tree, with ``poses.txt`` copied into each sequence folder.

``--dataset`` may also point straight at a sequence directory.

Everything except the poses is ``Generic_Handler``'s: ``.bin`` reading, the
submap accumulation, the keyframe rule, the count check.  KITTI's scans are
``x y z reflectance`` float32, which is already that reader's default.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from inlier.eval.datasets.generic import Generic_Handler

#: Sequences KITTI publishes ground-truth poses for.
POSED_SEQUENCES = tuple(f"{i:02d}" for i in range(11))

#: Sub-directory holding the scans, in place of generic's ``scans/``.
SCAN_SUBDIR = "velodyne"


def normalise_sequence(sequence) -> str:
    """``0`` / ``"0"`` / ``"00"`` -> ``"00"``.

    KITTI names its sequences with two digits, and typing ``--sequence 0`` is
    the obvious mistake to make; padding it beats a "no such directory" that
    is technically true and unhelpful.
    """
    text = str(sequence).strip()
    return text.zfill(2) if text.isdigit() and len(text) < 2 else text


def read_calib_tr(path: Path) -> np.ndarray:
    """The 4x4 velodyne -> rectified-camera transform from a ``calib.txt``.

    The odometry devkit writes it as a ``Tr:`` line of 12 floats (row-major
    3x4).  Raw-KITTI calibration files spell the same thing
    ``Tr_velo_to_cam``, so both are accepted -- a file that has neither is
    reported with the keys it *did* have, since the usual cause is pointing at
    the wrong calibration file rather than a corrupt one.
    """
    path = Path(path)
    keys: List[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, values = line.partition(":")
        key = key.strip()
        keys.append(key)
        if key not in ("Tr", "Tr_velo_to_cam"):
            continue
        numbers = np.array(values.split(), dtype=np.float64)
        if numbers.size != 12:
            raise ValueError(
                f"{path}: '{key}' has {numbers.size} values, expected 12 "
                f"(row-major 3x4)")
        T = np.eye(4, dtype=np.float64)
        T[:3, :4] = numbers.reshape(3, 4)
        return T

    raise ValueError(
        f"{path} has no 'Tr:' line, so the velodyne-to-camera transform is "
        f"unknown and KITTI's poses cannot be corrected. Keys found: "
        f"{', '.join(keys) if keys else '(none)'}. This is usually the wrong "
        f"calibration file -- the odometry benchmark's lives at "
        f"<root>/sequences/<seq>/calib.txt.")


def camera_to_velodyne(poses_cam: np.ndarray, Tr: np.ndarray) -> np.ndarray:
    """``(N,4,4)`` camera-frame poses -> velodyne-frame, via ``inv(Tr) P Tr``.

    ``Tr`` maps velodyne points into the camera frame and ``P_i`` maps camera
    *i* into camera 0, so conjugating by ``Tr`` re-expresses the same rigid
    motion between velodyne frames.  Free-standing and pure so the direction of
    the conjugation -- the one thing here that is easy to get backwards and
    impossible to notice -- can be pinned by a test on its own.
    """
    Tr_inv = np.linalg.inv(Tr)
    return Tr_inv @ np.asarray(poses_cam, dtype=np.float64) @ Tr


class KITTI_Handler(Generic_Handler):
    """``Generic_Handler`` over a KITTI sequence, with the poses corrected.

    Three overrides -- :meth:`scan_dir`, :meth:`_pose_file`, :meth:`load_poses`
    -- and that is provably enough: ``load_generic`` touches only those, plus
    ``list_scan_files``, ``load_scan_file`` and ``verbose``, all of which are
    already right for KITTI.  So the windows, the keyframe rule, the
    accumulation and the count check are inherited verbatim.

    That inheritance is what fixes the corrupted submaps for free: accumulation
    builds ``inv(poses[s]) @ poses[k]`` from whatever :meth:`load_poses`
    returned, so correcting the frame there corrects the clouds too.
    """

    _LOG_NAME = "KITTI_Handler"

    def __init__(self, dataset_path, sequence, verbose: bool = True,
                 bin_cols: Optional[int] = None) -> None:
        self.root = Path(dataset_path)
        self.sequence = normalise_sequence(sequence)
        # Deliberately no path resolution here: constructing a source must not
        # require the dataset to be present, or `from_describe` would explode
        # on a machine that has the results but not the scans, and `tag` --
        # which decides a cache filename -- would depend on disk state.
        super().__init__(verbose=verbose, bin_cols=bin_cols)

    # ------------------------------------------------------------------
    # Path resolution, done lazily on first use.  Every failure names each
    # path that was tried: with two layouts in circulation, "not found"
    # without the candidates is a guessing game.
    #
    # `.is_file()` throughout, never `.exists()` -- a SemanticKITTI tree can
    # carry a `poses/` *directory* next to `poses.txt`, and `.exists()` would
    # happily return it.
    # ------------------------------------------------------------------
    @property
    def seq_dir(self) -> Path:
        """``<root>/sequences/<seq>``, or ``<root>`` if it is already one."""
        if (self.root / SCAN_SUBDIR).is_dir():
            return self.root
        candidate = self.root / "sequences" / self.sequence
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(
            f"no KITTI sequence {self.sequence!r} under {self.root}: expected "
            f"{candidate}, and {self.root} itself holds no {SCAN_SUBDIR}/ "
            f"directory either.")

    @property
    def odom_root(self) -> Path:
        """The benchmark root, whichever way ``--dataset`` was pointed.

        ``poses/`` sits beside ``sequences/`` in the official layout, so a
        ``--dataset`` aimed straight at ``.../sequences/00`` still has to look
        two levels up to find them.  Without this, the sequence-directory
        convenience silently works on SemanticKITTI trees and fails on
        official ones.
        """
        seq_dir = self.seq_dir
        return self.root if seq_dir == self.root / "sequences" / self.sequence \
            else seq_dir.parent.parent

    @property
    def calib_file(self) -> Path:
        """The calibration to correct with, per-sequence first.

        Sequences really do have different ``Tr`` rows -- 00 and 08 differ in
        the first value -- so a root-level copy is a fallback, never a
        shortcut.
        """
        seq_dir = self.seq_dir
        for candidate in (seq_dir / "calib.txt", self.odom_root / "calib.txt"):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"no calib.txt for sequence {self.sequence} (tried "
            f"{seq_dir / 'calib.txt'} and {self.odom_root / 'calib.txt'}). "
            f"KITTI's "
            f"ground-truth poses are in the left rectified camera frame; "
            f"without Tr (velodyne->camera) they cannot be brought into the "
            f"frame the scans are in, and every XY distance would be measured "
            f"against the wrong two axes.")

    @property
    def times_file(self) -> Path:
        return self.seq_dir / "times.txt"

    def scan_dir(self, dataset_dir) -> Path:
        """``<seq>/velodyne`` -- KITTI's name for generic's ``scans/``."""
        return self.seq_dir / SCAN_SUBDIR

    def _pose_file(self, dataset_dir) -> Tuple[Path, str]:
        """``(path, "kitti")``, resolving the two layouts.

        Returns the base class's 2-tuple: ``load_generic`` re-calls this to
        name the pose file in its count-mismatch message.
        """
        official = self.odom_root / "poses" / f"{self.sequence}.txt"
        semantic = self.seq_dir / "poses.txt"
        for candidate in (official, semantic):
            if candidate.is_file():
                return candidate, "kitti"
        if self.sequence not in POSED_SEQUENCES:
            raise FileNotFoundError(
                f"no ground-truth poses for KITTI sequence {self.sequence}. "
                f"Looked for {official} and {semantic}. The odometry benchmark "
                f"publishes poses for sequences "
                f"{POSED_SEQUENCES[0]}-{POSED_SEQUENCES[-1]} only; "
                f"{self.sequence} is part of the held-out test set, and an "
                f"evaluation needs poses.")
        raise FileNotFoundError(
            f"no poses for sequence {self.sequence} (tried {official} and "
            f"{semantic}).")

    # ------------------------------------------------------------------
    # The one thing KITTI does differently
    # ------------------------------------------------------------------
    def load_poses(self, dataset_dir) -> Tuple[List[np.ndarray], List[float]]:
        """``(poses, timestamps)`` in the **velodyne** frame.

        The file on disk is in the camera frame; see the module docstring for
        why that is not a detail.  ``dataset_dir`` is accepted for signature
        parity with the base class and ignored -- the paths were resolved once,
        in ``__init__``, from the root and sequence.
        """
        pose_path, _ = self._pose_file(dataset_dir)
        calib_path = self.calib_file
        rows = np.atleast_2d(np.loadtxt(pose_path, dtype=np.float64))
        if rows.size and rows.shape[1] != 12:
            raise ValueError(
                f"{pose_path}: {rows.shape[1]} values per line, expected "
                f"12 (row-major 3x4). KITTI poses are 12 floats; a TUM file "
                f"has 8 and belongs to --dataset-type generic.")

        n = rows.shape[0] if rows.size else 0
        poses_cam = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
        if n:
            poses_cam[:, :3, :4] = rows.reshape(n, 3, 4)
        Tr = read_calib_tr(calib_path)
        if abs(float(np.linalg.det(Tr))) < 1e-12:
            raise ValueError(
                f"Tr in {calib_path} is singular, so it cannot be inverted. "
                f"A truncated or zero-filled calibration file is the usual "
                f"cause.")
        poses = camera_to_velodyne(poses_cam, Tr)

        stamps = self._load_times(n)
        if self.verbose:
            print(f"  [{self._LOG_NAME}] loaded {n} poses from "
                  f"{pose_path.name}, corrected camera->velodyne via "
                  f"{calib_path.name}"
                  + (f", timestamps from {self.times_file.name}" if stamps else ""))
        return list(poses), stamps

    def _load_times(self, n_poses: int) -> List[float]:
        """``times.txt`` -- seconds since the sequence started, one per scan.

        Optional: without it ``--exclusion seconds=`` is unavailable but
        everything else works, so a missing or mismatched file is a note
        rather than an error.
        """
        if not self.times_file.is_file():
            return []
        stamps = np.atleast_1d(np.loadtxt(self.times_file, dtype=np.float64))
        if stamps.size != n_poses:
            if self.verbose:
                print(f"  [{self._LOG_NAME}] {self.times_file.name} has "
                      f"{stamps.size} entries for {n_poses} poses; ignoring "
                      f"its timestamps")
            return []
        return [float(t) for t in stamps]


# ---------------------------------------------------------------------------
#  SequenceSource adapter
# ---------------------------------------------------------------------------

class KITTISource:
    """:class:`~inlier.eval.datasets.base.SequenceSource` over a KITTI sequence.

    ``n_scans``/``stride`` accumulate submaps exactly as the generic loader
    does -- KITTI's single Velodyne sweeps are sparse enough that the
    height-slice keypoints benefit from it.
    """

    name = "kitti"

    def __init__(
        self,
        dataset_path,
        sequence,
        n_scans: int = 1,
        stride: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.sequence = normalise_sequence(sequence)
        self.n_scans = int(n_scans)
        self.stride = int(stride) if stride is not None else int(n_scans)
        self.verbose = verbose
        self._handler = KITTI_Handler(self.dataset_path, self.sequence,
                                      verbose=verbose)

    def load(self, **_):
        from inlier.eval.datasets.base import Sequence

        data = self._handler.load_generic(self._handler.seq_dir,
                                          n_scans=self.n_scans,
                                          stride=self.stride)
        return Sequence.from_handler_dict(data, **self.describe())

    def describe(self):
        return {
            "dataset_type": self.name,
            "dataset_path": str(self.dataset_path),
            "sequence": self.sequence,
            # KITTI has one lidar, but the key is what the protocols and the
            # playback read to label a session; leaving it out would print
            # blanks everywhere HeLiPR prints a sensor.
            "sensor": SCAN_SUBDIR,
            "n_scans": self.n_scans,
            "stride": self.stride,
            # A results file is the only place a reader can later check that
            # the camera-frame correction was applied at all.  The resolved
            # calib *path* is deliberately not recorded: it is disk state, and
            # `from_describe(describe())` must round-trip without one.
            "pose_frame": SCAN_SUBDIR,
        }

    @classmethod
    def from_describe(cls, described, *, root=None, verbose=False):
        """Rebuild from a ``describe()`` block, so ``inlier play`` can reload."""
        return cls(root or described["dataset_path"],
                   described.get("sequence", ""),
                   described.get("n_scans", 1), described.get("stride"),
                   verbose=verbose)

    @property
    def tag(self) -> str:
        """Descriptor-cache identity.

        The ``kitti`` prefix is load-bearing, not decoration.  The cache path
        is ``desc_{tag}_{hash of encoder config}.npz`` -- the hash carries no
        dataset identity -- and the cache stores *poses*.  A tag that collided
        with the generic loader's for the same folder would silently serve up
        a cache full of uncorrected camera-frame poses, and the correction
        above would look like it had done nothing.
        """
        return f"kitti{self.sequence}_n{self.n_scans}s{self.stride}"
