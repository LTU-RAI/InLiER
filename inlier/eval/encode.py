"""Encoding a whole sequence, with a content-addressed descriptor cache.

Ported from ``encode_sequence`` (``evaluate_inlier_helipr.py`` :210), which was
duplicated in the generic driver.  Two changes, neither of which moves a
number:

* it takes a :class:`~inlier.eval.datasets.base.Sequence` rather than a HeLiPR
  handler, so one implementation serves every loader;
* the loading is lazy -- a cache hit never touches the dataset at all, which is
  what makes the golden-run regression cheap to re-run.

The cache key is unchanged: an md5 of the encoder config plus the preprocessing
voxel size.  It must stay that way, because ``tests/conftest.py`` pins the hash
``a6a8d4c7cbd5`` to identify the cached descriptors its matcher tests need --
change what goes into the key and those tests silently skip instead of failing.
Keying on the config is what makes the cache safe: re-tune the encoder and you
get a new file rather than stale descriptors.
"""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional

import numpy as np

from inlier.core.Dataclasses import InLiER_Config, InLiER_Tokens


class EncodedSequence(NamedTuple):
    """Everything downstream stages need from one encoded sequence."""

    tokens: List[InLiER_Tokens]      # per scan
    kp_aligned: List[np.ndarray]     # (K_i, 3) ground-aligned, for verification
    kp_sensor: List[np.ndarray]      # (K_i, 3) sensor frame, for GICP / pose error
    T_grounds: List[np.ndarray]      # (4, 4) sensor -> ground-aligned, per scan
    positions: np.ndarray            # (N, 3) pose translations
    poses: np.ndarray                # (N, 4, 4)

    def __len__(self) -> int:
        return len(self.tokens)


def empty_tokens() -> InLiER_Tokens:
    return InLiER_Tokens(token_id=np.zeros(0, dtype=np.uint32))


def cache_key(cfg: InLiER_Config, voxel_size: float) -> str:
    """Stable 12-hex digest of everything that changes a descriptor."""
    payload = asdict(cfg)
    payload["voxel_size"] = float(voxel_size)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def cache_path(cache_dir: Path, tag: str, cfg: InLiER_Config, voxel_size: float) -> Path:
    return Path(cache_dir) / f"desc_{tag}_{cache_key(cfg, voxel_size)}.npz"


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Grid-hash downsample, keeping the first point in each occupied voxel.

    Non-finite points are dropped first.  They are not merely useless here:
    ``np.floor(nan).astype(np.int64)`` is INT64_MIN, so every NaN point hashes
    into one bogus voxel that then wins ``np.unique``'s first-index draw
    against a real one.  A scan with NaNs therefore loses real points and
    encodes differently, silently.  Sensors mark invalid returns as NaN and
    the .pcd loaders pass them straight through.

    The HeLiPR undistorted binaries carry no non-finite points, so this
    changes nothing for the published results; it matters for the generic
    .pcd path.
    """
    if points.size == 0:
        return points
    finite = np.isfinite(points).all(axis=1)
    if not finite.all():
        points = points[finite]
    if voxel_size <= 0 or points.size == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(keep)]


def _from_cache(path: Path, verbose: bool) -> EncodedSequence:
    if verbose:
        print(f"  [cache] loading {path.name}")
    npz = np.load(path)
    positions = npz["positions"]
    offsets = npz["offsets"]
    all_tids = npz["token_ids"]
    n_scans = len(offsets) - 1

    def _slice(key: str, width: int) -> List[np.ndarray]:
        if key not in npz:
            # An old cache predating this field.  Filling with zeros would make
            # verification silently wrong, so say so loudly.
            print(f"  [cache] WARNING: {path.name} has no '{key}'; delete it and "
                  f"re-encode for correct verification / pose output")
            return [np.zeros((int(offsets[i + 1] - offsets[i]), width), dtype=np.float64)
                    for i in range(n_scans)]
        data = npz[key]
        return [data[offsets[i]:offsets[i + 1]] for i in range(n_scans)]

    tokens = [InLiER_Tokens(token_id=all_tids[offsets[i]:offsets[i + 1]].astype(np.uint32, copy=False))
              for i in range(n_scans)]
    kp_aligned = _slice("kp_aligned", 3)
    kp_sensor = _slice("kp_sensor", 3) if "kp_sensor" in npz else kp_aligned
    if "T_grounds" in npz:
        T_grounds = [npz["T_grounds"][i] for i in range(n_scans)]
    else:
        print(f"  [cache] WARNING: {path.name} has no 'T_grounds'; "
              f"delete it and re-encode")
        T_grounds = [np.eye(4, dtype=np.float64) for _ in range(n_scans)]
    if "poses" in npz:
        poses = npz["poses"]
    else:
        print(f"  [cache] WARNING: {path.name} has no 'poses'; "
              f"pose errors will be wrong")
        poses = np.tile(np.eye(4), (n_scans, 1, 1)).astype(np.float64)
        poses[:, :3, 3] = positions

    return EncodedSequence(tokens, kp_aligned, kp_sensor, T_grounds, positions, poses)


def _save_cache(path: Path, enc: EncodedSequence, verbose: bool) -> None:
    tids = [t.token_id for t in enc.tokens]
    lengths = np.array([len(t) for t in tids], dtype=np.int64)
    np.savez_compressed(
        path,
        positions=enc.positions,
        poses=enc.poses,
        offsets=np.concatenate([[0], lengths.cumsum()]),
        token_ids=np.concatenate(tids) if tids else np.zeros(0, dtype=np.uint32),
        kp_aligned=(np.concatenate(enc.kp_aligned) if enc.kp_aligned
                    else np.zeros((0, 3), dtype=np.float64)),
        kp_sensor=(np.concatenate(enc.kp_sensor) if enc.kp_sensor
                   else np.zeros((0, 3), dtype=np.float64)),
        T_grounds=np.stack(enc.T_grounds) if enc.T_grounds else np.zeros((0, 4, 4)),
    )
    if verbose:
        print(f"  [cache] saved {path.name}")


def encode_sequence(
    encoder,
    load: Callable[[], "object"],
    tag: str,
    voxel_size: float = 0.0,
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
    desc: str = "",
) -> EncodedSequence:
    """Encode a sequence, reusing a cached descriptor set when one matches.

    ``load`` is a thunk returning a
    :class:`~inlier.eval.datasets.base.Sequence`; it is only called on a cache
    miss, so a cached run never reads a scan from disk.
    """
    cfg = encoder.config
    path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_path(cache_dir, tag, cfg, voxel_size)
        if path.exists():
            return _from_cache(path, verbose)

    sequence = load()
    poses_list = sequence.poses
    clouds = sequence.point_clouds
    n = len(clouds)

    positions = np.asarray([p[:3, 3] for p in poses_list], dtype=np.float64)
    poses = np.asarray([p[:4, :4] for p in poses_list], dtype=np.float64)

    tokens: List[InLiER_Tokens] = []
    kp_aligned: List[np.ndarray] = []
    kp_sensor: List[np.ndarray] = []
    T_grounds: List[np.ndarray] = []

    iterator = range(n)
    if verbose:
        try:
            import tqdm
            iterator = tqdm.tqdm(iterator, desc=desc or f"  Encoding {tag}")
        except ImportError:
            pass

    for i in iterator:
        pts = np.asarray(clouds[i], dtype=np.float32)
        # HeLiPR pads short scans with exact zero rows; they are not measurements.
        pts = pts[np.any(pts != 0, axis=1)]
        if voxel_size > 0:
            pts = voxel_downsample(pts, voxel_size)

        if pts.shape[0] < 10:
            tokens.append(empty_tokens())
            kp_aligned.append(np.zeros((0, 3), dtype=np.float64))
            kp_sensor.append(np.zeros((0, 3), dtype=np.float64))
            T_grounds.append(np.eye(4, dtype=np.float64))
        else:
            kp, tok = encoder.encode(pts, verbose=False)
            tokens.append(tok)
            kp_aligned.append(np.asarray(kp.p_aligned, dtype=np.float64))
            kp_sensor.append(np.asarray(kp.p, dtype=np.float64))
            T_grounds.append(np.asarray(kp.T_ground, dtype=np.float64))

        # Whole sequences of dense scans are held at once; without this the
        # peak RSS on a 2700-scan sequence grows well past what the freed
        # arrays actually need.
        if (i + 1) % 50 == 0:
            gc.collect()
    gc.collect()

    encoded = EncodedSequence(tokens, kp_aligned, kp_sensor, T_grounds, positions, poses)
    if path is not None:
        _save_cache(path, encoded, verbose)
    return encoded
