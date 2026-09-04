"""Preprocessing that runs before the encoder sees a scan."""

import numpy as np
import pytest

from inlier.eval.encode import voxel_downsample


def test_voxel_downsample_keeps_one_point_per_voxel():
    points = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2],   # same 1 m voxel
                       [1.5, 0.1, 0.1],                     # next voxel over
                       [0.3, 0.3, 0.3]], dtype=np.float32)  # first voxel again
    kept = voxel_downsample(points, 1.0)
    assert kept.shape == (2, 3)
    # the first point of each voxel survives, in input order
    assert np.array_equal(kept[0], points[0])
    assert np.array_equal(kept[1], points[2])


def test_voxel_downsample_drops_non_finite_points():
    points = np.array([[0.1, 0.1, 0.1], [np.nan, 0.0, 0.0],
                       [1.5, 0.1, 0.1], [0.0, np.inf, 0.0]], dtype=np.float32)
    kept = voxel_downsample(points, 1.0)
    assert np.isfinite(kept).all()
    assert kept.shape == (2, 3)


def test_a_nan_point_does_not_displace_a_real_one():
    """The bug this guards: NaN floors to INT64_MIN and hashes into one voxel.

    `np.unique(..., return_index=True)` then hands that bogus voxel the first
    index it sees -- so a scan with NaNs silently loses real points and
    encodes differently from the same scan with the NaNs stripped.
    """
    real = np.array([[0.1, 0.1, 0.1], [5.0, 5.0, 5.0], [9.0, 0.0, 0.0]],
                    dtype=np.float32)
    with_nans = np.vstack([np.full((3, 3), np.nan, dtype=np.float32), real])

    assert np.array_equal(voxel_downsample(with_nans, 1.0),
                          voxel_downsample(real, 1.0))


def test_voxel_downsample_passes_an_empty_scan_through():
    empty = np.zeros((0, 3), dtype=np.float32)
    assert voxel_downsample(empty, 1.0).shape == (0, 3)


@pytest.mark.parametrize("voxel_size", [0.0, -1.0])
def test_a_disabled_voxel_size_still_drops_non_finite_points(voxel_size):
    points = np.array([[0.1, 0.1, 0.1], [np.nan, 0.0, 0.0]], dtype=np.float32)
    kept = voxel_downsample(points, voxel_size)
    assert kept.shape == (1, 3)


## --- the packed-key fast path ---


def _reference_voxel_downsample(points, voxel_size):
    """The straightforward implementation, kept as the thing to match.

    ``np.unique(axis=0)`` is obviously correct and eight times too slow to sit
    in front of the encoder: on a 575k-point Ouster scan it costs ~250 ms
    against the encoder's ~30 ms.  Same trade, and same guard, as
    ``playback.voxel_downsample_np`` already makes for the animation loop.
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


@pytest.mark.parametrize("voxel_size", [0.1, 0.5, 2.0])
def test_the_packed_key_path_matches_the_reference_exactly(voxel_size):
    """Not "close enough": which point survives decides the descriptor.

    Packing folds the three voxel coordinates into one integer so the unique
    becomes a 1-D sort.  It is injective while the grid fits in an int64, so
    the surviving indices must be the *same* indices, not merely as many.
    """
    rng = np.random.default_rng(0)
    cases = {
        "dense": rng.uniform(-100, 100, (20000, 3)).astype(np.float32),
        # Negative coordinates: the packing shifts by the minimum, and getting
        # that wrong shows up here and nowhere else.
        "all negative": rng.uniform(-500, -400, (5000, 3)).astype(np.float32),
        # Heavy collisions, so first-occurrence order actually matters.
        "clustered": np.repeat(rng.uniform(-5, 5, (200, 3)), 50,
                               axis=0).astype(np.float32),
        "with nan": np.vstack([rng.uniform(-10, 10, (1000, 3)),
                               np.full((50, 3), np.nan)]).astype(np.float32),
        "single point": np.ones((1, 3), np.float32),
    }
    for name, points in cases.items():
        got = voxel_downsample(points.copy(), voxel_size)
        want = _reference_voxel_downsample(points.copy(), voxel_size)
        assert np.array_equal(got, want), f"{name} at voxel_size={voxel_size}"


def test_a_grid_too_big_to_pack_falls_back_instead_of_colliding():
    """Two voxels folding onto one integer would silently drop real points."""
    rng = np.random.default_rng(1)
    points = (rng.uniform(-1.0, 1.0, (1000, 3)) * 1e17).astype(np.float64)

    got = voxel_downsample(points.copy(), 0.1)
    want = _reference_voxel_downsample(points.copy(), 0.1)
    assert np.array_equal(got, want)
