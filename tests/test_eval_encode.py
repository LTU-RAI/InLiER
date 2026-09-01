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
