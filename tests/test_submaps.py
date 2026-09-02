"""The submap window rule.

This is the one piece of arithmetic that the overlap ground truth and the
dataset loaders must agree on: the matrix is indexed by submap, so if the two
ever disagreed about how many submaps a sequence has, retrieval would be
scored against the wrong rows rather than fail.
"""

import numpy as np
import pytest

from inlier.eval.submaps import submap_count, submap_windows


def test_non_overlapping_is_the_default():
    assert submap_windows(6, 3) == [range(0, 3), range(3, 6)]


def test_the_trailing_window_is_kept_short_not_dropped():
    """Dropping it would change the submap count, and so the matrix shape."""
    windows = submap_windows(7, 3)
    assert windows == [range(0, 3), range(3, 6), range(6, 7)]
    assert len(windows[-1]) == 1


def test_a_stride_below_n_scans_overlaps():
    assert submap_windows(7, 3, stride=2) == [
        range(0, 3), range(2, 5), range(4, 7), range(6, 7)]


def test_a_stride_above_n_scans_leaves_gaps():
    windows = submap_windows(10, 2, stride=5)
    assert windows == [range(0, 2), range(5, 7)]
    covered = {i for w in windows for i in w}
    assert 2 not in covered and 3 not in covered


def test_single_scans_are_one_window_each():
    assert submap_windows(4, 1) == [range(0, 1), range(1, 2),
                                    range(2, 3), range(3, 4)]


def test_every_scan_is_covered_exactly_once_when_stride_equals_n():
    covered = [i for w in submap_windows(23, 4) for i in w]
    assert covered == list(range(23))


def test_the_first_scan_of_each_window_is_the_keyframe():
    """Callers rely on w[0]: its pose positions the submap."""
    assert [w[0] for w in submap_windows(10, 3)] == [0, 3, 6, 9]


def test_an_empty_sequence_produces_no_submaps():
    assert submap_windows(0, 5) == []
    assert submap_count(0, 5) == 0


@pytest.mark.parametrize("n_scans, stride", [(0, None), (-1, None), (3, 0), (3, -2)])
def test_degenerate_parameters_raise(n_scans, stride):
    with pytest.raises(ValueError):
        submap_windows(10, n_scans, stride)


def test_submap_count_agrees_with_the_windows():
    for count in (0, 1, 7, 23, 4041):
        for n in (1, 3, 10):
            for stride in (None, 1, 5):
                assert submap_count(count, n, stride) == \
                    len(submap_windows(count, n, stride))


def test_matches_the_overlap_builder_it_was_extracted_from():
    """The two used to be separate implementations; they must stay identical.

    `_group_into_submaps` now calls `submap_windows`, so this pins the
    contract rather than comparing two bodies of code -- if someone reverts
    it to a local loop, this catches the divergence.
    """
    from inlier.eval.overlap_build import _group_into_submaps

    for count in (1, 5, 7, 23):
        positions = np.arange(count * 3, dtype=np.float64).reshape(count, 3)
        poses = [np.eye(4) * i for i in range(count)]
        files = [f"{i:06d}.bin" for i in range(count)]
        for n in (1, 2, 3, 5):
            for stride in (None, 1, 2, 4):
                kf, _, submap_files = _group_into_submaps(
                    positions, poses, files, n, stride)
                windows = submap_windows(count, n, stride)
                assert submap_files == [[files[i] for i in w] for w in windows]
                assert np.array_equal(
                    kf, np.asarray([positions[w[0]] for w in windows]))
