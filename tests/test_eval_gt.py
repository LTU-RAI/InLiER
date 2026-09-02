"""Ground-truth policies, exclusion windows, and candidate filters."""

import numpy as np
import pytest

from inlier.eval import gt


def _ref_build_ground_truth(overlap_matrix, db_positions, q_positions,
                            overlap_threshold, max_pose_dist):
    """Frozen copy of evaluate_inlier_helipr.py :159."""
    _, n_q = overlap_matrix.shape
    out = {}
    for j in range(n_q):
        mask = overlap_matrix[:, j] >= overlap_threshold
        if max_pose_dist > 0.0:
            d = np.linalg.norm(db_positions[:, :2] - q_positions[j, :2], axis=1)
            mask = mask & (d <= max_pose_dist)
        out[j] = np.where(mask)[0]
    return out


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    return (rng.random((200, 150)), rng.random((200, 3)) * 100, rng.random((150, 3)) * 100)


@pytest.mark.parametrize("tau", [0.0, 0.2, 0.5, 0.9])
@pytest.mark.parametrize("max_pose_dist", [0.0, 10.0, 25.0, 1e9])
def test_overlap_and_distance_matches_reference(data, tau, max_pose_dist):
    overlap, dbp, qp = data
    ref = _ref_build_ground_truth(overlap, dbp, qp, tau, max_pose_dist)
    got = gt.build(gt.OverlapAndDistance(overlap, dbp, qp, tau, max_pose_dist), qp.shape[0])
    for j in range(qp.shape[0]):
        assert np.array_equal(ref[j], got[j])


def test_max_pose_dist_zero_disables_the_distance_criterion(data):
    overlap, dbp, qp = data
    got = gt.OverlapAndDistance(overlap, dbp, qp, 0.5, 0.0).positives(0)
    assert np.array_equal(got, np.where(overlap[:, 0] >= 0.5)[0])


def test_distance_only_ignores_overlap(data):
    _, dbp, qp = data
    got = gt.DistanceOnly(dbp, qp, 30.0).positives(7)
    d = np.linalg.norm(dbp[:, :2] - qp[7, :2], axis=1)
    assert np.array_equal(got, np.where(d <= 30.0)[0])


# ---------------------------------------------------------------------------
#  Exclusion
# ---------------------------------------------------------------------------

TS = np.arange(100, dtype=float) * 0.1     # 10 Hz
ARC = np.arange(100, dtype=float) * 2.0    # 2 m per frame


@pytest.mark.parametrize("t", [0, 1, 5, 20, 50, 99])
def test_exclusion_units_agree_on_the_same_window(t):
    """20 frames at 10 Hz and 2 m/frame is 2.0 s and 40 m -- same cutoff."""
    by_frames = gt.Exclusion(frames=20).cutoff(t)
    by_seconds = gt.Exclusion(seconds=2.0).cutoff(t, timestamps=TS)
    by_metres = gt.Exclusion(metres=40.0).cutoff(t, arc_length=ARC)
    assert by_frames == by_seconds == by_metres


def test_exclusion_cutoff_is_clamped_at_the_start():
    assert gt.Exclusion(frames=20).cutoff(5) == 0
    assert gt.Exclusion(frames=0).cutoff(0) == 0


def test_exclusion_zero_still_excludes_self():
    """cutoff(t) is exclusive, so frames=0 leaves [0, t) -- never t itself."""
    assert gt.Exclusion(frames=0).cutoff(10) == 10


@pytest.mark.parametrize("kwargs", [{}, {"frames": 1, "metres": 2.0},
                                    {"frames": 1, "seconds": 1.0, "metres": 2.0}])
def test_exclusion_requires_exactly_one_unit(kwargs):
    with pytest.raises(ValueError, match="exactly one"):
        gt.Exclusion(**kwargs)


def test_exclusion_rejects_negative_windows():
    with pytest.raises(ValueError, match=">= 0"):
        gt.Exclusion(frames=-1)


def test_exclusion_needs_the_data_its_unit_depends_on():
    with pytest.raises(ValueError, match="needs timestamps"):
        gt.Exclusion(seconds=1.0).cutoff(5)
    with pytest.raises(ValueError, match="needs arc_length"):
        gt.Exclusion(metres=1.0).cutoff(5)


# ---------------------------------------------------------------------------
#  Causal ground truth and filters
# ---------------------------------------------------------------------------

def test_causal_positives_are_past_and_near():
    rng = np.random.default_rng(11)
    pos = rng.random((100, 3)) * 50
    policy = gt.Causal(pos, gt.Exclusion(frames=10), max_pose_dist=15.0)
    for t in range(100):
        upper = max(0, t - 10)
        if upper:
            past = np.arange(upper)
            expected = past[np.linalg.norm(pos[past, :2] - pos[t, :2], axis=1) <= 15.0]
        else:
            expected = np.zeros(0, dtype=np.int64)
        assert np.array_equal(policy.positives(t), expected)


def test_causal_yields_nothing_inside_the_warmup():
    pos = np.zeros((100, 3))
    policy = gt.Causal(pos, gt.Exclusion(frames=30), max_pose_dist=1e9)
    assert policy.positives(10).size == 0
    assert policy.positives(40).size == 10


def test_no_filter_allows_everything():
    f = gt.NoFilter()
    assert f.bound(5) is None and f.allows(5, 999)


def test_causal_filter_bound_matches_exclusion():
    f = gt.CausalFilter(gt.Exclusion(frames=10))
    assert f.bound(50) == 40
    assert f.allows(50, 39) and not f.allows(50, 40)


def test_radius_filter_reports_that_it_uses_the_pose_oracle():
    pos = np.zeros((10, 3))
    described = gt.RadiusFilter(pos, 50.0).describe()
    assert described["uses_pose_oracle"] is True
    assert described["radius_m"] == 50.0


def test_radius_filter_mask_combines_radius_and_causality():
    pos = np.stack([np.arange(20, dtype=float), np.zeros(20), np.zeros(20)], axis=1)
    f = gt.RadiusFilter(pos, radius=5.0, exclusion=gt.Exclusion(frames=3))
    mask = f.allowed_mask(15, 20)
    # causal bound is 12; radius 5 around x=15 reaches back to x=10
    assert np.array_equal(np.flatnonzero(mask), np.arange(10, 12))
