"""Descriptor recovery and the encode figure.

The descriptor tests matter more than they look: the figure is only useful
if the matrices it draws are the ones the matcher scores on, and both the
packing and the histogram builders are easy to change without noticing that
the visualisation has drifted.
"""

import numpy as np
import pytest

from inlier.core.Dataclasses import InLiER_Config, InLiER_Keypoints, InLiER_Tokens
from inlier.core.reference.InLiER import InLiER as Encoder
from inlier.viz.descriptors import (
    describe,
    occupancy,
    popcount,
    shape_class_labels,
)

CFG = InLiER_Config()


def _tokens(hb, rb, sb, ab, cfg=CFG):
    return Encoder.pack_token_ids(
        np.asarray(hb), np.asarray(rb), np.asarray(sb), np.asarray(ab),
        cfg.N_r, cfg.N_s, cfg.N_a, np.uint32)


def test_popcount_matches_bin_across_widths():
    values = np.array([0, 1, 2, 3, 255, 2 ** 20, 2 ** 40, 2 ** 63], dtype=np.uint64)
    expected = [bin(int(v)).count("1") for v in values]
    assert popcount(values).tolist() == expected


def test_describe_recovers_the_bins_it_was_given():
    hb = [0, 3, 3, 9]
    rb = [1, 1, 4, 19]
    sb = [0, 6, 2, 3]
    ab = [0, 30, 30, 59]
    desc = describe(_tokens(hb, rb, sb, ab), CFG.N_h, CFG.N_r, CFG.N_s, CFG.N_a)

    assert desc.hb.tolist() == hb
    assert desc.rb.tolist() == rb
    assert desc.sb.tolist() == sb
    assert desc.ab.tolist() == ab
    assert desc.max_hb == 9


def test_full_histogram_counts_tokens_per_height_and_radial_shape_cell():
    """`full` is (N_h, N_r*N_s) with azimuth summed away."""
    # Two tokens in the same (hb, rb, sb) cell but different azimuths must
    # land in the same column: that collapse is what stage 1 scores on.
    desc = describe(_tokens([2, 2], [3, 3], [1, 1], [0, 45]),
                    CFG.N_h, CFG.N_r, CFG.N_s, CFG.N_a)

    assert desc.full.shape == (CFG.N_h, CFG.N_r * CFG.N_s)
    assert desc.full.sum() == 2
    assert desc.full[2, 3 * CFG.N_s + 1] == 2


def test_compact_row_is_the_full_histogram_summed_over_height():
    desc = describe(_tokens([0, 5, 9], [2, 2, 7], [0, 3, 6], [1, 2, 3]),
                    CFG.N_h, CFG.N_r, CFG.N_s, CFG.N_a)
    assert np.array_equal(desc.compact, desc.full.sum(axis=0))
    assert desc.compact.shape == (CFG.N_r * CFG.N_s,)


def test_beam_sets_one_bit_per_occupied_height_slice():
    # Same (rb, ab) cell, three different height slices -> three bits.
    desc = describe(_tokens([0, 1, 4], [6, 6, 6], [0, 1, 2], [12, 12, 12]),
                    CFG.N_h, CFG.N_r, CFG.N_s, CFG.N_a)

    assert desc.beam.shape == (CFG.N_r, CFG.N_a)
    assert int(desc.beam[6, 12]) == 0b10011
    assert desc.beam_popcount[6, 12] == 3
    assert desc.beam_popcount.sum() == 3


def test_empty_scan_describes_as_empty():
    desc = describe(np.array([], dtype=np.uint32),
                    CFG.N_h, CFG.N_r, CFG.N_s, CFG.N_a)
    assert desc.max_hb == -1
    assert desc.full.sum() == 0
    assert desc.beam_popcount.sum() == 0
    assert occupancy(desc, CFG.N_h, CFG.N_r, CFG.N_s, CFG.N_a) == 0.0


def test_occupancy_counts_distinct_tokens_not_keypoints():
    duplicated = _tokens([1, 1, 2], [0, 0, 0], [0, 0, 0], [5, 5, 5])
    desc = describe(duplicated, CFG.N_h, CFG.N_r, CFG.N_s, CFG.N_a)
    cells = CFG.N_h * CFG.N_r * CFG.N_s * CFG.N_a
    assert occupancy(desc, CFG.N_h, CFG.N_r, CFG.N_s, CFG.N_a) == 2 / cells


@pytest.mark.parametrize("n_s, expected", [
    (3, ["linear", "planar", "scatter"]),
    (5, ["lin 0-45", "lin 45-90", "pln 0-45", "pln 45-90", "scatter"]),
    (7, ["lin 0-30", "lin 30-60", "lin 60-90",
         "pln 0-30", "pln 30-60", "pln 60-90", "scatter"]),
])
def test_shape_class_labels_track_the_inclination_subdivision(n_s, expected):
    """Labels must follow `_compute_shape_pca`: scatter is always last."""
    assert shape_class_labels(n_s) == expected
    assert len(shape_class_labels(n_s)) == n_s


@pytest.fixture
def encoded():
    rng = np.random.default_rng(0)
    points = rng.uniform(-40, 40, size=(4000, 3))
    points[:, 2] = rng.uniform(0.0, 15.0, size=4000)
    keypoints = InLiER_Keypoints(p=points[:60], T_ground=np.eye(4))
    tokens = InLiER_Tokens(token_id=_tokens(
        rng.integers(0, CFG.N_h, 60), rng.integers(0, CFG.N_r, 60),
        rng.integers(0, CFG.N_s, 60), rng.integers(0, CFG.N_a, 60)))
    return points, keypoints, tokens


def test_encode_figure_draws_every_panel(encoded):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from inlier.viz import encode_figure

    figure = encode_figure(*encoded, CFG, title="scan.npy")
    try:
        titles = [ax.get_title() for ax in figure.axes if ax.get_title()]
        assert any("keypoints" in t for t in titles)
        assert any("height slicing" in t for t in titles)
        assert any("MINT" in t for t in titles)
        assert any("BEAM" in t for t in titles)
        assert "scan.npy" in figure._suptitle.get_text()
    finally:
        plt.close(figure)


def test_encode_figure_survives_a_token_keypoint_mismatch(encoded):
    """`point_mode="all_points"` breaks the 1:1 alignment; say so, don't crash."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from inlier.viz import encode_figure

    points, keypoints, tokens = encoded
    fewer = InLiER_Tokens(token_id=tokens.token_id[:10])
    figure = encode_figure(points, keypoints, fewer, CFG)
    try:
        texts = [t.get_text() for ax in figure.axes for t in ax.texts]
        assert any("not index-aligned" in t for t in texts)
    finally:
        plt.close(figure)
