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


# --- the evaluation's trajectory plot ---------------------------------------
# Two sessions stacked in z, one edge per decision at the operating threshold.
# The edge lists come from metrics.confusion, so the picture cannot disagree
# with the counts in its own title.

def _ring(radius, n=64, z=0.0):
    t = np.linspace(0.0, 2 * np.pi, n)
    return np.stack([radius * np.cos(t), radius * np.sin(t),
                     np.full(n, z)], axis=1)


def test_trajectory_plot_writes_a_png(tmp_path):
    from inlier.viz import write_trajectory_plot

    db, q = _ring(50.0), _ring(52.0)
    out = write_trajectory_plot(
        tmp_path / "nested" / "trajectory_run.png", db, q,
        tp_edges=[(0, 0), (5, 5)], fp_edges=[(9, 40)],
        title="InLiER Verify  A → B\nthr=0.300  TP=2  FP=1  FN=0  TN=0")

    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_trajectory_plot_handles_a_run_with_no_edges(tmp_path):
    """Nothing above threshold is a legitimate -- and informative -- run."""
    from inlier.viz import write_trajectory_plot

    out = write_trajectory_plot(tmp_path / "t.png", _ring(10.0), _ring(11.0),
                                tp_edges=[], fp_edges=[], title="empty")
    assert out.exists()


# --- the online protocol's single-session plot ------------------------------
# One trajectory, z is the frame index, so a closure edge spans the frames
# between leaving a place and returning to it.

def test_time_trajectory_plot_writes_a_png(tmp_path):
    from inlier.viz import write_time_trajectory_plot

    out = write_time_trajectory_plot(
        tmp_path / "nested" / "trajectory_lcd.png", _ring(50.0, n=64),
        tp_edges=[(60, 2), (61, 3)], fp_edges=[(55, 30)],
        title="InLiER online-LCD Verify  Roundabout01\n"
              "thr=0.300  TP=2  FP=1  FN=0  TN=0")

    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_time_trajectory_plot_handles_a_run_with_no_edges(tmp_path):
    from inlier.viz import write_time_trajectory_plot

    out = write_time_trajectory_plot(tmp_path / "t.png", _ring(10.0),
                                     tp_edges=[], fp_edges=[], title="empty")
    assert out.exists()


def test_time_trajectory_edges_span_their_own_frames(tmp_path):
    """z must come from the frame index, not a constant.

    A closure between frame 60 and frame 2 has to be drawn as a line rising
    58 frames; if z were fixed per trajectory the edge would be flat and the
    figure would say nothing about when the revisit happened.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from inlier.viz.trajectory import _draw_edges, _xyz

    positions = _ring(50.0, n=64)
    xyz = _xyz(positions, np.arange(len(positions), dtype=float))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    _draw_edges(ax, xyz, xyz, tp_edges=[(60, 2)], fp_edges=[])
    zs = [line.get_data_3d()[2] for line in ax.lines]
    plt.close(fig)

    assert len(zs) == 1
    assert list(zs[0]) == [60.0, 2.0]


def test_time_trajectory_accepts_real_timestamps(tmp_path):
    from inlier.viz import write_time_trajectory_plot

    positions = _ring(50.0, n=64)
    stamps = np.linspace(1000.0, 1063.0, len(positions))
    out = write_time_trajectory_plot(
        tmp_path / "ts.png", positions, tp_edges=[(60, 2)], fp_edges=[],
        title="stamped", times=stamps, z_label="Time (s)")
    assert out.exists()


def test_session_label_names_both_loaders():
    from inlier.eval.protocols.cross_session import _session_label

    class _Source:
        def __init__(self, described):
            self._described = described

        def describe(self):
            return self._described

    helipr = _Source({"sequence": "Roundabout01", "sensor": "Ouster"})
    generic = _Source({"path": "/data/campus_ouster"})
    assert _session_label(helipr) == "Roundabout01/Ouster"
    assert _session_label(generic) == "campus_ouster"


def test_a_missing_matplotlib_does_not_lose_the_run(tmp_path, monkeypatch, capsys):
    """The figure is written last, after a job that can take tens of minutes."""
    import sys

    from inlier.eval.protocols.cross_session import (
        CrossSessionSpec, _write_trajectory_plot,
    )

    monkeypatch.setitem(sys.modules, "inlier.viz", None)
    spec = CrossSessionSpec(resolved=None, db_source=None, q_source=None,
                            overlap_path=tmp_path, output_dir=tmp_path)
    written = _write_trajectory_plot(
        spec, tmp_path / "t.png", _ring(1.0), _ring(2.0), [], [],
        "Verify", 0.3, 1, 0, 0, 0)

    assert written is None
    assert not (tmp_path / "t.png").exists()
    assert "Trajectory plot skipped" in capsys.readouterr().out


# --- neutral edges, for a caller with no correctness to assert --------------


def _legend_labels(monkeypatch, tmp_path, **kw):
    """Render a time-trajectory plot and return its legend entries."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from inlier.viz import write_time_trajectory_plot

    captured = {}
    real_close = plt.close

    def spy(fig):
        ax = fig.axes[0]
        captured["labels"] = [t.get_text() for t in ax.get_legend().get_texts()]
        real_close(fig)

    monkeypatch.setattr(plt, "close", spy)
    positions = np.stack([np.arange(6.0), np.zeros(6), np.zeros(6)], axis=1)
    write_time_trajectory_plot(tmp_path / "t.png", positions,
                               [(4, 0), (5, 1)], [(3, 0)], "t", **kw)
    return captured["labels"]


def test_edges_are_labelled_tp_and_fp_by_default(monkeypatch, tmp_path):
    """The evaluation figures must not move."""
    labels = _legend_labels(monkeypatch, tmp_path)
    assert "TP (2)" in labels and "FP (1)" in labels


def test_an_empty_label_suppresses_its_legend_entry(monkeypatch, tmp_path):
    """`inlier run` has one class of edge and no "FP (0)" to explain."""
    labels = _legend_labels(monkeypatch, tmp_path,
                            edge_labels=("Closure", ""),
                            edge_colors=("#7f7f7f", "#7f7f7f"))
    assert "Closure (2)" in labels
    assert not any(label.startswith("FP") for label in labels)


# --- the per-stage score matrices ------------------------------------------


def _matrices(n=6):
    """MINT scores the causal past; the later stages score less of it."""
    mint = np.full((n, n), np.nan, dtype=np.float32)
    for q in range(1, n):
        mint[q, :q] = np.linspace(0.2, 0.8, q)
    beam = np.full((n, n), np.nan, dtype=np.float32)
    beam[3, 0] = 0.4
    beam[4, 1] = 0.1
    verify = np.full((n, n), np.nan, dtype=np.float32)
    verify[3, 0] = 0.7
    verify[4, 1] = 0.0                      # a real score, not "unscored"
    return {"mint": mint, "beam": beam, "verify": verify,
            "beam_shift": np.full((n, n), 3.0, dtype=np.float32)}


def test_score_matrix_figure_writes_a_png(tmp_path):
    from inlier.viz import write_score_matrix_figure

    out = write_score_matrix_figure(tmp_path / "s.png", _matrices(), "t")
    assert out.exists() and out.stat().st_size > 0


def test_the_shift_matrix_is_not_drawn_as_a_score(tmp_path, monkeypatch):
    """`beam_shift` is an azimuth index; a score ramp would misread it."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from inlier.viz import write_score_matrix_figure

    seen = {}
    real_close = plt.close

    def spy(fig):
        seen["n_panels"] = len([a for a in fig.axes if a.get_xlabel()])
        real_close(fig)

    monkeypatch.setattr(plt, "close", spy)
    write_score_matrix_figure(tmp_path / "s.png", _matrices(), "t")
    assert seen["n_panels"] == 3            # mint, beam, verify -- not shift


def test_nothing_to_draw_returns_none(tmp_path):
    """A run whose stages were all skipped needs no special-casing."""
    from inlier.viz import write_score_matrix_figure

    empty = {"mint": np.full((4, 4), np.nan, dtype=np.float32)}
    assert write_score_matrix_figure(tmp_path / "s.png", empty, "t") is None


def test_unscored_cells_get_their_own_colour_not_the_low_end():
    """The rule the arrays keep, the picture has to keep too.

    0.0 is a real score -- a failed verification returns exactly that -- so
    painting "never scored" with the bottom of the ramp would make the two
    indistinguishable in the one artifact people actually look at.
    """
    from matplotlib import colormaps

    from inlier.viz.scores import UNSCORED_COLOR

    cmap = colormaps["viridis"].with_extremes(bad=UNSCORED_COLOR)
    assert tuple(cmap.get_bad()) != tuple(cmap(0.0))


def test_a_huge_matrix_is_block_reduced_by_max(tmp_path):
    """Shrinking a sparse matrix must not throw the sparse signal away.

    Verify scores ~`topv` pairs out of hundreds per row, so stride-sampling
    would keep a quarter of the cells and lose three quarters of the scored
    pairs at random -- the picture would show a thinner funnel than the run
    produced. A block maximum keeps any scored pair in the block. And max
    rather than mean, because a mean would blend NaN into a number.
    """
    from inlier.viz.scores import _block_max

    m = np.full((100, 100), np.nan, dtype=np.float32)
    for q in range(100):
        m[q, q // 2] = 0.5 + q / 400          # one scored cell per row
    image, step = _block_max(m, max_cells=900)

    assert step > 1
    assert np.nanmax(image) == pytest.approx(np.nanmax(m))     # peak survives
    # Every block that held a scored cell still holds one; stride-sampling
    # keeps strictly fewer.
    assert np.isfinite(image).sum() > np.isfinite(m[::step, ::step]).sum()


def test_block_reduction_leaves_empty_blocks_unscored():
    """An all-NaN block is still NaN -- not 0.0, and not a warning."""
    import warnings

    from inlier.viz.scores import _block_max

    m = np.full((40, 40), np.nan, dtype=np.float32)
    m[0, 0] = 0.25
    with warnings.catch_warnings():
        warnings.simplefilter("error")        # all-NaN nanmax must not warn out
        image, step = _block_max(m, max_cells=100)
    assert image[0, 0] == pytest.approx(0.25)
    assert np.isnan(image[-1, -1])


def test_a_normal_session_is_never_reduced():
    """The threshold is high on purpose: a 4000-frame run draws every pair."""
    from inlier.viz.scores import MAX_CELLS, _block_max

    m = np.zeros((4000, 4000), dtype=np.float32)
    assert m.size <= MAX_CELLS
    _, step = _block_max(m)
    assert step == 1
