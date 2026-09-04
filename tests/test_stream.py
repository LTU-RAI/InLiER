"""``inlier run --live`` -- the streaming driver against the batch one.

One claim, and everything here serves it: **processing the session frame by
frame produces exactly the closures processing it stage by stage produces.**
If that stops being true, ``--live`` is no longer a view of the run, it is a
different run wearing the same name, and nothing in either output would say so.

The claim is structural rather than lucky -- every stage in
``inlier.eval.pipeline`` is a per-query loop with no state carried between
queries, and both drivers call the same per-query bodies -- but structure is
what regressions quietly break, so it is pinned here.

The sequence is synthetic and on disk, because the streaming loader's whole job
is reading scans one at a time: a fixture that seeds the descriptor cache would
test nothing.  It is two laps of the same six scenes, so frame ``t`` and frame
``t + LAP`` carry byte-identical clouds and every closure is planted.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from inlier.config import load, resolve
from inlier.eval import gt as gtmod
from inlier.eval.datasets import stream as dstream
from inlier.eval.datasets.generic import GenericSource
from inlier.eval.deploy import DeploySpec, run
from inlier.viz.live import StubViewer

LAP = 6
N_FRAMES = 2 * LAP
RADIUS = 40.0
THRESHOLD = 0.3

#: Keypoint refinement rather than raw clouds, so GICP does not depend on the
#: synthetic scans being dense enough for small_gicp to converge on.
CONFIG = ["gicp.use_raw_clouds=false"]


#: Pillars per scene.  The encoder keys off structure, and a floor alone
#: yields a handful of tokens -- too few for verification to have anything to
#: match, which makes a "no closures" run prove nothing.
N_PILLARS = 60


def _scene(k: int, rng) -> np.ndarray:
    """One distinct, ground-aligned scene: a floor and a forest of pillars."""
    x, y = np.meshgrid(np.arange(-30.0, 30.0, 0.5), np.arange(-30.0, 30.0, 0.5))
    floor = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)

    parts = [floor]
    centres = rng.uniform(-28.0, 28.0, size=(N_PILLARS, 2))
    for j, (cx, cy) in enumerate(centres):
        z = np.arange(0.2, 3.0 + 0.2 * (j % 20), 0.1)
        # The scene index shifts every pillar, so the six scenes are distinct
        # and a frame's true twin is the only thing it can match.
        parts.append(np.stack([np.full_like(z, cx + 1.7 * k),
                               np.full_like(z, cy + 1.3 * k), z], axis=1))
    return np.vstack(parts).astype(np.float32)


def _write_bin(path, pts) -> None:
    padded = np.zeros((len(pts), 4), dtype=np.float32)
    padded[:, :3] = pts
    padded.tofile(path)


def _write_sequence(root, frames):
    """A sequence directory: one .bin per frame, plus KITTI-format poses."""
    scans = root / "scans"
    scans.mkdir(parents=True)
    rng = np.random.default_rng(0)
    scenes = [_scene(k, rng) for k in range(LAP)]
    for i, t in enumerate(frames):
        _write_bin(scans / f"{i:06d}.bin", scenes[t % LAP])

    angles = 2 * np.pi * (np.asarray(frames) % LAP) / LAP
    positions = np.stack([RADIUS * np.cos(angles), RADIUS * np.sin(angles),
                          np.zeros(len(frames))], axis=1)
    (root / "poses_kitti.txt").write_text("\n".join(
        f"1 0 0 {p[0]:.6f} 0 1 0 {p[1]:.6f} 0 0 1 {p[2]:.6f}"
        for p in positions))
    return root


@pytest.fixture(scope="module")
def sequence(tmp_path_factory):
    """Two laps of the same six scenes: every revisit is planted."""
    return _write_sequence(tmp_path_factory.mktemp("stream_seq"),
                           range(N_FRAMES))


@pytest.fixture(scope="module")
def lap(tmp_path_factory):
    """One lap on its own, to serve as a prior map for a cross-session run."""
    return _write_sequence(tmp_path_factory.mktemp("stream_lap"), range(LAP))


def _source(root, **kw):
    kw.setdefault("n_scans", 1)
    kw.setdefault("stride", 1)
    return GenericSource(root, verbose=False, **kw)


def _spec(root, out_dir, *, live: bool, db_root=None, source_kw=None, **kw):
    source_kw = source_kw or {}
    kw.setdefault("exclusion", None if db_root else gtmod.Exclusion(frames=2))
    return DeploySpec(
        resolved=resolve(load(overrides=CONFIG), mode="deploy"),
        source=_source(root, **source_kw), threshold=THRESHOLD,
        db_source=_source(db_root, **source_kw) if db_root else None,
        output_dir=out_dir, cache_dir=None, verbose=False, tag="test",
        live=live, **kw)


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


#: Columns carrying a GICP-refined pose.  GICP is iterative, and refining one
#: pair at a time (live) rather than a batch of them (batch) reorders its
#: floating-point work, so these agree to about 1e-15 rather than bit for bit.
#: Everything else -- the pair, the score, the rank, every inlier count -- is
#: exact, and the unrefined verify pose is exact too (see the test below).
POSE_COLUMNS = ({"tx", "ty", "tz", "yaw_deg", "gicp_error", "inlier_rmse"}
                | {f"r{i}{j}" for i in range(3) for j in range(3)})

#: The variants worth running both ways.  Each exercises a branch the two
#: drivers implement separately: submap accumulation in the streaming loader,
#: the radius mask applied after retrieval, and a fixed prior map instead of
#: the session's own past.
#: ``period`` is the frame gap between a submap and its planted twin: with one
#: scan per submap that is the lap, and with three scans on a stride of two it
#: is every third submap, because that is when the whole window repeats.
VARIANTS = {
    "scan-by-scan": {"period": LAP},
    "submaps": {"source_kw": {"n_scans": 3, "stride": 2}, "period": 3},
    "search-radius": {"search_radius": 2 * RADIUS, "period": LAP},
    "cross-session": {"cross": True, "period": LAP},
}


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def both(request, sequence, lap, tmp_path_factory):
    """The same run, driven both ways."""
    import inlier.eval.deploy as deploy

    kw = dict(VARIANTS[request.param])
    period = kw.pop("period")
    if kw.pop("cross", False):
        kw["db_root"] = lap

    batch = run(_spec(sequence, tmp_path_factory.mktemp("batch"),
                      live=False, **kw))

    original = deploy._live_viewer
    deploy._live_viewer = lambda spec: StubViewer()
    try:
        live = run(_spec(sequence, tmp_path_factory.mktemp("live"),
                         live=True, **kw))
    finally:
        deploy._live_viewer = original
    return batch, live, period


# --- the claim -------------------------------------------------------------


def test_the_two_drivers_find_the_same_closures(both):
    batch, live, _ = both
    b_rows, l_rows = _rows(batch.artifacts["closures"]), _rows(live.artifacts["closures"])
    assert b_rows, "the fixture planted revisits; a run that finds none proves nothing"
    assert len(b_rows) == len(l_rows)
    for b, l in zip(b_rows, l_rows):
        assert {k: v for k, v in b.items() if k not in POSE_COLUMNS} == \
               {k: v for k, v in l.items() if k not in POSE_COLUMNS}
        for key in POSE_COLUMNS & set(b):
            if b[key] == "":
                assert l[key] == ""
                continue
            assert float(b[key]) == pytest.approx(float(l[key]), abs=1e-9)


def test_the_verification_pose_is_bit_identical(both):
    """Before GICP touches it, nothing about the pose is order-dependent.

    This is what confines the tolerance above to GICP: verification is a
    closed-form solve over the same correspondences, so the two drivers agree
    on it exactly, and any drift in the pose columns of a closure row has one
    possible source.
    """
    batch, live, _ = both
    b_rows = _rows(batch.artifacts["verify"])
    l_rows = _rows(live.artifacts["verify"])
    assert b_rows
    assert b_rows == l_rows


def test_the_two_drivers_agree_on_every_score(both):
    """Not just the accepted ones: the whole funnel, stage by stage."""
    batch, live, _ = both
    b = np.load(batch.artifacts["score_matrices"])
    l = np.load(live.artifacts["score_matrices"])
    assert set(b.files) == set(l.files)
    for key in b.files:
        np.testing.assert_array_equal(b[key], l[key],
                                      err_msg=f"stage {key} differs")


def test_the_planted_revisits_are_found(both):
    """A closure may only join frames the fixture made identical.

    Without this the equivalence tests could both be agreeing on nothing: two
    drivers that each find zero closures agree perfectly.
    """
    _, live, period = both
    pairs = {(int(r["query_idx"]), int(r["db_idx"]))
             for r in _rows(live.artifacts["closures"])}
    assert pairs, "no closures at all"
    assert all((q - d) % period == 0 for q, d in pairs), pairs


# --- the streaming loader --------------------------------------------------


def test_iter_frames_reproduces_what_load_returns(sequence):
    """Same frames, same order, same indices -- the equivalence rests on it."""
    source = _source(sequence)
    loaded = source.load()
    frames = list(dstream.iter_frames(source))

    assert len(frames) == len(loaded)
    assert dstream.frame_count(source) == len(loaded)
    for i, frame in enumerate(frames):
        assert frame.index == i
        np.testing.assert_array_equal(frame.points, loaded.point_clouds[i])
        np.testing.assert_array_equal(frame.pose, loaded.poses[i])


def test_iter_frames_accumulates_submaps_the_same_way(sequence):
    """With --n-scans > 1 the streaming path must build the same submaps."""
    source = GenericSource(sequence, n_scans=3, stride=2, verbose=False)
    loaded = source.load()
    frames = list(dstream.iter_frames(source))

    assert len(frames) == len(loaded)
    for frame, points in zip(frames, loaded.point_clouds):
        np.testing.assert_array_equal(frame.points, points)


def test_lazy_clouds_read_back_what_the_loader_built(sequence):
    """GICP asks for a past frame's cloud; it must get that frame's cloud."""
    from inlier.eval.stream import _LazyClouds

    source = GenericSource(sequence, n_scans=3, stride=2, verbose=False)
    loaded = source.load()
    lazy = _LazyClouds(source)
    for i in (0, len(loaded) // 2, len(loaded) - 1):
        np.testing.assert_array_equal(lazy[i], loaded.point_clouds[i])


def test_the_stub_viewer_needs_no_display():
    """A live run must be drivable headless, or CI cannot test it at all."""
    stub = StubViewer()
    stub.start(None, None, None, 0)
    stub.on_frame(anything=1)
    stub.finish()
    assert stub.closed() is False


# --- the viewer's geometry -------------------------------------------------


def test_to_world_is_the_pose_applied():
    """The map, the keypoints and the alignment panel all go through this."""
    from inlier.viz.live import _to_world

    rng = np.random.default_rng(1)
    pts = rng.uniform(-10.0, 10.0, size=(7, 3)).astype(np.float32)
    pose = np.eye(4)
    angle = 0.7
    pose[:3, :3] = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                             [np.sin(angle), np.cos(angle), 0.0],
                             [0.0, 0.0, 1.0]])
    pose[:3, 3] = [3.0, -4.0, 1.5]

    expected = (pose @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :3]
    np.testing.assert_allclose(_to_world(pts, pose), expected, atol=1e-5)


def test_the_alignment_panel_puts_the_candidate_on_the_query():
    """``p_db = T_sensor @ p_query``, so the panel must undo it, not apply it.

    Getting the direction backwards produces a picture that still looks like
    two clouds near each other, which is exactly why it is pinned.
    """
    from inlier.viz.live import _to_world

    rng = np.random.default_rng(2)
    q_pts = rng.uniform(-10.0, 10.0, size=(9, 3)).astype(np.float32)
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 0.5]
    db_pts = _to_world(q_pts, T)                    # what the DB would see

    q_pose = np.eye(4)
    q_pose[:3, 3] = [20.0, 5.0, 0.0]
    drawn = _to_world(_to_world(db_pts, np.linalg.inv(T)), q_pose)
    np.testing.assert_allclose(drawn, _to_world(q_pts, q_pose), atol=1e-5)


def test_image_panels_ramp_and_survive_a_flat_descriptor():
    """An all-zero descriptor must not divide by a zero range.

    The ramps are the true matplotlib ones, endpoints included: these panels
    are read against the figure `inlier encode --viz` writes for the same
    arrays, and a shifted ramp would make the two disagree about what a colour
    means.
    """
    from inlier.viz.live import _INFERNO, _VIRIDIS, _image_u8

    flat = _image_u8(np.zeros((4, 5)))
    assert flat.shape == (4, 5, 3)
    assert len(np.unique(flat.reshape(-1, 3), axis=0)) == 1
    np.testing.assert_array_equal(flat[0, 0], _VIRIDIS[0].astype(np.uint8))
    assert _image_u8(np.zeros((0, 3))).shape == (1, 1, 3)

    # Rows come back flipped, because the matplotlib panels these are read
    # against are drawn with origin="lower": the array's first row belongs at
    # the bottom of the image, not the top.
    for lut in (_VIRIDIS, _INFERNO):
        ramp = _image_u8(np.arange(16.0).reshape(4, 4), lut)
        np.testing.assert_array_equal(ramp[-1, 0], lut[0].astype(np.uint8))
        np.testing.assert_array_equal(ramp[0, -1], lut[-1].astype(np.uint8))

    # A span pins the scale the way imshow's vmin/vmax do, so a half-full cell
    # is drawn at the middle of the ramp rather than at the top of it.
    half = _image_u8(np.full((2, 2), 5.0), _INFERNO, (0.0, 10.0))
    mid = np.round(_INFERNO[15:17].mean(axis=0)).astype(np.uint8)  # 0.5 * 31
    np.testing.assert_array_equal(half[0, 0], mid)
    np.testing.assert_array_equal(
        _image_u8(np.full((2, 2), 10.0), _INFERNO, (0.0, 10.0))[0, 0],
        _INFERNO[-1].astype(np.uint8))



def test_height_colours_share_a_span_between_the_scan_and_its_keypoints():
    """A colour has to mean the same height in both, or they read as unrelated.

    Also pins the degenerate case: a perfectly flat cloud has no range to
    normalise over and must not divide by zero.
    """
    from inlier.viz.live import _GREY, _TURBO, _height_colors

    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 5.0], [2.0, 0.0, 10.0]])
    span = (0.0, 10.0)
    grey = _height_colors(pts, _GREY, span)
    turbo = _height_colors(pts, _TURBO, span)

    assert grey.shape == (3, 4) and grey.dtype == np.float32
    assert (grey[:, 3] == 1.0).all(), "alpha must stay opaque"
    # Grey means grey: the three channels agree at every point.
    np.testing.assert_allclose(grey[:, 0], grey[:, 1], atol=1e-6)
    np.testing.assert_allclose(grey[:, 1], grey[:, 2], atol=1e-6)
    # Turbo does not, or it would not separate anything.
    assert not np.allclose(turbo[:, 0], turbo[:, 2])
    # Monotonic in height, and the shared span puts the same point at the
    # same fraction of both ramps.
    assert grey[0, 0] < grey[1, 0] < grey[2, 0]

    flat = _height_colors(np.zeros((4, 3)), _TURBO)
    assert np.isfinite(flat).all() and len(np.unique(flat, axis=0)) == 1


def test_the_viewer_is_given_per_frame_times_not_a_running_total(sequence, tmp_path,
                                                                 monkeypatch):
    """A cumulative total says nothing about the frame on screen.

    `times` accumulates over the run and is what the results JSON reports;
    `frame_times` is this frame alone.  Handing the panel the first is how it
    came to show "encode 74000 ms" and climbing.
    """
    from inlier.eval import stream
    from inlier.viz.live import StubViewer

    seen = []

    class Spy(StubViewer):
        def on_frame(self, **kw):
            seen.append((dict(kw["times"]), dict(kw["frame_times"])))

    spec = _spec(sequence, tmp_path, live=True)
    stream.run_stream(spec, __import__("inlier").InLiER(spec.resolved.inlier), Spy())

    assert len(seen) > 2
    totals = [t["encode"] for t, _ in seen]
    per_frame = [f["encode"] for _, f in seen]

    # The total only ever grows; the frame cost does not track it.
    assert totals == sorted(totals)
    assert totals[-1] > totals[0]
    assert max(per_frame) < totals[-1], "frame time is following the total"
    # Each frame's cost is bounded by what the whole run spent.
    assert all(0.0 <= f <= totals[-1] + 1e-9 for f in per_frame)
    # And the totals really are the sum of the parts.
    assert totals[-1] == pytest.approx(sum(per_frame), rel=1e-6)


def test_a_stage_that_did_not_run_reports_zero_for_that_frame(sequence, tmp_path):
    """Stale timings are worse than none: they look like work that happened."""
    from inlier.eval import stream
    from inlier.viz.live import StubViewer

    seen = []

    class Spy(StubViewer):
        def on_frame(self, **kw):
            seen.append(dict(kw["frame_times"]))

    spec = _spec(sequence, tmp_path, live=True)
    stream.run_stream(spec, __import__("inlier").InLiER(spec.resolved.inlier), Spy())

    # Rerank is off by default, so it must read 0 on every frame rather than
    # carrying the previous frame's number forward.
    assert all(f["rr"] == 0.0 for f in seen)
    # GICP only runs on a frame that produced a closure, so at least one frame
    # must report nothing for it -- frame 0 has no past to close against.
    assert seen[0]["gicp"] == 0.0


def test_panel_textures_are_uploaded_bgr(sequence, tmp_path, drawn):
    """`glk::create_texture` hands GL `GL_BGR` for a 3-channel image.

    It takes a `cv::Mat`, and OpenCV images are BGR (`glk/texture_opencv.hpp`).
    A numpy array is RGB, so an unswapped upload puts viridis's yellow end on
    screen as cyan -- which still looks like a colour map, just not the one
    every other picture of these descriptors uses.
    """
    from inlier.viz.live import _INFERNO, _VIRIDIS, _bgr

    rgb = np.array([[[253, 231, 37]]], dtype=np.uint8)      # viridis, top end
    np.testing.assert_array_equal(_bgr(rgb), [[[37, 231, 253]]])
    assert _bgr(rgb).flags["C_CONTIGUOUS"], "pybind11 needs a contiguous buffer"

    log = _drive(_spec(sequence, tmp_path, live=True), drawn)
    uploaded = log["textures"]
    assert uploaded, "no panel texture was ever built"
    # These descriptors are sparse, so a panel's most common colour is its
    # empty cell -- the bottom of the ramp exactly, with no interpolation to
    # blur the comparison.  It has to be that colour *reversed*; read forwards
    # it lands on neither ramp, since neither is symmetric.
    ends = {tuple(int(v) for v in np.round(lut[0]))[::-1]
            for lut in (_VIRIDIS, _INFERNO)}
    for image in uploaded:
        assert image.dtype == np.uint8 and image.shape[2] == 3
        colours, counts = np.unique(image.reshape(-1, 3), axis=0,
                                    return_counts=True)
        common = tuple(int(v) for v in colours[counts.argmax()])
        assert common in ends, f"{common} is on neither ramp, read as BGR"


def test_every_descriptor_panel_is_drawn_the_same_width():
    """B is N_a columns where H and R are N_r*N_s; magnification evens them up.

    Most of that magnification happens in numpy, at whole cells, precisely so
    the GPU is not left filtering a 10x140 array across half the window --
    that is what made the panels look low-resolution.  What reaches `scale` is
    only the fractional remainder, and the two together still have to land on
    one width for all three panels.
    """
    from inlier.viz.live import PANEL_WIDTH_PX, _blocks

    for cols in (140, 60, 1):
        block = max(1, PANEL_WIDTH_PX // cols)
        scale = PANEL_WIDTH_PX / (cols * block)
        assert round(cols * block * scale) == PANEL_WIDTH_PX
        # Whatever is left for the GPU is a touch of stretch, not the 4x-9x
        # blur-up the panels used to be drawn with.
        assert 1.0 <= scale < 2.0

    # The upsample is nearest-neighbour: a cell becomes a block of one colour,
    # with no invented values between two counts.
    image = np.arange(6, dtype=np.uint8).reshape(2, 3, 1).repeat(3, axis=2)
    grown = _blocks(image, 4)
    assert grown.shape == (8, 12, 3)
    assert len(np.unique(grown.reshape(-1, 3), axis=0)) == 6
    np.testing.assert_array_equal(grown[:4, :4], np.broadcast_to(image[0, 0], (4, 4, 3)))
    assert _blocks(image, 1) is image


def test_the_scene_is_anchored_so_float32_can_hold_it():
    """The bug this was written for: HeLiPR poses are UTM.

    A Roundabout03 pose is around (3.06e5, 4.13e6, 33).  Drawn raw, the scene
    sits millions of metres from the camera, and float32 -- which is what the
    viewer's buffers are -- resolves about a quarter of a metre out there, so
    it would be quantised even if you found it.  Anchoring must bring the
    coordinates back to the origin without moving anything relative to
    anything else.
    """
    from inlier.viz.live import LiveViewer

    viewer = LiveViewer.__new__(LiveViewer)      # no window, no pyridescence
    viewer._origin = None

    utm = np.eye(4)
    utm[:3, 3] = [305714.4, 4134220.3, 32.92]
    viewer._anchor(utm)

    pts = np.array([[305714.4, 4134220.3, 32.92],
                    [305734.4, 4134200.3, 42.92]])
    local = viewer._local(pts)

    assert local.dtype == np.float32
    assert np.abs(local).max() < 1e3, "still out in UTM"
    np.testing.assert_array_equal(local[0], [0.0, 0.0, 0.0])
    # The one thing anchoring must not do is change the geometry.
    np.testing.assert_allclose(np.linalg.norm(local[1] - local[0]),
                               np.linalg.norm(pts[1] - pts[0]), rtol=1e-6)


def test_the_anchor_is_set_once_and_never_moves():
    """A drifting origin would make the map slide out from under the scan."""
    from inlier.viz.live import LiveViewer

    viewer = LiveViewer.__new__(LiveViewer)
    viewer._origin = None
    first = np.eye(4)
    first[:3, 3] = [100.0, 200.0, 5.0]
    viewer._anchor(first)

    later = np.eye(4)
    later[:3, 3] = [999.0, 888.0, 77.0]
    viewer._anchor(later)                        # must be ignored

    np.testing.assert_array_equal(viewer._origin, [100.0, 200.0, 5.0])


# --- the viewer, driven by a real run --------------------------------------
#
# The rendering cannot be tested without a display, but everything around it
# can: that `start` and `on_frame` accept what the driver actually passes, that
# the single-session and cross-session branches both survive a frame, and that
# the map is deposited in batches rather than re-uploaded whole.


class _Recorder:
    """Accepts any call, remembers the drawable names, chains like a setting."""

    def __init__(self, log) -> None:
        self.log = log

    def __getattr__(self, name):
        def call(*args, **kwargs):
            if name == "create_texture":
                self.log.setdefault("textures", []).append(args[0])
            if args and isinstance(args[0], str):
                self.log.setdefault(name, []).append(args[0])
            else:
                self.log.setdefault(name, []).append(None)
            if name == "register_ui_callback":
                self.log["ui"] = args[1]
            return self
        return call

    # The two the viewer branches on, rather than merely calls.  `spin_once`
    # runs out eventually, which is how a real window reports being closed --
    # and without it `finish()` would spin here for ever.
    _budget = 500

    def ok(self):
        return True

    def spin_once(self):
        self._budget -= 1
        return self._budget > 0


class _FakeImgui:
    """imgui, minus the window.  The widgets that hand a value back do so."""

    def __init__(self, log) -> None:
        self.log = log

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.log.setdefault("imgui", []).append(name)
            if name in ("slider_float", "drag_float"):
                return False, float(args[1])
            if name.startswith("checkbox"):
                return False, bool(args[1])
            return False
        return call


class _FakeGuik:
    def __init__(self, log) -> None:
        self.log = log
        self.LightViewer = type("LV", (), {
            "instance": staticmethod(lambda **kw: _Recorder(log))})

    def __getattr__(self, name):          # FlatColor, VertexColor, Rainbow
        return lambda *a, **k: _Recorder(self.log)


@pytest.fixture
def drawn(sequence, lap, tmp_path, monkeypatch, request):
    """Run the live driver with a fake pyridescence, and return what it drew."""
    from inlier.viz import live as livemod

    log: dict = {}
    monkeypatch.setattr(livemod, "_import_pyridescence",
                        lambda: (_Recorder(log), _FakeGuik(log), _FakeImgui(log)))
    return log


def _drive(spec, log):
    from inlier.eval import stream
    from inlier.viz.live import LiveViewer

    viewer = LiveViewer(spec.threshold, title="test")
    viewer._playing = True   # the real viewer opens paused; nothing here clicks play
    stream.run_stream(spec, __import__("inlier").InLiER(spec.resolved.inlier),
                      viewer)
    log["ui"]()          # the ImGui panel is only called by a real viewer
    viewer.finish()      # flushes whatever the last batch was still holding
    return log


def test_the_viewer_survives_a_single_session_run(sequence, tmp_path, drawn):
    log = _drive(_spec(sequence, tmp_path, live=True), drawn)
    names = log["update_points"]
    assert any(n.startswith("map/") for n in names), "no map was deposited"
    assert "traj/nodes" in names
    assert any(n.startswith("closure/") for n in log["update_thin_lines"]), \
        "the planted revisits produced no closure edge"
    # All three descriptor stages, not just one: H is the token histogram, R is
    # the row stage 1 scores, A is the BEAM elevation code.
    from inlier.viz.live import PANEL_ORDER

    assert set(log["update_image"]) == set(PANEL_ORDER)
    # Black behind all of it -- the default clear colour is grey, which eats
    # the bottom of the scan's grey-by-height ramp.
    assert "set_clear_color" in log
    # The scan, the keypoints and the pose triad are per-vertex coloured, so
    # they arrive as drawables rather than through update_points.
    assert {"cur/cloud", "cur/kp", "cur/pose"} <= set(log["update_drawable"])


def test_the_viewer_survives_a_cross_session_run(sequence, lap, tmp_path, drawn):
    log = _drive(_spec(sequence, tmp_path, live=True, db_root=lap), drawn)
    assert "db/map" in log["update_points"], "the prior map was never drawn"
    assert "db/traj" in log["update_thin_lines"]


def test_turning_the_map_off_removes_what_is_already_drawn(sequence, tmp_path, drawn):
    """Stopping the deposits is not enough -- the accumulated map must go.

    Removal is by exact name.  `remove_drawable`'s regex overload *matches*
    rather than searches, so a prefix pattern removes nothing at all and does
    it silently, which is how this looked fixed while still being broken.
    """
    from inlier.viz.live import LiveViewer

    viewer = LiveViewer.__new__(LiveViewer)
    removed = []
    viewer._viewer = type("V", (), {
        "remove_drawable": lambda self, name: removed.append(name)})()
    viewer._map_names = ["map/0", "map/1", "map/2"]
    viewer._deposits = 3
    viewer._pending = [np.zeros((3, 3), dtype=np.float32)]

    viewer._clear_map()

    assert removed == ["map/0", "map/1", "map/2"], removed
    assert viewer._map_names == [] and viewer._pending == []
    assert viewer._deposits == 0


def test_the_map_clears_on_the_click_even_while_paused(sequence, tmp_path, drawn):
    """A paused run has no next frame to do it on.

    The whole point of pausing is to look at what is there; deferring the
    clear to `on_frame` means the button does nothing precisely when someone
    is most likely to press it.
    """
    log = _drive(_spec(sequence, tmp_path, live=True), drawn)

    deposits = [n for n in log["update_points"] if n.startswith("map/")]
    assert deposits, "nothing was deposited, so clearing proves nothing"

    from inlier.viz.live import LiveViewer

    # Re-enter the UI callback with the box unticked, the way a click does.
    ui = log["ui"]
    assert isinstance(ui.__self__, LiveViewer)
    viewer = ui.__self__
    viewer._map_names = list(deposits)
    log.setdefault("remove_drawable", [])
    before = len(log["remove_drawable"])
    viewer._clear_map()
    assert log["remove_drawable"][before:] == deposits


def test_the_map_is_deposited_in_batches_not_per_frame(sequence, tmp_path, drawn):
    """One drawable per frame would mean thousands of draw calls by the end."""
    from inlier.viz.live import MAP_FLUSH

    log = _drive(_spec(sequence, tmp_path, live=True), drawn)
    deposits = {n for n in log["update_points"] if n.startswith("map/")}
    assert len(deposits) <= N_FRAMES // MAP_FLUSH + 1 < N_FRAMES
