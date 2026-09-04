"""The live viewer for ``inlier run --live``.

Built on `Iridescence <https://koide3.github.io/iridescence/>`_ through its
Python binding ``pyridescence`` -- the viewer GLIM uses, which is the reason it
is the one here: it is built for exactly this, watching a point-cloud algorithm
run, and its ImGui integration means the controls and the numbers cost a few
lines rather than a UI framework.

Two classes.  :class:`StubViewer` is the default and does nothing, so the
streaming driver has no ``if viewer is not None`` in its loop and a headless
run costs nothing.  :class:`LiveViewer` draws.

Two things here are not obvious and both are load-bearing.

**The scene is anchored.**  Every point and pose is drawn relative to the first
pose the viewer sees.  Datasets carry world coordinates in whatever frame they
were mapped in, and HeLiPR's are UTM -- a Roundabout03 pose is around
``(3.06e5, 4.13e6, 33)``.  The viewer's vertex buffers are float32, whose
resolution out there is about a quarter of a metre, so drawing raw coordinates
puts the scene millions of metres off camera *and* quantises it to mush.

**The map is deposited in batches.**  Each flush writes one drawable that is
never touched again (``map/0``, ``map/1``, ...).  One drawable per frame would
be thousands of draw calls by the end of a sequence; one growing buffer would
re-upload the whole map every frame.  Batching is neither, and it is what keeps
frame time flat instead of climbing with the sequence length.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

#: Voxel size for the points kept in the accumulated map, in metres.  The map
#: is context, not data: it exists to show where the robot has been.
MAP_VOXEL = 1.5
#: Voxel size for the current submap.  A HeLiPR scan is half a million points
#: and re-uploading all of them every frame is 7 MB of traffic for detail no
#: screen resolves; a quarter-metre grid keeps the shape and a tenth of the
#: points.
CUR_VOXEL = 0.5
#: Keyframes between prior-map deposits, cross-session only.  Reading every one
#: before the run starts costs minutes of disk for a picture a stride gives.
DB_MAP_STRIDE = 20
#: Frames per map deposit.  Every frame's points are kept, but they are
#: uploaded in batches: one drawable per frame would mean thousands of draw
#: calls by the end of a sequence, and one growing buffer would mean
#: re-uploading the whole map every frame.  Batching is neither.
MAP_FLUSH = 20

#: Point sizes, in metres.  `set_point_shape(size, metric=True, circle=True)`
#: is the API that actually sizes a point -- `set_point_scale` is only a
#: multiplier on the base size, which is why nothing drawn with it stood out.
#: Metric so the map stays readable as you zoom: keypoints and keyframes are
#: markers and should hold their real-world size, the clouds are texture.
KP_SIZE = 0.33
NODE_SIZE = 0.75
CLOUD_SIZE = 0.1
MAP_SIZE = 0.05

#: Flat colours, for the things that are not height-coded.  The scan and its
#: keypoints get per-vertex ramps instead (see `_height_colors`); what is left
#: is the map, which stays dim and uniform so it reads as background, and the
#: trajectory, which has to stay findable on top of everything.
TRAJ_COLOR = (1.00, 0.85, 0.10, 1.0)
MAP_COLOR = (0.38, 0.41, 0.46, 0.3)
DB_MAP_COLOR = (0.30, 0.34, 0.40, 1.0)
DB_TRAJ_COLOR = (0.20, 0.47, 0.72, 1.0)
CLOSURE_COLOR = (0.18, 0.90, 0.44, 1.0)

#: Colour tables, sampled at 16 points and interpolated.  Hardcoded rather than
#: taken from matplotlib because the ``[viz]`` extra does not pull ``[eval]``,
#: and three colour tables are not worth a dependency.
#:
#: Viridis draws all three descriptor panels -- they are the same quantity at
#: three stages of collapsing, so giving them separate ramps would invent a
#: distinction the data does not have.  Turbo codes keypoint height, where the
#: job is the opposite: pick the colours apart at a glance.
#: Viridis and inferno, sampled from matplotlib at 32 anchors and written out
#: here rather than imported: matplotlib is in the ``[eval]`` extra and the
#: viewer is in ``[viz]``, so a live run must not need the evaluation stack to
#: colour a panel.  32 rather than 16 because the ramps are not linear in RGB
#: -- interpolating between 16 anchors visibly missed viridis's blue-green
#: middle, and these panels are read against the matplotlib ones that
#: ``inlier encode --viz`` writes.
#:
#: Which panel gets which is that figure's choice, not a new one: H and R are
#: viridis there, A is inferno, and the endpoints are the true ones, so a
#: colour means the same thing in both pictures.
_VIRIDIS = np.array([
    [ 68,   1,  84], [ 71,  13,  96], [ 72,  24, 106], [ 72,  35, 116],
    [ 71,  46, 124], [ 69,  56, 130], [ 66,  65, 134], [ 62,  74, 137],
    [ 58,  84, 140], [ 54,  93, 141], [ 50, 101, 142], [ 46, 109, 142],
    [ 43, 117, 142], [ 40, 125, 142], [ 37, 132, 142], [ 34, 140, 141],
    [ 31, 148, 140], [ 30, 156, 137], [ 32, 163, 134], [ 37, 171, 130],
    [ 46, 179, 124], [ 58, 186, 118], [ 72, 193, 110], [ 88, 199, 101],
    [108, 205,  90], [127, 211,  78], [147, 215,  65], [168, 219,  52],
    [192, 223,  37], [213, 226,  26], [234, 229,  26], [253, 231,  37],
], dtype=np.float32)

_INFERNO = np.array([
    [  0,   0,   4], [  4,   3,  18], [ 11,   7,  36], [ 21,  11,  55],
    [ 35,  12,  76], [ 49,  10,  92], [ 62,   9, 102], [ 76,  12, 107],
    [ 90,  17, 110], [103,  22, 110], [116,  26, 110], [128,  31, 108],
    [143,  36, 105], [155,  41, 100], [168,  46,  95], [180,  51,  89],
    [193,  58,  80], [204,  66,  72], [215,  75,  63], [224,  85,  54],
    [233,  97,  43], [239, 110,  33], [245, 123,  23], [248, 137,  12],
    [251, 153,   6], [252, 168,  13], [251, 184,  29], [249, 199,  47],
    [245, 217,  73], [242, 232, 101], [243, 245, 134], [252, 255, 164],
], dtype=np.float32)

#: Turbo, for the keypoints: they sit on top of the grey scan and have to be
#: findable in it, which wants a ramp that changes hue fast.
_TURBO = np.array([
    [ 48,  18,  59], [ 65,  67, 167], [ 71, 113, 233], [ 62, 155, 254],
    [ 34, 197, 226], [ 26, 228, 182], [ 70, 248, 132], [136, 255,  78],
    [185, 246,  53], [225, 221,  55], [250, 186,  57], [253, 141,  39],
    [240,  91,  18], [214,  53,   6], [175,  24,   1], [122,   4,   3],
], dtype=np.float32)

#: The scan is grey-coded by height: it is the bulk of the scene and has to
#: stay legible underneath the keypoints without competing with them.  Not pure
#: black-to-white -- the floor is lifted so low points stay visible against the
#: background, and the ceiling stops short of white so the markers keep it.
_GREY = np.linspace(60.0, 210.0, 16)[:, None].repeat(3, axis=1).astype(np.float32)

#: The three descriptor stages, in pipeline order, with the display order and
#: magnification each needs to be readable.  The names are the ones
#: ``inlier/viz/figures.py`` uses for the same arrays, so a panel here and a
#: panel in an ``inlier encode --viz`` figure mean the same thing.
#:
#: Without an explicit order the viewer lays panels out in whatever sequence
#: they were first registered, which is what put them on top of each other.
PANEL_ORDER = {"H  histogram": 0, "R  MINT row": 1, "A  BEAM elevation": 2}
#: Every panel is drawn this wide, whatever its column count -- H and R are
#: N_r*N_s across, B is only N_a, and left to a shared magnification B came out
#: less than half the width of the other two.  Height follows from the aspect,
#: so B stays a square-ish grid rather than being stretched to match.
PANEL_WIDTH_PX = 560
#: Rows to tile the MINT row to.  It is one row of N_r*N_s values; at four
#: pixels tall it is a line, not a panel.
MINT_ROWS = 4

#: What each timed stage is called on screen.  The keys are the pipeline's
#: internal names; the labels are the ones `inlier run` prints and the paper
#: uses, so a row here and a line of the run's own log mean the same thing.
#: `rerank` is the 4-D token-histogram stage -- off by default, hence its
#: permanent 0.
STAGE_LABELS = (("voxel", "voxel"), ("encode", "encode"), ("s1", "MINT"),
                ("s2", "BEAM"), ("rr", "rerank"), ("verify", "verify"),
                ("gicp", "GICP"))

#: The pose triad: short enough not to spear the scan, thick enough to see.
#: Drawn as `glk.Lines`, which takes a real thickness -- the `coordinate_system`
#: primitive is one-pixel lines whose only knob is overall scale.
#: The window's background.  RGBA, as `set_clear_color` takes it.
BG_COLOR = (0.0, 0.0, 0.0, 1.0)

AXIS_LENGTH = 3.0
AXIS_THICKNESS = 0.25


class StubViewer:
    """The no-op viewer.  Every method is a hole the optimiser can see through."""

    def start(self, *args, **kwargs) -> None:
        pass

    def on_frame(self, **kwargs) -> None:
        pass

    def closed(self) -> bool:
        return False

    def finish(self, *args, **kwargs) -> None:
        pass


@contextlib.contextmanager
def _muted_stderr():
    """Silence fd 2, including writes from C++, for the duration of the block."""
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stderr.flush()
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def _import_pyridescence():
    """The three pyridescence modules, or a message saying how to get them.

    Only a *missing* module is turned into a user-facing message; an installed
    pyridescence that fails to import is a broken install and keeps its
    traceback, which is what a bug report needs.
    """
    try:
        # Importing the extension prints a C++ deprecation warning about
        # VoxelMapOptions on fd 2, from inside a library we do not call.  It is
        # not actionable and it lands in the middle of the run's own output, so
        # the descriptor is muted for the duration of the import only.
        with _muted_stderr():
            from pyridescence import glk, guik, imgui
    except ModuleNotFoundError as exc:  # pragma: no cover - install-dependent
        raise ModuleNotFoundError(
            "the live viewer needs pyridescence, which is not installed. "
            'Install it with `pip install -e ".[viz]"`, or drop --live to run '
            "the same pipeline without a window."
        ) from exc
    return glk, guik, imgui


def _voxel(points: np.ndarray, size: float) -> np.ndarray:
    from inlier.eval.encode import voxel_downsample

    return voxel_downsample(np.asarray(points, dtype=np.float32), size)


def _to_world(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    R = np.asarray(pose[:3, :3], dtype=np.float32)
    t = np.asarray(pose[:3, 3], dtype=np.float32)
    return (np.asarray(points, dtype=np.float32) @ R.T) + t


def _ramp(norm: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Look *norm* (already in 0..1) up in *lut*, interpolating between anchors."""
    x = np.clip(norm, 0.0, 1.0) * (len(lut) - 1)
    lo_i = np.floor(x).astype(np.int32)
    hi_i = np.minimum(lo_i + 1, len(lut) - 1)
    frac = (x - lo_i)[..., None]
    return lut[lo_i] * (1.0 - frac) + lut[hi_i] * frac


def _height_colors(points: np.ndarray, lut: np.ndarray,
                   span: Optional[tuple] = None) -> np.ndarray:
    """(m, 4) float32 RGBA, coding each point's height through *lut*.

    ``span`` pins the height range.  The scan and its keypoints pass the same
    one, so a colour means the same height in both and the markers read as
    sitting *in* the cloud rather than as a second, unrelated picture.
    """
    z = np.asarray(points, dtype=np.float64)[:, 2]
    if span is None:
        span = (float(z.min()), float(z.max())) if z.size else (0.0, 1.0)
    lo, hi = span
    norm = np.zeros_like(z) if hi <= lo else (z - lo) / (hi - lo)

    rgba = np.ones((len(z), 4), dtype=np.float32)
    rgba[:, :3] = _ramp(norm, lut) / 255.0
    return rgba


def _bgr(image: np.ndarray) -> np.ndarray:
    """RGB to the channel order the texture upload actually reads.

    ``glk::create_texture`` takes a ``cv::Mat`` and hands GL ``GL_BGR`` for a
    3-channel image (``glk/texture_opencv.hpp``), because OpenCV images are
    BGR.  A numpy array arrives as RGB, so without this swap viridis's yellow
    end uploads as cyan -- close enough to a colour map to look like one, and
    wrong everywhere.  Swapped here rather than in :func:`_image_u8` so that
    function keeps producing what matplotlib would, which is what the panels
    are checked against.
    """
    return np.ascontiguousarray(image[..., ::-1])


def _blocks(image: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour upsample, so a descriptor cell is a crisp square.

    ``update_image`` magnifies the texture on the GPU with the driver's
    filtering, which blurs a 10x140 array across half a window and reads as a
    low-resolution smear.  Growing the array here instead keeps every cell
    edge sharp -- these are discrete counts, and a gradient between two of
    them is a value the descriptor does not have.
    """
    if factor <= 1:
        return image
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def _image_u8(values: np.ndarray, lut: np.ndarray = _VIRIDIS,
              span: Optional[tuple] = None) -> np.ndarray:
    """A 2-D array as an RGB image, in matplotlib's row order.

    ``span`` pins the colour scale the way ``imshow(vmin=, vmax=)`` does; with
    no span the array is normalised to its own range, which is what matplotlib
    does when neither is given.

    The rows come back flipped because ``inlier/viz/figures.py`` draws every
    one of these arrays with ``origin="lower"`` -- row 0 at the bottom -- and
    the viewer's textures start at the top.  Without the flip the same
    descriptor is upside down in the two pictures.
    """
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    lo, hi = span if span is not None else (float(a.min()), float(a.max()))
    norm = np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo)
    return _ramp(np.flipud(norm), lut).round().astype(np.uint8)


class LiveViewer:
    """A window that shows one ``inlier run`` as it happens."""

    def __init__(self, threshold: float, title: str = "inlier run") -> None:
        self._glk, self._guik, self._imgui = _import_pyridescence()
        self._viewer = self._guik.LightViewer.instance(title=title)
        self._viewer.use_orbit_camera_control(200.0)
        # Black, not the default grey: the scan is grey-coded by height, and a
        # grey ground behind it takes the bottom of that ramp with it.
        self._viewer.set_clear_color(np.array(BG_COLOR, dtype=np.float32))
        self._threshold = float(threshold)

        # Paused on frame 0: a run opens with the first frame drawn and waits.
        # Starting mid-stream means the window appears with the sequence
        # already running and no chance to place the camera first.
        self._playing = False
        self._step = False
        self._fps_cap = 0.0          # 0 = unthrottled
        self._show_map = True
        self._show_kp = True
        self._show_scan = True
        self._axis_drawable = None
        self._last_frame_at = 0.0

        self._n_frames = 0
        self._index = 0
        self._n_closures = 0
        self._best_score = 0.0
        self._times: Dict[str, float] = {}
        self._frame_times: Dict[str, float] = {}
        self._latency_history: List[float] = []
        self._deposits = 0
        self._pending: List[np.ndarray] = []
        #: Names of the map deposits currently on the GPU, so they can be
        #: removed by name.  `remove_drawable` takes a regex, but the header
        #: says it *matches* rather than searches -- a prefix pattern silently
        #: removes nothing -- and guessing at that is not worth it when the
        #: names are ours to remember.
        self._map_names: List[str] = []
        self._grid: Optional[tuple] = None
        self._db_poses = None
        #: Scene anchor, subtracted from every point and pose before it is
        #: drawn.  Datasets carry world coordinates in whatever frame they were
        #: mapped in -- HeLiPR's are UTM, so a pose is around (3e5, 4.1e6) --
        #: and the viewer's buffers are float32, whose resolution out there is
        #: a quarter of a metre.  Drawing raw would put the scene millions of
        #: metres off camera *and* quantise it.  Anchoring at the first pose
        #: costs nothing and fixes both.
        self._origin: Optional[np.ndarray] = None
        self._follow = True

        self._viewer.register_ui_callback("inlier", self._ui)

    # -- lifecycle ---------------------------------------------------------

    def _anchor(self, pose_or_point) -> None:
        """Fix the scene anchor, once, on the first thing that has a position."""
        if self._origin is None:
            p = np.asarray(pose_or_point, dtype=np.float64)
            self._origin = (p[:3, 3] if p.shape == (4, 4) else p[:3]).copy()

    def _local(self, points: np.ndarray) -> np.ndarray:
        """World points, moved next to the origin so float32 can hold them."""
        return np.ascontiguousarray(
            (np.asarray(points, dtype=np.float64) - self._origin),
            dtype=np.float32)

    def start(self, spec, db_enc, db_clouds, n_frames: int) -> None:
        """Draw the prior map, once, before the query session starts."""
        self._n_frames = int(n_frames)
        self._grid = (spec.resolved.inlier.N_h, spec.resolved.inlier.N_r,
                      spec.resolved.inlier.N_s, spec.resolved.inlier.N_a)
        if db_enc is None or not len(db_enc.poses):
            return

        self._db_poses = np.asarray(db_enc.poses, dtype=np.float64)
        self._anchor(self._db_poses[0])
        traj = self._local(self._db_poses[:, :3, 3])
        if len(traj) > 1:
            self._viewer.update_thin_lines(
                "db/traj", traj, line_strip=True,
                shader_setting=self._guik.FlatColor(*DB_TRAJ_COLOR))
        self._viewer.update_points(
            "db/nodes", traj,
            self._guik.FlatColor(*DB_TRAJ_COLOR).set_point_shape(NODE_SIZE, True, True))
        self._viewer.lookat(traj[0])

        # The prior map really is available up front, so it is drawn up front
        # -- but every keyframe of it would be minutes of disk before the first
        # query frame appears, which is not a live view of anything.  A stride
        # gives the same picture for a twentieth of the wait.
        chunks = []
        for i in range(0, len(self._db_poses), DB_MAP_STRIDE):
            pts = _voxel(db_clouds[i], MAP_VOXEL)
            if pts.size:
                chunks.append(self._local(_to_world(pts, self._db_poses[i])))
        if chunks:
            self._viewer.update_points(
                "db/map", np.vstack(chunks),
                self._guik.FlatColor(*DB_MAP_COLOR).set_point_shape(MAP_SIZE, True, False))

    def closed(self) -> bool:
        return not self._viewer.ok()

    def _flush_map(self) -> None:
        if not self._pending:
            return
        name = f"map/{self._deposits}"
        self._viewer.update_points(
            name, np.vstack(self._pending),
            self._guik.FlatColor(*MAP_COLOR).set_point_shape(MAP_SIZE, True, False))
        self._map_names.append(name)
        self._deposits += 1
        self._pending = []

    def _draw_colored(self, name: str, points: np.ndarray, lut: np.ndarray,
                      span, size: float) -> None:
        """A point cloud coloured per vertex, rather than by one flat colour."""
        buf = self._glk.create_pointcloud_buffer(
            np.ascontiguousarray(points, dtype=np.float32),
            _height_colors(points, lut, span))
        self._viewer.update_drawable(
            name, buf,
            self._guik.VertexColor().set_point_shape(size, True, True))

    def _axis(self):
        """A pose triad with a real thickness, in the pose's own frame."""
        if self._axis_drawable is None:
            L = AXIS_LENGTH
            verts = np.array([[0, 0, 0], [L, 0, 0],
                              [0, 0, 0], [0, L, 0],
                              [0, 0, 0], [0, 0, L]], dtype=np.float32)
            cols = np.array([[1, 0, 0, 1], [1, 0, 0, 1],
                             [0, 1, 0, 1], [0, 1, 0, 1],
                             [0, 0.45, 1, 1], [0, 0.45, 1, 1]], dtype=np.float32)
            self._axis_drawable = self._glk.Lines(AXIS_THICKNESS, verts, cols)
        return self._axis_drawable

    def _clear_map(self) -> None:
        """Drop every deposit already on the GPU, not just future ones.

        Turning the map off has to remove what is drawn; stopping the deposits
        leaves the whole accumulated map sitting there, which is not what the
        checkbox appears to promise.  The points are gone rather than hidden --
        keeping them would mean holding the entire sequence in host memory for
        a checkbox -- so turning it back on accumulates afresh from that frame.
        """
        for name in self._map_names:
            self._viewer.remove_drawable(name)
        self._map_names = []
        self._pending = []
        self._deposits = 0

    def finish(self) -> None:
        """Hold the window open until it is closed, so the result stays up."""
        self._flush_map()
        self._playing = False
        while self._viewer.spin_once():
            pass

    # -- per frame ---------------------------------------------------------

    def on_frame(self, *, frame, index, tokens, kp_sensor, kp_aligned, pose,
                 ranked, sims_s1, sims_s2, sims_ver, verify, accepted, gicp,
                 db_poses, q_poses, times, bound, frame_times=None, **_) -> None:
        self._index = index
        self._times = times
        self._frame_times = frame_times or {}
        self._n_closures += len(accepted)
        best = max(sims_ver.values(), default=0.0)
        self._best_score = float(best)

        pose = np.asarray(pose, dtype=np.float64)
        self._anchor(pose)
        local_pose = pose.copy()
        local_pose[:3, 3] -= self._origin

        # Every frame's points go into the map, but a batch at a time: each
        # flush writes one drawable that is never touched again, so the frame
        # time stays flat however long the sequence runs.
        if self._show_map:
            pts = _voxel(frame.points, MAP_VOXEL)
            if pts.size:
                self._pending.append(self._local(_to_world(pts, pose)))
            if len(self._pending) >= MAP_FLUSH:
                self._flush_map()

        # The scan and its keypoints are height-coded against one shared range,
        # so a colour means the same height in both: grey for the cloud, which
        # is bulk and must not compete, turbo for the markers on top of it.
        scan = self._local(_to_world(_voxel(frame.points, CUR_VOXEL), pose))
        span = ((float(scan[:, 2].min()), float(scan[:, 2].max()))
                if len(scan) else None)
        if self._show_scan and len(scan):
            self._draw_colored("cur/cloud", scan, _GREY, span, CLOUD_SIZE)
        else:
            self._viewer.remove_drawable("cur/cloud")

        if self._show_kp and len(kp_sensor):
            self._draw_colored("cur/kp", self._local(_to_world(kp_sensor, pose)),
                               _TURBO, span, KP_SIZE)
        else:
            self._viewer.remove_drawable("cur/kp")

        self._viewer.update_drawable(
            "cur/pose", self._axis(),
            self._guik.VertexColor(local_pose.astype(np.float32)))

        # Trajectory: nodes stay visible, and the polyline joins them.
        nodes = self._local(np.asarray([p[:3, 3] for p in q_poses]))
        self._viewer.update_points(
            "traj/nodes", nodes,
            self._guik.FlatColor(*TRAJ_COLOR).set_point_shape(NODE_SIZE, True, True))
        if len(nodes) > 1:
            self._viewer.update_thin_lines(
                "traj/line", nodes, line_strip=True,
                shader_setting=self._guik.FlatColor(*TRAJ_COLOR))

        # Closures: one persistent edge per accepted pair.
        db_nodes = (self._local(np.asarray(db_poses, dtype=np.float64)[:, :3, 3])
                    if db_poses is not None else nodes)
        for q, d, _score, _rank in accepted:
            edge = np.ascontiguousarray([nodes[q], db_nodes[d]], dtype=np.float32)
            self._viewer.update_thin_lines(
                f"closure/{q}_{d}", edge,
                shader_setting=self._guik.FlatColor(*CLOSURE_COLOR))

        if self._follow:
            self._viewer.lookat(nodes[-1])

        self._panels(tokens)
        self._pump()

    # -- panels ------------------------------------------------------------

    def _panels(self, tokens) -> None:
        """The three descriptor stages, in the order the pipeline uses them."""
        from inlier.viz.descriptors import describe

        if self._grid is None or not len(tokens.token_id):
            return
        N_h, N_r, N_s, N_a = self._grid
        d = describe(tokens.token_id, N_h, N_r, N_s, N_a)

        # Colour map and scale per panel, both taken from the matplotlib figure
        # `inlier encode --viz` writes, so the same descriptor looks the same
        # in both.  H and R autoscale there; A is pinned to 0..N_h, which is
        # the popcount's real range -- autoscaling it would paint a cell
        # holding 3 of 10 slices in the colour of a full one.
        panels = (
            # H: the token histogram, azimuth collapsed. (N_h, N_r*N_s)
            ("H  histogram", d.full, _VIRIDIS, None),
            # R: the row stage 1 actually scores, height collapsed too.  One
            # row of N_r*N_s, tiled so it reads as a band and not a hairline.
            ("R  MINT row", np.tile(d.compact[None, :], (MINT_ROWS, 1)),
             _VIRIDIS, None),
            # A: how many height slices each (radial, azimuth) cell holds.  The
            # name is the one `inlier encode --viz` gives the same array.
            ("A  BEAM elevation", d.beam_popcount, _INFERNO,
             (0.0, float(max(N_h, 1)))),
        )
        for name, values, lut, span in panels:
            # Per-panel magnification, so all three come out the same width
            # however many columns they happen to have.  Most of it is done in
            # numpy, at whole cells: what is left for `scale` is the fractional
            # remainder, too small to blur anything visibly.
            cols = max(1, values.shape[1])
            block = max(1, PANEL_WIDTH_PX // cols)
            image = _bgr(_blocks(_image_u8(values, lut, span), block))
            scale = PANEL_WIDTH_PX / (cols * block)
            self._viewer.update_image(
                name, self._glk.create_texture(image),
                scale=scale, order=PANEL_ORDER[name])

    # -- ui ----------------------------------------------------------------

    def _ui(self) -> None:
        imgui = self._imgui
        imgui.text(f"frame {self._index + 1} / {self._n_frames}")
        imgui.text(f"closures {self._n_closures}   "
                   f"best {self._best_score:.3f} / thr {self._threshold:.3f}")
        imgui.separator()

        if imgui.button("pause" if self._playing else "play"):
            self._playing = not self._playing
        imgui.same_line()
        if imgui.button("step"):
            self._step = True
        imgui.same_line()
        if imgui.button("close"):
            self._viewer.close()

        _, self._fps_cap = imgui.slider_float("fps (0 = free)", self._fps_cap,
                                              0.0, 60.0, "%.0f")
        _, self._show_scan = imgui.checkbox("scan", self._show_scan)
        imgui.same_line()
        _, self._show_kp = imgui.checkbox("keypoints", self._show_kp)
        imgui.same_line()
        _, self._follow = imgui.checkbox("follow", self._follow)
        # Acted on here rather than in `on_frame`: the UI callback runs inside
        # `spin_once`, so it still fires while the run is paused -- which is
        # exactly when someone is most likely to reach for this.
        changed, self._show_map = imgui.checkbox("map", self._show_map)
        if changed and not self._show_map:
            self._clear_map()
        imgui.separator()
        # This frame only.  `voxel` is separate from `encode` because it is not
        # a rounding error next to it -- the downsample in front of the encoder
        # can cost several times the encoding.
        for key, label in STAGE_LABELS:
            imgui.text(f"{label:<7}{self._frame_times.get(key, 0.0) * 1e3:8.1f} ms")
        imgui.separator()
        total = sum(self._frame_times.get(key, 0.0) for key, _ in STAGE_LABELS)
        imgui.text(f"{'total':<7}{total * 1e3:8.1f} ms")

    def _pump(self) -> None:
        """Render, and honour pause / step / the speed cap."""
        if self._fps_cap > 0.0:
            wait = 1.0 / self._fps_cap - (time.perf_counter() - self._last_frame_at)
            deadline = time.perf_counter() + max(0.0, wait)
            while time.perf_counter() < deadline and self._viewer.spin_once():
                pass
        self._last_frame_at = time.perf_counter()

        if not self._viewer.spin_once():
            return
        while not self._playing and not self._step:
            if not self._viewer.spin_once():
                return
        self._step = False
