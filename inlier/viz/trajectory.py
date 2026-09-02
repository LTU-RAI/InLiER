"""The trajectory figure an evaluation run leaves behind.

One line drawn per decision the run made at its operating threshold: green for
a true positive, red for a false one.  It is the one artifact that shows
*where* in the sequence the matcher succeeded and failed, which no scalar in
``results_*.json`` can.

Two layouts, because the two protocol families mean different things by "z":

``write_trajectory_plot``
    Cross-session.  Two sessions stacked -- database at 0, query at
    ``z_offset`` -- and every edge crosses the gap between them.

``write_time_trajectory_plot``
    Single-session (online loop closure).  There is only one trajectory, so
    stacking it against itself would draw the same curve twice and every edge
    would be a meaningless vertical line.  Instead z *is* the frame index, so
    the curve climbs as the session runs and the height a closure edge spans
    is exactly how long the loop took to come back around.

The edge lists come straight from :func:`inlier.eval.metrics.confusion`, so the
picture cannot disagree with the confusion counts it is titled with.  One
asymmetry to keep in mind when reading it: a query with no ground-truth
positive that still produces a match counts in ``FP`` but has no edge to draw
(there is no correct database scan to draw it to), so the legend's FP tally can
be lower than the title's.  That is the population difference, not a bug.

matplotlib is imported inside the functions: ``inlier.viz`` must stay
importable without the ``[eval]`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

Z_OFFSET = 10.0
DPI = 200
DB_COLOR = "#1f77b4"
Q_COLOR = "#ff7f0e"
TP_COLOR = "g"
FP_COLOR = "r"


def _xyz(positions: np.ndarray, z) -> np.ndarray:
    """(N, 3) plotting coordinates: XY from the poses, z as given."""
    p = np.asarray(positions, dtype=float)
    z = np.full(len(p), float(z)) if np.isscalar(z) else np.asarray(z, dtype=float)
    return np.column_stack([p[:, 0], p[:, 1], z])


def _draw_edges(ax, q_xyz: np.ndarray, db_xyz: np.ndarray,
                tp_edges: Sequence[Tuple[int, int]],
                fp_edges: Sequence[Tuple[int, int]]) -> None:
    """Match edges as ``(query_index, db_index)`` pairs, in plotting space.

    False positives are drawn first so a true positive lying on top of one
    stays visible.
    """
    for color, edges in ((FP_COLOR, fp_edges), (TP_COLOR, tp_edges)):
        for qi, di in edges:
            ax.plot([q_xyz[qi, 0], db_xyz[di, 0]],
                    [q_xyz[qi, 1], db_xyz[di, 1]],
                    [q_xyz[qi, 2], db_xyz[di, 2]],
                    color=color, linewidth=1.0, alpha=0.35)


def _finish(fig, ax, path: Path, title: str, z_label: str,
            tp_edges: Sequence[Tuple[int, int]],
            fp_edges: Sequence[Tuple[int, int]], dpi: int) -> Path:
    """Label, legend, save, close.  Returns the path written."""
    # The edges are drawn at alpha 0.35; these opaque stubs give the legend a
    # readable swatch and carry the counts.
    if tp_edges:
        ax.plot([], [], [], color=TP_COLOR, linewidth=1.5,
                label=f"TP ({len(tp_edges)})")
    if fp_edges:
        ax.plot([], [], [], color=FP_COLOR, linewidth=1.5,
                label=f"FP ({len(fp_edges)})")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)

    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _new_axes():
    # No backend is forced: matplotlib already falls back to Agg when there is
    # no display, and savefig never needs a window.  Forcing one here would
    # clobber the backend of a caller that has its own.
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 10))
    return fig, fig.add_subplot(111, projection="3d")


def write_trajectory_plot(
    path: Path,
    db_positions: np.ndarray,
    q_positions: np.ndarray,
    tp_edges: Sequence[Tuple[int, int]],
    fp_edges: Sequence[Tuple[int, int]],
    title: str,
    *,
    z_offset: float = Z_OFFSET,
    dpi: int = DPI,
) -> Path:
    """Render two stacked trajectories and their match edges to ``path``.

    Edges are ``(query_index, db_index)`` pairs, the orientation
    ``metrics.confusion`` returns them in.  Returns the path written.
    """
    fig, ax = _new_axes()
    db_xyz = _xyz(db_positions, 0.0)
    q_xyz = _xyz(q_positions, z_offset)

    ax.plot(db_xyz[:, 0], db_xyz[:, 1], db_xyz[:, 2],
            color=DB_COLOR, linewidth=1.5, alpha=0.75, label="Database")
    ax.plot(q_xyz[:, 0], q_xyz[:, 1], q_xyz[:, 2],
            color=Q_COLOR, linewidth=1.5, alpha=0.75, label="Query")

    _draw_edges(ax, q_xyz, db_xyz, tp_edges, fp_edges)
    return _finish(fig, ax, path, title, "Session offset", tp_edges, fp_edges, dpi)


def write_time_trajectory_plot(
    path: Path,
    positions: np.ndarray,
    tp_edges: Sequence[Tuple[int, int]],
    fp_edges: Sequence[Tuple[int, int]],
    title: str,
    *,
    times: Optional[np.ndarray] = None,
    z_label: str = "Frame index",
    dpi: int = DPI,
) -> Path:
    """Render one trajectory with time on z, plus its loop-closure edges.

    The single-session layout.  ``times`` defaults to the frame index, which is
    what the descriptor cache can always supply -- it stores poses, not
    timestamps.  Pass real timestamps (and a matching ``z_label``) when the
    caller has them.

    Because both endpoints of an edge sit at their own height, a closure drawn
    here is legible as a revisit: it spans the frames between leaving a place
    and returning to it.  Edges are ``(query_index, db_index)`` pairs, and the
    database *is* the query sequence, so both index the same trajectory.
    """
    positions = np.asarray(positions, dtype=float)
    if times is None:
        times = np.arange(len(positions), dtype=float)

    fig, ax = _new_axes()
    xyz = _xyz(positions, times)

    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
            color=DB_COLOR, linewidth=1.5, alpha=0.75, label="Session")

    _draw_edges(ax, xyz, xyz, tp_edges, fp_edges)
    return _finish(fig, ax, path, title, z_label, tp_edges, fp_edges, dpi)
