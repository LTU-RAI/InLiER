"""The trajectory figure an evaluation run leaves behind.

Two sessions stacked in z -- database at 0, query at ``z_offset`` -- with one
line drawn per decision the run made at its operating threshold: green for a
true positive, red for a false one.  It is the one artifact that shows *where*
in the sequence the matcher succeeded and failed, which no scalar in
``results_*.json`` can.

The edge lists come straight from :func:`inlier.eval.metrics.confusion`, so the
picture cannot disagree with the confusion counts it is titled with.  One
asymmetry to keep in mind when reading it: a query with no ground-truth
positive that still produces a match counts in ``FP`` but has no edge to draw
(there is no correct database scan to draw it to), so the legend's FP tally can
be lower than the title's.  That is the population difference, not a bug.

matplotlib is imported inside the function: ``inlier.viz`` must stay importable
without the ``[eval]`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

Z_OFFSET = 10.0
DPI = 200
DB_COLOR = "#1f77b4"
Q_COLOR = "#ff7f0e"
TP_COLOR = "g"
FP_COLOR = "r"


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
    """Render the two trajectories and their match edges to ``path``.

    Edges are ``(query_index, db_index)`` pairs, the orientation
    ``metrics.confusion`` returns them in.  Returns the path written.
    """
    # No backend is forced: matplotlib already falls back to Agg when there is
    # no display, and savefig never needs a window.  Forcing one here would
    # clobber the backend of a caller that has its own.
    import matplotlib.pyplot as plt

    db_positions = np.asarray(db_positions, dtype=float)
    q_positions = np.asarray(q_positions, dtype=float)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(db_positions[:, 0], db_positions[:, 1], np.zeros(len(db_positions)),
            color=DB_COLOR, linewidth=1.5, alpha=0.75, label="Database")
    ax.plot(q_positions[:, 0], q_positions[:, 1],
            np.full(len(q_positions), z_offset),
            color=Q_COLOR, linewidth=1.5, alpha=0.75, label="Query")

    # False positives first, so a true positive on top of one stays visible.
    for color, edges in ((FP_COLOR, fp_edges), (TP_COLOR, tp_edges)):
        for qi, di in edges:
            ax.plot([q_positions[qi, 0], db_positions[di, 0]],
                    [q_positions[qi, 1], db_positions[di, 1]],
                    [z_offset, 0.0],
                    color=color, linewidth=1.0, alpha=0.35)

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
    ax.set_zlabel("Session offset")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
