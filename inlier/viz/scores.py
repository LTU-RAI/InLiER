"""The per-stage score matrices, drawn.

One panel per stage, in pipeline order, each an N_q x N_db image of what that
stage scored.  Reading them side by side is the point: the funnel narrows
visibly, and a candidate that MINT lit up but BEAM did not shows as a bright
cell in one panel and an unscored one in the next.

These are **not** confusion matrices, however much the shape suggests it.  No
threshold has been applied and no decision is recorded -- there are no labels
to be confused about.  Which closures a run accepted lives in
``closures_*.csv``, and the figure says so under the title so the two are not
read as the same thing.

The one rule the drawing has to respect is the one the arrays respect: **a cell
a stage never scored is not a cell it scored zero.**  ``0.0`` is a real score
(a failed verification returns exactly that), so unscored cells get their own
colour via ``cmap.set_bad`` rather than the bottom of the ramp, and the caption
names it.

matplotlib is imported inside the function, as everywhere in ``inlier.viz``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

#: Stages to draw, in the order the pipeline runs them.  ``beam_shift`` is
#: deliberately absent: it is an azimuth index, not a similarity, and putting
#: it on a 0-1 ramp beside four scores would invite reading it as one.
STAGE_ORDER = ("mint", "beam", "rerank", "verify")

STAGE_TITLES = {
    "mint": "MINT  (stage 1)",
    "beam": "BEAM  (stage 2)",
    "rerank": "rerank",
    "verify": "verify",
}

#: What each stage's number actually is, so the colourbars cannot be read as
#: three measurements of one thing.
METRIC_NAMES = {
    "mint": "MINT similarity",
    "beam": "bit Jaccard",
    "rerank": "rerank score",
    "verify": "keypoint inlier ratio",
}

#: Colour for a pair the stage never scored.  Deliberately not on the ramp.
UNSCORED_COLOR = "#e8e4dc"

#: Above this many cells, drop to nearest-neighbour subsampling before
#: rasterising.  A 4541-frame session is 20M cells per panel; matplotlib will
#: render it, slowly, into an image far larger than any screen.
MAX_CELLS = 4_000_000

DPI = 200


def _subsample(m: np.ndarray, max_cells: int = MAX_CELLS):
    """``(image, step)`` -- thinned by an integer stride if it is very large.

    Nearest-neighbour on purpose: averaging would blend scored cells with
    unscored ones and turn ``NaN`` into a number, which is the one thing these
    panels must not do.
    """
    if m.size <= max_cells:
        return m, 1
    step = int(np.ceil(np.sqrt(m.size / max_cells)))
    return m[::step, ::step], step


def write_score_matrix_figure(
    path: Path,
    matrices: Dict[str, np.ndarray],
    title: str = "",
    *,
    stages: Sequence[str] = STAGE_ORDER,
    dpi: int = DPI,
) -> Optional[Path]:
    """Render every stage's score matrix to one page.  Returns the path.

    ``None`` when there is nothing to draw, so a caller need not special-case
    a run whose stages were all skipped.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    panels = [(name, matrices[name]) for name in stages
              if name in matrices and np.isfinite(matrices[name]).any()]
    if not panels:
        return None

    cmap = colormaps["viridis"].with_extremes(bad=UNSCORED_COLOR)

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(5.4 * len(panels), 5.8), squeeze=False)
    for ax, (name, matrix) in zip(axes[0], panels):
        image, step = _subsample(matrix)
        # Scaled per panel, floor pinned at 0 so dark still means low.  A
        # shared 0-1 ramp is the tempting choice and the wrong one: the
        # stages do not measure the same quantity -- MINT is an L1 histogram
        # intersection, BEAM a bit-level Jaccard, verify a keypoint inlier
        # ratio -- so one scale would assert a comparability that does not
        # exist, and would flatten the two later panels into solid dark.  The
        # colourbar carries each panel's actual top instead.
        top = float(np.nanmax(matrix))
        art = ax.imshow(image, cmap=cmap, vmin=0.0, vmax=max(top, 1e-6),
                        interpolation="nearest", origin="upper", aspect="equal")
        scored = int(np.isfinite(matrix).sum())
        zeros = int((matrix == 0.0).sum())
        ax.set_title(
            f"{STAGE_TITLES.get(name, name)}\n"
            f"{scored:,} of {matrix.size:,} pairs scored"
            + (f"  ({zeros:,} scored 0.0)" if zeros else "")
            + (f"\nshown every {step}th frame" if step > 1 else ""),
            fontsize=10)
        ax.set_xlabel("database index")
        ax.set_ylabel("query index")
        fig.colorbar(art, ax=ax, fraction=0.046, pad=0.02).set_label(
            f"{METRIC_NAMES.get(name, 'score')}   (max {top:.3f})", fontsize=9)

    caption = (f"raw stage scores -- no threshold applied, no decision "
               f"recorded (accepted closures are in closures_*.csv)\n"
               f"pale cells were never scored by that stage, which is not the "
               f"same as scoring 0.0;  colour scales are per panel, since the "
               f"stages measure different quantities")
    fig.suptitle(f"{title}\n{caption}" if title else caption, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
