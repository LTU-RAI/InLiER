"""Visualisation helpers.

Kept out of ``inlier.core``: nothing here is needed to encode or match, and
the figure code needs matplotlib, which lives in the ``[eval]`` extra.  The
submodules import matplotlib lazily, so ``import inlier.viz`` alone is safe
on a bare install.
"""

from inlier.viz.descriptors import (  # noqa: F401
    Descriptors,
    describe,
    occupancy,
    popcount,
    shape_class_labels,
)

__all__ = ["Descriptors", "describe", "occupancy", "popcount",
           "shape_class_labels", "encode_figure", "write_trajectory_plot",
           "write_time_trajectory_plot", "write_score_matrix_figure",
           "LiveViewer", "StubViewer"]


def __getattr__(name: str):
    if name == "encode_figure":
        from inlier.viz.figures import encode_figure

        return encode_figure
    if name == "write_trajectory_plot":
        from inlier.viz.trajectory import write_trajectory_plot

        return write_trajectory_plot
    if name == "write_time_trajectory_plot":
        from inlier.viz.trajectory import write_time_trajectory_plot

        return write_time_trajectory_plot
    if name == "write_score_matrix_figure":
        from inlier.viz.scores import write_score_matrix_figure

        return write_score_matrix_figure
    if name in ("LiveViewer", "StubViewer"):
        from inlier.viz import live

        return getattr(live, name)
    raise AttributeError(f"module 'inlier.viz' has no attribute {name!r}")
