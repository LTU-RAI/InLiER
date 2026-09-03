"""The ``inlier match`` figure: two encoded scans, side by side and combined.

Reads top to bottom the way the pipeline runs.  The left column is the query,
the middle is the database entry, and the right column carries the answer: the
two clouds overlaid under the estimated pose, above the score each stage gave
the pair.

The descriptor rows are the same three matrices ``inlier encode --viz`` draws,
in the same order and on the same colour scale, because the point of putting
them next to each other is to see *where* two scans agree.  A shared scale is
what makes that comparison honest -- per-panel autoscaling would make a sparse
scan look as bright as a dense one.

The panel drawing is reused from :mod:`inlier.viz.figures` rather than copied,
so the two commands cannot drift apart.

matplotlib is imported inside the functions, as everywhere in ``inlier.viz``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from inlier.viz.descriptors import describe
from inlier.viz.figures import (CLOUD_COLOR, MAX_CLOUD_POINTS, _draw_beam,
                                _draw_full, _draw_mint, _extent, _subsample,
                                _to_ground_frame)

Q_COLOR = "#1f77b4"      # query
D_COLOR = "#ff7f0e"      # database
OK_COLOR = "#2ca02c"
BAD_COLOR = "#d62728"


def _apply(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """``p_db = T @ p_query``, for an (N, 3) array."""
    return (np.asarray(T)[:3, :3] @ np.asarray(points).T).T + np.asarray(T)[:3, 3]


def _bev(ax, cloud, kp, color, title: str, extent: float) -> None:
    """Top view only: the query/DB panels are for locating keypoints.

    ``extent`` is passed in rather than fitted per panel, so all three
    geometry panels share one scale.  Two scans drawn at different zooms
    cannot be compared by eye, which is the only thing this row is for.
    """
    if cloud is not None and len(cloud):
        ax.scatter(cloud[:, 0], cloud[:, 1], s=0.4, c=CLOUD_COLOR, alpha=0.4,
                   linewidths=0.0, rasterized=True, zorder=1)
    if len(kp):
        ax.scatter(kp[:, 0], kp[:, 1], s=14.0, c=color, linewidths=0.0,
                   zorder=3, label=f"{len(kp)} keypoints")
    _square(ax, extent, title)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)


def _square(ax, extent: float, title: str) -> None:
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=10)


def _draw_alignment(ax, q_cloud, q_kp, db_cloud, db_kp, pose, extent) -> None:
    """Both scans in the DB frame, with the query moved by the estimated pose.

    When there is no pose -- verification failed, or was skipped -- the two are
    still drawn, unmoved, so the panel shows what the matcher was looking at
    rather than going blank.
    """
    moved_cloud = None if q_cloud is None else (
        _apply(pose, q_cloud) if pose is not None else q_cloud)
    moved_kp = _apply(pose, q_kp) if (pose is not None and len(q_kp)) else q_kp

    for cloud, color in ((db_cloud, D_COLOR), (moved_cloud, Q_COLOR)):
        if cloud is not None and len(cloud):
            ax.scatter(cloud[:, 0], cloud[:, 1], s=0.4, c=color, alpha=0.30,
                       linewidths=0.0, rasterized=True, zorder=1)
    if len(db_kp):
        ax.scatter(db_kp[:, 0], db_kp[:, 1], s=14.0, c=D_COLOR,
                   linewidths=0.0, zorder=3, label="database")
    if len(moved_kp):
        ax.scatter(moved_kp[:, 0], moved_kp[:, 1], s=14.0, c=Q_COLOR,
                   linewidths=0.0, zorder=4,
                   label="query, transformed" if pose is not None else "query (no pose)")

    _square(ax, extent, "query aligned onto database" if pose is not None
            else "combined -- no pose estimated")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)


def aligned_query(pose, q_kp):
    """Query keypoints in the DB frame; unmoved when there is no pose."""
    return _apply(pose, q_kp) if (pose is not None and len(q_kp)) else q_kp


def _score_rows(result, cfg):
    """``(label, value, ceiling, detail)`` per stage, in pipeline order.

    ``ceiling`` is what the bar is drawn against.  Every score here is already
    a ratio in [0, 1], so the bars are directly comparable -- which is the
    whole reason to draw them rather than print four numbers.
    """
    rows = [("stage 1  MINT", result.mint, "cosine over the MINT row")]
    if result.beam is not None:
        rows.append(("stage 2  BEAM", result.beam,
                     f"bit Jaccard, yaw shift {result.beam_shift}/{cfg.N_a} "
                     f"({result.beam_shift * 360.0 / cfg.N_a:.0f}°)"))
    if result.rerank is not None:
        rows.append(("rerank", result.rerank,
                     f"4-D histogram, shift {result.rerank_shift}"))
    v = result.verify
    if v is not None:
        rows.append(("verify", float(v.keypoint_inlier_ratio),
                     f"{v.n_keypoint_inliers}/{v.n_total_keypoints} keypoint "
                     f"inliers, {v.n_ransac_inliers}/{v.n_correspondences} RANSAC"))
    return rows


def _draw_bars(ax, rows) -> None:
    """One horizontal bar per stage.

    Every score here is a ratio in [0, 1] -- a cosine, a Jaccard, an inlier
    fraction -- so the bars share an axis and can be read against each other.
    That is the reason to draw them rather than print four numbers.
    """
    n = len(rows)
    y = np.arange(n)[::-1]
    values = [max(0.0, min(1.0, float(r[1]))) for r in rows]
    ax.barh(y, values, color=Q_COLOR, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("score", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for yi, value in zip(y, values):
        ax.text(min(value + 0.02, 0.995), yi, f"{value:.4f}",
                va="center", ha="left" if value < 0.9 else "right", fontsize=8)
    ax.set_title("stage scores", fontsize=10)


def _draw_details(ax, result, rows) -> None:
    """The numbers behind the bars, plus the verdict and the pose."""
    ax.set_axis_off()
    lines = [f"{label:<14} {detail}" for label, _, detail in rows]

    v = result.verify
    if v is not None:
        lines.append("")
        lines.append("VERIFIED" if v.success else "NOT VERIFIED")
        if v.success:
            lines.append(f"  yaw  {np.degrees(v.yaw):+8.3f} deg")
            lines.append(f"  t    [{v.tx:+.3f}, {v.ty:+.3f}, {v.tz:+.3f}] m")
            lines.append(f"  RMSE {v.inlier_rmse:.4f} m")
    if result.gicp is not None:
        g = result.gicp
        lines.append("")
        lines.append(f"GICP on {result.gicp_on}")
        lines.append("  " + ("converged" if g.converged else "did NOT converge")
                     + f" in {g.n_iterations} iters")
        lines.append(f"  {g.n_inliers:,} inliers, error {g.final_error:.4f}")

    color = "0.15"
    if v is not None and not v.success:
        color = BAD_COLOR
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5, family="monospace",
            color=color, linespacing=1.55)


def _shared_scale(a, b):
    """One vmin/vmax for a pair of panels, so brightness means the same thing."""
    top = max(float(a.max()) if a.size else 0.0, float(b.max()) if b.size else 0.0)
    return {"vmin": 0.0, "vmax": max(top, 1.0)}


def match_figure(
    query,
    db,
    result,
    cfg,
    *,
    q_points: Optional[np.ndarray] = None,
    db_points: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    max_cloud_points: int = MAX_CLOUD_POINTS,
):
    """Render one pair: geometry, both descriptor stacks, and the scores.

    ``query``/``db`` are :class:`~inlier.eval.pair.EncodedScan`, ``result`` a
    :class:`~inlier.eval.pair.PairResult`.  Point clouds are optional: an
    ``.npz`` stores tokens and keypoints, not the scan, so the cloud is drawn
    only when the caller could recover it.  Returns the ``Figure``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    q_desc = describe(query.token_id, *(cfg.N_h, cfg.N_r, cfg.N_s, cfg.N_a))
    d_desc = describe(db.token_id, *(cfg.N_h, cfg.N_r, cfg.N_s, cfg.N_a))

    def cloud_of(points, T_ground):
        if points is None or not len(points):
            return None
        return _to_ground_frame(
            _subsample(np.asarray(points, dtype=np.float64), max_cloud_points),
            T_ground)

    q_cloud = cloud_of(q_points, query.T_ground)
    d_cloud = cloud_of(db_points, db.T_ground)
    q_kp = np.asarray(query.kp_aligned, dtype=np.float64)
    d_kp = np.asarray(db.kp_aligned, dtype=np.float64)

    # One extent for all three geometry panels, including the transformed
    # query, so nothing in the row is silently drawn at a different zoom.
    moved_kp = aligned_query(result.pose, q_kp)
    moved_cloud = (None if q_cloud is None else
                   (_apply(result.pose, q_cloud) if result.pose is not None
                    else q_cloud))
    spatial = [a[:, :2] for a in (q_cloud, d_cloud, moved_cloud,
                                  q_kp, d_kp, moved_kp)
               if a is not None and len(a)]
    extent = _extent(*spatial, cfg.xy_max * 2.5) if spatial else float(cfg.xy_max)

    fig = plt.figure(figsize=(19, 13))
    # wspace is generous because the histogram colourbars sit between the
    # columns; at the default they collide with the next panel's y labels.
    grid = GridSpec(4, 3, figure=fig, height_ratios=[3.0, 1.25, 0.30, 1.9],
                    hspace=0.55, wspace=0.42)

    _bev(fig.add_subplot(grid[0, 0]), q_cloud, q_kp, Q_COLOR,
         f"query: {query.label}", extent)
    _bev(fig.add_subplot(grid[0, 1]), d_cloud, d_kp, D_COLOR,
         f"database: {db.label}", extent)
    _draw_alignment(fig.add_subplot(grid[0, 2]), q_cloud, q_kp, d_cloud, d_kp,
                    result.pose, extent)

    # The descriptor rows share a colour scale across the two columns: a
    # difference in brightness must mean a difference in the descriptors,
    # not in how each panel autoscaled itself.
    full_scale = _shared_scale(q_desc.full, d_desc.full)
    mint_scale = _shared_scale(q_desc.compact, d_desc.compact)
    for col, desc, name in ((0, q_desc, "query"), (1, d_desc, "database")):
        ax_full = fig.add_subplot(grid[1, col])
        _draw_full(fig, ax_full, desc, cfg, **full_scale)
        ceiling = (f"   ceiling $h_b$={desc.max_hb}"
                   if 0 <= desc.max_hb < cfg.N_h - 1 else "")
        # The encode-figure titles carry the panel dimensions, which here
        # would be printed twice and run into the next column.  The shapes
        # move to the suptitle; the column keeps its identity.
        ax_full.set_title(f"$\\mathcal{{H}}$  token histogram -- {name}{ceiling}",
                          fontsize=10)

        ax_mint = fig.add_subplot(grid[2, col])
        _draw_mint(ax_mint, desc, cfg, **mint_scale)
        ax_mint.set_title(f"$\\mathcal{{R}}$  MINT row (stage 1) -- {name}",
                          fontsize=10)
        ax_mint.set_xlabel(f"$r_b$  ({cfg.r_max / cfg.N_r:.3g} m/bin)", fontsize=9)

        ax_beam = fig.add_subplot(grid[3, col])
        _draw_beam(fig, ax_beam, desc, cfg)
        ax_beam.set_title(f"$\\mathcal{{A}}$  BEAM codes (stage 2) -- {name}",
                          fontsize=10)

    rows = _score_rows(result, cfg)
    _draw_bars(fig.add_subplot(grid[1, 2]), rows)
    _draw_details(fig.add_subplot(grid[2:, 2]), result, rows)

    shapes = (f"$N_h$={cfg.N_h}  $N_r$={cfg.N_r}  $N_s$={cfg.N_s}  "
              f"$N_a$={cfg.N_a}")
    head = title or f"{query.label}   vs   {db.label}"
    fig.suptitle(f"{head}\n{len(query.token_id)} vs {len(db.token_id)} tokens"
                 f"   ({shapes})", fontsize=13)
    return fig
