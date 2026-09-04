"""The ``inlier encode --viz`` figure.

One page per scan, in the ground-aligned frame the encoder actually bins
in, so the panels line up: a keypoint at radius *r* in the top-left plot
lands in row *r_b* of the BEAM panel below it.

The geometry panels colour keypoints by height slice ``h_b``.  That is free
information -- ``unpack_token_ids`` recovers ``(h_b, r_b, s_b, a_b)`` per
token, and in ``point_mode="keypoints"`` the token array is index-aligned
with the keypoint array.  In ``"all_points"`` mode it is not, and the
keypoints are drawn in a single colour with a note saying why.

matplotlib is imported inside the functions: ``inlier.viz`` must stay
importable without the ``[eval]`` extra, so that ``inlier encode`` without
``--viz`` never needs it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from inlier.viz.descriptors import Descriptors, describe, occupancy, shape_class_labels

MAX_CLOUD_POINTS = 60_000
CLOUD_COLOR = "0.72"
CEILING_COLOR = "#d62728"
SLAB_COLOR = "#1f77b4"


def _subsample(points: np.ndarray, limit: int) -> np.ndarray:
    """Thin a dense cloud for scatter plotting, deterministically."""
    if points.shape[0] <= limit:
        return points
    rng = np.random.default_rng(0)
    keep = rng.choice(points.shape[0], limit, replace=False)
    return points[np.sort(keep)]


def _to_ground_frame(points: np.ndarray, T_ground: np.ndarray) -> np.ndarray:
    R, t = T_ground[:3, :3], T_ground[:3, 3]
    return (R @ points.T).T + t


def encode_figure(
    points: np.ndarray,
    keypoints,
    tokens,
    cfg,
    *,
    title: Optional[str] = None,
    max_cloud_points: int = MAX_CLOUD_POINTS,
):
    """Render scan, keypoints and descriptors for one encoded scan.

    Returns the matplotlib ``Figure``; the caller decides whether to show
    it or write it out.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    N_h, N_r, N_s, N_a = cfg.N_h, cfg.N_r, cfg.N_s, cfg.N_a
    desc = describe(tokens.token_id, N_h, N_r, N_s, N_a)

    cloud = _to_ground_frame(
        _subsample(np.asarray(points, dtype=np.float64), max_cloud_points),
        keypoints.T_ground)
    kp = np.asarray(keypoints.p_aligned, dtype=np.float64)

    # Only index-aligned in "keypoints" mode; see the module docstring.
    per_keypoint = len(tokens) == len(keypoints) and kp.shape[0] > 0
    kp_color = desc.hb if per_keypoint else None

    fig = plt.figure(figsize=(16, 12.5))
    grid = GridSpec(4, 4, figure=fig, height_ratios=[3.0, 1.15, 0.3, 1.9],
                    hspace=0.45, wspace=0.34)
    ax_bev = fig.add_subplot(grid[0, 0:2])
    ax_side = fig.add_subplot(grid[0, 2:4])
    ax_full = fig.add_subplot(grid[1, :])
    ax_mint = fig.add_subplot(grid[2, :])
    ax_beam = fig.add_subplot(grid[3, 0:2])
    ax_shape = fig.add_subplot(grid[3, 2])
    ax_slice = fig.add_subplot(grid[3, 3])

    _draw_bev(ax_bev, cloud, kp, kp_color, cfg)
    _draw_side(ax_side, cloud, kp, kp_color, cfg)
    _draw_full(fig, ax_full, desc, cfg)
    _draw_mint(ax_mint, desc, cfg)
    _draw_beam(fig, ax_beam, desc, cfg)
    _draw_shape(ax_shape, desc, cfg)
    _draw_slices(ax_slice, desc, cfg)

    if per_keypoint and kp.shape[0]:
        from matplotlib.ticker import MaxNLocator

        bar = fig.colorbar(ax_bev.collections[-1], ax=[ax_bev, ax_side],
                           fraction=0.025, pad=0.01)
        bar.locator = MaxNLocator(integer=True)
        bar.update_ticks()
        bar.set_label("height slice $h_b$")
    elif not per_keypoint:
        ax_bev.text(0.02, 0.02,
                    f"point_mode={cfg.point_mode!r}: {len(tokens)} tokens for "
                    f"{len(keypoints)} keypoints, not index-aligned",
                    transform=ax_bev.transAxes, fontsize=8, color=CEILING_COLOR)

    fig.suptitle(_headline(title, points, keypoints, tokens, desc, cfg),
                 fontsize=13)
    return fig


def _headline(title, points, keypoints, tokens, desc: Descriptors, cfg) -> str:
    occ = occupancy(desc, cfg.N_h, cfg.N_r, cfg.N_s, cfg.N_a)
    stats = (f"{len(points):,} pts  ->  {len(keypoints)} keypoints, "
             f"{len(tokens)} tokens, {np.unique(desc.hb).size}/{cfg.N_h} slices "
             f"occupied  ({occ * 100:.3f}% of the "
             f"{cfg.N_h * cfg.N_r * cfg.N_s * cfg.N_a:,}-cell token space)")
    return f"{title}\n{stats}" if title else stats


def _scatter_keypoints(ax, xy, color, cfg, size=16.0):
    if xy.shape[0] == 0:
        return None
    kwargs = dict(s=size, linewidths=0.0, zorder=3)
    if color is None:
        return ax.scatter(xy[:, 0], xy[:, 1], c=CEILING_COLOR, **kwargs)
    return ax.scatter(xy[:, 0], xy[:, 1], c=color, cmap="viridis",
                      vmin=0, vmax=max(cfg.N_h - 1, 1), **kwargs)


def _draw_bev(ax, cloud, kp, kp_color, cfg) -> None:
    """Top-down view with the ROI square and the radial bin rings."""
    from matplotlib.patches import Circle, Rectangle

    ax.scatter(cloud[:, 0], cloud[:, 1], s=0.4, c=CLOUD_COLOR, alpha=0.4,
               linewidths=0.0, rasterized=True, zorder=1)

    # Rings first, so the ROI edges read over them.
    if cfg.N_r <= 40:
        for i in range(1, cfg.N_r + 1):
            ax.add_patch(Circle((0, 0), cfg.r_max * i / cfg.N_r, fill=False,
                                edgecolor="0.55", lw=0.4, alpha=0.35, zorder=2))
    ax.add_patch(Circle((0, 0), cfg.r_max, fill=False, edgecolor=SLAB_COLOR,
                        lw=1.2, ls="--", zorder=4))
    ax.add_patch(Rectangle((-cfg.xy_max, -cfg.xy_max), 2 * cfg.xy_max,
                           2 * cfg.xy_max, fill=False, edgecolor=SLAB_COLOR,
                           lw=1.2, zorder=4))

    _scatter_keypoints(ax, kp[:, :2], kp_color, cfg)

    extent = _extent(cloud[:, :2], kp[:, :2], cfg.xy_max)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"scan + keypoints, ground-aligned  "
                 f"($r_{{max}}$={cfg.r_max:g} m, ROI $\\pm${cfg.xy_max:g} m)",
                 fontsize=11)


def _draw_side(ax, cloud, kp, kp_color, cfg) -> None:
    """Side view: shows which slab of the scan the height slices capture."""
    ax.scatter(cloud[:, 0], cloud[:, 2], s=0.4, c=CLOUD_COLOR, alpha=0.4,
               linewidths=0.0, rasterized=True, zorder=1)

    dz = (cfg.z_max - cfg.z_min) / max(cfg.N_h, 1)
    for i in range(1, cfg.N_h):
        ax.axhline(cfg.z_min + i * dz, color="0.55", lw=0.4, alpha=0.5, zorder=2)
    for z, label in ((cfg.z_min, "$z_{min}$"), (cfg.z_max, "$z_{max}$")):
        ax.axhline(z, color=SLAB_COLOR, lw=1.2, ls="--", zorder=4)
        ax.text(0.995, z, label, transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", color=SLAB_COLOR, fontsize=9)

    _scatter_keypoints(ax, kp[:, [0, 2]], kp_color, cfg)

    extent = _extent(cloud[:, :1], kp[:, :1], cfg.xy_max)
    span = cfg.z_max - cfg.z_min
    ax.set_xlim(-extent, extent)
    ax.set_ylim(cfg.z_min - 0.15 * span, cfg.z_max + 0.15 * span)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("height above ground [m]")
    ax.set_title(f"height slicing  ($N_h$={cfg.N_h}, $dz$={dz:.2f} m)",
                 fontsize=11)


def _extent(*arrays_and_cap) -> float:
    """Symmetric half-extent: the data, capped at the configured ROI.

    Falls back to the ROI for an empty scan, and never returns zero -- a
    degenerate limit is an error in matplotlib, not an empty plot.
    """
    *arrays, cap = arrays_and_cap
    values = [np.abs(a).max() for a in arrays if a.size]
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return float(cap)
    return float(max(min(max(values) * 1.05, cap), 1e-3))


def _draw_full(fig, ax, desc: Descriptors, cfg, vmin=None, vmax=None) -> None:
    """(N_h, N_r*N_s) histogram, with the height ceiling marked.

    ``vmin``/``vmax`` pin the colour scale.  ``inlier match`` draws two of
    these side by side and passes one scale to both, so a difference in
    brightness means a difference in the descriptors rather than in autoscaling.
    """
    image = ax.imshow(desc.full, aspect="auto", origin="lower", cmap="viridis",
                      interpolation="nearest", vmin=vmin, vmax=vmax)
    ceiling = ""
    if 0 <= desc.max_hb < cfg.N_h - 1:
        ax.axhline(desc.max_hb + 0.5, color=CEILING_COLOR, lw=1.5)
        ceiling = f",  ceiling $h_b$={desc.max_hb}"
    _bin_ticks(ax, cfg)
    ax.set_ylabel("$h_b$")
    ax.set_title("$\\mathcal{H}$ -- token histogram, azimuth collapsed "
                 f"($N_h \\times N_r N_s$ = {cfg.N_h}x{cfg.N_r * cfg.N_s})"
                 f"{ceiling}", fontsize=11)
    fig.colorbar(image, ax=ax, fraction=0.012, pad=0.008).set_label("tokens")


def _draw_mint(ax, desc: Descriptors, cfg, vmin=None, vmax=None) -> None:
    """The row stage 1 scores, after collapsing height.

    ``vmin``/``vmax`` as in :func:`_draw_full`.
    """
    ax.imshow(desc.compact[None, :], aspect="auto", origin="lower",
              cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
    _bin_ticks(ax, cfg)
    ax.set_yticks([])
    ax.set_xlabel(f"$r_b$  -- each tick is one radial bin of $N_s$={cfg.N_s} "
                  f"shape columns ({cfg.r_max / cfg.N_r:.3g} m/bin)")
    ax.set_title("$\\mathcal{R}$ -- MINT row (stage 1)", fontsize=11)


def _bin_ticks(ax, cfg) -> None:
    """Tick the (r_b, s_b) axis at radial-bin boundaries, not every column."""
    step = max(1, cfg.N_r // 10)
    positions = np.arange(0, cfg.N_r, step) * cfg.N_s
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{r}" for r in range(0, cfg.N_r, step)])
    ax.set_xlim(-0.5, cfg.N_r * cfg.N_s - 0.5)


def _draw_beam(fig, ax, desc: Descriptors, cfg) -> None:
    """(N_r, N_a) elevation codes, drawn as how many slices each cell holds."""
    image = ax.imshow(desc.beam_popcount, aspect="auto", origin="lower",
                      cmap="inferno", interpolation="nearest", vmin=0,
                      vmax=max(cfg.N_h, 1))
    step = max(1, cfg.N_a // 12)
    ax.set_xticks(np.arange(0, cfg.N_a, step))
    ax.set_xlabel(f"$a_b$  ($N_a$={cfg.N_a}, {360.0 / cfg.N_a:.3g}$\\degree$/bin)")
    ax.set_ylabel(f"$r_b$  ($N_r$={cfg.N_r}, {cfg.r_max / cfg.N_r:.3g} m/bin)")
    ax.set_title("$\\mathcal{A}$ -- BEAM elevation codes (stage 2): "
                 "height slices set per cell", fontsize=11)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015)


def _draw_shape(ax, desc: Descriptors, cfg) -> None:
    counts = np.bincount(desc.sb, minlength=cfg.N_s)[:cfg.N_s]
    labels = shape_class_labels(cfg.N_s)
    ax.bar(np.arange(cfg.N_s), counts, color="#4c72b0")
    ax.set_xticks(np.arange(cfg.N_s))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("tokens")
    ax.set_title(f"shape class $s_b$  ($N_s$={cfg.N_s})", fontsize=11)


def _draw_slices(ax, desc: Descriptors, cfg) -> None:
    from matplotlib import colormaps

    counts = np.bincount(desc.hb, minlength=cfg.N_h)[:cfg.N_h]
    colors = colormaps["viridis"](np.linspace(0, 1, max(cfg.N_h, 1)))
    ax.barh(np.arange(cfg.N_h), counts, color=colors)
    if 0 <= desc.max_hb < cfg.N_h - 1:
        ax.axhline(desc.max_hb + 0.5, color=CEILING_COLOR, lw=1.5)
    ax.set_yticks(np.arange(0, cfg.N_h, max(1, cfg.N_h // 10)))
    ax.set_ylabel("$h_b$")
    ax.set_xlabel("tokens")
    ax.set_title("tokens per height slice", fontsize=11)
