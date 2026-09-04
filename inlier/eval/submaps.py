"""How consecutive scans are grouped into submaps.

One rule, stated once.  It was previously written twice --
``overlap_build._group_into_submaps`` and ``Generic_Handler.load_generic`` --
which is a dangerous thing to duplicate: the overlap matrix is indexed *by
submap*, so if the two ever disagreed about how many submaps a sequence has,
the ground truth would silently misalign against retrieval rather than fail.

The index arithmetic is pure and cheap, which is what lets a caller decide
which submaps it wants before paying to load any points -- ``inlier encode``
needs one submap out of several hundred.  ``build_submap`` is the other half:
given a window, it reads exactly that window's scans and accumulates them.
Both the batch loader and the streaming one go through it, so the two cannot
disagree about what a submap is.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def submap_windows(count: int, n_scans: int = 1,
                   stride: Optional[int] = None) -> List[range]:
    """Index windows for grouping *count* scans into submaps.

    Each window's **first** scan is the keyframe: its pose defines the
    submap's position, and every other scan in the window is transformed
    into its frame.

    ``stride=None`` means ``stride=n_scans`` (non-overlapping submaps).
    ``stride < n_scans`` overlaps them; ``stride > n_scans`` leaves gaps.
    The final window is short when ``count`` is not a multiple of ``stride``
    -- it is kept, not dropped, because the overlap matrix's dimensions are
    counted the same way.

    >>> submap_windows(7, 3)
    [range(0, 3), range(3, 6), range(6, 7)]
    >>> submap_windows(7, 3, stride=2)
    [range(0, 3), range(2, 5), range(4, 7), range(6, 7)]
    >>> len(submap_windows(5, 1))
    5
    """
    n_scans = int(n_scans)
    if n_scans < 1:
        raise ValueError(f"n_scans must be >= 1, got {n_scans}")
    stride = n_scans if stride is None else int(stride)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    return [range(start, min(start + n_scans, count))
            for start in range(0, max(int(count), 0), stride)]


def submap_count(count: int, n_scans: int = 1,
                 stride: Optional[int] = None) -> int:
    """How many submaps *count* scans produce, without building the windows."""
    return len(submap_windows(count, n_scans, stride))


def build_submap(load_scan, scan_files, poses, window):
    """Accumulate one window's scans into its keyframe's frame.

    The keyframe is ``window[0]``: its pose is the submap's pose, and every
    other scan is brought into its frame by ``inv(T_keyframe) @ T_k``.  Empty
    scans are skipped; a window whose scans were all empty returns ``None``,
    which the caller drops rather than appending a zero-point submap.

    Lifted out of ``Generic_Handler.load_generic`` so the streaming loader can
    build submap ``i`` without re-listing the scan directory for every frame.
    The accumulation rule stays stated once, which is the point of this module.
    """
    s = window[0]
    ref_inv = np.linalg.inv(poses[s])
    window_pts = []
    for k in window:
        pts = load_scan(scan_files[k])
        if pts.size == 0:
            continue
        if k == s:
            window_pts.append(pts.astype(np.float32, copy=False))
        else:
            T_rel = ref_inv @ poses[k]  # keyframe <- scan k
            R = T_rel[:3, :3].astype(np.float32)
            t = T_rel[:3, 3].astype(np.float32)
            window_pts.append(((pts @ R.T) + t).astype(np.float32, copy=False))
    if not window_pts:
        return None
    return np.vstack(window_pts)
