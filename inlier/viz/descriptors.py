"""The matrices the matcher scores on, recovered from a token array alone.

Three of them, in the order the pipeline uses them:

``full``    (N_h, N_r·N_s) -- the token histogram with azimuth collapsed.
``compact`` (N_r·N_s,)     -- the MINT row: ``full`` summed over height,
                              after the pairwise height ceiling.  This is
                              what stage 1 scores.
``beam``    (N_r, N_a)     -- one bitmask per (radial, azimuth) cell, bit
                              *h* set iff a token exists at that height.
                              Stage 2 scores its bit-level Jaccard.

The ``full`` and ``beam`` builders are the matcher's own, imported rather
than re-derived: a change to the mixed-radix packing must not be able to
silently desynchronise what gets drawn from what gets scored.  The
reference (numpy) matcher is imported deliberately -- it needs no compiled
extension, and its private statics are pure functions of their arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from inlier.core.reference.InLiER import InLiER as _Encoder
from inlier.core.reference.InLiER_Matcher import InLiER_Matcher as _Matcher


@dataclass
class Descriptors:
    """Everything derivable from ``token_id`` plus the four radices."""

    full:    np.ndarray     # (N_h, N_r*N_s) float32
    compact: np.ndarray     # (N_r*N_s,)     float32 -- the MINT row
    beam:    np.ndarray     # (N_r, N_a)     uint    -- elevation bitmasks
    hb:      np.ndarray     # (K,) int64, per token
    rb:      np.ndarray
    sb:      np.ndarray
    ab:      np.ndarray
    max_hb:  int            # highest occupied slice; -1 when there are no tokens

    @property
    def beam_popcount(self) -> np.ndarray:
        """(N_r, N_a) int32 -- how many height slices each cell occupies."""
        return popcount(self.beam)


def popcount(values: np.ndarray) -> np.ndarray:
    """Set-bit count, for any width up to 64.

    ``InLiER_Matcher._popcount_u16`` only covers the ``N_h <= 16`` case;
    ``_build_beam`` widens to uint32/uint64 above that, and the figure has
    to keep working there.
    """
    x = np.asarray(values).astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + \
        ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int32)


def describe(token_id: np.ndarray, N_h: int, N_r: int, N_s: int,
             N_a: int) -> Descriptors:
    """Build every descriptor view of *token_id*."""
    tokens = np.asarray(token_id)
    N_h, N_r, N_s, N_a = int(N_h), int(N_r), int(N_s), int(N_a)
    Rw = N_r * N_s

    full = _Matcher._tokens_to_hist(tokens, N_a, N_h * Rw, N_h, Rw)
    hb, rb, sb, ab = _Encoder.unpack_token_ids(tokens, N_r, N_s, N_a)
    max_hb = int(hb.max()) if hb.size else -1
    beam = _Matcher._build_beam(hb, rb, ab, N_h, N_r, N_a)

    return Descriptors(full=full, compact=full.sum(axis=0), beam=beam,
                       hb=hb, rb=rb, sb=sb, ab=ab, max_hb=max_hb)


def shape_class_labels(N_s: int) -> List[str]:
    """Names for the ``sb`` axis, matching ``_compute_shape_pca``.

    The shape space is linear / planar / scatter, with linear and planar
    subdivided by inclination when ``N_s`` allows it; scatter is always the
    last class.  Inclination bins span 0-90 degrees in ``N_s // 2`` steps.
    """
    N_s = int(N_s)
    n_incl = 0 if N_s <= 3 else (2 if N_s <= 5 else 3)
    if n_incl == 0:
        return ["linear", "planar", "scatter"][:N_s]

    step = 90.0 / n_incl
    labels = [f"lin {int(i * step)}-{int((i + 1) * step)}" for i in range(n_incl)]
    labels += [f"pln {int(i * step)}-{int((i + 1) * step)}" for i in range(n_incl)]
    labels.append("scatter")
    return labels[:N_s]


def occupancy(descriptors: Descriptors, N_h: int, N_r: int, N_s: int,
              N_a: int) -> float:
    """Fraction of the token space a scan actually lights up."""
    cells = int(N_h) * int(N_r) * int(N_s) * int(N_a)
    if cells == 0 or descriptors.hb.size == 0:
        return 0.0
    ids = ((descriptors.hb * N_r + descriptors.rb) * N_s
           + descriptors.sb) * N_a + descriptors.ab
    return float(np.unique(ids).size) / cells
