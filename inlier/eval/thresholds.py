"""Operating-threshold selection, as a named policy.

Two different policies shipped in the surrounding work and neither said so.
InLiER's evaluation picked the threshold that maximises **precision**
(``_find_best_precision_threshold``, ``evaluate_inlier_helipr.py`` :935), while
the HeLiOS baseline it is compared against picks the threshold that maximises
**F1**.  Thresholds chosen by different criteria are not comparable, so a table
putting the two side by side is not comparing like with like unless the policy
is pinned.

Here the policy is explicit, recorded in the results JSON, and defaults to
``max_precision`` so previously published numbers reproduce unchanged.
``f1_max`` is computed and reported whatever the policy, so a comparison on the
common criterion is always available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from inlier.eval import metrics
from inlier.eval.retrieval import RankedResults

Policy = Literal["max_precision", "max_f1", "fixed"]
POLICIES = ("max_precision", "max_f1", "fixed")

# The PR sweep grid.  1001 points over [0, 1] -- the granularity the published
# runs used, kept so PR-AUC is comparable run to run.
DEFAULT_GRID = np.arange(0.0, 1.001, 0.001)


@dataclass
class ThresholdResult:
    policy: Policy
    threshold: float
    precision: float
    recall: float
    f1: float
    #: candidate thresholds actually evaluated (diagnostic)
    n_candidates: int = 0


def candidate_thresholds(ranked: RankedResults) -> np.ndarray:
    """The threshold pool ``max_precision`` searches.

    Reproduces the original pool exactly: the unique non-negative scores in the
    similarity map, bracketed by one value below the minimum and one above the
    maximum.  Negative scores are excluded because evaluation runs the stages
    with ``score_threshold = -2.0`` to keep every candidate for the sweep, and
    those sentinel-ish scores were never meant to be operating points.
    """
    scores = ranked.all_scores
    scores = scores[scores >= 0.0]
    if scores.size == 0:
        return np.zeros(0, dtype=np.float64)
    uniq = np.unique(scores)
    eps = 1e-9
    return np.concatenate(([uniq.min() - eps], uniq, [uniq.max() + eps]))


def select(
    ranked: RankedResults,
    policy: Policy = "max_precision",
    value: Optional[float] = None,
    grid: Optional[np.ndarray] = None,
) -> ThresholdResult:
    """Choose the operating threshold under ``policy``.

    ``fixed`` requires ``value`` and simply scores it.  The other two search --
    ``max_precision`` over the observed-score pool (ties broken by higher
    recall, earliest threshold winning, as the original loop did),
    ``max_f1`` over ``grid``.
    """
    if policy not in POLICIES:
        raise ValueError(f"threshold policy must be one of {POLICIES}, got {policy!r}")

    if policy == "fixed":
        if value is None:
            raise ValueError("threshold policy 'fixed' requires an explicit value")
        counts = ranked.sweep(np.asarray([value], dtype=np.float64))
        p, r, f = metrics.prf1(int(counts["tp"][0]), int(counts["fp"][0]), int(counts["fn"][0]))
        return ThresholdResult("fixed", float(value), p, r, f, n_candidates=1)

    if policy == "max_precision":
        cands = candidate_thresholds(ranked)
        if cands.size == 0:
            return ThresholdResult(policy, 0.0, 0.0, 0.0, 0.0, 0)
        counts = ranked.sweep(cands)
        tp = counts["tp"].astype(np.float64)
        fp = counts["fp"].astype(np.float64)
        fn = counts["fn"].astype(np.float64)
        # match the original's `tp / max(1, tp + fp)` -- same value, no warning
        prec = tp / np.maximum(1.0, tp + fp)
        rec = tp / np.maximum(1.0, tp + fn)
        # highest precision, ties to higher recall, then to the earliest
        # threshold -- exactly the original's strict-improvement loop.
        best_p = prec.max()
        at_best_p = np.flatnonzero(prec == best_p)
        best_r = rec[at_best_p].max()
        idx = int(at_best_p[np.flatnonzero(rec[at_best_p] == best_r)[0]])
        f1 = (2 * prec[idx] * rec[idx] / (prec[idx] + rec[idx])
              if (prec[idx] + rec[idx]) > 0 else 0.0)
        return ThresholdResult(policy, float(cands[idx]), float(prec[idx]),
                               float(rec[idx]), float(f1), int(cands.size))

    # max_f1
    grid = DEFAULT_GRID if grid is None else np.asarray(grid, dtype=np.float64)
    counts = ranked.sweep(grid)
    tp = counts["tp"].astype(np.float64)
    fp = counts["fp"].astype(np.float64)
    fn = counts["fn"].astype(np.float64)
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(tp), where=denom > 0)
    idx = int(np.argmax(f1))
    return ThresholdResult(policy, float(grid[idx]), float(prec[idx]),
                           float(rec[idx]), float(f1[idx]), int(grid.size))
