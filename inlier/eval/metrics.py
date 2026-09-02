"""Retrieval and detection metrics.

These are behaviour-preserving ports of the functions that produced the
published results, lifted out of ``evaluation/evaluate_inlier_helipr.py``
(``compute_recall_at_n`` :727, ``compute_recall_at_kpct`` :749,
``compute_pr_curve`` :777, ``_build_tp_fp_edges`` :888) where they existed in
two near-identical copies.  The semantics are deliberately unchanged, including
the two quirks documented below, because any change here changes a number in
the paper.

Query populations
-----------------
``pr_curve`` and ``confusion`` do **not** score the same set of queries:

* ``pr_curve`` iterates only queries that have at least one GT positive.  A
  query with no GT positive can never contribute, so it is invisible to the
  curve and to PR-AUC.
* ``confusion`` iterates *all* queries and counts a query with no GT positive
  that nonetheless produced an above-threshold match as a false positive (and
  as a true negative when it produced nothing).

So the reported confusion matrix describes a larger population than the curve
it is quoted alongside.  That is how the published numbers were produced; the
population is named explicitly in the results JSON rather than fixed silently.

Ranking conventions
-------------------
The two functions also disagree on how to combine ``rank_order`` with
``similarity_map`` when the two carry different candidate sets (verification
only scores the top-V, so it can return fewer candidates than the stage that
ranked them):

* ``pr_curve`` walks ``rank_order`` as given and treats a candidate missing
  from ``similarity_map`` as score ``0.0``.
* ``confusion`` drops candidates missing from ``similarity_map``, then appends
  any scored candidate that ``rank_order`` omitted, sorted by score.

Both are preserved exactly.  ``inlier.eval.retrieval`` materialises each one
once so the threshold sweep does not re-sort per threshold.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SimilarityMap = Dict[int, Dict[int, float]]
GroundTruth = Dict[int, np.ndarray]
RankOrder = Dict[int, List[int]]

DEFAULT_N_VALUES: Tuple[int, ...] = (1, 5, 10, 20, 50, 100)
DEFAULT_K_PCTS: Tuple[float, ...] = (1.0, 5.0, 10.0)


# ---------------------------------------------------------------------------
#  Recall
# ---------------------------------------------------------------------------

def recall_at_n(
    ranked_lists: Dict[int, List[int]],
    gt: GroundTruth,
    n_values: Sequence[int] = DEFAULT_N_VALUES,
) -> Dict[int, float]:
    """Fraction of queries with >=1 GT positive in the top-N retrieved.

    Queries with no GT positive are excluded from the denominator.
    """
    valid = [j for j in ranked_lists if gt[j].size > 0]
    if not valid:
        return {n: 0.0 for n in n_values}
    recalls: Dict[int, float] = {}
    for n in n_values:
        hits = sum(
            1 for j in valid
            if set(ranked_lists[j][:n]) & set(gt[j].tolist())
        )
        recalls[n] = hits / len(valid)
    return recalls


def recall_at_kpct(
    ranked_lists: Dict[int, List[int]],
    gt: GroundTruth,
    n_db: int,
    k_pcts: Sequence[float] = DEFAULT_K_PCTS,
    fallback_ranked_lists: Optional[Dict[int, List[int]]] = None,
) -> Dict[float, float]:
    """Recall at K% of the database size.

    ``fallback_ranked_lists`` backfills a query whose shortlist is shorter than
    N = ceil(k% * n_db) -- a later stage returns fewer candidates than the
    percentage asks for, and without the fallback those queries would be scored
    against a truncated list.
    """
    results: Dict[float, float] = {}
    for k in k_pcts:
        n = max(1, int(math.ceil(k / 100.0 * n_db)))
        if fallback_ranked_lists is None:
            results[k] = recall_at_n(ranked_lists, gt, [n])[n]
            continue
        valid = [j for j in ranked_lists if gt[j].size > 0]
        if not valid:
            results[k] = 0.0
            continue
        hits = 0
        for j in valid:
            candidates = ranked_lists[j]
            if len(candidates) < n and j in fallback_ranked_lists:
                candidates = fallback_ranked_lists[j]
            if set(candidates[:n]) & set(gt[j].tolist()):
                hits += 1
        results[k] = hits / len(valid)
    return results


# ---------------------------------------------------------------------------
#  Precision / recall
# ---------------------------------------------------------------------------

def pr_curve(
    similarity_map: SimilarityMap,
    gt: GroundTruth,
    thresholds: np.ndarray,
    rank_order: Optional[RankOrder] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sweep thresholds, taking one top-1 decision per query.

    Only queries with GT positives are scored -- see the module docstring.
    Returns ``(precisions, recalls, auc)`` aligned with ``thresholds``.
    """
    valid_queries = [j for j in gt if gt[j].size > 0]
    if not valid_queries:
        z = np.zeros_like(thresholds)
        return z, z, 0.0

    n_thr = len(thresholds)
    tp = np.zeros(n_thr, dtype=np.float64)
    fp = np.zeros(n_thr, dtype=np.float64)
    fn = np.zeros(n_thr, dtype=np.float64)

    for j in valid_queries:
        gt_set = set(gt[j].tolist())
        q_sims = similarity_map.get(j, {})

        if rank_order is not None and j in rank_order:
            ranked_d = rank_order[j]
        else:
            ranked_d = sorted(q_sims, key=lambda d: q_sims[d], reverse=True)

        for ti, thr in enumerate(thresholds):
            top1 = next((d for d in ranked_d if q_sims.get(d, 0.0) >= thr), None)
            if top1 is not None:
                if top1 in gt_set:
                    tp[ti] += 1
                else:
                    fp[ti] += 1
            else:
                fn[ti] += 1

    # np.where evaluates both branches, so the plain division warned on every
    # threshold that made no decision at all (0/0 -> nan, then discarded).
    # np.divide with `where=` skips those entries instead of computing them:
    # identical output, no RuntimeWarning on a run whose scores are all low.
    def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    precisions = _safe_ratio(tp, tp + fp)
    recalls = _safe_ratio(tp, tp + fn)
    order = np.argsort(recalls)
    auc = float(np.trapezoid(precisions[order], recalls[order]))
    return precisions, recalls, auc


def confusion(
    similarity_map: SimilarityMap,
    gt: GroundTruth,
    threshold: float,
    n_queries: int,
    rank_order: Optional[RankOrder] = None,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], int, int, int, int]:
    """``(tp_edges, fp_edges, tp, fp, fn, tn)`` at one threshold.

    Iterates *all* ``n_queries`` -- see the module docstring on populations.
    The edge lists drive the trajectory plot.
    """
    tp_edges: List[Tuple[int, int]] = []
    fp_edges: List[Tuple[int, int]] = []
    tp_count = fp_count = fn_count = tn_count = 0

    for j in range(n_queries):
        gt_set = set(gt[j].tolist())
        q_sims = similarity_map.get(j, {})

        if rank_order is not None and j in rank_order:
            ordered = [d for d in rank_order[j] if d in q_sims]
            seen = set(ordered)
            if len(ordered) < len(q_sims):
                tail = sorted((d for d in q_sims if d not in seen),
                              key=lambda d: q_sims[d], reverse=True)
                ranked_d = ordered + tail
            else:
                ranked_d = ordered
        else:
            ranked_d = sorted(q_sims, key=lambda d: q_sims[d], reverse=True)

        top1 = next((d for d in ranked_d if q_sims[d] >= threshold), None)

        if gt_set:
            if top1 is not None:
                if top1 in gt_set:
                    tp_count += 1
                    tp_edges.append((j, top1))
                else:
                    fp_count += 1
                    fp_edges.append((j, top1))
            else:
                fn_count += 1
        else:
            if top1 is not None:
                fp_count += 1
            else:
                tn_count += 1

    return tp_edges, fp_edges, tp_count, fp_count, fn_count, tn_count


def prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Precision, recall, F1 from raw counts.  0 where undefined."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def f1_from_curve(
    precisions: np.ndarray,
    recalls: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[float, float]:
    """``(f1_max, threshold_at_f1_max)`` from a swept PR curve.

    Reported unconditionally, whatever operating-threshold policy a run uses,
    so InLiER can be compared against baselines that select on F1.
    """
    denom = precisions + recalls
    f1 = np.where(denom > 0, 2 * precisions * recalls / np.where(denom > 0, denom, 1.0), 0.0)
    best = int(np.argmax(f1))
    return float(f1[best]), float(thresholds[best])


def max_recall_at_full_precision(
    precisions: np.ndarray,
    recalls: np.ndarray,
) -> float:
    """Highest recall reached while precision is still 1.0.

    The usual headline number for SLAM loop-closure detection, where a single
    false positive can corrupt the pose graph.
    """
    perfect = np.isclose(precisions, 1.0)
    if not perfect.any():
        return 0.0
    return float(np.max(recalls[perfect]))
