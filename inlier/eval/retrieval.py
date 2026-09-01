"""Materialised retrieval results and a vectorised threshold sweep.

Why this exists
---------------
Retrieval output was carried around as ``similarity_map: Dict[int, Dict[int,
float]]`` plus a parallel ``rank_order: Dict[int, List[int]]``, and every
consumer re-derived the same per-query ordering from scratch.
``_find_best_precision_threshold`` (``evaluate_inlier_helipr.py`` :935) called
``_build_tp_fp_edges`` once per *unique score in the whole run* -- tens of
thousands of full passes over every query, each one re-sorting the same lists.
It was the slowest thing in the script by a wide margin.

The fix is to notice what a threshold sweep actually needs.  For a fixed
candidate order, the top-1 decision at threshold ``t`` is the first candidate
whose score is ``>= t``.  A candidate at position ``i`` can only ever be that
winner if its score exceeds every score before it -- that is, if it is a
*strict running maximum* of the score sequence.  Those running maxima have
strictly increasing scores ``v_1 < v_2 < ...``, and candidate ``k`` wins
exactly for ``t`` in ``(v_{k-1}, v_k]``.

So the whole sweep collapses to one ``searchsorted`` per query against its
running-maximum values: ``O(K + T log K)`` instead of ``O(T * K)``, with no
re-sorting at any threshold.  ``tests/test_eval_retrieval.py`` pins the result
against the loop in ``inlier.eval.metrics``.

Ordering conventions
--------------------
``metrics.pr_curve`` and ``metrics.confusion`` order candidates differently
when ``rank_order`` and ``similarity_map`` disagree -- see the ``metrics``
module docstring.  Both are reproduced here and selected by ``convention``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

Convention = Literal["pr", "confusion"]

SimilarityMap = Dict[int, Dict[int, float]]
GroundTruth = Dict[int, np.ndarray]
RankOrder = Dict[int, List[int]]


def _ordered_candidates(
    q_sims: Dict[int, float],
    rank_order_j: Optional[List[int]],
    convention: Convention,
) -> Tuple[List[int], List[float]]:
    """The candidate order and aligned scores for one query."""
    if rank_order_j is None:
        order = sorted(q_sims, key=lambda d: q_sims[d], reverse=True)
        return order, [q_sims[d] for d in order]

    if convention == "pr":
        # rank_order walked as given; anything it names that was never scored
        # counts as 0.0.
        return list(rank_order_j), [q_sims.get(d, 0.0) for d in rank_order_j]

    # "confusion": drop unscored candidates, then append scored ones that
    # rank_order omitted, highest score first.
    ordered = [d for d in rank_order_j if d in q_sims]
    if len(ordered) < len(q_sims):
        seen = set(ordered)
        tail = sorted((d for d in q_sims if d not in seen),
                      key=lambda d: q_sims[d], reverse=True)
        ordered = ordered + tail
    return ordered, [q_sims[d] for d in ordered]


@dataclass
class _QueryBreakpoints:
    """Per-query running maxima: the only candidates that can ever win top-1."""

    values: np.ndarray      # (M,) strictly increasing scores
    db_idx: np.ndarray      # (M,) db index of each running maximum
    is_positive: np.ndarray  # (M,) bool, whether that db index is a GT positive
    has_gt: bool


@dataclass
class RankedResults:
    """Retrieval output for one stage, prepared for repeated threshold queries."""

    stage: str
    convention: Convention
    n_queries: int
    n_db: int
    _breaks: List[_QueryBreakpoints]
    # every score present in the similarity map, in insertion order.  The
    # operating-threshold search uses this exact pool (see thresholds.py).
    all_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # ------------------------------------------------------------------
    @classmethod
    def from_similarity_map(
        cls,
        similarity_map: SimilarityMap,
        gt: GroundTruth,
        n_queries: int,
        n_db: int,
        rank_order: Optional[RankOrder] = None,
        convention: Convention = "confusion",
        stage: str = "",
    ) -> "RankedResults":
        breaks: List[_QueryBreakpoints] = []
        pool: List[float] = []
        for j in range(n_queries):
            gt_arr = gt.get(j)
            gt_set = set(gt_arr.tolist()) if gt_arr is not None else set()
            q_sims = similarity_map.get(j, {})
            pool.extend(q_sims.values())
            ro = rank_order[j] if (rank_order is not None and j in rank_order) else None
            order, scores = _ordered_candidates(q_sims, ro, convention)

            vals: List[float] = []
            idxs: List[int] = []
            running = -np.inf
            for d, s in zip(order, scores):
                if s > running:       # strict: on a tie the earlier candidate wins
                    running = s
                    vals.append(s)
                    idxs.append(d)
            breaks.append(_QueryBreakpoints(
                values=np.asarray(vals, dtype=np.float64),
                db_idx=np.asarray(idxs, dtype=np.int64),
                is_positive=np.asarray([d in gt_set for d in idxs], dtype=bool),
                has_gt=bool(gt_set),
            ))
        return cls(stage=stage, convention=convention, n_queries=n_queries,
                   n_db=n_db, _breaks=breaks,
                   all_scores=np.asarray(pool, dtype=np.float64))

    # ------------------------------------------------------------------
    def top1(self, threshold: float) -> Dict[int, int]:
        """``{query_idx: db_idx}`` for queries with a match at ``threshold``."""
        out: Dict[int, int] = {}
        for j, bp in enumerate(self._breaks):
            k = int(np.searchsorted(bp.values, threshold, side="left"))
            if k < bp.values.size:
                out[j] = int(bp.db_idx[k])
        return out

    def sweep(self, thresholds: np.ndarray) -> Dict[str, np.ndarray]:
        """TP/FP/FN/TN counts at every threshold, in one pass per query.

        Counting follows ``metrics.confusion``: queries with GT positives
        contribute TP/FP/FN, queries without contribute FP (matched anyway) or
        TN.  ``gt_only=True`` on the caller's side reproduces
        ``metrics.pr_curve`` instead, which ignores the no-GT queries entirely.
        """
        thresholds = np.asarray(thresholds, dtype=np.float64)
        n_thr = thresholds.size
        tp = np.zeros(n_thr, dtype=np.int64)
        fp = np.zeros(n_thr, dtype=np.int64)
        fn = np.zeros(n_thr, dtype=np.int64)
        tn = np.zeros(n_thr, dtype=np.int64)

        for bp in self._breaks:
            # index of the first running maximum >= each threshold
            k = np.searchsorted(bp.values, thresholds, side="left")
            hit = k < bp.values.size
            if bp.has_gt:
                if bp.values.size:
                    pos = np.zeros(n_thr, dtype=bool)
                    pos[hit] = bp.is_positive[k[hit]]
                    tp += (hit & pos)
                    fp += (hit & ~pos)
                fn += ~hit
            else:
                fp += hit
                tn += ~hit

        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def sweep_gt_only(self, thresholds: np.ndarray) -> Dict[str, np.ndarray]:
        """As ``sweep`` but scoring only queries that have GT positives.

        This is the ``metrics.pr_curve`` population.
        """
        thresholds = np.asarray(thresholds, dtype=np.float64)
        n_thr = thresholds.size
        tp = np.zeros(n_thr, dtype=np.int64)
        fp = np.zeros(n_thr, dtype=np.int64)
        fn = np.zeros(n_thr, dtype=np.int64)

        for bp in self._breaks:
            if not bp.has_gt:
                continue
            k = np.searchsorted(bp.values, thresholds, side="left")
            hit = k < bp.values.size
            if bp.values.size:
                pos = np.zeros(n_thr, dtype=bool)
                pos[hit] = bp.is_positive[k[hit]]
                tp += (hit & pos)
                fp += (hit & ~pos)
            fn += ~hit

        return {"tp": tp, "fp": fp, "fn": fn, "tn": np.zeros(n_thr, dtype=np.int64)}

    # ------------------------------------------------------------------
    def candidate_scores(self) -> np.ndarray:
        """Every distinct score that can change a decision, ascending.

        The only thresholds worth evaluating: between two consecutive running
        maxima nothing changes, so a policy search over these values is exact.
        """
        if not self._breaks:
            return np.zeros(0, dtype=np.float64)
        allv = np.concatenate([bp.values for bp in self._breaks if bp.values.size]
                              or [np.zeros(0)])
        return np.unique(allv)
