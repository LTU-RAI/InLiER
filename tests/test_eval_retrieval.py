"""Pin the vectorised threshold sweep against the per-threshold loop.

``RankedResults.sweep`` replaces a loop that re-derived each query's candidate
order at every threshold.  These tests assert the replacement is exact, on the
same awkward inputs ``test_eval_metrics`` uses -- a ``rank_order`` naming
candidates the similarity map never scored, and a similarity map carrying
candidates ``rank_order`` omits, which are the only cases where the two
ordering conventions diverge.

``thresholds.select(policy="max_precision")`` is pinned against a frozen copy
of ``_find_best_precision_threshold`` (``evaluate_inlier_helipr.py`` :935),
including its tie-break: highest precision, then highest recall, then the
earliest threshold in ascending order.
"""

import numpy as np
import pytest

from inlier.eval import metrics, thresholds
from inlier.eval.retrieval import RankedResults

from test_eval_metrics import CASES, N_DB, N_Q, THRESHOLDS, _make_case, _ref_confusion


def _ref_best_precision_threshold(similarity_map, gt, n_queries, rank_order=None):
    """Frozen copy of evaluate_inlier_helipr.py :935."""
    all_scores = [float(s) for q in similarity_map.values() for s in q.values() if s >= 0.0]
    if not all_scores:
        return 0.0, 0.0, 0.0
    uniq = np.unique(np.asarray(all_scores, dtype=np.float64))
    eps = 1e-9
    grid = np.concatenate(([uniq.min() - eps], uniq, [uniq.max() + eps]))
    best_thr, best_prec, best_rec = float(grid[0]), -1.0, -1.0
    for thr in grid:
        _, _, tp, fp, fn, _ = _ref_confusion(similarity_map, gt, float(thr), n_queries, rank_order)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        if (prec > best_prec) or (prec == best_prec and rec > best_rec):
            best_prec, best_rec, best_thr = float(prec), float(rec), float(thr)
    return best_thr, best_prec, best_rec


@pytest.mark.parametrize("seed,sparse,partial", CASES)
@pytest.mark.parametrize("use_rank_order", [False, True])
def test_sweep_matches_confusion_loop(seed, sparse, partial, use_rank_order):
    gt, sim, rank, _ = _make_case(seed, sparse, partial)
    ro = rank if use_rank_order else None
    ranked = RankedResults.from_similarity_map(sim, gt, N_Q, N_DB, ro, "confusion")
    swept = ranked.sweep(THRESHOLDS)
    for ti, thr in enumerate(THRESHOLDS):
        _, _, tp, fp, fn, tn = metrics.confusion(sim, gt, float(thr), N_Q, ro)
        assert (tp, fp, fn, tn) == (
            int(swept["tp"][ti]), int(swept["fp"][ti]),
            int(swept["fn"][ti]), int(swept["tn"][ti])
        ), f"threshold {thr}"


@pytest.mark.parametrize("seed,sparse,partial", CASES)
@pytest.mark.parametrize("use_rank_order", [False, True])
def test_sweep_gt_only_matches_pr_curve(seed, sparse, partial, use_rank_order):
    gt, sim, rank, _ = _make_case(seed, sparse, partial)
    ro = rank if use_rank_order else None
    ranked = RankedResults.from_similarity_map(sim, gt, N_Q, N_DB, ro, "pr")
    swept = ranked.sweep_gt_only(THRESHOLDS)
    tp, fp, fn = swept["tp"], swept["fp"], swept["fn"]
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    with np.errstate(invalid="ignore"):
        p_ref, r_ref, _ = metrics.pr_curve(sim, gt, THRESHOLDS, ro)
    assert np.array_equal(prec, p_ref)
    assert np.array_equal(rec, r_ref)


@pytest.mark.parametrize("seed,sparse,partial", CASES)
@pytest.mark.parametrize("use_rank_order", [False, True])
def test_max_precision_policy_matches_reference(seed, sparse, partial, use_rank_order):
    gt, sim, rank, _ = _make_case(seed, sparse, partial)
    ro = rank if use_rank_order else None
    thr_ref, prec_ref, rec_ref = _ref_best_precision_threshold(sim, gt, N_Q, ro)
    ranked = RankedResults.from_similarity_map(sim, gt, N_Q, N_DB, ro, "confusion")
    got = thresholds.select(ranked, "max_precision")
    assert got.threshold == pytest.approx(thr_ref, abs=1e-12)
    assert got.precision == prec_ref
    assert got.recall == rec_ref


def test_fixed_policy_scores_the_given_value():
    gt, sim, rank, _ = _make_case(0, False, False)
    ranked = RankedResults.from_similarity_map(sim, gt, N_Q, N_DB, None, "confusion")
    got = thresholds.select(ranked, "fixed", value=0.42)
    assert got.threshold == 0.42
    _, _, tp, fp, fn, _ = metrics.confusion(sim, gt, 0.42, N_Q, None)
    assert (got.precision, got.recall) == metrics.prf1(tp, fp, fn)[:2]


def test_max_f1_policy_beats_or_matches_others_on_f1():
    gt, sim, rank, _ = _make_case(1, False, False)
    ranked = RankedResults.from_similarity_map(sim, gt, N_Q, N_DB, None, "confusion")
    best_f1 = thresholds.select(ranked, "max_f1")
    by_precision = thresholds.select(ranked, "max_precision")
    assert best_f1.f1 >= by_precision.f1 - 1e-12


def test_fixed_policy_requires_a_value():
    gt, sim, _, _ = _make_case(0, False, False)
    ranked = RankedResults.from_similarity_map(sim, gt, N_Q, N_DB, None, "confusion")
    with pytest.raises(ValueError, match="requires an explicit value"):
        thresholds.select(ranked, "fixed")


def test_unknown_policy_rejected():
    gt, sim, _, _ = _make_case(0, False, False)
    ranked = RankedResults.from_similarity_map(sim, gt, N_Q, N_DB, None, "confusion")
    with pytest.raises(ValueError, match="threshold policy"):
        thresholds.select(ranked, "max_recall")


def test_top1_agrees_with_confusion_edges():
    gt, sim, rank, _ = _make_case(3, True, True)
    ranked = RankedResults.from_similarity_map(sim, gt, N_Q, N_DB, rank, "confusion")
    thr = 0.5
    tp_edges, fp_edges, *_ = metrics.confusion(sim, gt, thr, N_Q, rank)
    expected = dict(tp_edges) | dict(fp_edges)
    got = ranked.top1(thr)
    for q, d in expected.items():
        assert got[q] == d
