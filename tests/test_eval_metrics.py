"""Pin inlier.eval.metrics against the implementations that produced the paper.

The repo already uses this pattern for the C++ core: ``inlier/core/reference/``
keeps the original numpy implementation as the ground truth its replacement is
tested against.  This file does the same for the metrics, which were lifted out
of ``evaluation/evaluate_inlier_helipr.py``.  The reference bodies below are
verbatim copies of those functions, kept here so the test survives the removal
of the old scripts in 1.1.0.

The generated cases deliberately include the awkward shapes: a ``rank_order``
that lists candidates ``similarity_map`` never scored, and a ``similarity_map``
carrying candidates ``rank_order`` omits.  Those are the only inputs on which
``pr_curve`` and ``confusion`` disagree about ordering, so they are what would
catch a "tidy-up" of either function.
"""

import numpy as np
import pytest

from inlier.eval import metrics

N_Q, N_DB = 60, 120
THRESHOLDS = np.arange(0.0, 1.001, 0.01)


# ---------------------------------------------------------------------------
#  Frozen reference implementations (evaluate_inlier_helipr.py :727/:749/:777/:888)
# ---------------------------------------------------------------------------

def _ref_recall_at_n(ranked_lists, gt, n_values):
    valid = [j for j in ranked_lists if gt[j].size > 0]
    if not valid:
        return {n: 0.0 for n in n_values}
    recalls = {}
    for n in n_values:
        hits = sum(1 for j in valid if set(ranked_lists[j][:n]) & set(gt[j].tolist()))
        recalls[n] = hits / len(valid)
    return recalls


def _ref_recall_at_kpct(ranked_lists, gt, n_db, k_pcts, fallback=None):
    import math
    results = {}
    for k in k_pcts:
        n = max(1, int(math.ceil(k / 100.0 * n_db)))
        if fallback is None:
            results[k] = _ref_recall_at_n(ranked_lists, gt, [n])[n]
            continue
        valid = [j for j in ranked_lists if gt[j].size > 0]
        if not valid:
            results[k] = 0.0
            continue
        hits = 0
        for j in valid:
            candidates = ranked_lists[j]
            if len(candidates) < n and j in fallback:
                candidates = fallback[j]
            if set(candidates[:n]) & set(gt[j].tolist()):
                hits += 1
        results[k] = hits / len(valid)
    return results


def _ref_pr_curve(similarity_map, gt, thresholds, rank_order=None):
    valid_queries = [j for j in gt if gt[j].size > 0]
    if not valid_queries:
        z = np.zeros_like(thresholds)
        return z, z, 0.0
    n_thr = len(thresholds)
    tp = np.zeros(n_thr); fp = np.zeros(n_thr); fn = np.zeros(n_thr)
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
    precisions = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recalls = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    order = np.argsort(recalls)
    return precisions, recalls, float(np.trapezoid(precisions[order], recalls[order]))


def _ref_confusion(similarity_map, gt, best_thr, n_queries, rank_order=None):
    tp_edges = []; fp_edges = []
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
        top1 = next((d for d in ranked_d if q_sims[d] >= best_thr), None)
        if gt_set:
            if top1 is not None:
                if top1 in gt_set:
                    tp_count += 1; tp_edges.append((j, top1))
                else:
                    fp_count += 1; fp_edges.append((j, top1))
            else:
                fn_count += 1
        else:
            if top1 is not None:
                fp_count += 1
            else:
                tn_count += 1
    return tp_edges, fp_edges, tp_count, fp_count, fn_count, tn_count


# ---------------------------------------------------------------------------
#  Case generation
# ---------------------------------------------------------------------------

def _make_case(seed, sparse_sim, partial_rank):
    r = np.random.default_rng(seed)
    gt = {j: np.unique(r.integers(0, N_DB, r.integers(0, 4))) for j in range(N_Q)}
    sim, rank = {}, {}
    for j in range(N_Q):
        k = int(r.integers(1, 25))
        cands = r.choice(N_DB, size=k, replace=False).tolist()
        scores = r.random(k).round(3)
        sim[j] = {int(d): float(s) for d, s in zip(cands, scores)}
        order = cands[:]
        r.shuffle(order)
        if partial_rank:  # rank_order references candidates similarity_map lacks
            order = order[: max(1, k // 2)] + r.choice(N_DB, 3).tolist()
        if sparse_sim:    # similarity_map carries candidates rank_order omits
            for extra in r.choice(N_DB, 3):
                sim[j][int(extra)] = float(r.random())
        rank[j] = [int(d) for d in order]
    ranked = {j: [d for d, _ in sorted(sim[j].items(), key=lambda kv: -kv[1])] for j in sim}
    return gt, sim, rank, ranked


CASES = [(s, sp, pr)
         for s in range(6)
         for sp, pr in ((False, False), (True, False), (False, True), (True, True))]


@pytest.mark.parametrize("seed,sparse,partial", CASES)
def test_recall_matches_reference(seed, sparse, partial):
    gt, sim, rank, ranked = _make_case(seed, sparse, partial)
    n_values = [1, 5, 10, 20]
    assert metrics.recall_at_n(ranked, gt, n_values) == _ref_recall_at_n(ranked, gt, n_values)
    k_pcts = [1.0, 5.0, 10.0]
    assert (metrics.recall_at_kpct(ranked, gt, N_DB, k_pcts, ranked)
            == _ref_recall_at_kpct(ranked, gt, N_DB, k_pcts, ranked))


@pytest.mark.parametrize("seed,sparse,partial", CASES)
@pytest.mark.parametrize("use_rank_order", [False, True])
def test_pr_curve_matches_reference(seed, sparse, partial, use_rank_order):
    gt, sim, rank, _ = _make_case(seed, sparse, partial)
    ro = rank if use_rank_order else None
    with np.errstate(invalid="ignore"):
        p_new, r_new, auc_new = metrics.pr_curve(sim, gt, THRESHOLDS, ro)
        p_ref, r_ref, auc_ref = _ref_pr_curve(sim, gt, THRESHOLDS, ro)
    assert np.array_equal(p_new, p_ref)
    assert np.array_equal(r_new, r_ref)
    assert auc_new == auc_ref


@pytest.mark.parametrize("seed,sparse,partial", CASES)
@pytest.mark.parametrize("use_rank_order", [False, True])
@pytest.mark.parametrize("thr", [0.0, 0.3, 0.55, 0.9, 1.5])
def test_confusion_matches_reference(seed, sparse, partial, use_rank_order, thr):
    gt, sim, rank, _ = _make_case(seed, sparse, partial)
    ro = rank if use_rank_order else None
    assert metrics.confusion(sim, gt, thr, N_Q, ro) == _ref_confusion(sim, gt, thr, N_Q, ro)


# ---------------------------------------------------------------------------
#  New metrics (no reference to match; check the definitions directly)
# ---------------------------------------------------------------------------

def test_prf1_edge_cases():
    assert metrics.prf1(0, 0, 0) == (0.0, 0.0, 0.0)
    assert metrics.prf1(5, 0, 0) == (1.0, 1.0, 1.0)
    p, r, f = metrics.prf1(1, 1, 2)
    assert (p, r) == (0.5, 1 / 3)
    assert f == pytest.approx(2 * 0.5 * (1 / 3) / (0.5 + 1 / 3))


def test_f1_from_curve_picks_the_max():
    thr = np.array([0.0, 0.5, 1.0])
    prec = np.array([0.5, 0.9, 1.0])
    rec = np.array([1.0, 0.9, 0.1])
    f1, at = metrics.f1_from_curve(prec, rec, thr)
    assert f1 == pytest.approx(0.9)
    assert at == 0.5


def test_max_recall_at_full_precision():
    prec = np.array([0.5, 1.0, 1.0, 0.8])
    rec = np.array([0.9, 0.6, 0.2, 0.95])
    assert metrics.max_recall_at_full_precision(prec, rec) == pytest.approx(0.6)
    assert metrics.max_recall_at_full_precision(np.array([0.4]), np.array([0.9])) == 0.0


def test_pr_curve_is_quiet_when_a_threshold_makes_no_decision():
    """0/0 precision must not warn: it is a legitimate end of the sweep.

    Above the highest score no query produces a match, so tp+fp is 0 there.
    The value is 0.0 either way; this pins that computing it stays silent, so
    a run whose scores are all low does not spray RuntimeWarnings.
    """
    import warnings

    from inlier.eval import metrics

    sims = {0: {5: 0.10}, 1: {7: 0.05}}
    gt = {0: np.array([5]), 1: np.array([9])}
    thresholds = np.linspace(0.0, 1.0, 21)      # most are above every score

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prec, rec, auc = metrics.pr_curve(sims, gt, thresholds)

    assert prec.shape == rec.shape == thresholds.shape
    assert np.isfinite(prec).all() and np.isfinite(rec).all()
    ## nothing clears a threshold above 0.10: no decisions, precision 0
    assert prec[thresholds > 0.10].tolist() == [0.0] * int((thresholds > 0.10).sum())
