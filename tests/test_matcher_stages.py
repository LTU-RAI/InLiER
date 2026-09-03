""" Matcher retrieval stages (add/finalize/shortlist/beam/rerank) —
C++ vs the Python implementation, on REAL cached descriptors
(cache_inlier/*.npz).

The numpy side is ``inlier.core.reference.InLiER_Matcher``, imported
explicitly and never ``inlier.core.InLiER_Matcher``: with the extension built,
every stage method on the latter is a C++ override, so pairing it against
``ip._Matcher`` compares the core against itself and cannot fail. Measured on
this fixture, those two agree bit-for-bit while the reference differs by ~6e-8
on MINT (float32 accumulation) and not at all on BEAM or rerank — which is the
signal the tolerances below are actually sized for.

Shortlist/rerank scores use float32 accumulation in Python, double in
C++, so comparisons are allclose(rtol=1e-5). BEAM is pure integer
popcount Jaccard, so it's compared bit-exact. Ranking ties may legally
reorder; results are therefore compared as {id: ...} dicts, not
ordered sequences.
"""

from __future__ import annotations

import numpy as np
import pytest

from inlier import _inlier_pybind as ip
from inlier.core import _cfg_bridge as bridge
from inlier.core.Dataclasses import (
    BEAMScoreConfig,
    InLiER_Config,
    InLiER_Tokens,
    RerankConfig,
    ShortlistConfig,
)
from inlier.core.InLiER_Matcher import InLiER_Matcher
from inlier.core.reference.InLiER_Matcher import (
    InLiER_Matcher as ReferenceMatcher,
)

N_DB = 60      # scans loaded into the matcher
N_QUERY = 8    # queries evaluated per stage

## Non-default gates so the gating branches are actually exercised.
SL_KW = dict(min_shared_rows=3, mint_mode="compact",
             mint_scoring="l1_intersection")
BEAM_KW = dict(min_shared_bins=4, min_shared_az_cols=3, score_threshold=0.0)


@pytest.fixture(scope="module")
def db_and_queries(cached_descriptors):
    """(db token lists, query token lists) sliced from the real cache."""
    d = cached_descriptors
    offsets, tids = d["offsets"], d["token_ids"]
    n = len(offsets) - 1
    assert n >= N_DB + N_QUERY
    step = n // (N_DB + N_QUERY)
    picks = [i * step for i in range(N_DB + N_QUERY)]
    scans = [tids[offsets[i]:offsets[i + 1]].astype(np.uint32)
             for i in picks]
    return scans[:N_DB], scans[N_DB:]


@pytest.fixture(scope="module")
def matchers(db_and_queries):
    db, _ = db_and_queries
    py_cfg = InLiER_Config()
    py_m = ReferenceMatcher(
        inlier_config=py_cfg,
        shortlist_config=ShortlistConfig(**SL_KW),
        beam_score_config=BEAMScoreConfig(**BEAM_KW),
        rerank_config=RerankConfig(),
    )
    cpp_m = ip._Matcher(bridge.to_cpp_inlier_config(py_cfg))
    for i, tid in enumerate(db):
        py_m.add(i, InLiER_Tokens(token_id=tid))
        cpp_m.add(i, tid.astype(np.uint64))
    py_m.finalize(verbose=False)
    cpp_m.finalize()
    return py_m, cpp_m


## --- add/finalize bookkeeping ---


def test_add_finalize_state(matchers, db_and_queries):
    py_m, cpp_m = matchers
    db, _ = db_and_queries
    assert len(cpp_m) == len(py_m) == len(db)
    np.testing.assert_array_equal(cpp_m.db_ids(), py_m.db_ids)

    for i in (0, len(db) // 2, len(db) - 1):
        tid, hb, rb, sb, ab, max_hb = cpp_m.get_scan_data(i)
        scan = py_m.get_scan_data(i)
        np.testing.assert_array_equal(tid, scan["token_id"].astype(np.uint64))
        np.testing.assert_array_equal(hb, scan["hb"])
        np.testing.assert_array_equal(rb, scan["rb"])
        np.testing.assert_array_equal(sb, scan["sb"])
        np.testing.assert_array_equal(ab, scan["ab"])
        assert max_hb == scan["max_active_hb"]


def test_add_errors(db_and_queries):
    db, _ = db_and_queries
    cpp_m = ip._Matcher(bridge.to_cpp_inlier_config(InLiER_Config()))
    cpp_m.add(0, db[0].astype(np.uint64))
    with pytest.raises(Exception, match="already exists"):
        cpp_m.add(0, db[1].astype(np.uint64))
    cpp_m.finalize()
    ## add after finalize is legal (incremental DB) and re-opens the matcher
    cpp_m.add(1, db[1].astype(np.uint64))
    assert len(cpp_m) == 2 and not cpp_m.finalized
    cpp_m.reset()
    assert len(cpp_m) == 0
    cpp_m.add(1, db[1].astype(np.uint64))  # works again after reset


def test_get_scan_data_missing_id(matchers):
    _, cpp_m = matchers
    with pytest.raises(Exception, match="not found"):
        cpp_m.get_scan_data(10**6)


## --- Stage 1: MINT shortlist ---


@pytest.mark.parametrize("mint_mode", ["compact", "full"])
@pytest.mark.parametrize(
    "mint_scoring", ["l1_intersection", "raw_intersection", "cosine"])
def test_shortlist_scores_match(db_and_queries, matchers, mint_mode,
                                mint_scoring):
    py_m, cpp_m = matchers
    _, queries = db_and_queries
    n_db = len(py_m)

    sl_py_cfg = ShortlistConfig(
        min_shared_rows=SL_KW["min_shared_rows"], mint_mode=mint_mode,
        mint_scoring=mint_scoring)
    py_m._shortlist_cfg = sl_py_cfg
    sl_cpp_cfg = bridge.to_cpp_stage_config(ip.ShortlistConfig, sl_py_cfg)

    for q in queries:
        out_py = py_m.shortlist(
            InLiER_Tokens(token_id=q), topk=n_db, verbose=False)
        out_cpp = cpp_m.shortlist(q.astype(np.uint64), sl_cpp_cfg, topk=n_db)

        py_map = dict(zip(out_py.ids, out_py.scores))
        cpp_map = dict(zip(out_cpp.ids, out_cpp.scores))
        assert set(py_map) == set(cpp_map)
        for i in py_map:
            np.testing.assert_allclose(
                cpp_map[i], py_map[i], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(
            out_cpp.scores, out_py.scores, rtol=1e-5, atol=1e-7)


def test_shortlist_topk_and_pct(db_and_queries, matchers):
    py_m, cpp_m = matchers
    _, queries = db_and_queries
    q = queries[0]
    sl_cfg = ShortlistConfig(**SL_KW)
    py_m._shortlist_cfg = sl_cfg
    cpp_cfg = bridge.to_cpp_stage_config(ip.ShortlistConfig, sl_cfg)

    out_py = py_m.shortlist(InLiER_Tokens(token_id=q), topk=10, verbose=False)
    out_cpp = cpp_m.shortlist(q.astype(np.uint64), cpp_cfg, topk=10)
    assert len(out_cpp.ids) == len(out_py.ids) == 10
    np.testing.assert_allclose(out_cpp.scores, out_py.scores,
                               rtol=1e-5, atol=1e-7)

    out_py = py_m.shortlist(
        InLiER_Tokens(token_id=q), topk_pct=0.15, verbose=False)
    out_cpp = cpp_m.shortlist(q.astype(np.uint64), cpp_cfg, topk_pct=0.15)
    assert len(out_cpp.ids) == len(out_py.ids)


def test_shortlist_empty_query(matchers):
    py_m, cpp_m = matchers
    empty = np.zeros(0, dtype=np.uint32)
    out_py = py_m.shortlist(
        InLiER_Tokens(token_id=empty), topk=5, verbose=False)
    out_cpp = cpp_m.shortlist(
        empty.astype(np.uint64),
        bridge.to_cpp_stage_config(ip.ShortlistConfig,
                                   ShortlistConfig(**SL_KW)),
        topk=5)
    assert out_cpp.scores == out_py.scores  # all zeros


def test_shortlist_empty_db():
    cpp_m = ip._Matcher(bridge.to_cpp_inlier_config(InLiER_Config()))
    cpp_m.finalize()
    out = cpp_m.shortlist(
        np.zeros(0, dtype=np.uint64),
        bridge.to_cpp_stage_config(ip.ShortlistConfig, ShortlistConfig()))
    assert out.ids == [] and out.scores == []


## --- Stage 2: BEAM ---


def test_beam_scores_match(db_and_queries, matchers):
    py_m, cpp_m = matchers
    _, queries = db_and_queries
    n_db = len(py_m)
    cand_ids = list(range(n_db))
    beam_cpp_cfg = bridge.to_cpp_stage_config(ip.BEAMScoreConfig,
                                              BEAMScoreConfig(**BEAM_KW))

    for q in queries:
        out_py = py_m.beam_score(
            InLiER_Tokens(token_id=q), cand_ids, topk=n_db, verbose=False)
        out_cpp = cpp_m.beam_score(
            q.astype(np.uint64), np.asarray(cand_ids, dtype=np.int64),
            beam_cpp_cfg, topk=n_db)

        py_map = {i: (s, y, sh) for i, s, y, sh in zip(
            out_py.ids, out_py.scores, out_py.yaw_estimates,
            out_py.best_shifts)}
        cpp_map = {i: (s, y, sh) for i, s, y, sh in zip(
            out_cpp.ids, out_cpp.scores, out_cpp.yaw_estimates,
            out_cpp.best_shifts)}
        assert set(py_map) == set(cpp_map)
        for i in py_map:
            ps, py_yaw, psh = py_map[i]
            cs, cy, csh = cpp_map[i]
            ## bit-exact: integer popcount Jaccard
            assert cs == pytest.approx(ps, abs=0), f"id {i}"
            assert csh == psh, f"id {i} shift"
            assert cy == pytest.approx(py_yaw, abs=1e-12), f"id {i} yaw"
        ## ranking sequence identical apart from ties
        np.testing.assert_allclose(out_cpp.scores, out_py.scores, atol=0)


def test_beam_empty_query_tokens(matchers):
    py_m, cpp_m = matchers
    empty = np.zeros(0, dtype=np.uint32)
    out_py = py_m.beam_score(
        InLiER_Tokens(token_id=empty), [0, 1, 2], topk=3, verbose=False)
    out_cpp = cpp_m.beam_score(
        empty.astype(np.uint64), np.array([0, 1, 2], dtype=np.int64),
        bridge.to_cpp_stage_config(ip.BEAMScoreConfig,
                                   BEAMScoreConfig(**BEAM_KW)),
        topk=3)
    assert out_cpp.scores == out_py.scores == [0.0, 0.0, 0.0]
    assert out_cpp.best_shifts == out_py.best_shifts


## --- Rerank ---


@pytest.mark.parametrize("scoring_mode", ["jaccard4d", "cosine4d"])
@pytest.mark.parametrize("spatial_tol", [0, 1])
def test_rerank_matches(db_and_queries, matchers, scoring_mode, spatial_tol):
    py_m, cpp_m = matchers
    _, queries = db_and_queries
    cand_ids = list(range(0, len(py_m), 3))

    rr_py_cfg = RerankConfig(scoring_mode=scoring_mode,
                             spatial_tol=spatial_tol)
    py_m._rerank_cfg = rr_py_cfg
    rr_cpp_cfg = bridge.to_cpp_stage_config(ip.RerankConfig, rr_py_cfg)

    for q in queries[:4]:
        ## realistic shifts: take them from BEAM
        beam = py_m.beam_score(InLiER_Tokens(token_id=q), cand_ids,
                               topk=len(cand_ids), verbose=False)
        ids, shifts = beam.ids, beam.best_shifts

        out_py = py_m.rerank(InLiER_Tokens(token_id=q), ids, shifts,
                             topk=len(ids), verbose=False)
        out_cpp = cpp_m.rerank(
            q.astype(np.uint64), np.asarray(ids, dtype=np.int64),
            np.asarray(shifts, dtype=np.int32), rr_cpp_cfg, topk=len(ids))

        py_map = {i: (s, h, r, c) for i, s, h, r, c in zip(
            out_py.ids, out_py.scores, out_py.hist_scores,
            out_py.inlier_ratios, out_py.inlier_counts)}
        cpp_map = {i: (s, h, r, c) for i, s, h, r, c in zip(
            out_cpp.ids, out_cpp.scores, out_cpp.hist_scores,
            out_cpp.inlier_ratios, out_cpp.inlier_counts)}
        assert set(py_map) == set(cpp_map)
        for i in py_map:
            ps, ph, pr, pc = py_map[i]
            cs, ch, cr, cc = cpp_map[i]
            assert cc == pc, f"id {i} inlier count"
            np.testing.assert_allclose(cr, pr, atol=0)
            np.testing.assert_allclose(ch, ph, rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(cs, ps, rtol=1e-5, atol=1e-7)


## --- Phase 2.1: incremental database ---
##
## The DB-build path feeds shortlist (via the stacked histogram matrix) and
## beam/rerank (via get_scan_data).  verify() takes its tokens and keypoints
## as explicit arguments and never reads the DB store, so it cannot observe
## how the database was built and is not re-tested here.

MATCHER_CLASSES = [
    pytest.param(InLiER_Matcher, id="cpp"),
    pytest.param(ReferenceMatcher, id="numpy"),
]


def _new_matcher(cls):
    return cls(
        inlier_config=InLiER_Config(),
        shortlist_config=ShortlistConfig(**SL_KW),
        beam_score_config=BEAMScoreConfig(**BEAM_KW),
        rerank_config=RerankConfig(),
    )


def _fill(m, scans):
    for i, tid in enumerate(scans):
        m.add(i, InLiER_Tokens(token_id=tid))
    m.finalize(verbose=False)
    return m


def _stages(m, q, k):
    """shortlist → beam → rerank for one query."""
    sl = m.shortlist(InLiER_Tokens(token_id=q), topk=k, verbose=False)
    beam = m.beam_score(InLiER_Tokens(token_id=q), sl.ids,
                        topk=len(sl.ids), verbose=False)
    rr = m.rerank(InLiER_Tokens(token_id=q), beam.ids, beam.best_shifts,
                  topk=len(beam.ids), verbose=False)
    return sl, beam, rr


def _assert_stages_identical(a, b):
    """Exact, not allclose: same backend, same arithmetic, same order."""
    (sl_a, beam_a, rr_a), (sl_b, beam_b, rr_b) = a, b
    assert sl_a.ids == sl_b.ids and sl_a.scores == sl_b.scores
    assert beam_a.ids == beam_b.ids and beam_a.scores == beam_b.scores
    assert beam_a.best_shifts == beam_b.best_shifts
    assert rr_a.ids == rr_b.ids and rr_a.scores == rr_b.scores
    assert rr_a.inlier_counts == rr_b.inlier_counts


@pytest.mark.parametrize("cls", MATCHER_CLASSES)
def test_incremental_add_matches_bulk(db_and_queries, cls):
    """finalize() after every add == one finalize() over the whole DB."""
    db, queries = db_and_queries

    inc = _new_matcher(cls)
    for i, tid in enumerate(db):
        inc.add(i, InLiER_Tokens(token_id=tid))
        inc.finalize(verbose=False)      # appends exactly one row each time
    assert len(inc) == len(db)

    bulk = _new_matcher(cls)
    bulk.reset()                         # the plan's stated baseline
    _fill(bulk, db)

    np.testing.assert_array_equal(inc.db_ids, bulk.db_ids)
    for i in (0, len(db) // 2, len(db) - 1):
        a, b = inc.get_scan_data(i), bulk.get_scan_data(i)
        for key in ("hb", "rb", "sb", "ab", "token_id"):
            np.testing.assert_array_equal(a[key], b[key])
        assert a["max_active_hb"] == b["max_active_hb"]

    for q in queries[:4]:
        _assert_stages_identical(_stages(inc, q, len(db)),
                                 _stages(bulk, q, len(db)))


@pytest.mark.parametrize("cls", MATCHER_CLASSES)
@pytest.mark.parametrize("k", [1, 7, N_DB // 2, N_DB])
def test_shortlist_max_db_index_equals_prefix_db(db_and_queries, cls, k):
    """A bounded search == the same search on a DB of only those scans."""
    db, queries = db_and_queries
    full = _fill(_new_matcher(cls), db)
    prefix = _fill(_new_matcher(cls), db[:k])

    for q in queries[:3]:
        tok = InLiER_Tokens(token_id=q)
        got = full.shortlist(tok, topk=k, verbose=False, max_db_index=k)
        want = prefix.shortlist(tok, topk=k, verbose=False)
        assert got.ids == want.ids and got.scores == want.scores
        assert all(i < k for i in got.ids)

        ## topk_pct must resolve against the *bounded* database, not the
        ## full one -- otherwise an online run silently returns k/2 more
        ## candidates than the same offline run over the same scans.
        got = full.shortlist(tok, topk_pct=0.5, verbose=False, max_db_index=k)
        want = prefix.shortlist(tok, topk_pct=0.5, verbose=False)
        assert got.ids == want.ids and got.scores == want.scores


def test_max_db_index_edges(matchers, db_and_queries):
    _, cpp_m = matchers
    _, queries = db_and_queries
    q = queries[0].astype(np.uint64)
    cfg = bridge.to_cpp_stage_config(ip.ShortlistConfig,
                                     ShortlistConfig(**SL_KW))

    assert cpp_m.shortlist(q, cfg, topk=5, max_db_index=0).ids == []

    unbounded = cpp_m.shortlist(q, cfg, topk=N_DB)
    for bound in (N_DB, 10 ** 6, -1):    # clamped, over-sized, "no bound"
        out = cpp_m.shortlist(q, cfg, topk=N_DB, max_db_index=bound)
        assert list(out.ids) == list(unbounded.ids)
        assert list(out.scores) == list(unbounded.scores)


def test_reserve_does_not_change_results(db_and_queries):
    """reserve() is pure pre-allocation: identical scores, no reallocation."""
    db, queries = db_and_queries
    cfg = bridge.to_cpp_inlier_config(InLiER_Config())
    sl_cfg = bridge.to_cpp_stage_config(ip.ShortlistConfig,
                                        ShortlistConfig(**SL_KW))

    reserved = ip._Matcher(cfg)
    reserved.reserve(len(db))
    plain = ip._Matcher(cfg)
    for i, tid in enumerate(db):
        reserved.add(i, tid.astype(np.uint64))
        reserved.finalize()              # the online, one-row-at-a-time path
        plain.add(i, tid.astype(np.uint64))
    plain.finalize()

    k = N_DB // 3
    for q in queries[:3]:
        qq = q.astype(np.uint64)
        a = reserved.shortlist(qq, sl_cfg, topk=k, max_db_index=k)
        b = plain.shortlist(qq, sl_cfg, topk=k, max_db_index=k)
        assert list(a.ids) == list(b.ids)
        assert list(a.scores) == list(b.scores)
        assert max(a.ids) < k
