"""The retrieval pipeline: MINT -> BEAM -> rerank -> verify -> GICP.

One implementation of each stage, shared by every protocol and by
``inlier run``.  Ported from ``evaluate_inlier_helipr.py`` (:363 ``compute_ranked_lists``,
:388 ``compute_beam_ranked_lists``, :427 ``compute_rerank_ranked_lists``,
:497 ``compute_verify_similarity_map``), where each existed twice.  Behaviour is
unchanged.

Evaluation vs deployment
------------------------
Every stage here is run in *evaluation* mode: candidates are scored with
``topk=len(shortlist)`` and the stage ``score_threshold`` set to -2.0 by
:func:`inlier.config.resolve`, so nothing is filtered out before the metrics
see it.  A deployment run wants the opposite -- the configured ``topk`` and
threshold, so the stage actually prunes.  That is the ``mode`` argument on
``resolve``, not a difference in this code.

Scoring note
------------
A verification that fails scores ``0.0``, not the ``-1.0`` sentinel the
original docstring described -- the code never wrote -1.0.  Since the threshold
sweep runs over non-negative scores and 0.0 is below every candidate threshold
in practice, the two behave the same in the sweep; the docstring was simply
describing an intent the implementation did not have.  The implemented
behaviour is kept, because it is what produced the published numbers.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from inlier.core.Dataclasses import (
    InLiER_Keypoints,
    InLiER_Tokens,
    VerifyConfig,
)

RankedLists = Dict[int, List[int]]
SimilarityMap = Dict[int, Dict[int, float]]
ShiftsMap = Dict[int, Dict[int, int]]


def _progress(iterable, desc: str, verbose: bool):
    if not verbose:
        return iterable
    try:
        import tqdm
        return tqdm.tqdm(iterable, desc=desc)
    except ImportError:
        return iterable


def minimal_keypoints(
    p_aligned: np.ndarray,
    p_sensor: Optional[np.ndarray] = None,
    T_ground: Optional[np.ndarray] = None,
) -> InLiER_Keypoints:
    """Keypoints for verification / GICP.

    With ``p_sensor`` and ``T_ground`` the keypoints carry real frame
    information, so verification returns a pose in the sensor frame.  Without
    them the legacy path assumes ``T_ground = I``, which makes the returned
    pose ground-aligned rather than sensor-frame -- correct only if the cache
    predates those fields.
    """
    if p_sensor is not None and T_ground is not None:
        return InLiER_Keypoints(p=p_sensor, T_ground=T_ground)
    return InLiER_Keypoints(p=p_aligned, T_ground=np.eye(4, dtype=np.float64))


# ---------------------------------------------------------------------------
#  Stage 1 -- MINT shortlist
# ---------------------------------------------------------------------------

def shortlist_one(matcher, tokens: InLiER_Tokens, topk: int,
                  max_db_index: Optional[int] = None):
    """One query's MINT shortlist, as ``(ids, scores)``.

    ``max_db_index`` bounds the search *inside* the matcher's scoring loop, so
    an excluded recent neighbour cannot crowd a real closure out of the top-k.
    """
    kw = {} if max_db_index is None else {"max_db_index": max_db_index}
    out = matcher.shortlist(tokens, topk=topk, verbose=False, **kw)
    return list(out.ids), [float(s) for s in out.scores]


def shortlist_stage(
    matcher,
    q_tokens: List[InLiER_Tokens],
    n_db: int,
    verbose: bool = True,
) -> Tuple[RankedLists, SimilarityMap]:
    """Rank every database scan by MINT score, per query."""
    ranked: RankedLists = {}
    sims: SimilarityMap = {}

    for j in _progress(range(len(q_tokens)), "  Stage-1 (MINT) retrieval", verbose):
        ids, scores = shortlist_one(matcher, q_tokens[j], topk=n_db)
        ranked[j] = ids
        sims[j] = {d: s for d, s in zip(ids, scores)}

    return ranked, sims


def online_shortlist_stage(
    matcher,
    tokens: List[InLiER_Tokens],
    bounds: Sequence[int],
    verbose: bool = True,
    allowed: Optional[Callable[[int, int], np.ndarray]] = None,
) -> Tuple[RankedLists, SimilarityMap, np.ndarray]:
    """Stream one sequence: query the past, then append the frame.

    ``bounds[t]`` is the exclusive database bound for frame ``t``.  The matcher
    applies it inside its own scoring loop, so an excluded recent neighbour
    cannot crowd a real loop closure out of the top-k -- filtering the results
    afterwards would silently cost recall exactly where the window is tight.

    ``allowed(t, bound) -> (bound,) bool`` narrows the pool further than a
    prefix can express -- a search radius keeps scattered indices, not a
    contiguous range, so the matcher's bound cannot carry it.  Applying it
    after retrieval is nevertheless *exact* here, because the call above asks
    for ``topk=bound``: the whole causal set is scored and returned, so nothing
    a filter would have kept was ever dropped for want of a rank.

    The returned per-frame wall clock covers the query *and* the insertion,
    which is the whole cost an online system pays per frame.  It is only
    meaningful because the database really does grow one frame at a time.
    With ``allowed`` set it over-reports: the search still scans every causal
    frame, where a deployed radius-limited system would not.
    """
    ranked: RankedLists = {}
    sims: SimilarityMap = {}
    latency_ms = np.zeros(len(tokens), dtype=np.float64)

    matcher.reserve(len(tokens))
    for t in _progress(range(len(tokens)), "  Online MINT retrieval", verbose):
        bound = int(bounds[t])
        t0 = time.perf_counter()
        if bound > 0:
            ids, scores = shortlist_one(matcher, tokens[t], topk=bound,
                                        max_db_index=bound)
            if allowed is not None:
                mask = allowed(t, bound)
                keep = [i for i, d in enumerate(ids) if mask[d]]
                ids = [ids[i] for i in keep]
                scores = [scores[i] for i in keep]
            ranked[t] = ids
            sims[t] = {d: s for d, s in zip(ids, scores)}
        else:
            ranked[t], sims[t] = [], {}
        matcher.add(t, tokens[t])
        matcher.finalize(verbose=False)
        latency_ms[t] = (time.perf_counter() - t0) * 1e3

    return ranked, sims, latency_ms


# ---------------------------------------------------------------------------
#  Stage 2 -- BEAM azimuth-shift rerank
# ---------------------------------------------------------------------------

def beam_one(matcher, tokens: InLiER_Tokens, shortlist: List[int]):
    """One query's BEAM rerank, as ``(ids, scores, best_shifts)``.

    Every shortlisted candidate is scored (``topk=len(shortlist)``) and nothing
    is rank-filtered.
    """
    out = matcher.beam_score(tokens, shortlist, topk=len(shortlist), verbose=False)
    return list(out.ids), list(out.scores), list(out.best_shifts)


def beam_stage(
    matcher,
    q_tokens: List[InLiER_Tokens],
    ranked_s1: RankedLists,
    stage1_topk: int,
    verbose: bool = True,
) -> Tuple[RankedLists, SimilarityMap, ShiftsMap]:
    """Rerank the stage-1 shortlist by bitmask alignment, and estimate yaw.

    Every shortlisted candidate is scored (``topk=len(shortlist)``) and nothing
    is rank-filtered, so Recall@N and the PR sweep see the complete stage-2
    ordering rather than a list truncated at the deployment ``topk``.
    """
    ranked: RankedLists = {}
    sims: SimilarityMap = {}
    shifts: ShiftsMap = {}

    for j in _progress(range(len(q_tokens)), "  Stage-2 (BEAM) reranking", verbose):
        ids, scores, best_shifts = beam_one(matcher, q_tokens[j],
                                            ranked_s1[j][:stage1_topk])
        ranked[j] = ids
        sims[j] = {sid: sc for sid, sc in zip(ids, scores)}
        shifts[j] = {sid: sh for sid, sh in zip(ids, best_shifts)}

    return ranked, sims, shifts


# ---------------------------------------------------------------------------
#  Rerank -- 4-D histogram scoring (off by default)
# ---------------------------------------------------------------------------

def rerank_one(matcher, tokens: InLiER_Tokens, shortlist: List[int],
               prior_shifts: Optional[Dict[int, int]]):
    """One query's 4-D histogram rerank, as ``(ids, scores, best_shifts)``.

    ``prior_shifts`` is the BEAM shift per candidate; ``None`` means no prior,
    which is the same zero the batch path passes.
    """
    if not shortlist:
        return [], [], []
    prior = ([prior_shifts.get(sid, 0) for sid in shortlist]
             if prior_shifts is not None else [0] * len(shortlist))
    out = matcher.rerank(tokens, shortlist, prior, topk=len(shortlist),
                         verbose=False)
    return list(out.ids), list(out.scores), list(out.best_shifts)


def rerank_stage(
    matcher,
    q_tokens: List[InLiER_Tokens],
    ranked_prev: RankedLists,
    shifts_prev: Optional[ShiftsMap],
    input_topk: int,
    verbose: bool = True,
) -> Tuple[RankedLists, SimilarityMap, ShiftsMap]:
    """Rerank with 4-D token-histogram scores, pre-aligned by the BEAM shift."""
    ranked: RankedLists = {}
    sims: SimilarityMap = {}
    shifts: ShiftsMap = {}

    for j in _progress(range(len(q_tokens)), "  Rerank", verbose):
        ids, scores, best_shifts = rerank_one(
            matcher, q_tokens[j], ranked_prev[j][:input_topk],
            shifts_prev[j] if shifts_prev is not None else None)
        ranked[j] = ids
        sims[j] = {sid: sc for sid, sc in zip(ids, scores)}
        shifts[j] = {sid: sh for sid, sh in zip(ids, best_shifts)}

    return ranked, sims, shifts


# ---------------------------------------------------------------------------
#  Verify -- token-guided geometric verification
# ---------------------------------------------------------------------------

def verify_one(
    matcher,
    q_tokens: InLiER_Tokens,
    q_kp: InLiER_Keypoints,
    db_tokens: List[InLiER_Tokens],
    db_kp: Callable[[int], InLiER_Keypoints],
    cands: List[int],
    shifts: Optional[Dict[int, int]],
    verify_cfg: VerifyConfig,
    top_v: int = 1,
):
    """Verify one query's top-V candidates.

    Returns ``(scores, verified, outputs)``: the per-candidate score (``0.0``
    when verification fails, which is what the code has always returned), the
    candidates actually verified in *retrieval* rank order, and the raw
    ``VerifyOutput`` per candidate.

    ``db_kp`` is a callable rather than four parallel lists so a streaming
    caller, which has the database in a different shape, can serve it without
    rebuilding them.
    """
    if not cands:
        return {}, [], {}

    verified = cands[:top_v]
    scores: Dict[int, float] = {}
    outputs: Dict[int, Any] = {}
    for db_id in verified:
        shift = shifts.get(db_id, 0) if shifts is not None else 0
        out = matcher.verify(
            q_tokens, q_kp, db_tokens[db_id], db_kp(db_id),
            azimuth_shift=shift, config=verify_cfg, verbose=False,
        )
        scores[db_id] = out.keypoint_inlier_ratio if out.success else 0.0
        outputs[db_id] = out
    return scores, verified, outputs


def verify_stage(
    matcher,
    q_tokens: List[InLiER_Tokens],
    q_kp_aligned: List[np.ndarray],
    db_tokens: List[InLiER_Tokens],
    db_kp_aligned: List[np.ndarray],
    ranked: RankedLists,
    shifts_map: Optional[ShiftsMap],
    verify_cfg: VerifyConfig,
    top_v: int = 1,
    q_kp_sensor: Optional[List[np.ndarray]] = None,
    db_kp_sensor: Optional[List[np.ndarray]] = None,
    q_T_grounds: Optional[List[np.ndarray]] = None,
    db_T_grounds: Optional[List[np.ndarray]] = None,
    verbose: bool = True,
) -> Tuple[SimilarityMap, RankedLists, Dict[Tuple[int, int], Any]]:
    """Verify the top-V candidates of every query.

    Returns the per-pair score (``keypoint_inlier_ratio``, or ``0.0`` when
    verification fails), the candidates in *retrieval* rank order -- the PR
    sweep walks them in that order, falling through to the next candidate when
    the first is below threshold -- and the raw ``VerifyOutput`` per pair, which
    is what the pose-error metrics and the per-pair CSV are built from.

    Every query is verified, including those with no GT positive: their false
    positives are exactly what the confusion matrix needs to count.
    """
    n_queries = len(q_tokens)
    sims: SimilarityMap = {}
    rank_order: RankedLists = {}
    outputs: Dict[Tuple[int, int], Any] = {}

    have_sensor = (q_kp_sensor is not None and db_kp_sensor is not None
                   and q_T_grounds is not None and db_T_grounds is not None)

    desc = f"  Verify (top-{top_v})" if top_v > 1 else "  Verify (top-1)"

    def db_kp(db_id: int) -> InLiER_Keypoints:
        if have_sensor:
            return minimal_keypoints(db_kp_aligned[db_id], db_kp_sensor[db_id],
                                     db_T_grounds[db_id])
        return minimal_keypoints(db_kp_aligned[db_id])

    for j in _progress(range(n_queries), desc, verbose):
        q_kp = (minimal_keypoints(q_kp_aligned[j], q_kp_sensor[j], q_T_grounds[j])
                if have_sensor else minimal_keypoints(q_kp_aligned[j]))
        q_sims, to_verify, per_db = verify_one(
            matcher, q_tokens[j], q_kp, db_tokens, db_kp,
            ranked.get(j, []),
            shifts_map.get(j) if shifts_map is not None else None,
            verify_cfg, top_v)
        sims[j] = q_sims
        rank_order[j] = to_verify
        for db_id, out in per_db.items():
            outputs[(j, db_id)] = out

    return sims, rank_order, outputs


# ---------------------------------------------------------------------------
#  Database construction
# ---------------------------------------------------------------------------

def build_matcher(resolved, db_tokens: List[InLiER_Tokens], verbose: bool = True):
    """Populate and finalize a matcher from an encoded database.

    ``rerank_config`` falls back to the dataclass default when reranking is
    off: the matcher stores it unconditionally, and ``None`` would only fail
    later, at a ``rerank()`` call that in that configuration never happens.
    ``verify_config`` is deliberately not passed -- verification takes its
    config per call, so binding one here would be a second source of truth.
    """
    from inlier.core.Dataclasses import RerankConfig
    from inlier.core.InLiER_Matcher import InLiER_Matcher

    matcher = InLiER_Matcher(
        inlier_config=resolved.inlier,
        shortlist_config=resolved.shortlist,
        beam_score_config=resolved.beam,
        rerank_config=resolved.rerank if resolved.rerank is not None else RerankConfig(),
    )
    for i, tok in enumerate(db_tokens):
        matcher.add(i, tok)
    matcher.finalize(verbose=verbose)
    return matcher
