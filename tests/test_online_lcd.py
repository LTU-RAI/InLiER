"""Online loop-closure protocol -- causality, streaming, and the run itself.

The session is synthetic but the descriptors are real: frames are drawn from
the cached HeLiPR descriptors and laid out on a circle traversed **twice**, so
frame ``t`` and frame ``t + LAP`` sit at the same position with byte-identical
tokens.  Every revisit is therefore known exactly, which is what makes the
causality assertions below sharp -- a protocol that leaked a future frame, or
that filtered the exclusion window after retrieval instead of inside it, would
still look plausible on real data but fails here.

The encoded session is written straight into a temporary descriptor cache, so
``run()`` never touches a point cloud and the whole file stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from inlier.config import load, resolve
from inlier.eval import gt as gtmod
from inlier.eval.encode import EncodedSequence, _save_cache, cache_path
from inlier.eval.pipeline import build_matcher, online_shortlist_stage
from inlier.eval.protocols.online_lcd import OnlineLCDSpec, run

LAP = 12                 # frames per lap
N_FRAMES = 2 * LAP       # two laps: every frame in lap 2 revisits one in lap 1
RADIUS = 40.0            # circle radius, metres
STEP = 2 * np.pi * RADIUS / LAP     # arc length between consecutive frames
# A revisit is the *same* place (distance 0).  Half a step is comfortably under
# the 2*R*sin(pi/LAP) chord to the circle-adjacent frame, so the only positives
# are the planted twins and the ground truth stays exactly known.
GT_RADIUS = STEP * 0.5


class _CachedSource:
    """A source whose descriptors are already cached, so ``load`` never runs."""

    name = "stub"

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def load(self):
        raise AssertionError("load() must not be called: the cache is seeded")

    def describe(self):
        return {"path": self.tag, "sensor": ""}


@pytest.fixture(scope="module")
def session(cached_descriptors, tmp_path_factory):
    """(spec-ready cache dir, source, resolved config, positions)."""
    d = cached_descriptors
    offsets, tids = d["offsets"], d["token_ids"]
    n_avail = len(offsets) - 1
    assert n_avail >= LAP, f"need {LAP} cached scans, have {n_avail}"

    step = n_avail // LAP
    scans = [tids[offsets[i * step]:offsets[i * step + 1]].astype(np.uint32)
             for i in range(LAP)]

    from inlier.core.Dataclasses import InLiER_Tokens

    angles = 2 * np.pi * (np.arange(N_FRAMES) % LAP) / LAP
    positions = np.stack([RADIUS * np.cos(angles),
                          RADIUS * np.sin(angles),
                          np.zeros(N_FRAMES)], axis=1)
    poses = np.tile(np.eye(4), (N_FRAMES, 1, 1))
    poses[:, :3, 3] = positions

    # Verification indexes keypoints by token, so there must be exactly one
    # keypoint per token.  A lap's keypoints are reused on the revisit, which
    # is what makes the planted loop closure geometrically consistent too.
    rng = np.random.default_rng(0)
    lap_kp = [rng.uniform(-30.0, 30.0, size=(len(s), 3)) for s in scans]

    tokens, kp = [], []
    for t in range(N_FRAMES):
        tokens.append(InLiER_Tokens(token_id=scans[t % LAP]))
        kp.append(lap_kp[t % LAP])
    T_grounds = [np.eye(4) for _ in range(N_FRAMES)]
    enc = EncodedSequence(tokens, kp, kp, T_grounds, positions, poses)

    resolved = resolve(load(), mode="eval")
    cache_dir = tmp_path_factory.mktemp("lcd_cache")
    source = _CachedSource("loop")
    _save_cache(cache_path(cache_dir, f"{source.tag}_Undistorted",
                           resolved.inlier, resolved.voxel_size),
                enc, verbose=False)
    return cache_dir, source, resolved, positions, enc


def _spec(session, tmp_path, exclusion, **kw):
    cache_dir, source, resolved, _, _ = session
    return OnlineLCDSpec(
        resolved=resolved, source=source, exclusion=exclusion,
        output_dir=tmp_path, max_pose_dist=GT_RADIUS,
        cache_dir=cache_dir, verbose=False, tag="test", **kw)


# --- causality -------------------------------------------------------------


def test_no_candidate_comes_from_the_future(session, tmp_path):
    """Retrieval and ground truth must obey the same cutoff."""
    cache_dir, source, resolved, positions, enc = session
    exclusion = gtmod.Exclusion(frames=3)
    bounds = [exclusion.cutoff(t) for t in range(N_FRAMES)]

    matcher = build_matcher(resolved, [], verbose=False)
    ranked, sims, latency = online_shortlist_stage(
        matcher, enc.tokens, bounds, verbose=False)

    policy = gtmod.Causal(positions=positions, exclusion=exclusion,
                          max_pose_dist=GT_RADIUS)
    ground_truth = gtmod.build(policy, N_FRAMES)

    for t in range(N_FRAMES):
        assert all(d < bounds[t] for d in ranked[t]), f"frame {t} saw the future"
        assert all(d < bounds[t] for d in ground_truth[t].tolist())
        assert set(sims[t]) == set(ranked[t])
    assert latency.shape == (N_FRAMES,)
    assert (latency >= 0).all()


def test_streaming_equals_prefix_database(session):
    """The growing database gives exactly what a prebuilt prefix would."""
    cache_dir, source, resolved, positions, enc = session
    exclusion = gtmod.Exclusion(frames=3)
    bounds = [exclusion.cutoff(t) for t in range(N_FRAMES)]

    matcher = build_matcher(resolved, [], verbose=False)
    ranked, sims, _ = online_shortlist_stage(
        matcher, enc.tokens, bounds, verbose=False)

    for t in range(N_FRAMES):
        if bounds[t] <= 0:
            assert ranked[t] == [] and sims[t] == {}
            continue
        prefix = build_matcher(resolved, enc.tokens[:bounds[t]], verbose=False)
        want = prefix.shortlist(enc.tokens[t], topk=bounds[t], verbose=False)
        assert ranked[t] == list(want.ids), f"frame {t} ranking differs"
        assert [sims[t][d] for d in ranked[t]] == [float(s) for s in want.scores]


def test_frames_and_metres_windows_agree(session):
    """The same physical window in two units gives the same cutoffs.

    Frames are evenly spaced on the circle, so a k-frame window is exactly a
    k*STEP window.  ``metres`` is measured on travelled arc length, which on a
    two-lap circle keeps increasing -- it does not reset at the revisit.
    """
    from inlier.eval.datasets.base import arc_length

    _, _, _, positions, _ = session
    arc = arc_length(positions)
    for k in (1, 3, 5):
        by_frames = [gtmod.Exclusion(frames=k).cutoff(t) for t in range(N_FRAMES)]
        by_metres = [gtmod.Exclusion(metres=k * STEP).cutoff(t, None, arc)
                     for t in range(N_FRAMES)]
        assert by_frames == by_metres, f"k={k}: {by_frames} != {by_metres}"


# --- the protocol end to end ----------------------------------------------


def test_run_finds_the_planted_revisits(session, tmp_path):
    """Second-lap frames must retrieve their first-lap twins."""
    result = run(_spec(session, tmp_path, gtmod.Exclusion(frames=3)))

    assert result.protocol == "online_lcd"
    res = result.results
    assert res["protocol"] == "online_lcd"
    assert res["candidate_filter"]["filter"] == "causal"
    assert res["dataset"]["n_frames"] == N_FRAMES

    lc = res["loop_closure"]
    # Frame t+LAP has byte-identical tokens to frame t, so top-1 must find it.
    assert lc["recall_at_1"] == pytest.approx(1.0), lc
    assert 0.0 <= lc["max_recall_at_full_precision"] <= 1.0
    assert lc["f1_max"] > 0.5, lc

    # These planted twins are perfectly separable, so recall is constant across
    # the whole threshold sweep and the trapezoid over that degenerate PR curve
    # is 0 by construction.  A property of the metric on synthetic data, not of
    # the protocol -- pinned so a future reader does not file it as a bug.
    assert res["stage1"]["pr_auc"] == 0.0

    lat = res["latency"]
    assert lat["n_frames"] == sum(1 for t in range(N_FRAMES) if t - 3 > 0)
    assert lat["max_ms"] >= lat["median_ms"] >= 0.0

    for name in ("results", "candidates", "ranked"):
        assert result.artifacts[name].exists(), name


def test_zero_exclusion_withholds_nothing(session, tmp_path):
    """``frames=0`` is the degenerate window: every past frame is a candidate.

    On a real, densely sampled sequence this is what makes the benchmark
    meaningless -- the immediate predecessor is a metre away and matches
    trivially.  These synthetic frames are a full STEP apart, so the collapse
    shows up as coverage rather than as inflated recall: every frame from the
    first onwards now queries, and the planted twins are still all found.  The
    dense-sequence version of this check belongs on real HeLiPR data.
    """
    res = run(_spec(session, tmp_path, gtmod.Exclusion(frames=0))).results
    assert res["config"]["exclusion"] == {"unit": "frames", "value": 0}
    assert res["dataset"]["n_queried"] == N_FRAMES - 1     # every frame but t=0
    assert res["loop_closure"]["recall_at_1"] == pytest.approx(1.0)


def test_no_revisit_is_an_error_not_an_empty_run(session, tmp_path):
    """A window longer than the loop must say so, not report zeros."""
    with pytest.raises(ValueError, match="no frame has a revisit"):
        run(_spec(session, tmp_path, gtmod.Exclusion(frames=N_FRAMES + 5)))


# --- the search radius -----------------------------------------------------


def test_radius_zero_is_the_global_search(session, tmp_path):
    """0 must be exactly the unrestricted run, not a near-miss of it."""
    a = run(_spec(session, tmp_path / "a", gtmod.Exclusion(frames=3))).results
    b = run(_spec(session, tmp_path / "b", gtmod.Exclusion(frames=3),
                  search_radius=0.0)).results

    assert a["loop_closure"] == b["loop_closure"]
    assert a["confusion"] == b["confusion"]
    assert a["stage1"] == b["stage1"]
    assert a["candidate_filter"]["filter"] == "causal"
    assert a["candidate_filter"]["uses_pose_oracle"] is False


def test_radius_restricts_candidates_to_the_local_map(session):
    """Only frames within the radius of the query's true pose survive."""
    _, _, resolved, positions, enc = session
    exclusion = gtmod.Exclusion(frames=3)
    bounds = [exclusion.cutoff(t) for t in range(N_FRAMES)]
    radius = GT_RADIUS * 2

    cand = gtmod.RadiusFilter(positions=positions, radius=radius,
                              exclusion=exclusion)
    matcher = build_matcher(resolved, [], verbose=False)
    ranked, _, _ = online_shortlist_stage(
        matcher, enc.tokens, bounds, verbose=False, allowed=cand.allowed_mask)

    for t in range(N_FRAMES):
        for d in ranked[t]:
            assert d < bounds[t]
            gap = np.linalg.norm(positions[t, :2] - positions[d, :2])
            assert gap <= radius, f"frame {t} kept {d} at {gap:.1f} m"

    # And it really is a restriction: the global run returns strictly more.
    m2 = build_matcher(resolved, [], verbose=False)
    global_ranked, _, _ = online_shortlist_stage(
        m2, enc.tokens, bounds, verbose=False)
    assert sum(map(len, global_ranked.values())) > sum(map(len, ranked.values()))


def test_radius_keeps_the_ranking_of_what_it_kept(session):
    """Filtering is exact: the survivors keep their global order and scores.

    The stage asks for ``topk=bound``, so the whole causal set is scored before
    anything is dropped.  A radius run must therefore be a sub-sequence of the
    global run, never a re-ranking of it.
    """
    _, _, resolved, positions, enc = session
    exclusion = gtmod.Exclusion(frames=3)
    bounds = [exclusion.cutoff(t) for t in range(N_FRAMES)]
    cand = gtmod.RadiusFilter(positions=positions, radius=GT_RADIUS * 2,
                              exclusion=exclusion)

    m1 = build_matcher(resolved, [], verbose=False)
    r_glob, s_glob, _ = online_shortlist_stage(m1, enc.tokens, bounds, verbose=False)
    m2 = build_matcher(resolved, [], verbose=False)
    r_rad, s_rad, _ = online_shortlist_stage(
        m2, enc.tokens, bounds, verbose=False, allowed=cand.allowed_mask)

    for t in range(N_FRAMES):
        kept = set(r_rad[t])
        assert r_rad[t] == [d for d in r_glob[t] if d in kept]
        for d in kept:
            assert s_rad[t][d] == s_glob[t][d]


def test_radius_run_is_flagged_as_an_oracle(session, tmp_path):
    """A number produced with the radius must never look like one without."""
    res = run(_spec(session, tmp_path, gtmod.Exclusion(frames=3),
                    search_radius=GT_RADIUS * 2)).results
    cf = res["candidate_filter"]
    assert cf["filter"] == "radius"
    assert cf["uses_pose_oracle"] is True
    assert cf["radius_m"] == pytest.approx(GT_RADIUS * 2)
    assert res["config"]["search_radius"] == pytest.approx(GT_RADIUS * 2)
    # The planted twins are at distance 0, so a radius run still finds them.
    assert res["loop_closure"]["recall_at_1"] == pytest.approx(1.0)


def test_radius_below_the_gt_distance_is_rejected(session, tmp_path):
    """A radius that hides real revisits is a misconfiguration, not a result."""
    with pytest.raises(ValueError, match="smaller than"):
        run(_spec(session, tmp_path, gtmod.Exclusion(frames=3),
                  search_radius=GT_RADIUS / 2))


# --- exclusion units the data cannot support -------------------------------


class _NoTimestampSource(_CachedSource):
    """What the generic loader hands back: zeros 'for API parity'."""

    def __init__(self, tag, n):
        super().__init__(tag)
        self._n = n

    def load(self):
        from inlier.eval.datasets.base import Sequence

        return Sequence(poses=[np.eye(4)] * self._n,
                        point_clouds=[np.zeros((1, 3))] * self._n)


def test_seconds_window_on_a_dataset_without_timestamps(session, tmp_path):
    """Zeros must be rejected as absent, not silently read as t=0.

    Every cutoff would collapse to 0, no frame would ever query, and the run
    would then fail complaining that the *window* is too long -- which sends
    the reader looking in exactly the wrong place.
    """
    cache_dir, source, resolved, _, _ = session
    spec = OnlineLCDSpec(
        resolved=resolved, source=_NoTimestampSource(source.tag, N_FRAMES),
        exclusion=gtmod.Exclusion(seconds=30.0), output_dir=tmp_path,
        max_pose_dist=GT_RADIUS, cache_dir=cache_dir, verbose=False, tag="t")

    with pytest.raises(ValueError, match="no usable pose timestamps"):
        run(spec)


def test_metres_window_on_a_stationary_sequence(session, tmp_path, monkeypatch):
    """A trajectory that never moves cannot carry a distance window either."""
    import inlier.eval.protocols.online_lcd as mod

    monkeypatch.setattr(mod, "arc_length",
                        lambda positions: np.zeros(len(positions)))
    with pytest.raises(ValueError, match="no usable travelled distance"):
        run(_spec(session, tmp_path, gtmod.Exclusion(metres=50.0)))


def test_metres_window_works_on_a_moving_sequence(session, tmp_path):
    """The guard must not fire on a real trajectory."""
    res = run(_spec(session, tmp_path, gtmod.Exclusion(metres=STEP * 3))).results
    assert res["config"]["exclusion"] == {"unit": "metres",
                                          "value": pytest.approx(STEP * 3)}
    assert res["dataset"]["n_queried"] > 0


# --- the loop-closure curve's population -----------------------------------


def test_loop_closure_curve_scores_every_frame(session, tmp_path):
    """Frames that close no loop must be able to fail.

    ``metrics.pr_curve`` scores only queries that have a ground-truth positive,
    which on a session where most frames close nothing makes the dominant
    failure -- firing where there is no loop -- unmeasurable, and drives F1max
    to the accept-everything point at threshold 0.  The loop-closure block is
    swept over all frames instead, so a threshold of 0 is now punished by the
    lap-1 frames that have no twin behind them.
    """
    res = run(_spec(session, tmp_path, gtmod.Exclusion(frames=3))).results
    lc = res["loop_closure"]

    assert lc["population"] == "all_queries"
    # The fixture has such frames: lap 1 revisits nothing.
    assert res["dataset"]["n_with_ground_truth"] < res["dataset"]["n_frames"]
    assert lc["f1_max_threshold"] > 0.0, lc

    # The stage PR-AUCs keep the narrower retrieval population, so switching
    # the headline did not quietly redefine them too.
    assert "population" not in res["stage1"]
