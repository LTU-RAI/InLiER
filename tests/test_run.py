"""``inlier run`` -- loop closures with no ground truth.

Everything here guards one claim: **the poses are odometry, and no pose ever
decides anything.** They accumulate submaps, they may bound the search, they
fill two diagnostic columns -- and a closure is accepted by its verification
score against the threshold, full stop. If that ever stops being true, the
command silently becomes an evaluation wearing a deployment's name, and
nothing in its output would say so.

The second claim, almost as easy to break, is that the score artifacts record
what the *matcher computed* rather than what this *run decided*: no threshold
touches them, and "never scored" is `NaN`/blank, never `0.0` -- a value the
stages genuinely return.

The session is the planted two-lap circle from ``test_online_lcd``: frame `t`
and frame `t + LAP` are the same place with byte-identical tokens, so every
closure is known without any of it reaching the code under test as ground
truth.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from inlier.config import load, resolve
from inlier.eval import gt as gtmod
from inlier.eval.deploy import DeploySpec, accept, run, score_matrices
from tests.test_online_lcd import GT_RADIUS, LAP, N_FRAMES, session  # noqa: F401

THRESHOLD = 0.3

#: The fixture seeds a descriptor cache and refuses `load()`, so GICP must not
#: ask for raw clouds.  Keypoint refinement is the documented fallback and
#: exercises the same code path.  Deploy mode, because that is what `run` uses.
CONFIG = ["gicp.use_raw_clouds=false"]


def _resolved(*overrides):
    return resolve(load(overrides=CONFIG + list(overrides)), mode="deploy")


def _spec(session, tmp_path, **kw):
    cache_dir, source, _, _, _ = session
    kw.setdefault("exclusion", gtmod.Exclusion(frames=3))
    kw.setdefault("threshold", THRESHOLD)
    kw.setdefault("resolved", _resolved())
    return DeploySpec(
        source=source, output_dir=tmp_path, cache_dir=cache_dir,
        verbose=False, tag="test", **kw)


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# --- the claim this command lives or dies on -------------------------------


def test_odometry_never_gates_a_closure(session, tmp_path):
    """Drift the poses arbitrarily; the closures must not move.

    Poses reach the pipeline through submap accumulation and (optionally) the
    search radius. With the radius off and the descriptors already cached, a
    pose can only influence the result by leaking into the accept decision --
    which is exactly what must never happen. Here the poses are shifted by a
    kilometre, which would wreck any distance-based gate.
    """
    from inlier.eval.encode import EncodedSequence

    cache_dir, source, _, _, enc = session
    honest = run(_spec(session, tmp_path / "a")).results

    drifted_poses = [np.array(p, dtype=np.float64) for p in enc.poses]
    for p in drifted_poses:
        p[:3, 3] += 1000.0
    drifted = EncodedSequence(enc.tokens, enc.kp_aligned, enc.kp_sensor,
                              enc.T_grounds, enc.positions + 1000.0,
                              drifted_poses)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("inlier.eval.deploy.encode_sequence",
                   lambda *a, **k: drifted)
        moved = run(_spec(session, tmp_path / "b")).results

    assert moved["closures"]["n_closures"] == honest["closures"]["n_closures"]
    assert _closure_pairs(tmp_path / "b") == _closure_pairs(tmp_path / "a")


def _closure_pairs(out_dir):
    return {(r["query_idx"], r["db_idx"], r["score"])
            for r in _rows(out_dir / "closures_test.csv")}


def test_accept_takes_every_verified_candidate_at_or_above_the_threshold():
    """The whole decision, in isolation.

    Success is required explicitly rather than inferred from the threshold
    being above zero -- a failed verification scores exactly 0.0, and relying
    on the threshold to exclude it is how one ends up in the output.
    """
    ok = type("V", (), {"success": True})()
    bad = type("V", (), {"success": False})()
    sims = {0: {5: 0.9, 6: 0.4, 7: 0.2}, 1: {5: 0.8}}
    outs = {(0, 5): ok, (0, 6): ok, (0, 7): ok, (1, 5): bad}

    got = accept(sims, outs, 0.3)
    assert got == [(0, 5, 0.9, 0), (0, 6, 0.4, 1)]     # 0.2 below, (1,5) failed


def test_accept_ranks_by_descending_score():
    ok = type("V", (), {"success": True})()
    sims = {0: {1: 0.4, 2: 0.9, 3: 0.6}}
    outs = {(0, d): ok for d in (1, 2, 3)}
    assert [d for _, d, _, _ in accept(sims, outs, 0.0)] == [2, 3, 1]


# --- the closures ----------------------------------------------------------


def test_every_row_is_verified_and_above_the_threshold(session, tmp_path):
    result = run(_spec(session, tmp_path))
    rows = _rows(result.artifacts["closures"])
    assert rows
    for row in rows:
        assert float(row["score"]) >= THRESHOLD


def test_the_planted_revisits_are_found(session, tmp_path):
    """Sanity: the closures are the twins, not noise."""
    rows = _rows(run(_spec(session, tmp_path)).artifacts["closures"])
    best = {int(r["query_idx"]): int(r["db_idx"])
            for r in rows if int(r["rank"]) == 0}
    assert best, "no closure at all"
    for q, d in best.items():
        assert q - d == LAP, (q, d)


def test_nothing_qualifying_is_dropped(session, tmp_path):
    """Every verified pair at or above the threshold reaches the CSV.

    On this fixture only the planted twin verifies successfully per query, so
    the multi-candidate case is exercised against `accept` directly above;
    what matters here is the completeness of the file against the run's own
    verification output.
    """
    result = run(_spec(session, tmp_path, threshold=1e-6))
    rows = _rows(result.artifacts["closures"])
    verified = {(r["query_idx"], r["db_idx"])
                for r in _rows(result.artifacts["verify"]) if r["success"] == "1"}
    assert {(r["query_idx"], r["db_idx"]) for r in rows} == verified

    per_query = {}
    for row in rows:
        per_query.setdefault(int(row["query_idx"]), []).append(row)
    for rows_q in per_query.values():
        ordered = sorted(rows_q, key=lambda r: int(r["rank"]))
        scores = [float(r["score"]) for r in ordered]
        assert scores == sorted(scores, reverse=True)
        assert [int(r["rank"]) for r in ordered] == list(range(len(ordered)))


def test_a_threshold_nothing_clears_is_not_an_error(session, tmp_path):
    """A session that revisited nothing is a valid answer, not a failure."""
    # Scores are ratios capped at 1.0, and this fixture's planted twins are
    # byte-identical, so they verify at exactly 1.0 -- nothing below that is
    # unreachable here.
    result = run(_spec(session, tmp_path, threshold=1.5))
    assert result.results["closures"]["n_closures"] == 0
    assert _rows(result.artifacts["closures"]) == []      # header only
    # The highest score seen is what lets the threshold be judged.
    assert result.results["closures"]["highest_score_seen"] > 0.0


def test_yaw_comes_from_the_emitted_rotation(session, tmp_path):
    """Not from VerifyOutput.yaw, which is a different frame and goes stale."""
    rows = _rows(run(_spec(session, tmp_path)).artifacts["closures"])
    for row in rows:
        expected = np.degrees(np.arctan2(float(row["r10"]), float(row["r00"])))
        assert float(row["yaw_deg"]) == pytest.approx(expected, abs=1e-3)


def test_gicp_columns_are_empty_only_when_gicp_did_not_run(session, tmp_path):
    """Empty means "not attempted"; a failed attempt keeps its diagnostics.

    Writing `0` for a pair GICP never saw would read as "converged in 0
    iterations with 0 error". But a pair it tried and failed on has real
    numbers worth keeping, so the two cases must not be collapsed.
    """
    gicp_cols = ("gicp_converged", "gicp_error", "gicp_inliers",
                 "gicp_iterations")
    for row in _rows(run(_spec(session, tmp_path)).artifacts["closures"]):
        empty = [c for c in gicp_cols if row[c] == ""]
        assert empty in ([], list(gicp_cols)), row      # all or nothing
        if row["refined"] == "1":
            assert row["gicp_converged"] == "1"


def test_the_unrefined_pose_survives_in_the_verify_csv(session, tmp_path):
    """Refinement must not destroy the record of what verification produced."""
    result = run(_spec(session, tmp_path))
    verify = {(r["query_idx"], r["db_idx"]): r
              for r in _rows(result.artifacts["verify"])}
    for row in _rows(result.artifacts["closures"]):
        assert (row["query_idx"], row["db_idx"]) in verify


# --- the score artifacts: decision-free --------------------------------------


def test_the_threshold_does_not_touch_the_score_matrices(session, tmp_path):
    """Two thresholds, same scores, different closures.

    The score files record what the matcher computed; the closures record what
    this run decided. Letting the first depend on the second would make it a
    record of neither.
    """
    # The second threshold has to be one nothing clears, or the control below
    # proves nothing: this fixture's twins verify at exactly 1.0, so any
    # threshold at or under that accepts the same 12 closures.
    a = run(_spec(session, tmp_path / "low", threshold=1e-6))
    b = run(_spec(session, tmp_path / "high", threshold=1.5))

    lo = np.load(a.artifacts["score_matrices"])
    hi = np.load(b.artifacts["score_matrices"])
    assert sorted(lo.files) == sorted(hi.files)
    for key in lo.files:
        assert np.array_equal(lo[key], hi[key], equal_nan=True), key

    assert a.results["closures"]["n_closures"] > 0
    assert b.results["closures"]["n_closures"] == 0


def test_not_scored_is_nan_and_a_real_zero_is_zero():
    """The one way these files can lie.

    A failed verification scores exactly 0.0. If "never scored" also wrote
    0.0, the two would be indistinguishable and the matrices would claim the
    matcher looked at pairs it never saw.
    """
    matrices = score_matrices({"mint": {0: {0: 0.5, 1: 0.0}},
                               "verify": {0: {1: 0.0}}}, None, 2, 2)
    assert matrices["mint"][0, 0] == pytest.approx(0.5)
    assert matrices["mint"][0, 1] == 0.0            # a real score of zero
    assert np.isnan(matrices["mint"][1, 0])         # query 1 never ran
    assert matrices["verify"][0, 1] == 0.0          # verification failed
    assert np.isnan(matrices["verify"][0, 0])       # verify never saw it


def test_single_session_matrices_are_strictly_causal(session, tmp_path):
    """A frame may only match its own past, so the rest is NaN, not a low score."""
    result = run(_spec(session, tmp_path))
    mint = np.load(result.artifacts["score_matrices"])["mint"]
    upper = np.triu_indices_from(mint)
    assert np.isnan(mint[upper]).all()


def test_the_scores_csv_records_no_decision(session, tmp_path):
    result = run(_spec(session, tmp_path))
    with open(result.artifacts["scores"], newline="") as fh:
        header = next(csv.reader(fh))
    for banned in ("accepted", "match_type", "threshold"):
        assert banned not in header
    assert {"mint", "beam", "verify", "rank_mint"} <= set(header)


def test_the_scores_csv_shows_the_funnel(session, tmp_path):
    """Later stages score fewer pairs, so their cells thin out."""
    rows = _rows(run(_spec(session, tmp_path)).artifacts["scores"])
    scored = {name: sum(1 for r in rows if r[name] != "")
              for name in ("mint", "beam", "verify")}
    assert scored["mint"] >= scored["beam"] >= scored["verify"] > 0


# --- the record ------------------------------------------------------------


def test_the_record_carries_no_metrics(session, tmp_path):
    """An absent block beats a zeroed one; there is nothing to score."""
    results = run(_spec(session, tmp_path)).results
    for key in ("confusion", "loop_closure", "stage1", "stage2", "verify",
                "combined"):
        assert key not in results, key
    assert results["ground_truth"] is None       # explicit, not merely absent
    assert results["protocol"] == "run"
    assert results["mode"] == "single_session"
    assert results["odometry"]["pose_source"] == "odometry"
    assert results["odometry"]["never_used_for"].startswith("accepting")


def test_no_edge_weight_is_invented(session, tmp_path):
    """InLiER does not estimate a closure covariance, so it does not ship one.

    A constant here would be a guess wearing a measurement's clothes, and a
    back-end would weight its graph with it. The per-closure quality columns
    are the honest raw material instead, and a weight belongs to the back-end
    that can calibrate one.
    """
    poses = run(_spec(session, tmp_path)).results["poses"]
    assert poses["edge_weight"] is None
    assert "p_db = T @ p_query" in poses["convention"]   # this part is load-bearing


def test_config_mode_is_deploy(session, tmp_path):
    """`inlier eval` relaxes the stage thresholds; a deployment must not."""
    assert run(_spec(session, tmp_path)).results["config_mode"] == "deploy"


# --- the radius, and why it is not an oracle here ---------------------------


def test_a_radius_over_odometry_is_not_flagged_as_an_oracle(session, tmp_path):
    """Same filter as `inlier eval --search-radius`, opposite status.

    There it measures against ground truth the system does not have; here it
    measures against the drifted estimate it does. Only `pose_source` tells
    the two apart, so it has to be right.
    """
    cf = run(_spec(session, tmp_path, search_radius=GT_RADIUS * 4)).results["candidate_filter"]
    assert cf["filter"] == "radius"
    assert cf["pose_source"] == "odometry"
    assert cf["uses_pose_oracle"] is False


def test_the_evaluation_radius_is_still_an_oracle():
    """The other half of the distinction, guarded here too."""
    f = gtmod.RadiusFilter(positions=np.zeros((3, 3)), radius=5.0)
    assert f.describe()["pose_source"] == "ground_truth"
    assert f.describe()["uses_pose_oracle"] is True


def test_an_unknown_pose_source_is_refused():
    with pytest.raises(ValueError, match="pose_source must be"):
        gtmod.RadiusFilter(positions=np.zeros((3, 3)), radius=5.0,
                           pose_source="vibes")


# --- refusals --------------------------------------------------------------


def test_skipping_verification_is_refused_before_any_work(session, tmp_path):
    """Poses come only from verification, and the encode takes minutes."""
    cache_dir, source, _, _, _ = session
    spec = _spec(session, tmp_path, resolved=_resolved("verify.skip=true"))
    with pytest.raises(ValueError, match="6-DoF"):
        run(spec)


def test_a_threshold_of_zero_is_refused(session, tmp_path):
    with pytest.raises(ValueError, match="threshold must be > 0"):
        run(_spec(session, tmp_path, threshold=0.0))


def test_an_exclusion_makes_no_sense_against_a_prior_map(session, tmp_path):
    cache_dir, source, _, _, _ = session
    spec = _spec(session, tmp_path)
    spec.db_source = source
    with pytest.raises(ValueError, match="different session"):
        run(spec)


# --- artifacts -------------------------------------------------------------


def test_every_artifact_is_written(session, tmp_path):
    result = run(_spec(session, tmp_path))
    for name in ("closures", "scores", "score_matrices", "verify", "ranked",
                 "results", "trajectory"):
        assert result.artifacts[name].exists(), name
    payload = json.loads(result.artifacts["results"].read_text())
    assert payload["closures"]["n_closures"] >= 0


def test_score_matrices_can_be_skipped(session, tmp_path):
    result = run(_spec(session, tmp_path, score_matrices=False))
    assert "score_matrices" not in result.artifacts


def test_the_summary_reports_closures_not_metrics(session, tmp_path):
    text = run(_spec(session, tmp_path)).summary()
    assert "closures :" in text
    assert "F1max" not in text and "TP=" not in text


def test_the_score_matrices_are_also_drawn(session, tmp_path):
    """The .npz is the data; the .png is what anyone actually looks at."""
    result = run(_spec(session, tmp_path))
    assert result.artifacts["score_figure"].exists()
    assert result.artifacts["score_figure"].stat().st_size > 0


def test_no_score_figure_when_the_matrices_are_skipped(session, tmp_path):
    result = run(_spec(session, tmp_path, score_matrices=False))
    assert "score_figure" not in result.artifacts
