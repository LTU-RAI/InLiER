"""``inlier match`` -- pairwise scoring of two encoded scans.

The tool is a diagnostic, so what matters is that its numbers are the *same*
numbers the evaluation would produce, not merely plausible ones.  The
load-bearing test is ``test_scores_agree_with_the_evaluation_pipeline``: the
same pair, scored through the protocol stage functions, must give exactly what
``match_pair`` gives.  Everything else guards a way of reporting a confident
number about a pair that cannot be compared at all.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from inlier.cli.main import main
from inlier.config import load, resolve
from inlier.eval.pair import (EncodedScan, PairResult, check_compatible,
                              load_encoded, match_pair)


def _cli(capsys, *argv):
    try:
        code = main(list(argv))
    except SystemExit as exc:
        code = exc.code
    return code, capsys.readouterr()


@pytest.fixture(scope="module")
def resolved():
    return resolve(load(), mode="eval")


@pytest.fixture(scope="module")
def encodings(cached_descriptors, tmp_path_factory, resolved):
    """Two real encodings written as ``inlier encode`` writes them.

    Drawn from the cached HeLiPR descriptors, so the tokens and keypoints are
    the encoder's own rather than synthetic noise that no stage would score
    meaningfully.
    """
    d = cached_descriptors
    offsets, tids = d["offsets"], d["token_ids"]
    assert len(offsets) - 1 >= 2, "need two cached scans"

    out = tmp_path_factory.mktemp("encodings")
    cfg = resolved.inlier
    paths = []
    for i in range(2):
        lo, hi = offsets[i], offsets[i + 1]
        path = out / f"scan{i}.npz"
        np.savez_compressed(
            path,
            token_id=tids[lo:hi],
            # Real keypoints, not zeros: verification matches token-consistent
            # pairs geometrically, and a cloud collapsed to the origin has no
            # geometry to verify.
            kp_sensor=np.asarray(d["kp_sensor"][lo:hi], dtype=np.float64),
            kp_aligned=np.asarray(d["kp_aligned"][lo:hi], dtype=np.float64),
            T_ground=np.asarray(d["T_grounds"][i], dtype=np.float64),
            N_h=cfg.N_h, N_r=cfg.N_r, N_s=cfg.N_s, N_a=cfg.N_a,
            voxel_size=resolved.voxel_size)
        paths.append(path)
    return paths


# --- the contract that matters ---------------------------------------------


def test_scores_agree_with_the_evaluation_pipeline(encodings, resolved):
    """A pair scored alone must score exactly as it does inside a run.

    ``match_pair`` builds a one-entry database and calls the same stage
    functions the protocols call.  If that ever stops being true, the tool
    starts reporting numbers that no evaluation would reproduce -- which is
    worse than reporting nothing.
    """
    from inlier.eval.pipeline import (beam_stage, build_matcher,
                                      shortlist_stage)

    q, db = (load_encoded(p) for p in encodings)
    result = match_pair(resolved, q, db)

    matcher = build_matcher(resolved, [db.tokens], verbose=False)
    _, sims = shortlist_stage(matcher, [q.tokens], 1, verbose=False)
    assert result.mint == pytest.approx(sims[0][0])

    ranked = {0: [0]}
    _, s2, shifts = beam_stage(matcher, [q.tokens], ranked, 1, verbose=False)
    assert result.beam == pytest.approx(s2[0][0])
    assert result.beam_shift == shifts[0][0]


def test_a_scan_matched_against_itself_scores_perfectly(encodings, resolved):
    """The sanity check anyone runs first."""
    q = load_encoded(encodings[0])
    result = match_pair(resolved, q, q)
    assert result.mint == pytest.approx(1.0, abs=1e-6)
    assert result.beam == pytest.approx(1.0, abs=1e-6)
    assert result.beam_shift == 0
    assert result.verify is not None and result.verify.success
    # Identity, so the recovered transform must be the identity.
    assert np.allclose(result.pose, np.eye(4), atol=1e-6)


# --- refusing what cannot be compared --------------------------------------


def test_mismatched_radices_are_refused(encodings, resolved, tmp_path):
    """Two token arrays packed differently index different spaces.

    Every stage would still return a number; it would just be arithmetic on
    unrelated integers.  Failing loudly is the only useful answer.
    """
    q = load_encoded(encodings[0])
    odd = tmp_path / "odd.npz"
    with np.load(encodings[1]) as data:
        held = dict(data)
    held["N_a"] = int(held["N_a"]) + 1
    np.savez_compressed(odd, **held)

    with pytest.raises(ValueError, match="different token radices"):
        check_compatible(q, load_encoded(odd), resolved.inlier)


def test_a_config_that_does_not_match_the_files_is_refused(encodings, resolved):
    import dataclasses

    q, db = (load_encoded(p) for p in encodings)
    other = dataclasses.replace(resolved.inlier, N_r=resolved.inlier.N_r + 1)
    with pytest.raises(ValueError, match="active config does not match"):
        check_compatible(q, db, other)


def test_a_file_that_is_not_an_encoding_is_refused(tmp_path):
    path = tmp_path / "junk.npz"
    np.savez_compressed(path, something=np.zeros(3))
    with pytest.raises(ValueError, match="does not look like"):
        load_encoded(path)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such encoding"):
        load_encoded(tmp_path / "nope.npz")


# --- loading ---------------------------------------------------------------


def test_provenance_survives_the_round_trip(encodings):
    q = load_encoded(encodings[0])
    assert q.radices == tuple(int(np.load(encodings[0])[k])
                              for k in ("N_h", "N_r", "N_s", "N_a"))
    assert q.voxel_size is not None
    assert q.label == encodings[0].name          # no provenance -> the filename


def test_a_submap_encoding_labels_itself_by_its_submap(tmp_path, encodings):
    with np.load(encodings[0]) as data:
        held = dict(data)
    path = tmp_path / "sub.npz"
    np.savez_compressed(path, dataset="/data/campus", submap_index=7,
                        n_scans=5, stride=5, **held)
    assert load_encoded(path).label == "campus submap 7"


# --- PairResult ------------------------------------------------------------


def test_pose_prefers_the_refined_transform():
    """GICP runs after verification, so its pose is the one to report."""
    verify = type("V", (), {"success": True, "T_sensor": np.eye(4)})()
    gicp = type("G", (), {"success": True, "T_sensor": np.full((4, 4), 2.0)})()
    assert np.allclose(PairResult(0.5, verify=verify).pose, np.eye(4))
    assert np.allclose(PairResult(0.5, verify=verify, gicp=gicp).pose, 2.0)


def test_pose_is_none_when_nothing_succeeded():
    failed = type("V", (), {"success": False, "T_sensor": np.eye(4)})()
    assert PairResult(0.5).pose is None
    assert PairResult(0.5, verify=failed).pose is None


def test_the_shift_handed_to_verify_is_the_last_one_estimated():
    """Rerank re-estimates the yaw shift, so it supersedes BEAM's."""
    assert PairResult(0.5, beam=0.4, beam_shift=3).shift == 3
    assert PairResult(0.5, beam=0.4, beam_shift=3, rerank=0.6,
                      rerank_shift=9).shift == 9
    assert PairResult(0.5).shift == 0


# --- CLI -------------------------------------------------------------------


def test_cli_prints_every_stage(capsys, encodings):
    code, out = _cli(capsys, "match", str(encodings[0]), str(encodings[1]), "-q")
    assert code == 0
    # -q suppresses the report, so nothing but the matcher's own output.
    code, out = _cli(capsys, "match", str(encodings[0]), str(encodings[1]))
    assert code == 0
    for stage in ("stage 1  MINT", "stage 2  BEAM", "verify"):
        assert stage in out.out


def test_cli_writes_json(capsys, encodings, tmp_path):
    target = tmp_path / "scores.json"
    code, _ = _cli(capsys, "match", str(encodings[0]), str(encodings[1]),
                   "-o", str(target), "-q")
    assert code == 0
    payload = json.loads(target.read_text())
    assert payload["query"]["path"] == str(encodings[0])
    assert 0.0 <= payload["scores"]["stage1_mint"] <= 1.0
    assert "verify" in payload["scores"]


def test_cli_writes_a_figure(capsys, encodings, tmp_path):
    target = tmp_path / "cmp.png"
    code, _ = _cli(capsys, "match", str(encodings[0]), str(encodings[1]),
                   "--viz-save", str(target), "-q")
    assert code == 0
    assert target.exists() and target.stat().st_size > 0


def test_the_two_output_flags_are_not_interchangeable(capsys, encodings, tmp_path):
    """`-o` is the data and `--viz-save` is the picture, as in `inlier encode`.

    Pointing `-o` at a .png would otherwise write JSON under an image name and
    look like it had worked.
    """
    code, out = _cli(capsys, "match", str(encodings[0]), str(encodings[1]),
                     "-o", str(tmp_path / "cmp.png"))
    assert code == 1
    assert "--viz-save" in out.err

    code, out = _cli(capsys, "match", str(encodings[0]), str(encodings[1]),
                     "--viz-save", str(tmp_path / "cmp.txt"))
    assert code == 1
    assert "must name an image file" in out.err


def test_viz_flags_match_encode(capsys):
    """The two commands' figure flags must stay spelled the same."""
    for command in ("encode", "match"):
        with pytest.raises(SystemExit):
            main([command, "--help"])
        body = capsys.readouterr().out
        for flag in ("--viz", "--viz-save", "--viz-dpi"):
            assert flag in body, (command, flag)


def test_cli_takes_exactly_two_files(capsys, encodings):
    """Pairs only: directories and all-vs-all belong to `inlier eval`."""
    code, _ = _cli(capsys, "match", str(encodings[0]))
    assert code == 2                               # argparse: missing argument
    code, _ = _cli(capsys, "match", str(encodings[0]), str(encodings[1]),
                   str(encodings[0]))
    assert code == 2                               # argparse: unrecognised
