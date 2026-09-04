"""KITTI odometry: the camera-frame correction, and the two layouts.

KITTI ships ground-truth poses in the left rectified camera frame and scans in
the velodyne frame.  Read verbatim -- which is what ``--dataset-type generic``
does -- the two disagree, and nothing downstream notices: the positions are
still finite, still monotonic, still plot.  What breaks is that InLiER's
distances are all XY, so on the real sequence 00 ground truth ends up measured
across a 565 x 15 m sliver of facade instead of the 498 x 565 m ground plane.

Everything here is synthetic; the real dataset is never required.  The fixture
uses an exact axis-permutation ``Tr``, so every assertion is exact rather than
tolerance-fudged, and the *camera-frame* poses it writes have their vertical on
y -- the same signature the real data has -- so a test that passes here would
have caught the real bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from inlier.eval.datasets import get_source
from inlier.eval.datasets.generic import Generic_Handler, GenericSource
from inlier.eval.datasets.kitti import (KITTI_Handler, KITTISource,
                                        camera_to_velodyne, normalise_sequence,
                                        read_calib_tr)

N_SCANS = 5

#: velodyne -> camera as an exact axis permutation: (x, y, z) -> (-y, -z, x).
#: The real one is a near-identical rotation plus a small offset; this one is
#: exact, so `inv(Tr) @ P @ Tr` can be asserted without a tolerance.
TR = np.array([
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def _velodyne_poses(n: int = N_SCANS) -> np.ndarray:
    """Ground truth: a flat drive.  100 m in x, 100 m in y, 1 m of climb."""
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.linspace(0.0, 100.0, n)
    poses[:, 1, 3] = np.linspace(0.0, 100.0, n)
    poses[:, 2, 3] = np.linspace(0.0, 1.0, n)
    return poses


def _to_camera(poses_velo: np.ndarray) -> np.ndarray:
    """The inverse of the loader's correction -- what KITTI actually stores."""
    return TR @ poses_velo @ np.linalg.inv(TR)


def _calib_text(Tr: np.ndarray = TR) -> str:
    """A calib.txt with the P-matrices KITTI writes, so `Tr` must be found."""
    p = " ".join(["1.0", "0.0", "0.0", "0.0"] * 3)
    rows = [f"P{i}: {p}" for i in range(4)]
    rows.append("Tr: " + " ".join(f"{v:.12e}" for v in Tr[:3, :4].ravel()))
    return "\n".join(rows) + "\n"


def _pose_text(poses: np.ndarray) -> str:
    return "\n".join(" ".join(f"{v:.12e}" for v in T[:3, :4].ravel())
                     for T in poses) + "\n"


def _write_bin(path, pts) -> None:
    """KITTI velodyne format: flat float32, x y z reflectance."""
    padded = np.zeros((len(pts), 4), dtype=np.float32)
    padded[:, :3] = pts
    padded.tofile(path)


def _scan_points(i: int) -> np.ndarray:
    """A distinct 4-point cloud per scan, so accumulation is traceable."""
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0], [float(i), 0.0, 0.0]], dtype=np.float32)


def _build_tree(root, *, layout: str = "official", n: int = N_SCANS,
                sequence: str = "00", calib: np.ndarray = TR,
                poses_velo=None, times: bool = True):
    """A synthetic KITTI tree in one of the two layouts."""
    seq_dir = root / "sequences" / sequence
    (seq_dir / "velodyne").mkdir(parents=True)
    for i in range(n):
        _write_bin(seq_dir / "velodyne" / f"{i:06d}.bin", _scan_points(i))
    (seq_dir / "calib.txt").write_text(_calib_text(calib))
    if times:
        (seq_dir / "times.txt").write_text(
            "\n".join(f"{0.1 * i:.6e}" for i in range(n)) + "\n")

    poses_velo = _velodyne_poses(n) if poses_velo is None else poses_velo
    text = _pose_text(_to_camera(poses_velo))
    if layout == "official":
        (root / "poses").mkdir(exist_ok=True)
        (root / "poses" / f"{sequence}.txt").write_text(text)
    else:                                   # semantickitti
        (seq_dir / "poses.txt").write_text(text)
    return seq_dir


@pytest.fixture
def kitti_tree(tmp_path):
    root = tmp_path / "kitti"
    _build_tree(root)
    return root


# --- the correction: what this whole module exists for ---------------------


def test_camera_to_velodyne_is_the_conjugation_and_not_its_inverse():
    """Pinned on its own: the direction is the one thing easy to get backwards.

    `inv(Tr) @ P @ Tr` and `Tr @ P @ inv(Tr)` are both plausible-looking, both
    produce finite trajectories, and only one is right.
    """
    velo = _velodyne_poses()
    got = camera_to_velodyne(_to_camera(velo), TR)
    assert np.allclose(got, velo, atol=1e-9)
    wrong = TR @ _to_camera(velo) @ np.linalg.inv(TR)
    assert not np.allclose(wrong, velo)


def test_poses_load_in_the_velodyne_frame(kitti_tree):
    poses, _ = KITTI_Handler(kitti_tree, "00", verbose=False).load_poses(None)
    assert np.allclose(np.array(poses), _velodyne_poses(), atol=1e-9)


def test_the_vertical_axis_ends_up_on_z(kitti_tree):
    """The real dataset's signature: y is vertical before, z after."""
    poses, _ = KITTI_Handler(kitti_tree, "00", verbose=False).load_poses(None)
    spans = np.ptp(np.array([p[:3, 3] for p in poses]), axis=0)
    assert spans[2] == pytest.approx(1.0)          # the climb
    assert spans[0] == pytest.approx(100.0)
    assert spans[1] == pytest.approx(100.0)
    assert int(np.argmin(spans)) == 2


def test_reading_the_same_file_uncorrected_puts_the_vertical_on_y(kitti_tree):
    """Guards the correction against being quietly deleted.

    The generic loader reads exactly this file and is exactly what the user
    was using; it must show the broken signature, or this suite would pass
    with the bug reinstated.
    """
    raw = Generic_Handler(verbose=False,
                          pose_file=kitti_tree / "poses" / "00.txt")
    poses, _ = raw.load_poses(kitti_tree)
    spans = np.ptp(np.array([p[:3, 3] for p in poses]), axis=0)
    assert int(np.argmin(spans)) == 1              # y, not z


def test_submaps_accumulate_in_the_velodyne_frame(kitti_tree):
    """The `--n-scans > 1` corruption, pinned.

    Accumulation applies `inv(T_s) @ T_k` to the points, so an uncorrected
    pose does not merely mislabel a submap -- it builds the wrong cloud.
    """
    data = KITTI_Handler(kitti_tree, "00", verbose=False).load_generic(
        kitti_tree, n_scans=2, stride=2)
    velo = _velodyne_poses()
    T_rel = np.linalg.inv(velo[0]) @ velo[1]
    expected = (T_rel[:3, :3] @ _scan_points(1).T).T + T_rel[:3, 3]

    submap = data["point_clouds"][0]
    assert len(submap) == 8                        # two scans of four points
    assert np.allclose(submap[4:], expected, atol=1e-5)

    cam = _to_camera(velo)
    T_rel_cam = np.linalg.inv(cam[0]) @ cam[1]
    wrong = (T_rel_cam[:3, :3] @ _scan_points(1).T).T + T_rel_cam[:3, 3]
    assert not np.allclose(submap[4:], wrong, atol=1e-3)


def test_kitti_equals_generic_once_the_frame_is_right(kitti_tree, tmp_path):
    """The reuse contract: only the poses differ, everything else is shared.

    A generic dataset holding the same scans and the *already corrected*
    poses must produce identical submaps -- which is what proves the subclass
    inherits the accumulation rather than reimplementing it.
    """
    plain = tmp_path / "plain"
    (plain / "scans").mkdir(parents=True)
    for i in range(N_SCANS):
        _write_bin(plain / "scans" / f"{i:06d}.bin", _scan_points(i))
    (plain / "poses_kitti.txt").write_text(_pose_text(_velodyne_poses()))

    a = KITTI_Handler(kitti_tree, "00", verbose=False).load_generic(
        kitti_tree, n_scans=3, stride=1)
    b = Generic_Handler(verbose=False).load_generic(plain, n_scans=3, stride=1)

    assert len(a["point_clouds"]) == len(b["point_clouds"]) > 0
    for pa, pb in zip(a["point_clouds"], b["point_clouds"]):
        assert np.allclose(pa, pb, atol=1e-9)
    for qa, qb in zip(a["poses"], b["poses"]):
        assert np.allclose(qa, qb, atol=1e-9)


# --- layouts ---------------------------------------------------------------


def test_semantickitti_keeps_its_poses_beside_the_scans(tmp_path):
    root = tmp_path / "sem"
    _build_tree(root, layout="semantickitti")
    poses, _ = KITTI_Handler(root, "00", verbose=False).load_poses(None)
    assert np.allclose(np.array(poses), _velodyne_poses(), atol=1e-9)


def test_a_poses_directory_is_not_mistaken_for_a_pose_file(tmp_path):
    """SemanticKITTI trees really do carry a `poses/` dir next to `poses.txt`.

    `.exists()` would return the directory and the load would fail obscurely.
    """
    root = tmp_path / "sem"
    seq = _build_tree(root, layout="semantickitti")
    (seq / "poses").mkdir()
    path, fmt = KITTI_Handler(root, "00", verbose=False)._pose_file(None)
    assert path == seq / "poses.txt" and fmt == "kitti"


def test_dataset_may_point_straight_at_a_sequence_directory(kitti_tree):
    seq = kitti_tree / "sequences" / "00"
    handler = KITTI_Handler(seq, "00", verbose=False)
    assert handler.seq_dir == seq
    assert len(handler.list_scan_files(seq)) == N_SCANS


def test_the_per_sequence_calib_wins_over_a_root_copy(tmp_path):
    """Sequences genuinely differ -- 00 and 08 have different Tr rows."""
    root = tmp_path / "kitti"
    seq = _build_tree(root)
    (root / "calib.txt").write_text(_calib_text(np.eye(4)))    # a wrong one
    poses, _ = KITTI_Handler(root, "00", verbose=False).load_poses(None)
    assert KITTI_Handler(root, "00", verbose=False).calib_file == seq / "calib.txt"
    assert np.allclose(np.array(poses), _velodyne_poses(), atol=1e-9)


def test_a_root_calib_is_used_when_the_sequence_has_none(tmp_path):
    root = tmp_path / "kitti"
    seq = _build_tree(root)
    (seq / "calib.txt").unlink()
    (root / "calib.txt").write_text(_calib_text())
    poses, _ = KITTI_Handler(root, "00", verbose=False).load_poses(None)
    assert np.allclose(np.array(poses), _velodyne_poses(), atol=1e-9)


@pytest.mark.parametrize("given,want", [(0, "00"), ("0", "00"), ("00", "00"),
                                        (8, "08"), ("21", "21")])
def test_sequence_ids_are_zero_padded(given, want):
    assert normalise_sequence(given) == want


# --- timestamps ------------------------------------------------------------


def test_times_txt_becomes_the_pose_timestamps(kitti_tree):
    """This is what makes `--exclusion seconds=` usable on KITTI at all."""
    src = KITTISource(kitti_tree, "00", verbose=False)
    seq = src.load()
    assert seq.pose_timestamps == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])


def test_accumulated_submaps_take_their_keyframe_timestamp(kitti_tree):
    """Five scans at n=2/stride=2 give three windows -- the last one partial."""
    seq = KITTISource(kitti_tree, "00", n_scans=2, stride=2, verbose=False).load()
    assert seq.pose_timestamps == pytest.approx([0.0, 0.2, 0.4])


def test_a_times_mismatch_is_dropped_rather_than_misaligned(tmp_path):
    root = tmp_path / "kitti"
    seq = _build_tree(root)
    (seq / "times.txt").write_text("0.0\n0.1\n")          # 2 for 5 poses
    _, stamps = KITTI_Handler(root, "00", verbose=False).load_poses(None)
    assert stamps == []


def test_a_missing_times_file_is_not_an_error(tmp_path):
    root = tmp_path / "kitti"
    _build_tree(root, times=False)
    poses, stamps = KITTI_Handler(root, "00", verbose=False).load_poses(None)
    assert len(poses) == N_SCANS and stamps == []


# --- calibration failures --------------------------------------------------


def test_calib_accepts_the_raw_kitti_spelling(tmp_path):
    p = tmp_path / "calib.txt"
    p.write_text("Tr_velo_to_cam: " + " ".join(f"{v}" for v in TR[:3, :4].ravel()))
    assert np.allclose(read_calib_tr(p), TR)


def test_calib_without_a_tr_line_lists_what_it_found(tmp_path):
    p = tmp_path / "calib.txt"
    p.write_text("P0: 1 2 3\nP1: 4 5 6\n")
    with pytest.raises(ValueError, match="P0, P1"):
        read_calib_tr(p)


def test_a_short_tr_line_is_rejected(tmp_path):
    p = tmp_path / "calib.txt"
    p.write_text("Tr: 1 2 3 4\n")
    with pytest.raises(ValueError, match="expected 12"):
        read_calib_tr(p)


def test_a_missing_calib_explains_why_it_is_required(tmp_path):
    root = tmp_path / "kitti"
    seq = _build_tree(root)
    (seq / "calib.txt").unlink()
    with pytest.raises(FileNotFoundError, match="camera frame"):
        KITTI_Handler(root, "00", verbose=False).load_poses(None)


def test_a_singular_tr_is_rejected(tmp_path):
    root = tmp_path / "kitti"
    _build_tree(root, calib=np.zeros((4, 4)))
    with pytest.raises(ValueError, match="singular"):
        KITTI_Handler(root, "00", verbose=False).load_poses(None)


# --- missing data ----------------------------------------------------------


def test_a_test_sequence_says_it_has_no_ground_truth(tmp_path):
    """11-21 are the held-out split; "file not found" would not explain it."""
    root = tmp_path / "kitti"
    _build_tree(root, sequence="11")
    (root / "poses" / "11.txt").unlink()
    with pytest.raises(FileNotFoundError, match="held-out test set"):
        KITTI_Handler(root, "11", verbose=False)._pose_file(None)


def test_a_posed_sequence_with_no_pose_file_names_both_candidates(tmp_path):
    root = tmp_path / "kitti"
    _build_tree(root)
    (root / "poses" / "00.txt").unlink()
    with pytest.raises(FileNotFoundError, match="poses/00.txt"):
        KITTI_Handler(root, "00", verbose=False)._pose_file(None)


def test_an_unknown_sequence_names_the_directory_it_wanted(tmp_path):
    root = tmp_path / "kitti"
    _build_tree(root)
    with pytest.raises(FileNotFoundError, match="sequences/07"):
        KITTI_Handler(root, "07", verbose=False).seq_dir


def test_a_pose_scan_count_mismatch_is_caught(tmp_path):
    root = tmp_path / "kitti"
    seq = _build_tree(root)
    (seq / "velodyne" / "000004.bin").unlink()
    with pytest.raises(RuntimeError, match="mismatch"):
        KITTI_Handler(root, "00", verbose=False).load_generic(root)


# --- the source, and the cache tag -----------------------------------------


def test_the_tag_cannot_collide_with_a_generic_tag(kitti_tree):
    """The stale-cache guard, pinned as a literal.

    The descriptor cache is `desc_{tag}_{hash of encoder config}.npz` and the
    hash carries no dataset identity, while the cache stores *poses*.  A tag
    equal to the generic loader's for the same tree would silently serve back
    the uncorrected camera-frame poses and the correction would look inert.
    """
    src = KITTISource(kitti_tree, "00", n_scans=10, stride=5, verbose=False)
    assert src.tag == "kitti00_n10s5"

    seq_dir = kitti_tree / "sequences" / "00"
    for other in (GenericSource(seq_dir, 10, 5, verbose=False),
                  GenericSource(kitti_tree, 10, 5, verbose=False)):
        assert src.tag != other.tag


def test_the_source_needs_no_dataset_on_disk():
    """`tag` names a cache file and `describe()` is written into results JSON.

    Neither may depend on what happens to be mounted, or a replay on another
    machine would fail before it could say why.
    """
    src = KITTISource("/nowhere/at/all", 0, 4, 2, verbose=False)
    assert src.tag == "kitti00_n4s2"
    assert src.describe()["sequence"] == "00"


def test_describe_round_trips(kitti_tree):
    src = KITTISource(kitti_tree, "00", n_scans=3, stride=2, verbose=False)
    back = KITTISource.from_describe(src.describe(), verbose=False)
    assert back.describe() == src.describe()
    assert back.tag == src.tag


def test_from_describe_honours_a_moved_root(kitti_tree, tmp_path):
    src = KITTISource("/old/location", "00", verbose=False)
    back = KITTISource.from_describe(src.describe(), root=kitti_tree)
    assert len(back.load()) == N_SCANS


def test_describe_names_the_sequence_and_the_frame(kitti_tree):
    d = KITTISource(kitti_tree, "00", verbose=False).describe()
    assert d["dataset_type"] == "kitti"
    assert d["sequence"] == "00"
    assert d["sensor"] == "velodyne"
    # The only record, in a finished run, that the correction was applied.
    assert d["pose_frame"] == "velodyne"


def test_the_registry_offers_kitti():
    assert get_source("kitti") is KITTISource


def test_load_returns_a_sequence_with_matching_poses_and_clouds(kitti_tree):
    seq = KITTISource(kitti_tree, "00", verbose=False).load()
    assert len(seq) == N_SCANS
    assert np.allclose(seq.positions[:, 2].max(), 1.0)


# --- CLI -------------------------------------------------------------------


def _cli(capsys, *argv):
    from inlier.cli.main import main

    try:
        code = main(list(argv))
    except SystemExit as exc:
        code = exc.code
    return code, capsys.readouterr()


def test_doctor_checks_a_kitti_tree(capsys, kitti_tree):
    code, out = _cli(capsys, "doctor", "--dataset-type", "kitti",
                     "--dataset", str(kitti_tree), "--sequence", "00")
    assert code == 0, out.out
    assert "5 .bin files" in out.out
    assert "Tr found" in out.out
    assert "z is vertical" in out.out
    assert "--exclusion seconds= is available" in out.out


def test_doctor_flags_a_kitti_tree_checked_as_helipr(capsys, kitti_tree):
    """This used to report "all checks passed" -- sequences/ read as a sequence."""
    code, out = _cli(capsys, "doctor", "--dataset", str(kitti_tree))
    assert code == 1
    assert "--dataset-type kitti" in out.out


def test_doctor_lists_the_sequences_when_none_was_given(capsys, kitti_tree):
    code, out = _cli(capsys, "doctor", "--dataset-type", "kitti",
                     "--dataset", str(kitti_tree))
    assert code == 1
    assert "needs --sequence" in out.out and "00" in out.out


def test_doctor_warns_when_the_frame_looks_wrong(capsys, tmp_path):
    """An identity Tr leaves the poses in the camera frame -- y still vertical."""
    root = tmp_path / "kitti"
    _build_tree(root, calib=np.eye(4))
    code, out = _cli(capsys, "doctor", "--dataset-type", "kitti",
                     "--dataset", str(root), "--sequence", "00")
    assert code == 0                       # a warning, never a failure
    assert "z is not clearly the vertical axis" in out.out


def test_encode_builds_a_kitti_submap(capsys, kitti_tree, tmp_path):
    target = tmp_path / "submap.npz"
    code, out = _cli(capsys, "encode", "--dataset-type", "kitti",
                     "--dataset", str(kitti_tree), "--sequence", "00",
                     "--n-scans", "2", "-o", str(target), "-q")
    assert code == 0, out.err
    assert target.exists()
    # The provenance names the sequence, not the benchmark root.
    assert str(np.load(target)["dataset"]).endswith("sequences/00")


def test_kitti_needs_a_sequence_unless_the_dataset_is_one(capsys, kitti_tree,
                                                          tmp_path):
    code, out = _cli(capsys, "encode", "--dataset-type", "kitti",
                     "--dataset", str(kitti_tree), "-o", str(tmp_path / "x.npz"))
    assert code == 1
    assert "needs --sequence" in out.err

    code, out = _cli(capsys, "encode", "--dataset-type", "kitti",
                     "--dataset", str(kitti_tree / "sequences" / "00"),
                     "-o", str(tmp_path / "y.npz"), "-q")
    assert code == 0, out.err


def test_online_lcd_rejects_generic_layout_flags(capsys, kitti_tree, tmp_path):
    """`--scans`/`--poses` would silently bypass the calibration."""
    code, out = _cli(capsys, "eval", "online-lcd", "--dataset-type", "kitti",
                     "--dataset", str(kitti_tree), "--sequence", "00",
                     "--scans", str(kitti_tree), "-o", str(tmp_path))
    assert code == 1
    assert "does not apply to --dataset-type kitti" in out.err


def test_online_lcd_rejects_a_contradictory_sequence(capsys, kitti_tree,
                                                     tmp_path):
    code, out = _cli(capsys, "eval", "online-lcd", "--dataset-type", "kitti",
                     "--dataset", str(kitti_tree / "sequences" / "00"),
                     "--sequence", "07", "-o", str(tmp_path))
    assert code == 1
    assert "is itself sequence" in out.err


def test_cross_session_still_refuses_kitti(capsys):
    """Out of scope by decision, and argparse should say so rather than crash."""
    code, out = _cli(capsys, "eval", "cross-session", "--dataset-type", "kitti")
    assert code == 2
    assert "invalid choice" in out.err


# --- inlier match ----------------------------------------------------------


def _encode(capsys, kitti_tree, index, target, extra=()):
    code, out = _cli(capsys, "encode", "--dataset-type", "kitti",
                     "--dataset", str(kitti_tree), "--sequence", "00",
                     "--n-scans", "2", "--index", str(index),
                     *extra, "-o", str(target), "-q")
    assert code == 0, out.err
    return target


def test_encode_records_which_loader_built_it(capsys, kitti_tree, tmp_path):
    """`inlier match` reloads clouds with it; guessing would be wrong for KITTI."""
    npz = np.load(_encode(capsys, kitti_tree, 0, tmp_path / "a.npz"))
    assert str(npz["dataset_type"]) == "kitti"


def test_match_reloads_kitti_clouds_through_the_kitti_loader(capsys, kitti_tree,
                                                             tmp_path):
    """The generic loader would find no scans/ beside a sequence directory.

    Worse, if it found any it would rebuild the submap with uncorrected
    camera-frame poses -- so this asserts the note is absent, i.e. the clouds
    really were recovered.
    """
    from inlier.cli.cmd_match import _handler_for, _reload_points
    from inlier.eval.datasets.kitti import KITTI_Handler
    from inlier.eval.pair import load_encoded

    a = load_encoded(_encode(capsys, kitti_tree, 0, tmp_path / "a.npz"))
    assert isinstance(_handler_for(a.provenance), KITTI_Handler)
    points, note = _reload_points(a)
    assert note == ""
    assert points is not None and len(points) == 8      # two scans of four


def test_match_falls_back_to_sniffing_an_older_encoding(capsys, kitti_tree,
                                                        tmp_path):
    """Files written before the loader was recorded must still reload."""
    from inlier.cli.cmd_match import _handler_for
    from inlier.eval.datasets.kitti import KITTI_Handler

    npz = _encode(capsys, kitti_tree, 0, tmp_path / "a.npz")
    with np.load(npz) as data:
        held = {k: data[k] for k in data.files if k != "dataset_type"}
    np.savez_compressed(npz, **held)

    from inlier.eval.pair import load_encoded
    prov = load_encoded(npz).provenance
    assert "dataset_type" not in prov
    assert isinstance(_handler_for(prov), KITTI_Handler)


def test_match_runs_on_two_kitti_encodings(capsys, kitti_tree, tmp_path):
    a = _encode(capsys, kitti_tree, 0, tmp_path / "a.npz")
    b = _encode(capsys, kitti_tree, 1, tmp_path / "b.npz")
    code, out = _cli(capsys, "match", str(a), str(b),
                     "--viz-save", str(tmp_path / "cmp.png"))
    assert code == 0, out.err
    assert "point cloud not reloaded" not in out.out
    assert (tmp_path / "cmp.png").exists()


def test_a_gated_stage_one_says_so_instead_of_printing_zero(capsys, kitti_tree,
                                                            tmp_path):
    """A flat sensor can occupy too few height slices to be scored at all.

    Stage 1 then returns 0.0, which is indistinguishable from "unrelated
    places" -- and that ambiguity is precisely what sends someone hunting for
    a bug that is really a configuration mismatch.
    """
    a = _encode(capsys, kitti_tree, 0, tmp_path / "a.npz")
    code, out = _cli(capsys, "match", str(a), str(a))
    assert code == 0
    assert "stage 1  MINT   0.000000" in out.out
    assert "not scored" in out.out
    assert "min_shared_rows" in out.out


def test_the_gate_is_recorded_in_the_json(capsys, kitti_tree, tmp_path):
    import json

    a = _encode(capsys, kitti_tree, 0, tmp_path / "a.npz")
    target = tmp_path / "scores.json"
    code, _ = _cli(capsys, "match", str(a), str(a), "-o", str(target), "-q")
    assert code == 0
    gated = json.loads(target.read_text())["scores"]["stage1_gated"]
    assert gated["shared_height_rows"] < gated["min_shared_rows"]


def test_an_ungated_pair_reports_no_gate(cached_descriptors, tmp_path):
    """The note must not fire on data that simply scores low."""
    from inlier.config import load, resolve
    from inlier.eval.pair import load_encoded, match_pair

    r = resolve(load(), mode="eval")
    d, cfg = cached_descriptors, resolve(load(), mode="eval").inlier
    offsets, tids = d["offsets"], d["token_ids"]
    paths = []
    for i in range(2):
        lo, hi = offsets[i], offsets[i + 1]
        path = tmp_path / f"s{i}.npz"
        np.savez_compressed(
            path, token_id=tids[lo:hi],
            kp_sensor=np.asarray(d["kp_sensor"][lo:hi], dtype=np.float64),
            kp_aligned=np.asarray(d["kp_aligned"][lo:hi], dtype=np.float64),
            T_ground=np.asarray(d["T_grounds"][i], dtype=np.float64),
            N_h=cfg.N_h, N_r=cfg.N_r, N_s=cfg.N_s, N_a=cfg.N_a,
            voxel_size=r.voxel_size)
        paths.append(path)
    q = load_encoded(paths[0])
    result = match_pair(r, q, q)
    assert result.mint == pytest.approx(1.0, abs=1e-6)
    assert result.mint_gate is None
