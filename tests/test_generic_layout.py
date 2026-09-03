"""Explicit scan/pose paths and ``.bin`` scans for the generic loader.

The conventional layout -- ``<root>/scans/*.pcd`` beside ``poses_kitti.txt`` --
is a convenience, and data that was not exported that way should not have to be
rearranged before it can be evaluated.  These tests pin the two escapes:
``scans_dir``/``pose_file`` overrides, and reading KITTI-style ``.bin`` dumps.

The load-bearing assertion is ``test_explicit_paths_match_the_conventional_layout``:
the same data, described either way, must produce identical submaps.  Everything
else here guards a way of getting that silently wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from inlier.eval.datasets.generic import Generic_Handler, GenericSource

# Poses for N scans on a straight line, 1 m apart.
def _kitti(n: int) -> str:
    return "\n".join(f"1 0 0 {i} 0 1 0 0 0 0 1 0" for i in range(n))


def _tum(n: int, t0: float = 100.0) -> str:
    return "\n".join(f"{t0 + i:.3f} {i} 0 0 0 0 0 1" for i in range(n))


def _write_bin(path, pts, cols: int = 4) -> None:
    """Write (N, 3) points as a flat float32 dump with ``cols`` per point."""
    padded = np.zeros((len(pts), cols), dtype=np.float32)
    padded[:, :3] = pts
    padded.tofile(path)


@pytest.fixture
def bin_sequence(tmp_path):
    """3 scans of 5 points each, as .bin, with the poses in a separate tree."""
    scans = tmp_path / "velodyne"
    scans.mkdir()
    clouds = []
    for i in range(3):
        pts = np.arange(15, dtype=np.float32).reshape(5, 3) + i
        clouds.append(pts)
        _write_bin(scans / f"{i:06d}.bin", pts)
    poses = tmp_path / "gt" / "odometry.txt"
    poses.parent.mkdir()
    poses.write_text(_kitti(3))
    return scans, poses, clouds


# --- .bin reading ----------------------------------------------------------


def test_bin_scan_round_trips(bin_sequence):
    scans, poses, clouds = bin_sequence
    h = Generic_Handler(verbose=False, scans_dir=scans, pose_file=poses)
    got = h.load_scan_file(scans / "000000.bin")
    assert got.shape == (5, 3)
    assert np.array_equal(got, clouds[0])


def test_bin_drops_non_finite_points(tmp_path):
    pts = np.array([[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0], [4.0, 5.0, 6.0]],
                   dtype=np.float32)
    _write_bin(tmp_path / "s.bin", pts)
    got = Generic_Handler(verbose=False).load_scan_file(tmp_path / "s.bin")
    assert np.array_equal(got, [[1, 2, 3], [4, 5, 6]])


def test_bin_defaults_to_four_floats_per_point(tmp_path):
    """12 floats is consistent with 3, 4 and 6 wide; 4 is the one to pick.

    Inferring here would be a coin flip that silently scrambles coordinates,
    so the KITTI convention wins whenever the size allows it.
    """
    np.arange(12, dtype=np.float32).tofile(tmp_path / "s.bin")
    got = Generic_Handler(verbose=False).load_scan_file(tmp_path / "s.bin")
    assert got.shape == (3, 3)
    assert np.array_equal(got[0], [0, 1, 2])       # not [0, 1, 2] of a 3-wide read
    assert np.array_equal(got[1], [4, 5, 6])       # 4-wide: index 4, not 3


def test_bin_infers_a_width_four_cannot_explain(tmp_path):
    # 10 floats: not 4-wide, and of 3/5/6 only 5 divides it.
    np.arange(10, dtype=np.float32).tofile(tmp_path / "s.bin")
    got = Generic_Handler(verbose=False).load_scan_file(tmp_path / "s.bin")
    assert got.shape == (2, 3)
    assert np.array_equal(got[1], [5, 6, 7])


def test_bin_refuses_an_ambiguous_width(tmp_path):
    # 15 floats: not 4-wide, and both 3 and 5 fit.  A coin flip is not an answer.
    np.arange(15, dtype=np.float32).tofile(tmp_path / "s.bin")
    with pytest.raises(ValueError, match="ambiguous"):
        Generic_Handler(verbose=False).load_scan_file(tmp_path / "s.bin")


def test_bin_cols_overrides_the_guess(tmp_path):
    np.arange(12, dtype=np.float32).tofile(tmp_path / "s.bin")
    got = Generic_Handler(verbose=False, bin_cols=3).load_scan_file(tmp_path / "s.bin")
    assert got.shape == (4, 3)
    assert np.array_equal(got[1], [3, 4, 5])


def test_bin_cols_must_fit_the_file(tmp_path):
    np.arange(10, dtype=np.float32).tofile(tmp_path / "s.bin")
    with pytest.raises(ValueError, match="not a multiple"):
        Generic_Handler(verbose=False, bin_cols=4).load_scan_file(tmp_path / "s.bin")


def test_empty_bin_is_an_empty_cloud_not_a_crash(tmp_path):
    (tmp_path / "s.bin").write_bytes(b"")
    got = Generic_Handler(verbose=False).load_scan_file(tmp_path / "s.bin")
    assert got.shape == (0, 3)


# --- explicit paths --------------------------------------------------------


def test_explicit_paths_find_scans_and_poses(bin_sequence):
    scans, poses, _ = bin_sequence
    h = Generic_Handler(verbose=False, scans_dir=scans, pose_file=poses)
    # The dataset_dir argument is unused for both, so anything may be passed.
    assert len(h.list_scan_files(scans.parent / "nonexistent")) == 3
    loaded, _ = h.load_poses(scans.parent / "nonexistent")
    assert len(loaded) == 3
    assert loaded[2][0, 3] == 2.0


def test_explicit_paths_match_the_conventional_layout(tmp_path):
    """The same data, described two ways, must load identically."""
    conventional = tmp_path / "seq"
    (conventional / "scans").mkdir(parents=True)
    for i in range(4):
        _write_bin(conventional / "scans" / f"{i:06d}.bin",
                   np.full((3, 3), float(i), dtype=np.float32))
    (conventional / "poses_kitti.txt").write_text(_kitti(4))

    a = Generic_Handler(verbose=False).load_generic(conventional, n_scans=2)
    b = Generic_Handler(
        verbose=False,
        scans_dir=conventional / "scans",
        pose_file=conventional / "poses_kitti.txt",
    ).load_generic(tmp_path / "unused", n_scans=2)

    assert len(a["point_clouds"]) == len(b["point_clouds"]) == 2
    for pa, pb in zip(a["point_clouds"], b["point_clouds"]):
        assert np.array_equal(pa, pb)
    for qa, qb in zip(a["poses"], b["poses"]):
        assert np.array_equal(qa, qb)


def test_only_the_poses_may_be_overridden(tmp_path):
    """Naming the poses while the scans stay conventional is an ordinary case."""
    root = tmp_path / "seq"
    (root / "scans").mkdir(parents=True)
    for i in range(2):
        _write_bin(root / "scans" / f"{i:06d}.bin", np.zeros((2, 3), np.float32))
    elsewhere = tmp_path / "poses.txt"
    elsewhere.write_text(_kitti(2))

    h = Generic_Handler(verbose=False, pose_file=elsewhere)
    assert len(h.list_scan_files(root)) == 2
    assert len(h.load_poses(root)[0]) == 2


def test_a_missing_explicit_pose_file_says_so(tmp_path):
    h = Generic_Handler(verbose=False, pose_file=tmp_path / "nope.txt")
    with pytest.raises(FileNotFoundError, match="nope.txt"):
        h.load_poses(tmp_path)


def test_count_mismatch_names_both_paths(tmp_path):
    scans = tmp_path / "s"
    scans.mkdir()
    _write_bin(scans / "000000.bin", np.zeros((2, 3), np.float32))
    poses = tmp_path / "p.txt"
    poses.write_text(_kitti(5))
    h = Generic_Handler(verbose=False, scans_dir=scans, pose_file=poses)
    with pytest.raises(RuntimeError) as exc:
        h.load_generic(tmp_path)
    # With overrides the two need not share a parent, so naming one is useless.
    assert str(poses) in str(exc.value) and str(scans) in str(exc.value)


# --- pose-format sniffing --------------------------------------------------


@pytest.mark.parametrize("text,fmt", [(_kitti(2), "kitti"), (_tum(2), "tum")])
def test_format_is_read_from_the_contents_not_the_name(tmp_path, text, fmt):
    p = tmp_path / "poses.txt"           # a name that says nothing
    p.write_text(text)
    assert Generic_Handler._sniff_pose_format(p) == fmt


def test_comments_and_blanks_do_not_confuse_the_sniff(tmp_path):
    p = tmp_path / "poses.txt"
    p.write_text("# timestamp x y z qx qy qz qw\n\n" + _tum(2))
    assert Generic_Handler._sniff_pose_format(p) == "tum"


def test_an_unrecognised_pose_line_is_rejected(tmp_path):
    p = tmp_path / "poses.txt"
    p.write_text("1.0 2.0 3.0\n")
    with pytest.raises(ValueError, match="neither KITTI"):
        Generic_Handler._sniff_pose_format(p)


def test_an_empty_pose_file_is_rejected(tmp_path):
    p = tmp_path / "poses.txt"
    p.write_text("# nothing but a header\n\n")
    with pytest.raises(ValueError, match="no pose lines"):
        Generic_Handler._sniff_pose_format(p)


def test_tum_timestamps_are_borrowed_beside_an_explicit_kitti_file(tmp_path):
    """The sibling lookup follows the pose file, not the dataset directory."""
    d = tmp_path / "gt"
    d.mkdir()
    (d / "poses_kitti.txt").write_text(_kitti(3))
    (d / "poses_tum.txt").write_text(_tum(3))
    h = Generic_Handler(verbose=False, pose_file=d / "poses_kitti.txt")
    _, stamps = h.load_poses(tmp_path)
    assert stamps == [100.0, 101.0, 102.0]


# --- scan discovery --------------------------------------------------------


def test_mixed_scan_formats_are_rejected(tmp_path):
    scans = tmp_path / "s"
    scans.mkdir()
    _write_bin(scans / "000000.bin", np.zeros((2, 3), np.float32))
    (scans / "000000.pcd").write_text("junk")
    h = Generic_Handler(verbose=False, scans_dir=scans)
    with pytest.raises(RuntimeError, match="more than one scan format"):
        h.list_scan_files(tmp_path)


def test_an_empty_scan_directory_names_both_formats(tmp_path):
    scans = tmp_path / "s"
    scans.mkdir()
    h = Generic_Handler(verbose=False, scans_dir=scans)
    with pytest.raises(FileNotFoundError, match=r"\.pcd or \.bin"):
        h.list_scan_files(tmp_path)


def test_unrelated_files_are_ignored(tmp_path):
    scans = tmp_path / "s"
    scans.mkdir()
    _write_bin(scans / "000000.bin", np.zeros((2, 3), np.float32))
    (scans / "README.md").write_text("hi")
    (scans / "calib.yaml").write_text("hi")
    assert len(Generic_Handler(verbose=False, scans_dir=scans)
               .list_scan_files(tmp_path)) == 1


# --- GenericSource ---------------------------------------------------------


def test_from_paths_takes_its_identity_from_the_scans_parent(bin_sequence):
    scans, poses, _ = bin_sequence
    src = GenericSource.from_paths(scans, poses, 1, 1, verbose=False)
    assert src.path == scans.parent
    assert src.tag.startswith(scans.parent.name)


def test_from_paths_identity_can_be_named(bin_sequence, tmp_path):
    scans, poses, _ = bin_sequence
    src = GenericSource.from_paths(scans, poses, 1, 1, path=tmp_path / "campus",
                                   verbose=False)
    assert src.path.name == "campus"


def test_describe_records_the_explicit_paths(bin_sequence):
    scans, poses, _ = bin_sequence
    d = GenericSource.from_paths(scans, poses, 1, 1, verbose=False).describe()
    assert d["scans_dir"] == str(scans)
    assert d["pose_file"] == str(poses)


def test_describe_stays_quiet_for_the_conventional_layout(tmp_path):
    """A run against <root>/scans keeps the description it has always had."""
    d = GenericSource(tmp_path, 1, 1, verbose=False).describe()
    assert set(d) == {"dataset_type", "path", "n_scans", "stride", "transform"}


def test_from_describe_restores_the_paths(bin_sequence):
    """Without this a replay could not find the scans of an explicit run."""
    scans, poses, _ = bin_sequence
    src = GenericSource.from_paths(scans, poses, 2, 1, verbose=False)
    back = GenericSource.from_describe(src.describe(), verbose=False)
    assert back.scans_dir == scans
    assert back.pose_file == poses
    assert (back.n_scans, back.stride) == (2, 1)
    assert len(back._handler.list_scan_files(back.path)) == 3


# --- CLI -------------------------------------------------------------------
#
#  The loader can be told where the data is; these check that each command
#  actually asks it, and refuses the half-specified cases.

def _cli(capsys, *argv):
    from inlier.cli.main import main

    try:
        code = main(list(argv))
    except SystemExit as exc:               # argparse errors exit rather than return
        code = exc.code
    return code, capsys.readouterr()


def test_doctor_checks_an_explicit_layout(capsys, bin_sequence):
    scans, poses, _ = bin_sequence
    code, out = _cli(capsys, "doctor", "--dataset-type", "generic",
                     "--scans", str(scans), "--poses", str(poses))
    assert code == 0, out.out
    assert "explicit paths" in out.out
    assert "3 .bin files" in out.out
    assert "(kitti), 3 poses" in out.out


def test_doctor_still_catches_a_count_mismatch_with_explicit_paths(capsys, tmp_path):
    scans = tmp_path / "s"
    scans.mkdir()
    _write_bin(scans / "000000.bin", np.zeros((2, 3), np.float32))
    poses = tmp_path / "p.txt"
    poses.write_text(_kitti(4))
    code, out = _cli(capsys, "doctor", "--dataset-type", "generic",
                     "--scans", str(scans), "--poses", str(poses))
    assert code == 1
    assert "4 poses but 1 scans" in out.out


def test_encode_accepts_explicit_paths(capsys, bin_sequence, tmp_path):
    scans, poses, _ = bin_sequence
    out_npz = tmp_path / "submap.npz"
    code, out = _cli(capsys, "encode", "--scans", str(scans),
                     "--poses", str(poses), "-o", str(out_npz), "-q")
    assert code == 0, out.err
    assert out_npz.exists()
    # The provenance names the derived identity, not a dataset that never was.
    assert str(np.load(out_npz)["dataset"]) == str(scans.parent)


def test_encode_scans_without_poses_is_refused(capsys, bin_sequence, tmp_path):
    scans, _, _ = bin_sequence
    code, out = _cli(capsys, "encode", "--scans", str(scans),
                     "-o", str(tmp_path / "x.npz"))
    assert code == 1
    assert "also needs --poses" in out.err


def test_online_lcd_rejects_scans_without_poses(capsys, bin_sequence, tmp_path):
    scans, _, _ = bin_sequence
    code, out = _cli(capsys, "eval", "online-lcd", "--dataset-type", "generic",
                     "--scans", str(scans), "-o", str(tmp_path))
    assert code == 1
    assert "also needs --poses" in out.err


def test_online_lcd_rejects_neither_dataset_nor_scans(capsys, tmp_path):
    code, out = _cli(capsys, "eval", "online-lcd", "--dataset-type", "generic",
                     "-o", str(tmp_path))
    assert code == 1
    assert "got neither" in out.err


@pytest.mark.parametrize("cmd", ["build", "validate"])
def test_gt_rejects_scans_without_poses(capsys, bin_sequence, cmd):
    scans, _, _ = bin_sequence
    code, out = _cli(capsys, "gt", cmd, "--dataset-type", "generic",
                     "--db-scans", str(scans), "--q-path", "/nope")
    assert code == 2
    assert "--db-scans without --db-path also needs --db-poses" in out.err
