"""CLI smoke tests: dispatch, flag placement, exit codes, artifacts."""

import numpy as np
import pytest

from inlier.cli.main import main


@pytest.fixture(autouse=True)
def _reset_verbosity():
    """Keep the global verbosity from leaking between tests.

    `--version` exits during argument parsing, so the `-q` pre-scan that made
    it quiet never reaches the usual per-command reset.  Harmless in a real
    one-shot process; not harmless when every test shares an interpreter.
    """
    from inlier import verbosity

    verbosity.set_verbosity(verbosity.NORMAL)
    yield
    verbosity.set_verbosity(verbosity.NORMAL)


LOGO_MARKER = "Intermediate LiDAR Encoding for Retrieval"


def _run(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr()


def test_no_command_prints_help(capsys):
    code, out = _run(capsys, )
    assert code == 0
    assert "usage: inlier" in out.out


def test_version_flag_prints_the_banner(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert LOGO_MARKER in out
    assert "version :" in out


def test_quiet_version_stays_machine_readable(capsys):
    """`inlier -q --version` is the form a script or CI check should parse."""
    import inlier

    with pytest.raises(SystemExit) as exc:
        main(["-q", "--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert LOGO_MARKER not in out
    assert out.strip() == f"inlier {inlier.__version__}"


def test_config_show_reports_the_resolved_values(capsys):
    code, out = _run(capsys, "config", "show")
    assert code == 0
    assert LOGO_MARKER in out.out
    assert "encoder (InLiER_Config)" in out.out
    assert "mint_scoring" in out.out
    # eval mode must say it moved the stage thresholds
    assert "-2.0" in out.out


def test_config_dump_is_loadable_yaml(capsys, tmp_path):
    """dump has no banner: its output has to stay usable as a --config file."""
    code, out = _run(capsys, "config", "dump")
    assert code == 0
    assert LOGO_MARKER not in out.out
    import yaml
    cfg = yaml.safe_load(out.out)
    assert cfg["encoder"]["N_h"] == 10


def test_config_show_banner_suppressed_by_quiet(capsys):
    code, out = _run(capsys, "config", "show", "-q")
    assert code == 0
    assert LOGO_MARKER not in out.out
    assert "encoder (InLiER_Config)" in out.out


@pytest.mark.parametrize("argv", [
    ["config", "show", "--set", "stage1.topk=7"],
    ["--set", "stage1.topk=7", "config", "show"],
])
def test_global_flags_work_on_either_side_of_the_command(capsys, argv):
    """A value given before the command name must not be clobbered by the
    subparser's default -- the reason the subcommand copies use SUPPRESS."""
    code, out = _run(capsys, *argv)
    assert code == 0
    assert "stage1.topk=7" in out.out


def test_bad_config_key_exits_nonzero_without_a_traceback(capsys):
    code, out = _run(capsys, "config", "show", "--set", "stage1.topkk=7")
    assert code == 1
    assert "unknown config key" in out.err
    assert "Traceback" not in out.err


def test_doctor_runs_and_reports_a_backend(capsys):
    code, out = _run(capsys, "doctor")
    assert code in (0, 1)
    assert "backend" in out.out
    assert ("cpp" in out.out) or ("python" in out.out)


def test_doctor_flags_a_missing_dataset(capsys, tmp_path):
    code, out = _run(capsys, "doctor", "--dataset", str(tmp_path / "nope"))
    assert code == 1
    assert "does not exist" in out.out


def test_doctor_flags_a_helipr_tree_without_undistorted(capsys, tmp_path):
    """Every evaluation reads Undistorted/, never the distorted LiDAR/."""
    (tmp_path / "Seq01" / "LiDAR" / "Ouster").mkdir(parents=True)
    code, out = _run(capsys, "doctor", "--dataset", str(tmp_path))
    assert code == 1
    assert "Undistorted" in out.out


@pytest.fixture
def scan_file(tmp_path):
    rng = np.random.default_rng(5)
    pts = np.concatenate([
        np.stack([rng.uniform(-40, 40, 8000), rng.uniform(-40, 40, 8000),
                  rng.normal(0, 0.05, 8000)], axis=1),
        np.stack([rng.normal(10, 0.05, 2000), rng.normal(5, 0.05, 2000),
                  rng.uniform(0.2, 8.0, 2000)], axis=1),
    ]).astype(np.float32)
    path = tmp_path / "scan.npy"
    np.save(path, pts)
    return path


def test_encode_writes_tokens_with_their_radices(capsys, scan_file, tmp_path):
    out_path = tmp_path / "tokens.npz"
    code, _ = _run(capsys, "encode", str(scan_file), "-o", str(out_path), "--quiet")
    assert code == 0
    data = np.load(out_path)
    assert data["token_id"].ndim == 1
    assert data["kp_sensor"].shape[1] == 3
    # tokens are meaningless without the radices they were packed with
    for key in ("N_h", "N_r", "N_s", "N_a", "voxel_size"):
        assert key in data


def test_encode_a_directory_writes_one_npz_per_scan(capsys, scan_file, tmp_path):
    out_dir = tmp_path / "out"
    code, _ = _run(capsys, "encode", str(scan_file.parent), "-o", str(out_dir), "--quiet")
    assert code == 0
    assert (out_dir / "scan.npz").exists()


def test_encode_reports_a_missing_input(capsys, tmp_path):
    code, out = _run(capsys, "encode", str(tmp_path / "none.pcd"),
                     "-o", str(tmp_path / "x.npz"))
    assert code == 1
    assert "Traceback" not in out.err


def test_encode_viz_writes_a_figure(capsys, scan_file, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    figure = tmp_path / "scan.png"
    code, _ = _run(capsys, "encode", str(scan_file), "--viz-save", str(figure),
                   "--quiet")
    assert code == 0
    assert figure.stat().st_size > 0


def test_encode_viz_needs_no_output_npz(capsys, scan_file, tmp_path):
    """Looking at a scan should not force writing tokens you do not want."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    figure = tmp_path / "look.png"
    code, _ = _run(capsys, "encode", str(scan_file), "--viz-save", str(figure),
                   "--quiet")
    assert code == 0
    assert not list(tmp_path.glob("*.npz"))


def test_encode_without_output_or_viz_is_an_error(capsys, scan_file):
    code, out = _run(capsys, "encode", str(scan_file), "--quiet")
    assert code == 1
    assert "-o/--output" in out.err


def test_encode_interactive_viz_refuses_a_directory(capsys, scan_file):
    """One window per scan is never what the user meant."""
    code, out = _run(capsys, "encode", str(scan_file.parent), "--viz", "--quiet")
    assert code == 1
    assert "--viz-save" in out.err


def test_encode_viz_save_on_a_directory_writes_one_figure_per_scan(
        capsys, scan_file, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    figures = tmp_path / "figs"
    code, _ = _run(capsys, "encode", str(scan_file.parent),
                   "--viz-save", str(figures), "--quiet")
    assert code == 0
    assert (figures / f"{scan_file.stem}.png").exists()


# --- tilde expansion -------------------------------------------------------
# bash expands `--opt ~/x` but not `--opt=~/x`, so the second form reaches
# argparse with a literal `~` and fails as a missing path that is plainly
# there.  zsh expands both, which is why this only bites some users.

def test_expand_user_handles_the_equals_form():
    import os

    from inlier.cli._common import expand_user

    home = os.path.expanduser("~")
    assert expand_user(["--dataset=~/data"]) == [f"--dataset={home}/data"]
    assert expand_user(["~/data"]) == [f"{home}/data"]


def test_expand_user_leaves_non_paths_alone():
    from inlier.cli._common import expand_user

    argv = ["--set", "stage1.topk=50", "--pair=O-Aeva", "note=~weird", "-q"]
    assert expand_user(argv) == argv


def test_tilde_path_reaches_the_command(capsys, scan_file, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(scan_file.parent))
    code, _ = _run(capsys, "encode", f"--output={tmp_path / 'out.npz'}",
                   f"~/{scan_file.name}", "--quiet")
    assert code == 0
    assert (tmp_path / "out.npz").exists()


# --- non-finite points -----------------------------------------------------
# Sensors write NaN for invalid returns and .pcd carries them through.  Left
# in, they poison the voxel-grid cast (NaN -> INT64_MIN) and every axis limit.

@pytest.fixture
def scan_with_nans(tmp_path):
    rng = np.random.default_rng(1)
    points = rng.uniform(-20, 20, size=(3000, 3)).astype(np.float32)
    points[:, 2] = rng.uniform(0.0, 8.0, size=3000)
    points[::7] = np.nan
    path = tmp_path / "holes" / "scan.npy"
    path.parent.mkdir()
    np.save(path, points)
    return path


def test_encode_drops_non_finite_points_and_says_so(capsys, scan_with_nans, tmp_path):
    out_path = tmp_path / "tokens.npz"
    code, out = _run(capsys, "encode", str(scan_with_nans), "-o", str(out_path))
    assert code == 0
    assert "non-finite" in out.out
    data = np.load(out_path)
    assert np.isfinite(data["kp_sensor"]).all()


def test_encode_refuses_an_all_nan_scan(capsys, tmp_path):
    path = tmp_path / "empty.npy"
    np.save(path, np.full((100, 3), np.nan, dtype=np.float32))
    code, out = _run(capsys, "encode", str(path), "-o", str(tmp_path / "x.npz"))
    assert code == 1
    assert "no finite points" in out.err


def test_encode_viz_survives_non_finite_points(capsys, scan_with_nans, tmp_path):
    """NaNs reached `_extent` as NaN axis limits, which matplotlib rejects."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    figure = tmp_path / "holes.png"
    code, _ = _run(capsys, "encode", str(scan_with_nans),
                   "--viz-save", str(figure), "--quiet")
    assert code == 0
    assert figure.stat().st_size > 0


# --- doctor dataset layouts ------------------------------------------------

MINIMAL_PCD = """\
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 3
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 3
DATA ascii
0 0 0
1 0 0
0 1 0
"""


@pytest.fixture
def generic_dataset(tmp_path):
    root = tmp_path / "campus"
    (root / "scans").mkdir(parents=True)
    for i in range(3):
        (root / "scans" / f"{i:06d}.pcd").write_text(MINIMAL_PCD)
    (root / "poses_kitti.txt").write_text(
        "\n".join(" ".join(["1", "0", "0", str(i), "0", "1", "0", "0",
                            "0", "0", "1", "0"]) for i in range(3)))
    return root


@pytest.fixture
def helipr_dataset(tmp_path):
    root = tmp_path / "HeLiPR"
    (root / "Roundabout01" / "Undistorted" / "Ouster").mkdir(parents=True)
    (root / "Roundabout01" / "LiDAR_GT").mkdir()
    return root


def test_doctor_checks_the_generic_layout(capsys, generic_dataset):
    code, out = _run(capsys, "doctor", "--dataset", str(generic_dataset),
                     "--dataset-type", "generic")
    assert code == 0
    assert "3 .pcd files" in out.out
    assert "poses_kitti.txt" in out.out


def test_doctor_names_the_layout_it_is_checking(capsys, helipr_dataset):
    code, out = _run(capsys, "doctor", "--dataset", str(helipr_dataset))
    assert code == 0
    assert "helipr -- <root>/<sequence>/Undistorted" in out.out


def test_doctor_reports_a_generic_tree_checked_as_helipr(capsys, generic_dataset):
    """Checking the wrong layout used to report the dataset as empty."""
    code, out = _run(capsys, "doctor", "--dataset", str(generic_dataset))
    assert code == 1
    assert "layout mismatch" in out.out
    assert "--dataset-type generic" in out.out


def test_doctor_reports_a_helipr_tree_checked_as_generic(capsys, helipr_dataset):
    code, out = _run(capsys, "doctor", "--dataset", str(helipr_dataset),
                     "--dataset-type", "generic")
    assert code == 1
    assert "layout mismatch" in out.out
    assert "--dataset-type helipr" in out.out


def test_doctor_catches_a_pose_scan_count_mismatch(capsys, generic_dataset):
    """`load_generic` raises on this, but only after the submap build starts."""
    (generic_dataset / "scans" / "000003.pcd").write_text(MINIMAL_PCD)
    code, out = _run(capsys, "doctor", "--dataset", str(generic_dataset),
                     "--dataset-type", "generic")
    assert code == 1
    assert "3 poses but 4 scans" in out.out


def test_doctor_reports_a_generic_dataset_with_no_poses(capsys, generic_dataset):
    (generic_dataset / "poses_kitti.txt").unlink()
    code, out = _run(capsys, "doctor", "--dataset", str(generic_dataset),
                     "--dataset-type", "generic")
    assert code == 1
    assert "no poses_kitti.txt or poses_tum.txt" in out.out
