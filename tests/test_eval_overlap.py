"""Overlap matrices and the provenance sidecar.

The matrix is indexed by submap, so an evaluation that accumulates scans
differently reads row i as a different piece of trajectory and every metric is
wrong with no error raised.  README.md warns about this in prose; these tests
cover the check that replaces the warning.
"""

import json

import numpy as np
import pytest

from inlier.eval import overlap as ov


@pytest.fixture
def matrix():
    return np.round(np.random.default_rng(0).random((12, 9)), 6)


@pytest.fixture
def built(tmp_path, matrix):
    prov = ov.OverlapProvenance(
        n_db=10, n_q=10, stride_db=1, stride_q=1,
        voxel_size=0.5, max_range=100.0, distance_threshold=100.0,
        db_id="A", q_id="B",
    )
    path = ov.save(matrix, tmp_path / ov.generic_name("A", "B", 10, 10, 1, 1), prov)
    return path


def test_naming_conventions():
    assert (ov.helipr_name("Roundabout01", "Ouster", "Roundabout03", "Aeva")
            == "overlap_Roundabout01_Ouster_Roundabout03_Aeva.txt")
    # submap parameters land in the name so two builds cannot collide
    assert ov.generic_name("a", "b", 1, 1, 1, 1) == "overlap_a_b.txt"
    assert ov.generic_name("a", "b", 10, 10, 1, 1) == "overlap_a_b_Ndb10_Nq10_Sdb1_Sq1.txt"
    assert ov.generic_name("a", "b", 10, 5, 10, 5) == "overlap_a_b_Ndb10_Nq5.txt"


def test_save_load_round_trip(built, matrix):
    assert np.allclose(ov.load(built), matrix, atol=1e-6)


def test_sidecar_written_and_read_back(built):
    side = ov.sidecar_path(built)
    assert side.exists()
    prov = ov.load_provenance(built)
    assert (prov.n_db, prov.n_q, prov.stride_db, prov.stride_q) == (10, 10, 1, 1)
    assert prov.shape == (12, 9)
    assert prov.from_legacy_header is False
    assert "from_legacy_header" not in json.loads(side.read_text())


def test_check_accepts_matching_parameters(built, matrix):
    ov.check(built, ov.load(built),
             ov.OverlapProvenance(n_db=10, n_q=10, stride_db=1, stride_q=1, shape=(12, 9)))


@pytest.mark.parametrize("field,value", [
    ("n_db", 1), ("n_q", 5), ("stride_db", 3), ("stride_q", 2),
])
def test_check_rejects_each_critical_mismatch(built, field, value):
    expected = ov.OverlapProvenance(n_db=10, n_q=10, stride_db=1, stride_q=1, shape=(12, 9))
    setattr(expected, field, value)
    with pytest.raises(ov.OverlapMismatch, match=field):
        ov.check(built, ov.load(built), expected)


def test_check_can_downgrade_a_mismatch_to_a_warning(built):
    expected = ov.OverlapProvenance(n_db=1, n_q=1, stride_db=1, stride_q=1, shape=(12, 9))
    with pytest.warns(UserWarning, match="submap parameters disagree"):
        ov.check(built, ov.load(built), expected, strict=False)


def test_shape_mismatch_is_always_an_error(built):
    """Shape needs no sidecar and catches the coarsest misalignment."""
    expected = ov.OverlapProvenance(n_db=10, n_q=10, stride_db=1, stride_q=1, shape=(99, 9))
    with pytest.raises(ov.OverlapMismatch, match="matrix is 12x9"):
        ov.check(built, ov.load(built), expected)


def test_missing_matrix_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="inlier gt build"):
        ov.load(tmp_path / "absent.txt")


def test_provenance_recovered_from_a_legacy_header(tmp_path, matrix):
    """Matrices built before the sidecar still carry the facts, as prose."""
    path = tmp_path / "overlap_legacy.txt"
    header = "\n".join([
        "HeLiOS overlap matrix",
        "Database: Seq01 / Ouster  (12 submaps, n_db=7, stride_db=3)",
        "Query:    Seq02 / Aeva  (9 submaps, n_q=4, stride_q=2)",
        "Voxel size (delta): 0.5 m   tau: 0.75 m",
        "Pose distance threshold: 100.0 m",
        "Max point range: 80.0 m",
    ])
    np.savetxt(path, matrix, fmt="%.6f", header=header)

    prov = ov.load_provenance(path)
    assert prov is not None and prov.from_legacy_header
    assert (prov.n_db, prov.stride_db, prov.n_q, prov.stride_q) == (7, 3, 4, 2)
    assert prov.voxel_size == 0.5 and prov.max_range == 80.0

    with pytest.raises(ov.OverlapMismatch, match="text header"):
        ov.check(path, matrix,
                 ov.OverlapProvenance(n_db=1, n_q=1, stride_db=1, stride_q=1, shape=(12, 9)))


def test_unparsable_matrix_warns_rather_than_failing(tmp_path, matrix):
    """The matrices shipped in overlap_matrices/ must keep working."""
    path = tmp_path / "overlap_bare.txt"
    np.savetxt(path, matrix, fmt="%.6f")
    assert ov.load_provenance(path) is None
    with pytest.warns(UserWarning, match="cannot verify"):
        ov.check(path, matrix,
                 ov.OverlapProvenance(n_db=10, n_q=10, stride_db=1, stride_q=1, shape=(12, 9)))


def test_shipped_matrix_still_loads_and_reports_its_parameters():
    """The matrix checked into the repo predates the sidecar."""
    from pathlib import Path

    repo_matrix = (Path(__file__).resolve().parents[1] / "overlap_matrices"
                   / "overlap_Roundabout01_Ouster_Roundabout03_Aeva.txt")
    if not repo_matrix.exists():
        pytest.skip("shipped overlap matrix not present")
    prov = ov.load_provenance(repo_matrix)
    assert prov is not None
    assert prov.from_legacy_header
    assert (prov.n_db, prov.n_q) == (1, 1)  # HeLiPR scans are pre-accumulated
