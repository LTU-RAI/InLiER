"""Regression: the refactored pipeline must reproduce the published numbers.

Phase 1 of the CLI work moved the evaluation into the package and rewrote how
it is configured and orchestrated.  None of that is allowed to move a metric.
This test compares a fresh run against the checked-in results of the
Roundabout01 (Ouster) <- Roundabout03 (Aeva) pair.

It is skipped unless both the HeLiPR dataset and the descriptor cache are
available; with the cache present the run needs no scans at all for the
retrieval stages, which is what makes re-running it cheap enough to be a test.
Point it at a dataset with ``INLIER_TEST_DATASET=/path/to/HeLiPR``.

Two things about how it runs
----------------------------
**The backend is pinned to the numpy reference**, because that is what produced
the checked-in results.  The C++ and numpy verification stages are not
bit-identical -- RANSAC draws different samples -- and the repo already treats
that as expected: ``tests/test_verify.py`` asserts only 95% agreement on the
success flag and bounds the pose difference rather than requiring equality.  On
this pair the two backends disagree on roughly 17% of verified pairs, which
moves Recall@1 by about 0.2 points.  Comparing a C++ run against a numpy golden
file would therefore fail for a reason that has nothing to do with this
refactor.

**It runs the CLI as a subprocess**, because ``--backend`` works by setting
``INLIER_FORCE_PYTHON`` before ``inlier.core`` is first imported.  In-process
that is unreliable: any earlier test that touched the core has already fixed
the backend for the interpreter.

The operating threshold is passed explicitly (``--threshold 0.3``) to match how
the golden file was produced; its ``confusion.threshold`` of exactly 0.3 is a
pinned value, not one the max-precision search would return.
"""

import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDEN = (REPO / "results" / "HeLiPR"
          / "dbR01-O-qR03-Aeva_vs0.5_cs1_nh10_nr20_na60_ns7"
          / "results_Roundabout01_Ouster_Roundabout03_Aeva_ov0.2_pd10.0m.json")
CACHE = REPO / "cache_inlier"
CACHE_FILES = ["desc_Roundabout01_Ouster_Undistorted_a6a8d4c7cbd5.npz",
               "desc_Roundabout03_Aeva_Undistorted_a6a8d4c7cbd5.npz"]

DATASET = Path(os.environ.get("INLIER_TEST_DATASET",
                              Path.home() / "Documents" / "datasets" / "HeLiPR"))

pytestmark = pytest.mark.slow


def _requirements_met():
    return (GOLDEN.exists() and DATASET.exists()
            and all((CACHE / name).exists() for name in CACHE_FILES))


needs_data = pytest.mark.skipif(
    not _requirements_met(),
    reason="needs the HeLiPR dataset, the golden results file, and the descriptor cache",
)


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    import subprocess
    import sys

    out = tmp_path_factory.mktemp("golden")
    proc = subprocess.run(
        [sys.executable, "-m", "inlier.cli.main", "eval", "cross-session",
         "--backend", "python",
         "--dataset", str(DATASET),
         "--db-sequence", "Roundabout01", "--q-sequence", "Roundabout03",
         "--pair", "O-Aeva",
         "--overlap-dir", str(REPO / "overlap_matrices"),
         "--overlap-threshold", "0.2", "--max-pose-dist", "10.0",
         "--threshold", "0.3",
         "--cache-dir", str(CACHE),
         "-o", str(out), "--quiet"],
        cwd=REPO, capture_output=True, text=True, timeout=7200,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    produced = list(out.rglob("results_*.json"))
    assert len(produced) == 1, produced
    return json.loads(produced[0].read_text())


@needs_data
def test_dataset_shape_matches(golden, fresh):
    assert fresh["dataset_info"] == golden["dataset_info"]


@needs_data
@pytest.mark.parametrize("stage", ["stage1", "stage2", "verify"])
def test_stage_metrics_match_exactly(golden, fresh, stage):
    want, got = golden[stage], fresh[stage]
    if want is None:
        assert got is None
        return
    assert got["recall_at_n"] == want["recall_at_n"], f"{stage} Recall@N moved"
    assert got["recall_at_kpct"] == want["recall_at_kpct"], f"{stage} Recall@K% moved"
    assert got["pr_auc"] == want["pr_auc"], f"{stage} PR-AUC moved"


@needs_data
def test_confusion_matches(golden, fresh):
    want, got = golden["confusion"], fresh["confusion"]
    assert got["stage"] == want["stage"]
    assert got["threshold"] == want["threshold"], "operating threshold moved"
    for key in ("TP", "FP", "FN", "TN", "precision", "recall"):
        assert got[key] == want[key], f"confusion {key} moved"


@needs_data
def test_tp_match_distances_match(golden, fresh):
    assert fresh["confusion"]["tp_match_distance"] == golden["confusion"]["tp_match_distance"]


@needs_data
def test_verify_pose_errors_match(golden, fresh):
    want = golden["confusion"]["tp_pose_error_verify"]
    got = fresh["confusion"]["tp_pose_error_verify"]
    if want is None:
        assert got is None
        return
    for key in want:
        assert got[key] == want[key], f"verify pose error {key} moved"


@needs_data
def test_new_provenance_is_present_without_disturbing_v1_keys(fresh):
    """Schema v2 adds provenance; it must not have displaced anything."""
    assert fresh["schema_version"] == 2
    assert fresh["protocol"] == "cross_session"
    assert fresh["threshold_policy"] == "fixed"
    assert fresh["backend"] == "python"
    assert fresh["ground_truth"]["policy"] == "overlap_and_distance"
    # f1_max is reported regardless of which policy chose the operating point
    assert "f1_max" in fresh["confusion"]


@needs_data
def test_config_block_preserved(golden, fresh):
    want, got = golden["config"], fresh["config"]
    for key in ("db_sequence", "db_sensor", "q_sequence", "q_sensor",
                "overlap_threshold", "max_pose_dist", "voxel_size"):
        assert got[key] == want[key], f"config.{key} changed"
    assert got["inlier"] == want["inlier"]
    assert got["shortlist"] == want["shortlist"]
