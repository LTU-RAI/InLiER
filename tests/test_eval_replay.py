"""What a finished run has to record for `inlier play` to replay it.

Playback used to rebuild the artifact tag and the descriptor-cache names from
the sequence and sensor fields -- a second copy of a naming rule, and one that
produced nonsense for a loader with no sensor.  The run now records them, and
records enough to rebuild its own loader, so these are the contract between
``cross_session`` and ``playback``.
"""

import numpy as np
import pytest

from inlier.eval.datasets import get_source, source_from_describe
from inlier.eval.datasets.generic import GenericSource
from inlier.eval.datasets.helipr import HeLiPRSource
from inlier.eval.protocols.cross_session import (
    CrossSessionSpec, _artifact_provenance, _cache_tag,
)

HELIPR = {"dataset_type": "helipr", "dataset_path": "/data/HeLiPR",
          "sequence": "Roundabout01", "sensor": "Ouster",
          "scan_type": "Undistorted"}
GENERIC = {"dataset_type": "generic", "path": "/data/campus_ouster",
           "n_scans": 40, "stride": 5, "transform": False}


@pytest.mark.parametrize("described", [HELIPR, GENERIC])
def test_describe_round_trips_through_from_describe(described):
    rebuilt = source_from_describe(described)
    assert rebuilt.describe() == described


def test_from_describe_restores_the_submap_accumulation():
    """The whole point: a replay must not re-window the sequence."""
    rebuilt = source_from_describe(GENERIC)
    assert (rebuilt.n_scans, rebuilt.stride) == (40, 5)
    assert rebuilt.tag == "campus_ouster_n40s5"


def test_from_describe_honours_a_moved_dataset_root():
    assert source_from_describe(HELIPR, root="/mnt/other").dataset_path.as_posix() \
        == "/mnt/other"
    assert source_from_describe(GENERIC, root="/mnt/other").path.as_posix() \
        == "/mnt/other"


def test_the_registry_covers_every_source_the_cli_offers():
    assert get_source("helipr") is HeLiPRSource
    assert get_source("generic") is GenericSource


def test_cache_tag_matches_what_the_encoder_was_given():
    assert _cache_tag(source_from_describe(HELIPR)) == "Roundabout01_Ouster_Undistorted"
    assert _cache_tag(source_from_describe(GENERIC)) == "campus_ouster_n40s5_Undistorted"


def _spec(tmp_path, transform=None):
    return CrossSessionSpec(
        resolved=None,
        db_source=source_from_describe(GENERIC),
        q_source=source_from_describe(HELIPR),
        overlap_path=tmp_path, output_dir=tmp_path,
        db_transform=transform, tag="run_tag")


def test_artifact_provenance_names_the_files_the_run_wrote(tmp_path):
    block = _artifact_provenance(_spec(tmp_path))
    assert block["tag"] == "run_tag"
    assert block["db_cache"] == "campus_ouster_n40s5_Undistorted"
    assert block["q_cache"] == "Roundabout01_Ouster_Undistorted"
    assert block["db_transform"] is None


def test_artifact_provenance_carries_the_transform_as_json(tmp_path):
    """The caches hold untransformed DB poses; a replay has to reapply it."""
    T = np.eye(4)
    T[0, 3] = 12.5
    block = _artifact_provenance(_spec(tmp_path, transform=T))

    assert isinstance(block["db_transform"], list)
    assert np.allclose(np.asarray(block["db_transform"]), T)


# --- the 0.2.x results schema ----------------------------------------------
# The published results checked into this repo predate the db/query/artifacts
# blocks, and README points `inlier play` straight at them.

V1 = {"config": {"db_sequence": "Roundabout01", "db_sensor": "Ouster",
                 "q_sequence": "Roundabout03", "q_sensor": "Aeva",
                 "overlap_threshold": 0.2, "max_pose_dist": 10.0}}


def test_run_identity_prefers_what_the_run_recorded():
    from inlier.eval.playback import run_identity

    data = {"db": GENERIC, "query": HELIPR,
            "artifacts": {"tag": "t", "db_cache": "a", "q_cache": "b",
                          "db_transform": None},
            "config": V1["config"]}
    db, q, written = run_identity(data, dataset_root=None)
    assert (db, q) == (GENERIC, HELIPR)
    assert written["tag"] == "t"


def test_run_identity_reconstructs_a_0_2_x_run():
    from inlier.eval.playback import run_identity

    db, q, written = run_identity(V1, dataset_root="/data/HeLiPR")
    assert db["dataset_type"] == "helipr" and db["sequence"] == "Roundabout01"
    assert q["sensor"] == "Aeva"
    assert written["tag"] == "Roundabout01_Ouster_Roundabout03_Aeva_ov0.2_pd10.0m"
    assert written["db_cache"] == "Roundabout01_Ouster_Undistorted"
    assert written["db_transform"] is None


def test_run_identity_reconstruction_needs_a_dataset_root():
    """A 0.2.x run did not record where its dataset was."""
    from inlier.eval.playback import run_identity

    with pytest.raises(ValueError, match="--dataset"):
        run_identity(V1, dataset_root=None)


def test_the_published_results_still_replay():
    """README points `inlier play` at the checked-in HeLiPR run."""
    import json
    from pathlib import Path

    from inlier.eval.playback import run_identity

    hits = sorted(Path("results").glob("HeLiPR/*/results_*.json"))
    if not hits:
        pytest.skip("published results not present in this checkout")

    _, _, written = run_identity(json.loads(hits[0].read_text()),
                                 dataset_root="/data/HeLiPR")
    assert (hits[0].parent / f"candidates_{written['tag']}.csv").exists()


def test_session_label_falls_back_to_the_folder_without_a_sensor():
    from inlier.eval.playback import session_label

    assert session_label(HELIPR) == "Ouster"
    assert session_label(GENERIC) == "campus_ouster"


## --- single-session runs (online-lcd) ---


def test_run_identity_accepts_a_single_session_run():
    """One `session` block stands in for both db and query.

    The replay reads two sessions because cross-session has two; an online-lcd
    run has one, and every layer of the animation should read it.
    """
    from inlier.eval.playback import run_identity

    described = {"dataset_type": "generic", "path": "/data/campus",
                 "n_scans": 40, "stride": 10}
    data = {"protocol": "online_lcd", "session": described,
            "artifacts": {"tag": "campus_lcd", "cache": "campus_n40s10_Undistorted"}}

    db, q, written = run_identity(data, None)
    assert db is q is described
    assert written["tag"] == "campus_lcd"
    assert written["db_cache"] == written["q_cache"] == "campus_n40s10_Undistorted"
    ## one session cannot have been mapped in a different frame from itself
    assert written["db_transform"] is None


def test_run_identity_still_prefers_the_two_session_blocks():
    """A cross-session run must not be re-read as single-session."""
    from inlier.eval.playback import run_identity

    data = {"db": {"a": 1}, "query": {"b": 2},
            "artifacts": {"tag": "t", "db_cache": "d", "q_cache": "q"}}
    db, q, written = run_identity(data, None)
    assert db == {"a": 1} and q == {"b": 2}
    assert written["db_cache"] == "d" and written["q_cache"] == "q"


def test_frame_index_z_spans_the_plot_range_in_order():
    """Single-session playback puts the frame index on z.

    Raw indices would run to N and leave the axes' z-limits, so they are
    scaled; what has to survive is the ordering and the proportions, since
    an edge's height is read as "how long the loop took".
    """
    import numpy as np

    from inlier.eval.playback import TRAJ_Z_LIFT, Z_OFFSET, frame_index_z

    z = frame_index_z(405)
    assert z.shape == (405,)
    assert z[0] == pytest.approx(TRAJ_Z_LIFT)
    assert z[-1] == pytest.approx(TRAJ_Z_LIFT + Z_OFFSET)
    assert np.all(np.diff(z) > 0)                    # strictly climbing
    ## evenly spaced: a 10-frame gap is the same height anywhere in the run
    assert np.allclose(np.diff(z), np.diff(z)[0])


def test_frame_index_z_degenerate_lengths():
    """A one-frame run must not divide by zero; an empty one stays empty."""
    from inlier.eval.playback import TRAJ_Z_LIFT, frame_index_z

    assert frame_index_z(1).tolist() == [pytest.approx(TRAJ_Z_LIFT)]
    assert frame_index_z(0).shape == (0,)
