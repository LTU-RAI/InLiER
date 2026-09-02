"""Config loading, merging, validation, and resolution.

The point of this layer is that "the same config" means the same run.  Before
it, the YAML -> dataclass mapping existed in four hand-written copies whose
fallbacks had already drifted (``stage1.mint_scoring`` defaulted to
``l1_intersection`` in the HeLiPR script and ``cosine`` in the generic one, and
only the generic one read ``encoder.window`` at all).
"""

import pytest

from inlier.config import DEFAULT_CONFIG_PATH, deep_merge, dump, load, parse_override, resolve
from inlier.config.schema import EVAL_SCORE_THRESHOLD, validate


def test_packaged_defaults_are_complete_and_valid():
    cfg = load()
    assert DEFAULT_CONFIG_PATH.exists()
    # every section the resolver indexes must be present, so resolve() never
    # needs an inline fallback
    for section in ("encoder", "stage1", "stage2", "rerank", "verify", "gicp"):
        assert section in cfg, section
    assert "voxel_size" in cfg
    resolve(cfg)  # must not raise


def test_resolve_produces_every_core_dataclass():
    r = resolve(load())
    assert r.inlier.N_h == 10 and r.inlier.N_r == 20 and r.inlier.N_a == 60
    assert r.shortlist.mint_scoring == "l1_intersection"
    assert r.verify.min_ransac_inliers == 16
    assert r.gicp.registration_type == "GICP"
    assert r.rerank is None  # default.yaml has rerank.run: false


def test_values_derived_from_voxel_size():
    """These lived only inside run_evaluation(), invisible from the YAML."""
    cfg = deep_merge(load(), {"voxel_size": 0.25})
    del cfg["encoder"]["cell_size"]  # let the derivation apply
    r = resolve(cfg)
    assert r.inlier.cell_size == pytest.approx(0.5)
    assert r.inlier.ransac_dist_thresh == pytest.approx(0.5)
    assert r.inlier.shape_radius == pytest.approx(0.75)


def test_explicit_cell_size_wins_over_the_derivation():
    """docs/configuration.md documents cell_size as tunable; run_evaluation ignored it."""
    r = resolve(deep_merge(load(), {"voxel_size": 0.25, "encoder": {"cell_size": 3.0}}))
    assert r.inlier.cell_size == 3.0


def test_eval_mode_opens_the_stage_thresholds_for_the_pr_sweep():
    ev = resolve(load(), mode="eval")
    dep = resolve(load(), mode="deploy")
    assert ev.beam.score_threshold == EVAL_SCORE_THRESHOLD
    assert dep.beam.score_threshold == 0.0
    # the deployment value is still reported, whichever mode ran
    assert ev.stage2_score_threshold_deploy == 0.0


def test_rerank_dataclass_appears_only_when_enabled():
    assert resolve(load(overrides=["rerank.run=true"])).rerank is not None
    assert resolve(load(overrides=["rerank.run=false"])).rerank is None


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown config key"):
        validate({"encoder": {"N_h": 10}, "nonsense": 1})
    with pytest.raises(ValueError, match=r"stage1\.topkk"):
        load(overrides=["stage1.topkk=50"])


def test_section_must_be_a_mapping():
    with pytest.raises(ValueError, match="must be a mapping"):
        validate({"encoder": 5})


@pytest.mark.parametrize("item,expected", [
    ("stage1.topk=50", {"stage1": {"topk": 50}}),
    ("voxel_size=0.25", {"voxel_size": 0.25}),
    ("verify.skip=true", {"verify": {"skip": True}}),
    ("stage2.topk_pct=null", {"stage2": {"topk_pct": None}}),
])
def test_parse_override_uses_yaml_scalars(item, expected):
    assert parse_override(item) == expected


def test_parse_override_rejects_malformed_input():
    with pytest.raises(ValueError, match="key=value"):
        parse_override("stage1.topk")


def test_overrides_apply_after_the_config_file():
    r = resolve(load(overrides=["stage1.topk=7", "verify.skip=true"]))
    assert r.shortlist.topk == 7
    assert r.skip_verify is True


def test_deep_merge_does_not_mutate_its_inputs():
    base = {"a": {"x": 1, "y": 2}}
    over = {"a": {"y": 3}}
    merged = deep_merge(base, over)
    assert merged == {"a": {"x": 1, "y": 3}}
    assert base == {"a": {"x": 1, "y": 2}}


def test_dump_round_trips_through_load(tmp_path):
    cfg = load(overrides=["stage1.topk=33"])
    path = tmp_path / "round.yaml"
    path.write_text(dump(cfg))
    assert resolve(load(path)).shortlist.topk == 33


def test_missing_config_file_is_reported():
    with pytest.raises(FileNotFoundError, match="config file not found"):
        load("/nonexistent/never.yaml")


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        resolve(load(), mode="production")
