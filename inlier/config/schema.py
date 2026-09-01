"""The one YAML -> dataclass mapping.

Before this module the mapping was written out by hand in four places
(``evaluate_inlier_helipr.py`` main() + run_evaluation(), and the same pair in
``evaluate_inlier_generic.py``) and the copies had already drifted: the HeLiPR
script defaulted ``stage1.mint_scoring`` to ``l1_intersection`` while the
generic one defaulted it to ``cosine``, and only the generic script read
``encoder.window`` at all.  Both drifts were latent -- ``config/default.yaml``
sets both keys explicitly, so a run that passes a full config was unaffected --
but "the same config" was not guaranteed to mean the same run.

Two rules keep that from coming back:

1. **The packaged ``default.yaml`` is the defaults.**  A user config is deep
   merged onto it, so there are no second-guess ``cfg.get(key, literal)``
   fallbacks anywhere and no place for a fallback to drift from the shipped
   file.
2. **Unknown keys are an error.**  The YAML is deliberately a superset of the
   dataclasses -- it carries orchestration flags (``stage2.skip``,
   ``rerank.run``, ``verify.skip``, ``verify.topv``, ``gicp.skip``) that no
   dataclass has -- so the accepted keys are enumerated here explicitly.  A
   typo used to silently no-op; now it raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from inlier.core.Dataclasses import (
    BEAMScoreConfig,
    GICPRefineConfig,
    InLiER_Config,
    RerankConfig,
    ShortlistConfig,
    VerifyConfig,
)

Mode = Literal["eval", "deploy"]

# --------------------------------------------------------------------------
#  Accepted keys.  Anything not listed here is rejected by validate().
# --------------------------------------------------------------------------

SCHEMA: Dict[str, Any] = {
    "voxel_size": None,  # scalar at top level
    "encoder": {
        "N_h", "z_min", "z_max", "N_r", "N_s", "N_a", "r_max", "xy_max",
        "cell_size", "window", "point_mode",
        # passed straight through to InLiER_Config when present
        "max_kp_per_slice", "max_kp_total", "ransac_iters",
        "ransac_dist_thresh", "ransac_min_inliers",
        "shape_radius", "shape_min_neighbors",
    },
    "stage1": {"topk", "topk_pct", "min_shared_rows", "mint_mode", "mint_scoring"},
    "stage2": {"skip", "topk", "topk_pct", "min_shared_bins", "min_shared_az_cols",
               "score_threshold"},
    "rerank": {"run", "topk", "topk_pct", "scoring_mode", "min_shared_rows",
               "spatial_tol", "score_threshold"},
    "verify": {"skip", "topv", "ransac_iters", "inlier_dist", "min_correspondences",
               "min_ransac_inliers", "min_keypoint_inliers", "spatial_tol", "seed"},
    "gicp": {"skip", "registration_type", "max_correspondence_distance",
             "use_raw_clouds", "downsampling_resolution", "voxel_resolution",
             "num_threads", "max_iterations"},
}

# ``score_threshold`` is forced to this during evaluation so that every
# candidate survives the stage and the PR sweep can see the full score range.
# Deployment uses the value from the config.  This was an inline constant in
# run_evaluation(); it is a named mode here so a results file can say which
# one produced it.
EVAL_SCORE_THRESHOLD = -2.0


def validate(cfg: Dict[str, Any]) -> None:
    """Raise ``ValueError`` on any key the schema does not define."""
    unknown = []
    for key, value in cfg.items():
        if key not in SCHEMA:
            unknown.append(key)
            continue
        allowed = SCHEMA[key]
        if allowed is None:  # scalar leaf
            continue
        if not isinstance(value, dict):
            raise ValueError(f"config section '{key}' must be a mapping, got {type(value).__name__}")
        for sub in value:
            if sub not in allowed:
                unknown.append(f"{key}.{sub}")
    if unknown:
        known = ", ".join(
            k if SCHEMA[k] is None else f"{k}.{{{','.join(sorted(SCHEMA[k]))}}}"
            for k in SCHEMA
        )
        raise ValueError(
            "unknown config key(s): " + ", ".join(sorted(unknown))
            + "\naccepted keys: " + known
        )


@dataclass
class ResolvedConfig:
    """Every dataclass the pipeline needs, plus the orchestration flags."""

    mode: Mode
    voxel_size: float

    inlier: InLiER_Config
    shortlist: ShortlistConfig
    beam: BEAMScoreConfig
    rerank: Optional[RerankConfig]
    verify: VerifyConfig
    gicp: GICPRefineConfig

    # orchestration (no dataclass owns these)
    skip_stage2: bool = False
    run_rerank: bool = False
    skip_verify: bool = False
    skip_gicp: bool = False
    verify_topv: int = 20

    # the deploy-mode thresholds, kept for reporting even in eval mode
    stage2_score_threshold_deploy: float = 0.0
    rerank_score_threshold_deploy: float = 0.0

    raw: Dict[str, Any] = field(default_factory=dict)


def resolve(cfg: Dict[str, Any], mode: Mode = "eval") -> ResolvedConfig:
    """Turn a merged, validated config dict into the six core dataclasses.

    ``mode="eval"`` forces the stage-2 / rerank ``score_threshold`` to
    ``EVAL_SCORE_THRESHOLD`` so the PR sweep sees every candidate;
    ``mode="deploy"`` keeps the thresholds the config asks for.
    """
    if mode not in ("eval", "deploy"):
        raise ValueError(f"mode must be 'eval' or 'deploy', got {mode!r}")
    validate(cfg)

    voxel_size = float(cfg["voxel_size"])
    enc = cfg["encoder"]
    s1 = cfg["stage1"]
    s2 = cfg["stage2"]
    rr = cfg["rerank"]
    ver = cfg["verify"]
    gic = cfg["gicp"]

    # Values derived from voxel_size.  These lived only inside run_evaluation()
    # and were therefore invisible to anyone reading the YAML.
    #
    # NOTE: run_evaluation() hardcoded `cell_size = 2 * voxel_size` and ignored
    # the YAML key entirely -- even though README.md documents `cell_size` as a
    # tunable ("BEV cell size (m) ... keep it ~ 2 x voxel_size").  Here an
    # explicit config value wins and the derivation is the fallback, so the
    # documented knob actually does something.  With the shipped default.yaml
    # (voxel_size 0.5, cell_size 1.0) the two agree, so no existing result
    # changes; only a config that sets them inconsistently behaves differently.
    # `ransac_dist_thresh` and `shape_radius` are absent from default.yaml, so
    # they stay derived unless a user opts in.
    inlier_cfg = InLiER_Config(
        N_h=enc["N_h"],
        z_min=enc["z_min"],
        z_max=enc["z_max"],
        r_max=enc["r_max"],
        N_r=enc["N_r"],
        N_a=enc["N_a"],
        N_s=enc["N_s"],
        xy_max=enc["xy_max"],
        window=enc["window"],
        point_mode=enc["point_mode"],
        cell_size=enc.get("cell_size", 2.0 * voxel_size),
        ransac_dist_thresh=enc.get("ransac_dist_thresh", 2.0 * voxel_size),
        shape_radius=enc.get("shape_radius", 3.0 * voxel_size),
        **{k: enc[k] for k in (
            "max_kp_per_slice", "max_kp_total", "ransac_iters",
            "ransac_min_inliers", "shape_min_neighbors",
        ) if k in enc},
    )

    shortlist_cfg = ShortlistConfig(
        topk=s1["topk"],
        topk_pct=s1["topk_pct"],
        min_shared_rows=s1["min_shared_rows"],
        mint_mode=s1["mint_mode"],
        mint_scoring=s1["mint_scoring"],
    )

    s2_thr = EVAL_SCORE_THRESHOLD if mode == "eval" else s2["score_threshold"]
    beam_cfg = BEAMScoreConfig(
        topk=s2["topk"],
        topk_pct=s2["topk_pct"],
        min_shared_bins=s2["min_shared_bins"],
        min_shared_az_cols=s2["min_shared_az_cols"],
        score_threshold=s2_thr,
    )

    run_rerank = bool(rr["run"])
    rerank_cfg = None
    if run_rerank:
        rr_thr = EVAL_SCORE_THRESHOLD if mode == "eval" else rr["score_threshold"]
        rerank_cfg = RerankConfig(
            topk=rr["topk"],
            scoring_mode=rr["scoring_mode"],
            score_threshold=rr_thr,
            **{k: rr[k] for k in ("topk_pct", "min_shared_rows", "spatial_tol") if k in rr},
        )

    verify_cfg = VerifyConfig(
        ransac_iters=ver["ransac_iters"],
        inlier_dist_thresh=ver["inlier_dist"],
        min_correspondences=ver["min_correspondences"],
        min_ransac_inliers=ver["min_ransac_inliers"],
        min_keypoint_inliers=ver["min_keypoint_inliers"],
        spatial_tol=ver["spatial_tol"],
        **({"seed": ver["seed"]} if "seed" in ver else {}),
    )

    gicp_cfg = GICPRefineConfig(
        registration_type=gic["registration_type"],
        max_correspondence_distance=gic["max_correspondence_distance"],
        downsampling_resolution=gic["downsampling_resolution"],
        voxel_resolution=gic["voxel_resolution"],
        num_threads=gic["num_threads"],
        use_raw_clouds=gic["use_raw_clouds"],
        max_iterations=gic["max_iterations"],
    )

    return ResolvedConfig(
        mode=mode,
        voxel_size=voxel_size,
        inlier=inlier_cfg,
        shortlist=shortlist_cfg,
        beam=beam_cfg,
        rerank=rerank_cfg,
        verify=verify_cfg,
        gicp=gicp_cfg,
        skip_stage2=bool(s2["skip"]),
        run_rerank=run_rerank,
        skip_verify=bool(ver["skip"]),
        skip_gicp=bool(gic["skip"]),
        verify_topv=int(ver["topv"]),
        stage2_score_threshold_deploy=float(s2["score_threshold"]),
        rerank_score_threshold_deploy=float(rr["score_threshold"]),
        raw=cfg,
    )
