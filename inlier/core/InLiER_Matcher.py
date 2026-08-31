"""InLiER multi-stage matcher — thin wrapper over the C++ core.

The retrieval/verification stages (MINT shortlist, BEAM, rerank,
token-guided RANSAC verify) delegate to ``inlier._inlier_pybind``; the
registration refinement (``refine_gicp``) is inherited unchanged from
the reference implementation — it wraps the already-compiled
``small_gicp`` package.

The original pure-numpy implementation lives in
``inlier.core.reference`` and is used as the automatic fallback when
the compiled extension is unavailable or ``INLIER_FORCE_PYTHON=1``.

Public API (unchanged)
----------------------
add(database_id, tokens); finalize(); reset(); get_scan_data(id)
shortlist(query_tokens, …)     → ShortlistOutput
beam_score(query_tokens, …)    → BEAMScoreOutput
rerank(query_tokens, …)        → RerankOutput
verify(…)                      → VerifyOutput
refine_gicp(…)                 (static, Python)
"""

from __future__ import annotations

import os as _os
import time as _time
import warnings as _warnings
from typing import Dict, List, Optional

import numpy as np

from inlier.core.Dataclasses import (
    BEAMScoreConfig,
    BEAMScoreOutput,
    InLiER_Config,
    InLiER_Keypoints,
    InLiER_Tokens,
    RerankConfig,
    RerankOutput,
    ShortlistConfig,
    ShortlistOutput,
    VerifyConfig,
    VerifyOutput,
)
from inlier.core.reference.InLiER_Matcher import (
    InLiER_Matcher as _ReferenceMatcher,
)

_BACKEND = "python"
if _os.environ.get("INLIER_FORCE_PYTHON", "0") != "1":
    try:
        from inlier import _inlier_pybind as _ip
        from inlier.core import _cfg_bridge as _bridge

        _BACKEND = "cpp"
    except ImportError as _err:  # pragma: no cover - build-dependent
        _warnings.warn(
            f"inlier: C++ extension unavailable ({_err}); using the "
            f"pure-Python reference implementation.")


if _BACKEND == "python":

    InLiER_Matcher = _ReferenceMatcher

else:

    class InLiER_Matcher(_ReferenceMatcher):
        """Multi-stage LiDAR place-recognition matcher (C++-accelerated)."""

        def __init__(
            self,
            inlier_config:      InLiER_Config    = InLiER_Config(),
            shortlist_config:   ShortlistConfig  = ShortlistConfig(),
            beam_score_config:  BEAMScoreConfig  = BEAMScoreConfig(),
            rerank_config:      RerankConfig     = RerankConfig(),
            verify_config:      Optional[VerifyConfig] = None,
        ) -> None:
            super().__init__(inlier_config, shortlist_config,
                             beam_score_config, rerank_config, verify_config)
            self._cpp = _ip._Matcher(
                _bridge.to_cpp_inlier_config(inlier_config))
            self._token_dtype = _bridge.token_dtype(inlier_config)

        ## ---- database management ----

        def __len__(self) -> int:
            return len(self._cpp)

        def add(self, database_id: int, tokens: InLiER_Tokens) -> None:
            ## the C++ side raises the same finalized/duplicate errors
            self._cpp.add(int(database_id),
                          np.ascontiguousarray(tokens.token_id))

        def finalize(self, verbose: Optional[bool] = True) -> None:
            if self._finalized:
                return
            t0 = _time.perf_counter()
            self._cpp.finalize()
            self.db_ids = self._cpp.db_ids()
            self._finalized = True
            if verbose:
                dt = _time.perf_counter() - t0
                print(f"[InLiER_Matcher] finalize: {len(self._cpp)} scans "
                      f"in {dt * 1000:.0f}ms")

        def get_scan_data(self, database_id: int) -> Dict[str, object]:
            try:
                tid, hb, rb, sb, ab, max_hb = self._cpp.get_scan_data(
                    int(database_id))
            except IndexError:
                raise KeyError(f"database_id {database_id} not found.")
            return {
                "hb": hb, "rb": rb, "sb": sb, "ab": ab,
                "max_active_hb": int(max_hb),
                "token_id": tid.astype(self._token_dtype),
            }

        ## ---- retrieval stages ----

        def shortlist(
            self,
            query_tokens: InLiER_Tokens,
            topk: Optional[int] = None,
            topk_pct: Optional[float] = None,
            verbose: Optional[bool] = True,
        ) -> ShortlistOutput:
            t0 = _time.perf_counter()
            self.finalize(verbose=verbose)
            N = len(self._cpp)
            if N == 0:
                return ShortlistOutput(ids=[], scores=[])

            res = self._cpp.shortlist(
                np.ascontiguousarray(query_tokens.token_id),
                _bridge.to_cpp_stage_config(
                    _ip.ShortlistConfig, self._shortlist_cfg),
                -1 if topk is None else int(topk),
                -1.0 if topk_pct is None else float(topk_pct),
            )
            out = ShortlistOutput(ids=list(res.ids), scores=list(res.scores))
            if verbose:
                dt = _time.perf_counter() - t0
                print(
                    f"[InLiER_Matcher] MINT shortlist: top-{len(out.ids)} "
                    f"from {N} scans in {dt * 1000:.0f}ms"
                )
            return out

        def beam_score(
            self,
            query_tokens: InLiER_Tokens,
            candidate_ids: List[int],
            topk: Optional[int] = None,
            topk_pct: Optional[float] = None,
            verbose: Optional[bool] = True,
        ) -> BEAMScoreOutput:
            t0 = _time.perf_counter()
            res = self._cpp.beam_score(
                np.ascontiguousarray(query_tokens.token_id),
                np.asarray(candidate_ids, dtype=np.int64),
                _bridge.to_cpp_stage_config(
                    _ip.BEAMScoreConfig, self._shift_score_cfg),
                -1 if topk is None else int(topk),
                -1.0 if topk_pct is None else float(topk_pct),
            )
            out = BEAMScoreOutput(
                ids=list(res.ids),
                scores=list(res.scores),
                yaw_estimates=list(res.yaw_estimates),
                best_shifts=list(res.best_shifts),
            )
            if verbose:
                dt = _time.perf_counter() - t0
                print(
                    f"[InLiER_Matcher] BEAM score: {len(candidate_ids)} → "
                    f"top-{len(out.ids)} in {dt * 1000:.0f}ms"
                )
            return out

        def rerank(
            self,
            query_tokens: InLiER_Tokens,
            candidate_ids: List[int],
            candidate_shifts: List[int],
            topk: Optional[int] = None,
            topk_pct: Optional[float] = None,
            verbose: Optional[bool] = True,
        ) -> RerankOutput:
            t0 = _time.perf_counter()
            res = self._cpp.rerank(
                np.ascontiguousarray(query_tokens.token_id),
                np.asarray(candidate_ids, dtype=np.int64),
                np.asarray(candidate_shifts, dtype=np.int32),
                _bridge.to_cpp_stage_config(_ip.RerankConfig,
                                            self._rerank_cfg),
                -1 if topk is None else int(topk),
                -1.0 if topk_pct is None else float(topk_pct),
            )
            out = RerankOutput(
                ids=list(res.ids),
                scores=list(res.scores),
                hist_scores=list(res.hist_scores),
                inlier_ratios=list(res.inlier_ratios),
                inlier_counts=list(res.inlier_counts),
                yaw_estimates=list(res.yaw_estimates),
                best_shifts=list(res.best_shifts),
            )
            if verbose:
                dt = _time.perf_counter() - t0
                print(
                    f"[InLiER_Matcher] rerank: {len(candidate_ids)} → "
                    f"top-{len(out.ids)} in {dt * 1000:.0f}ms"
                )
            return out

        ## ---- geometric verification ----

        def verify(
            self,
            query_tokens: InLiER_Tokens,
            query_keypoints: InLiER_Keypoints,
            db_tokens: InLiER_Tokens,
            db_keypoints: InLiER_Keypoints,
            azimuth_shift: int,
            config: Optional[VerifyConfig] = None,
            verbose: Optional[bool] = True,
        ) -> VerifyOutput:
            t0 = _time.perf_counter()
            cfg = config if config is not None else self._verify_cfg
            if cfg is None:
                raise RuntimeError(
                    "verify_config not set; pass it to __init__ or verify().")

            res = _ip.verify(
                np.ascontiguousarray(query_tokens.token_id),
                np.asarray(query_keypoints.p, dtype=np.float64),
                np.asarray(query_keypoints.T_ground, dtype=np.float64),
                np.ascontiguousarray(db_tokens.token_id),
                np.asarray(db_keypoints.p, dtype=np.float64),
                np.asarray(db_keypoints.T_ground, dtype=np.float64),
                int(azimuth_shift),
                _bridge.to_cpp_inlier_config(self._inlier_cfg),
                _bridge.to_cpp_stage_config(_ip.VerifyConfig, cfg),
            )

            out = VerifyOutput(
                success=res.success,
                T_sensor=np.asarray(res.T_sensor, dtype=np.float64),
                yaw=res.yaw, tx=res.tx, ty=res.ty, tz=res.tz,
                n_correspondences=res.n_correspondences,
                n_ransac_inliers=res.n_ransac_inliers,
                n_keypoint_inliers=res.n_keypoint_inliers,
                n_total_keypoints=res.n_total_keypoints,
                ransac_inlier_ratio=res.ransac_inlier_ratio,
                keypoint_inlier_ratio=res.keypoint_inlier_ratio,
                inlier_rmse=res.inlier_rmse,
            )

            if verbose:
                dt = (_time.perf_counter() - t0) * 1000
                if res.fail_stage == 1:
                    print(
                        f"[InLiER_Matcher] verify: failed — "
                        f"{res.n_correspondences} correspondences "
                        f"(min {cfg.min_correspondences}) in {dt:.0f}ms"
                    )
                elif res.fail_stage == 2:
                    print(
                        f"[InLiER_Matcher] verify: failed — "
                        f"{res.ransac_inliers_found} RANSAC inliers "
                        f"(min {cfg.min_ransac_inliers}) in {dt:.0f}ms"
                    )
                elif res.fail_stage == 3:
                    print(
                        f"[InLiER_Matcher] verify: failed — keypoint "
                        f"inliers below min {cfg.min_keypoint_inliers} "
                        f"in {dt:.0f}ms"
                    )
                else:
                    print(
                        f"[InLiER_Matcher] verify: {res.n_correspondences} "
                        f"corr, {res.n_ransac_inliers} RANSAC inliers, "
                        f"{res.n_keypoint_inliers}/{res.n_total_keypoints} "
                        f"kp inliers, rmse={res.inlier_rmse:.3f}m "
                        f"in {dt:.0f}ms"
                    )
            return out

        ## refine_gicp is inherited from the reference implementation (a
        ## thin Python wrapper over the already-compiled small_gicp
        ## package).
