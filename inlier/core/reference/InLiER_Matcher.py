"""InLiER Matcher — multi-stage place recognition pipeline.

Stages
------
mint_shortlist   : Rotation-invariant Minimum-ceiling INTersection (MINT) retrieval (full database).
beam_score : Binary Elevation Azimuth Matching (BEAM) on the shortlist.
rerank      : Full 4-D histogram scoring (jaccard4d / cosine4d). (not in the paper, can boost performance slightly)
verify      : Token-guided RANSAC pose estimation.
refine_gicp : GICP-based 6-DOF pose refinement (via ``small_gicp``).

Workflow
--------
1. ``add(database_id, tokens)``  → register each DB scan.
2. ``finalize()``                → materialize stacked histograms.
3. ``mint_shortlist(query_tokens)``   → ShortlistOutput
4. ``beam_score(query_tokens, shortlist_ids)`` → BEAMScoreOutput
5. ``rerank(query_tokens, shift_ids, shifts)``  → RerankOutput
6. ``verify(q_tok, q_kp, db_tok, db_kp, shift)`` → VerifyOutput
7. ``refine_gicp(verify_out, q_kp, db_kp, ...)``  → GICPRefineOutput
"""

from __future__ import annotations

import math
import time as _time
from typing import Dict, List, Optional, Tuple

import numpy as np
import small_gicp

from inlier.core.Dataclasses import (
    InLiER_Config,
    ShortlistConfig, ShortlistOutput,
    BEAMScoreConfig, BEAMScoreOutput,
    RerankConfig, RerankOutput,
    VerifyConfig, VerifyOutput,
    GICPRefineConfig, GICPRefineOutput,
    InLiER_Keypoints, InLiER_Tokens,
)


## ---- vectorised popcount for uint16/uint32/uint64 arrays ----

_BIT_TABLE = np.unpackbits(
    np.arange(256, dtype=np.uint8).reshape(-1, 1),
    axis=1, bitorder="big",
).sum(axis=1).astype(np.int32)          # (256,) lookup: byte -> popcount

def _popcount_u16(arr: np.ndarray) -> np.ndarray:
    """Element-wise popcount for a uint16, uint32, or uint64 ndarray.  Returns int32."""
    itemsize = arr.dtype.itemsize
    flat_bits = _BIT_TABLE[arr.ravel().view(np.uint8)].reshape(-1, itemsize)
    return flat_bits.sum(axis=1).reshape(arr.shape)

def _popcount_sum_rows(arr_2d: np.ndarray) -> np.ndarray:
    """For a (M, N) uint16, uint32, or uint64 array, return (M,) total popcount per row."""
    M, N = arr_2d.shape
    flat_bits = _BIT_TABLE[arr_2d.ravel().view(np.uint8)]
    return flat_bits.reshape(M, -1).sum(axis=1)


class InLiER_Matcher:
    """Multi-stage LiDAR Heterogeneous Place Recognition Matcher.

    Public API
    ----------
    add(database_id, tokens)
    finalize()
    mint-shortlist(query_tokens)  → ShortlistOutput
    beam_score(query_tokens, …)   → BEAMScoreOutput
    rerank(query_tokens, …)       → RerankOutput
    verify(…)                     → VerifyOutput
    """

    def __init__(
        self,
        inlier_config:      InLiER_Config    = InLiER_Config(),
        shortlist_config:   ShortlistConfig  = ShortlistConfig(),
        beam_score_config:  BEAMScoreConfig  = BEAMScoreConfig(),
        rerank_config:      RerankConfig     = RerankConfig(),
        verify_config:      Optional[VerifyConfig] = None,
    ) -> None:
        self._inlier_cfg      = inlier_config
        self._Nh: int = inlier_config.N_h
        self._Nr: int = inlier_config.N_r
        self._Ns: int = inlier_config.N_s
        self._Na: int = inlier_config.N_a

        self._shortlist_cfg   = shortlist_config
        self._shift_score_cfg = beam_score_config
        self._rerank_cfg      = rerank_config
        self._verify_cfg      = verify_config

        ## per-scan storage (before finalize)
        self._db_ids:          List[int]        = []
        self._token_ids:       List[np.ndarray] = []
        self._max_active_hbs:  List[int]        = []
        self._hb_arrays:       List[np.ndarray] = []
        self._rb_arrays:       List[np.ndarray] = []
        self._sb_arrays:       List[np.ndarray] = []
        self._ab_arrays:       List[np.ndarray] = []

        self._finalized: bool = False
        self.db_ids: Optional[np.ndarray] = None

        ## stacked arrays built at finalize()
        self._HM:      Optional[np.ndarray] = None   # (N, Nh, Nr*Ns) float32
        self._max_hbs: Optional[np.ndarray] = None   # (N,) int32

        self._id_to_index: Dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._db_ids)

    ## ---- database management ----

    def add(self, database_id: int, tokens: InLiER_Tokens) -> None:
        """Register one DB scan.  Must be called before finalize().

        Unpacks (hb, rb, sb, ab) from ``token_id`` once and caches them
        alongside ``max_active_hb`` so that matching stages never
        redundantly decode the same scan.
        """
        if self._finalized:
            raise RuntimeError("Matcher is finalized; call reset() to rebuild.")

        db_id = int(database_id)
        if db_id in self._id_to_index:
            raise ValueError(f"database_id {db_id} already exists.")

        self._id_to_index[db_id] = len(self._db_ids)
        self._db_ids.append(db_id)
        tid_dtype = np.uint64 if tokens.token_id.dtype.itemsize > 4 else np.uint32
        tid = tokens.token_id.astype(tid_dtype, copy=False)
        self._token_ids.append(tid)

        ## Unpack bins from token_id once
        hb, rb, sb, ab = self._unpack(tid)
        self._max_active_hbs.append(int(hb.max()) if hb.shape[0] > 0 else -1)
        self._hb_arrays.append(hb)
        self._rb_arrays.append(rb)
        self._sb_arrays.append(sb)
        self._ab_arrays.append(ab)

    def reset(self) -> None:
        """Clear all stored scans and reset to pre-finalize state."""
        self.__init__(
            self._inlier_cfg, self._shortlist_cfg, self._shift_score_cfg,
            self._rerank_cfg, self._verify_cfg,
        )

    def _unpack(self, token_id: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Unpack token_id → (hb, rb, sb, ab) as int16 arrays."""
        Nr, Ns, Na = self._Nr, self._Ns, self._Na
        tid = token_id.astype(np.int64)
        ab = (tid % Na).astype(np.int16)
        tmp = tid // Na
        sb = (tmp % Ns).astype(np.int16)
        tmp //= Ns
        rb = (tmp % Nr).astype(np.int16)
        hb = (tmp // Nr).astype(np.int16)
        return hb, rb, sb, ab

    def _unpack_query(self, tokens: InLiER_Tokens) -> Dict[str, object]:
        """Unpack a query token once for use across a single stage call.

        Returns a dict with hb, rb, sb, ab arrays and max_active_hb.
        """
        hb, rb, sb, ab = self._unpack(tokens.token_id)
        return {
            "token_id":      tokens.token_id,
            "hb": hb, "rb": rb, "sb": sb, "ab": ab,
            "max_active_hb": int(hb.max()) if len(tokens) > 0 else -1,
        }

    def get_scan_data(self, database_id: int) -> Dict[str, object]:
        """Retrieve per-scan token arrays for a given database ID.

        Used internally by beam_score / rerank / verify to access
        individual DB scan representations without duplicating storage.
        """
        idx = self._id_to_index.get(int(database_id))
        if idx is None:
            raise KeyError(f"database_id {database_id} not found.")
        return {
            "hb":            self._hb_arrays[idx],
            "rb":            self._rb_arrays[idx],
            "sb":            self._sb_arrays[idx],
            "ab":            self._ab_arrays[idx],
            "max_active_hb": self._max_active_hbs[idx],
            "token_id":      self._token_ids[idx],
        }

    def finalize(self, verbose: Optional[bool] = True) -> None:
        """Materialize stacked hist_matrix from raw token_ids."""
        if self._finalized:
            return

        t0 = _time.perf_counter()
        N   = len(self._db_ids)
        Nh, Nr, Ns, Na = self._Nh, self._Nr, self._Ns, self._Na
        V = Nh * Nr * Ns

        self.db_ids   = np.asarray(self._db_ids, dtype=np.int64)
        self._max_hbs = np.asarray(self._max_active_hbs, dtype=np.int32)

        if N == 0:
            self._HM = np.zeros((0, Nh, Nr * Ns), dtype=np.float32)
            self._finalized = True
            return

        hm_list = [
            self._tokens_to_hist(self._token_ids[i], Na, V, Nh, Nr * Ns)
            for i in range(N)
        ]
        self._HM = np.stack(hm_list, axis=0)
        self._finalized = True

        if verbose:
            dt = _time.perf_counter() - t0
            print(f"[InLiER_Matcher] finalize: {N} scans in {dt * 1000:.0f}ms")

    ## ---- shortlist — rotation-invariant MINT retrieval ----

    def shortlist(
        self,
        query_tokens: InLiER_Tokens,
        topk: Optional[int] = None,
        topk_pct: Optional[float] = None,
        verbose: Optional[bool] = True,
    ) -> ShortlistOutput:
        """Query the full database; return a shortlist sorted by MINT score.

        Parameters
        ----------
        query_tokens : InLiER_Tokens for the query scan.
        topk         : Override config topk.
        topk_pct     : Override config topk_pct.
        """
        t0 = _time.perf_counter()
        self.finalize()
        assert self.db_ids is not None and self._HM is not None

        N = int(self.db_ids.shape[0])
        if N == 0:
            return ShortlistOutput(ids=[], scores=[])

        k = self._resolve_topk(
            topk, topk_pct,
            self._shortlist_cfg.topk, self._shortlist_cfg.topk_pct, N,
        )

        ## reconstruct query hist_matrix
        Nh, Nr, Ns, Na = self._Nh, self._Nr, self._Ns, self._Na
        V = Nh * Nr * Ns
        q = self._unpack_query(query_tokens)

        q_hm = self._tokens_to_hist(
            query_tokens.token_id, Na, V, Nh, Nr * Ns,
        )

        scores = self._score_mint(q_hm, q["max_active_hb"])
        ids, sc = self._topk_sorted(scores, k)
        out = ShortlistOutput(
            ids=[int(x) for x in ids],
            scores=[float(x) for x in sc],
        )
        if verbose:
            dt = _time.perf_counter() - t0
            print(
                f"[InLiER_Matcher] MINT shortlist: top-{k} from {N} scans "
                f"in {dt * 1000:.0f}ms"
            )
        return out

    ## ---- beam score — Binary Elevation Azimuth Matching ----

    def beam_score(
        self,
        query_tokens: InLiER_Tokens,
        candidate_ids: List[int],
        topk: Optional[int] = None,
        topk_pct: Optional[float] = None,
        verbose: Optional[bool] = True,
    ) -> BEAMScoreOutput:
        """Rerank candidates using BEAM bit-level Jaccard with circular
        azimuth shift search.

        Parameters
        ----------
        query_tokens  : InLiER_Tokens for the query scan.
        candidate_ids : DB IDs from the shortlist stage.
        topk / topk_pct : Override config.
        """
        t0 = _time.perf_counter()
        cfg = self._shift_score_cfg
        Nh, Nr, Na = self._Nh, self._Nr, self._Na

        n_cands = len(candidate_ids)
        k = self._resolve_topk(
            topk, topk_pct, cfg.topk, cfg.topk_pct, n_cands,
        )

        ## Unpack query once, build BEAM
        q = self._unpack_query(query_tokens)
        q_beam = self._build_beam(q["hb"], q["rb"], q["ab"], Nh, Nr, Na)
        q_max_hb = q["max_active_hb"]

        scores = np.full(n_cands, 0.0, dtype=np.float64)
        shifts = np.zeros(n_cands, dtype=np.int32)

        for i, db_id in enumerate(candidate_ids):
            scan = self.get_scan_data(db_id)
            db_beam = self._build_beam(scan["hb"], scan["rb"],
                                      scan["ab"], Nh, Nr, Na)
            db_max_hb = int(scan["max_active_hb"])

            best_score, best_n = self._score_beam_shifts(
                q_beam, q_max_hb, db_beam, db_max_hb, cfg,
            )
            scores[i] = best_score
            shifts[i] = best_n

        ## Yaw from shift
        yaws = self._shifts_to_yaw(shifts, Na)

        ## Score threshold + top-k
        scores = np.where(scores >= cfg.score_threshold, scores, 0.0)
        order = self._topk_order(scores, k)

        out = BEAMScoreOutput(
            ids           = [int(candidate_ids[j]) for j in order],
            scores        = [float(scores[j])      for j in order],
            yaw_estimates = [float(yaws[j])        for j in order],
            best_shifts   = [int(shifts[j])        for j in order],
        )
        if verbose:
            dt = _time.perf_counter() - t0
            print(
                f"[InLiER_Matcher] BEAM score: {n_cands} → top-{k} "
                f"in {dt * 1000:.0f}ms"
            )
        return out

    ## ---- rerank — full 4-D histogram scoring (jaccard4d / cosine4d) ----

    def rerank(
        self,
        query_tokens: InLiER_Tokens,
        candidate_ids: List[int],
        candidate_shifts: List[int],
        topk: Optional[int] = None,
        topk_pct: Optional[float] = None,
        verbose: Optional[bool] = True,
    ) -> RerankOutput:
        """Rerank candidates using combined 4-D histogram + token inlier scoring.

        For each candidate at its estimated azimuth shift, computes:
          - hist_score  : 4-D histogram similarity (jaccard4d or cosine4d)
          - inlier_ratio: token correspondence inlier ratio (pointwise)

        Combined score = hist_score × inlier_ratio.
        Both must be positive for a nonzero result.

        Parameters
        ----------
        query_tokens     : InLiER_Tokens for the query scan.
        candidate_ids    : DB IDs from the shift_score stage.
        candidate_shifts : Azimuth bin shifts per candidate.
        topk / topk_pct  : Override config.
        """
        cfg = self._rerank_cfg
        mode = cfg.scoring_mode.lower()
        Nh, Nr, Na, Ns = self._Nh, self._Nr, self._Na, self._Ns
        tol = int(cfg.spatial_tol)

        t0 = _time.perf_counter()
        
        n_cands = len(candidate_ids)
        k = self._resolve_topk(
            topk, topk_pct, cfg.topk, getattr(cfg, 'topk_pct', None), n_cands,
        )

        ## Unpack query once for histogram + inlier counting
        q = self._unpack_query(query_tokens)
        q_hist = self._build_4d_hist(
            q["token_id"], Nh, Nr, Ns, Na,
        )
        q_max_hb = q["max_active_hb"]

        q_hb = q["hb"].astype(np.int32)
        q_rb = q["rb"].astype(np.int32)
        q_sb = q["sb"].astype(np.int32)
        q_ab = q["ab"].astype(np.int32)
        Kq = len(query_tokens)

        hist_scores    = np.zeros(n_cands, dtype=np.float64)
        inlier_ratios  = np.zeros(n_cands, dtype=np.float64)
        inlier_counts  = np.zeros(n_cands, dtype=np.int32)
        combined       = np.full(n_cands, 0.0, dtype=np.float64)
        best_shifts    = np.array(candidate_shifts, dtype=np.int32)

        for i, db_id in enumerate(candidate_ids):
            scan = self.get_scan_data(db_id)
            db_max_hb = int(scan["max_active_hb"])
            shift = int(candidate_shifts[i])

            ## --- Histogram score ---
            db_hist = self._build_4d_hist(
                scan["token_id"], Nh, Nr, Ns, Na,
            )
            if shift != 0:
                db_hist = np.roll(db_hist, shift=shift, axis=2)

            if mode == "jaccard4d":
                h_score, _ = self._score_jaccard4d(
                    q_hist, q_max_hb, db_hist, db_max_hb, Nh)
            elif mode == "cosine4d":
                h_score, _ = self._score_cosine4d(
                    q_hist, q_max_hb, db_hist, db_max_hb, Nh, cfg.eps)
            else:
                raise ValueError(
                    f'scoring_mode must be "jaccard4d" | "cosine4d", got "{mode}"'
                )
            hist_scores[i] = h_score

            ## --- Token inlier score ---
            db_hb = scan["hb"].astype(np.int32)
            db_rb = scan["rb"].astype(np.int32)
            db_sb = scan["sb"].astype(np.int32)
            db_ab = (scan["ab"].astype(np.int32) + shift) % Na

            count = self._verify_tokens_vectorized(
                q_hb, q_rb, q_sb, q_ab,
                db_hb, db_rb, db_sb, db_ab,
                Nh, Nr, Na, tol,
            )
            inlier_counts[i] = count
            inlier_ratios[i] = float(count) / max(Kq, 1)

            ## --- Product fusion ---
            combined[i] =  hist_scores[i] * inlier_ratios[i]

        ## Yaw from shift
        yaws = self._shifts_to_yaw(best_shifts, Na)

        ## Score threshold + top-k
        combined = np.where(combined >= cfg.score_threshold, combined, 0.0)
        order = self._topk_order(combined, k)

        if verbose:
            print(
                f"[InLiER_Matcher] rerank: {n_cands} → top-{k} "
                f"in {(_time.perf_counter() - t0) * 1000:.0f}ms"
            )

        return RerankOutput(
            ids            = [int(candidate_ids[j])  for j in order],
            scores         = [float(combined[j])     for j in order],
            hist_scores    = [float(hist_scores[j])  for j in order],
            inlier_ratios  = [float(inlier_ratios[j]) for j in order],
            inlier_counts  = [int(inlier_counts[j])  for j in order],
            yaw_estimates  = [float(yaws[j])         for j in order],
            best_shifts    = [int(best_shifts[j])    for j in order],
        )

    ## ---- verify — token-guided RANSAC pose estimation ----

    @staticmethod
    def _build_T_aligned(yaw: float, tx: float, ty: float, tz: float) -> np.ndarray:
        """Build the 4×4 SE(3) transform in the ground-aligned frame from
        the 4-DOF estimate (yaw, tx, ty, tz).

            p_db_aligned = T_aligned @ p_query_aligned
        """
        c, s = math.cos(yaw), math.sin(yaw)
        T = np.eye(4, dtype=np.float64)
        T[0, 0] = c;  T[0, 1] = -s;  T[0, 3] = tx
        T[1, 0] = s;  T[1, 1] = c;   T[1, 3] = ty
        T[2, 3] = tz
        return T

    @staticmethod
    def _aligned_to_sensor_T(
        T_aligned: np.ndarray,
        T_ground_query: np.ndarray,
        T_ground_db: np.ndarray,
    ) -> np.ndarray:
        """Convert a ground-aligned-frame transform to the original sensor frame.

        Given:
            T_aligned maps query_aligned → db_aligned
            T_ground_query maps query_sensor → query_aligned
            T_ground_db    maps db_sensor   → db_aligned

        We want T_sensor that maps query_sensor → db_sensor:

            T_sensor = T_ground_db⁻¹ @ T_aligned @ T_ground_query

        Returns (4, 4) float64.
        """
        ## T_ground_db inverse (rotation-only inversion, more stable)
        T_db_inv = np.eye(4, dtype=np.float64)
        R_db = T_ground_db[:3, :3]
        t_db = T_ground_db[:3, 3]
        T_db_inv[:3, :3] = R_db.T
        T_db_inv[:3, 3] = -(R_db.T @ t_db)

        return T_db_inv @ T_aligned @ T_ground_query

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
        """Estimate 4-DOF pose (yaw, tx, ty, tz) via token-guided RANSAC.

        1. Establish putative 3-D correspondences from token matches
           at the given azimuth alignment.
        2. RANSAC for 2-D rigid transform (yaw + tx, ty).
        3. Refine from inliers via SVD, estimate tz from median residual.
        4. Count keypoint inliers by transforming all query keypoints
           with the refined pose and measuring proximity to DB keypoints.
        5. Convert to the original sensor frame via ``T_ground``.

        Returns
        -------
        VerifyOutput
            ``T_sensor`` is the 4×4 SE(3) in the original sensor frame.
        """
        t0 = _time.perf_counter()
        cfg = config if config is not None else self._verify_cfg
        if cfg is None:
            raise RuntimeError("verify_config not set; pass it to __init__ or verify().")

        n_q_kp = len(query_keypoints)
        _T_identity = np.eye(4, dtype=np.float64)

        ## Unpack both token sets once
        q = self._unpack_query(query_tokens)
        db = self._unpack_query(db_tokens)

        ## putative correspondences
        q_corr, db_corr = self._find_correspondences(
            q["hb"], q["rb"], q["ab"],
            db["hb"], db["rb"], db["ab"],
            azimuth_shift, cfg,
            self._Nh, self._Nr, self._Na,
        )

        n_corr = q_corr.shape[0]
        fail = VerifyOutput(
            success=False, T_sensor=_T_identity.copy(),
            yaw=0.0, tx=0.0, ty=0.0, tz=0.0,
            n_correspondences=n_corr, n_ransac_inliers=0,
            n_keypoint_inliers=0, n_total_keypoints=n_q_kp,
            ransac_inlier_ratio=0.0, keypoint_inlier_ratio=0.0,
            inlier_rmse=float('inf'),
        )
        if n_corr < max(2, cfg.min_correspondences):
            if verbose:
                dt = _time.perf_counter() - t0
                print(
                    f"[InLiER_Matcher] verify: failed — "
                    f"{n_corr} correspondences (min {cfg.min_correspondences}) "
                    f"in {dt * 1000:.0f}ms"
                )
            return fail

        ## RANSAC (ground-aligned frame)
        q_pts = query_keypoints.p_aligned[q_corr]
        db_pts = db_keypoints.p_aligned[db_corr]

        best_inliers, best_yaw, best_tx, best_ty = self._ransac_2d_rigid(
            q_pts, db_pts, cfg,
        )
        n_ransac_inliers = int(best_inliers.sum()) if best_inliers is not None else 0
        if n_ransac_inliers < cfg.min_ransac_inliers:
            if verbose:
                dt = _time.perf_counter() - t0
                print(
                    f"[InLiER_Matcher] verify: failed — "
                    f"{n_ransac_inliers} RANSAC inliers (min {cfg.min_ransac_inliers}) "
                    f"in {dt * 1000:.0f}ms"
                )
            return fail

        ## SVD refinement from inliers
        inl_q  = q_pts[best_inliers]
        inl_db = db_pts[best_inliers]

        yaw_ref, tx_ref, ty_ref = self._refine_2d_rigid(inl_q, inl_db)

        ## tz from median height residual
        dz = inl_db[:, 2] - inl_q[:, 2]
        tz_ref = float(np.median(dz))

        ## Keypoint inliers: transform ALL query keypoints, find nearest DB
        n_kp_inliers, rmse = self._count_keypoint_inliers(
            query_keypoints.p_aligned,
            db_keypoints.p_aligned,
            yaw_ref, tx_ref, ty_ref, tz_ref,
            cfg.inlier_dist_thresh,
        )

        if n_kp_inliers < cfg.min_keypoint_inliers:
            if verbose:
                dt = _time.perf_counter() - t0
                print(
                    f"[InLiER_Matcher] verify: failed — "
                    f"{n_kp_inliers} keypoint inliers (min {cfg.min_keypoint_inliers}) "
                    f"in {dt * 1000:.0f}ms"
                )
            return fail

        ## Convert to sensor frame:
        ##   T_sensor = T_ground_db⁻¹ @ T_aligned @ T_ground_query
        T_aligned = self._build_T_aligned(yaw_ref, tx_ref, ty_ref, tz_ref)
        T_sensor = self._aligned_to_sensor_T(
            T_aligned, query_keypoints.T_ground, db_keypoints.T_ground,
        )

        out = VerifyOutput(
            success=True,
            T_sensor=T_sensor,
            yaw=float(yaw_ref),
            tx=float(tx_ref),
            ty=float(ty_ref),
            tz=float(tz_ref),
            n_correspondences=n_corr,
            n_ransac_inliers=n_ransac_inliers,
            n_keypoint_inliers=n_kp_inliers,
            n_total_keypoints=n_q_kp,
            ransac_inlier_ratio=float(n_ransac_inliers) / max(n_corr, 1),
            keypoint_inlier_ratio=float(n_kp_inliers) / max(n_q_kp, 1),
            inlier_rmse=rmse,
        )
        if verbose:
            dt = _time.perf_counter() - t0
            print(
                f"[InLiER_Matcher] verify: {n_corr} corr, "
                f"{n_ransac_inliers} RANSAC inliers, "
                f"{n_kp_inliers}/{n_q_kp} kp inliers, "
                f"rmse={rmse:.3f}m in {dt * 1000:.0f}ms"
            )
        return out

    ## ---- refine_gicp — GICP-based 6-DOF pose refinement ----

    @staticmethod
    def refine_gicp(
        verify_output: Optional[VerifyOutput],
        query_keypoints: InLiER_Keypoints,
        db_keypoints: InLiER_Keypoints,
        query_raw: Optional[np.ndarray] = None,
        db_raw: Optional[np.ndarray] = None,
        config: Optional[GICPRefineConfig] = None,
        verbose: Optional[bool] = True,
        init_T_override: Optional[np.ndarray] = None,
    ) -> GICPRefineOutput:
        """Refine the initial pose from ``verify`` to full 6-DOF using GICP.

        Uses ``small_gicp`` (Koide et al.) to run Generalized-ICP seeded
        with the 4-DOF initial estimate.  The result is a full SE(3)
        transform in the **original sensor frame**.

        Parameters
        ----------
        verify_output    : VerifyOutput from the verify stage (must have
                           ``success=True``) when ``init_T_override`` is
                           *None*; ``T_sensor`` is then used as the initial
                           guess.  May be *None* when ``init_T_override`` is
                           supplied.
        query_keypoints  : Query scan keypoints (used when
                           ``config.use_raw_clouds=False``).
        db_keypoints     : DB scan keypoints (used when
                           ``config.use_raw_clouds=False``).
        query_raw        : (N, 3) raw query point cloud in sensor frame.
                           Required when ``config.use_raw_clouds=True``.
        db_raw           : (M, 3) raw DB point cloud in sensor frame.
                           Required when ``config.use_raw_clouds=True``.
        config           : GICPRefineConfig.  Uses defaults if *None*.
        verbose          : Print timing / diagnostics.
        init_T_override  : Optional (4, 4) SE(3) query→DB initial guess.
                           When provided it is used as the GICP seed and the
                           ``verify_output.success`` gate is **bypassed**
                           (``verify_output`` may be *None*).  Used for
                           KISS-seeded refinement (conditions 5/8, pass the
                           KISS pose) and the identity-seeded HeLiOS baseline
                           (condition 6, pass ``np.eye(4)``).

        Returns
        -------
        GICPRefineOutput
            ``T_sensor`` is the refined 6-DOF SE(3) in the sensor frame.

        Raises
        ------
        ValueError
            If raw clouds are required but not provided, or if verify
            did not succeed.
        """
        t0 = _time.perf_counter()
        cfg = config if config is not None else GICPRefineConfig()

        _T_identity = np.eye(4, dtype=np.float64)
        fail = GICPRefineOutput(
            success=False, T_sensor=_T_identity.copy(),
            converged=False, n_iterations=0,
            final_error=float('inf'), n_inliers=0,
        )

        ## When an explicit init is supplied we bypass the verify gate
        ## (verify_output may be None).  Otherwise we require a successful
        ## verify whose pose seeds the alignment.
        if init_T_override is None:
            if verify_output is None or not verify_output.success:
                if verbose:
                    print("[InLiER_Matcher] refine_gicp: skipped — verify did not succeed")
                return fail

        ## ---- Select point clouds ----
        if cfg.use_raw_clouds:
            if query_raw is None or db_raw is None:
                raise ValueError(
                    "use_raw_clouds=True requires both query_raw and db_raw "
                    "(Nx3 numpy arrays in the original sensor frame)."
                )
            source_pts = np.asarray(query_raw, dtype=np.float64)
            target_pts = np.asarray(db_raw, dtype=np.float64)
        else:
            source_pts = np.asarray(query_keypoints.p, dtype=np.float64)
            target_pts = np.asarray(db_keypoints.p, dtype=np.float64)

        if source_pts.ndim != 2 or source_pts.shape[1] != 3:
            raise ValueError(f"source points must be (N, 3), got {source_pts.shape}")
        if target_pts.ndim != 2 or target_pts.shape[1] != 3:
            raise ValueError(f"target points must be (M, 3), got {target_pts.shape}")

        if source_pts.shape[0] < 10 or target_pts.shape[0] < 10:
            if verbose:
                print(
                    f"[InLiER_Matcher] refine_gicp: skipped — "
                    f"too few points (source={source_pts.shape[0]}, "
                    f"target={target_pts.shape[0]})"
                )
            return fail

        ## ---- Initial guess: explicit override else verify pose ----
        if init_T_override is not None:
            init_T = np.asarray(init_T_override, dtype=np.float64)
            if init_T.shape != (4, 4):
                raise ValueError(
                    f"init_T_override must be (4, 4), got {init_T.shape}"
                )
        else:
            init_T = verify_output.T_sensor.astype(np.float64)

        ## ---- Preprocess + align ----
        ## small_gicp convention: result.T_target_source transforms
        ## source points into the target frame, i.e.
        ##     p_target = T_target_source @ p_source
        ## which matches our T_sensor (query→DB).
        ##
        ## When use_raw_clouds=True we use the convenience function
        ## ``small_gicp.align`` with numpy arrays which internally
        ## downsamples, estimates normals/covariances, and builds KD-trees.
        ##
        ## When use_raw_clouds=False (keypoint clouds, typically <1k pts)
        ## we preprocess manually to skip downsampling and use a finer
        ## correspondence distance.

        reg_type = cfg.registration_type.upper()

        if cfg.use_raw_clouds:
            ## Convenience path — handles preprocessing internally
            result = small_gicp.align(
                target_pts, source_pts,
                init_T_target_source=init_T,
                registration_type=reg_type,
                downsampling_resolution=cfg.downsampling_resolution,
                max_correspondence_distance=cfg.max_correspondence_distance,
                max_iterations=cfg.max_iterations,
                num_threads=cfg.num_threads,
            )
        else:
            ## Manual preprocessing — no downsampling on sparse keypoints
            target_pc, target_tree = small_gicp.preprocess_points(
                target_pts,
                downsampling_resolution=-1.0,   # skip downsampling
                num_threads=cfg.num_threads,
            )
            source_pc, source_tree = small_gicp.preprocess_points(
                source_pts,
                downsampling_resolution=-1.0,
                num_threads=cfg.num_threads,
            )
            result = small_gicp.align(
                target_pc, source_pc, target_tree,
                init_T_target_source=init_T,
                registration_type=reg_type,
                max_correspondence_distance=cfg.max_correspondence_distance,
                max_iterations=cfg.max_iterations,
                num_threads=cfg.num_threads,
            )

        ## ---- Pack output ----
        T_refined = np.array(result.T_target_source, dtype=np.float64)
        converged = bool(result.converged)
        n_iters = int(result.iterations)
        final_err = float(result.error)
        n_inliers = int(result.num_inliers)

        out = GICPRefineOutput(
            success=converged,
            T_sensor=T_refined,
            converged=converged,
            n_iterations=n_iters,
            final_error=final_err,
            n_inliers=n_inliers,
        )

        if verbose:
            dt = _time.perf_counter() - t0
            print(
                f"[InLiER_Matcher] refine_gicp ({reg_type}): "
                f"converged={converged}, {n_iters} iters, "
                f"error={final_err:.4f}, {n_inliers} inliers, "
                f"{dt * 1000:.0f}ms"
            )
        return out

    ## ---- private MINT scoring (shortlist) ----
    ##
    ## All MINT variants share the same first steps:
    ##   1. Pairwise height-ceiling pruning
    ##   2. Collapse over height → 1-D vector of length Rw = Nr·Ns
    ##
    ## They differ only in the scoring function applied to the
    ## collapsed (range × shape) vectors.
    ##
    ## ``mint_scoring``:
    ##   "cosine"           — cosine similarity on raw counts (recommended)
    ##   "raw_intersection" — Σ min(q,d) / Σ q  on raw counts
    ##   "l1_intersection"  — Σ min(q̃,d̃) on L1-normalised vectors (original)

    def _score_mint(
        self,
        q_hm: np.ndarray,
        q_max_hb: int,
    ) -> np.ndarray:
        """Ceiling-prune, then score using the configured mode and function.

        ``mint_mode`` controls the descriptor shape before scoring:

        ``"compact"``
            Collapse height → 1-D vector of length Rw = Nr·Ns.
            10× smaller, faster comparison.

        ``"full"``
            Flatten the full ceiling-pruned Nh×Rw → 1-D vector of
            length Nh·Rw.  Preserves height structure for more
            discriminative scoring.

        ``mint_scoring`` selects the scoring function applied to the
        resulting vectors: ``"cosine"`` | ``"raw_intersection"`` |
        ``"l1_intersection"``.

        Parameters
        ----------
        q_hm      : (Nh, Rw) query histogram (Rw = Nr·Ns).
        q_max_hb  : highest occupied height bin of the query.

        Returns
        -------
        (N,) float32 scores in [0, 1].
        """
        assert self._HM is not None and self._max_hbs is not None
        cfg = self._shortlist_cfg
        eps        = float(cfg.eps)
        min_shared = int(cfg.min_shared_rows)

        N, Nh, Rw = self._HM.shape     # (N_db, N_h, N_r·N_s)

        if q_max_hb < 0:
            return np.zeros(N, dtype=np.float32)

        ## ----------------------------------------------------------
        ## Step 1: Pairwise height-ceiling pruning
        ## ----------------------------------------------------------
        ceilings    = np.minimum(q_max_hb, self._max_hbs)       # (N,)
        hb_idx      = np.arange(Nh, dtype=np.int32)
        height_mask = (hb_idx[None, :] <= ceilings[:, None])    # (N, Nh)

        ## Gate: require at least min_shared active height rows
        q_row_active = (q_hm.sum(axis=1) >= 0.5)                # (Nh,)
        n_valid = (height_mask & q_row_active[None, :]).sum(axis=1).astype(np.int32)

        ## ----------------------------------------------------------
        ## Step 2: Build descriptor vectors based on mint_mode
        ## ----------------------------------------------------------
        row_mask = height_mask[:, :, None].astype(np.float32)   # (N, Nh, 1)

        mode = cfg.mint_mode.lower()
        if mode == "compact":
            # Collapse over height → (N, Rw)
            db_vec = (self._HM * row_mask).sum(axis=1)            # (N, Rw)
            q_vec  = (q_hm[None, :, :] * row_mask).sum(axis=1)   # (N, Rw)
        elif mode == "full":
            # Flatten ceiling-pruned (N, Nh, Rw) → (N, Nh*Rw)
            db_vec = (self._HM * row_mask).reshape(N, -1)         # (N, Nh*Rw)
            q_vec  = (q_hm[None, :, :] * row_mask).reshape(N, -1)  # (N, Nh*Rw)
        else:
            raise ValueError(
                f'mint_mode must be "compact" | "full", got "{mode}"'
            )

        ## ----------------------------------------------------------
        ## Step 3: Score using the selected function
        ## ----------------------------------------------------------
        scoring = cfg.mint_scoring.lower()

        if scoring == "cosine":
            scores = self._mint_cosine(q_vec, db_vec, eps)
        elif scoring == "raw_intersection":
            scores = self._mint_raw_intersection(q_vec, db_vec, eps)
        elif scoring == "l1_intersection":
            scores = self._mint_l1_intersection(q_vec, db_vec, eps)
        else:
            raise ValueError(
                f'mint_scoring must be "cosine" | "raw_intersection" '
                f'| "l1_intersection", got "{scoring}"'
            )

        ## Apply min_shared_rows gate
        scores = np.where(n_valid >= min_shared, scores, 0.0).astype(np.float32)
        return scores

    ## ---- MINT scoring functions ----
    ## Each takes (N, Rw) collapsed vectors and returns (N,) scores.

    @staticmethod
    def _mint_cosine(
        q: np.ndarray,
        db: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """Cosine similarity on raw (unnormalised) collapsed vectors.

            S = (q · d) / (‖q‖ · ‖d‖)

        Invariant to total token count (L2-normalised implicitly).
        Not penalised by FOV asymmetry — only the *direction* of the
        range×shape profile matters, not its magnitude.

        Returns (N,) float64 scores in [-1, 1]  (typically [0, 1] for
        non-negative histograms).
        """
        dot    = (q * db).sum(axis=1)                           # (N,)
        q_norm = np.sqrt((q * q).sum(axis=1))  + eps            # (N,)
        d_norm = np.sqrt((db * db).sum(axis=1)) + eps           # (N,)
        return dot / (q_norm * d_norm)

    @staticmethod
    def _mint_raw_intersection(
        q: np.ndarray,
        db: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """Histogram intersection on raw counts, normalised by query total.

            S = Σ min(q_i, d_i) / Σ q_i

        Asks: "what fraction of the query's tokens are explained by the
        database entry?"  Extra structure in the database does not hurt.
        Naturally tolerant to FOV asymmetry — a narrow-FOV query that is
        fully contained in a wide-FOV database scores 1.0.

        Returns (N,) float64 scores in [0, 1].
        """
        inter   = np.minimum(q, db).sum(axis=1)                # (N,)
        q_total = q.sum(axis=1) + eps                           # (N,)
        return inter / q_total

    @staticmethod
    def _mint_l1_intersection(
        q: np.ndarray,
        db: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """Histogram intersection on L1-normalised vectors (original MINT).

            S = Σ min(q̃_i, d̃_i)     where q̃ = q / ‖q‖₁

        Compares the *shape* of two probability distributions.
        Score is 1.0 iff both distributions are identical.

        Note: can underperform under FOV asymmetry because L1-norm
        concentrates mass into the few occupied bins of a narrow-FOV
        sensor, causing a distribution mismatch with a wide-FOV sensor
        even when observing the same place.

        Returns (N,) float64 scores in [0, 1].
        """
        q_norm  = q  / (q.sum(axis=1,  keepdims=True) + eps)
        db_norm = db / (db.sum(axis=1, keepdims=True) + eps)
        return np.minimum(q_norm, db_norm).sum(axis=1)

    @staticmethod
    def _tokens_to_hist(
        token_id: np.ndarray,
        Na: int, V: int, Nh: int, Rw: int,
    ) -> np.ndarray:
        """Build hist_matrix (Nh, Rw) from token_ids.

        stage1_id = token_id // Na  (strips azimuth).
        """
        if token_id.shape[0] == 0:
            return np.zeros((Nh, Rw), dtype=np.float32)
        stage1_ids = (token_id // token_id.dtype.type(Na)).astype(np.int64)
        counts = np.bincount(stage1_ids, minlength=V)
        return counts.reshape(Nh, Rw).astype(np.float32)

    ## ---- private — BEAM scoring (beam_score) ----

    @staticmethod
    def _build_beam(
        hb: np.ndarray,
        rb: np.ndarray,
        ab: np.ndarray,
        Nh: int, Nr: int, Na: int,
    ) -> np.ndarray:
        """Build (Nr, Na) binary elevation vector matrix.

        Each cell beam[r, a] is a bitmask where bit h is set iff at least
        one token exists at (h, r, a).  Uses uint16 when Nh<=16, uint32 when
        Nh<=32, uint64 otherwise.
        """
        dtype = np.uint16 if Nh <= 16 else (np.uint32 if Nh <= 32 else np.uint64)
        beam_flat = np.zeros(Nr * Na, dtype=dtype)
        if hb.shape[0] == 0:
            return beam_flat.reshape(Nr, Na)
        bits = dtype(1) << hb.astype(dtype)
        ra_flat = rb.astype(np.int32) * Na + ab.astype(np.int32)
        np.bitwise_or.at(beam_flat, ra_flat, bits)
        return beam_flat.reshape(Nr, Na)

    @staticmethod
    def _score_beam_shifts(
        q_beam: np.ndarray,
        q_max_hb: int,
        db_beam: np.ndarray,
        db_max_hb: int,
        cfg: BEAMScoreConfig,
    ) -> Tuple[float, int]:
        """Score all N_a shifts using bit-level Jaccard on BEAM matrices.

        Returns (best_score, best_shift).
        """
        Nr, Na = q_beam.shape

        ceiling = min(q_max_hb, db_max_hb)
        if ceiling < 0:
            return 0.0, 0

        ceil_mask = q_beam.dtype.type((1 << (ceiling + 1)) - 1)
        q  = np.ascontiguousarray(q_beam  & ceil_mask)
        db = np.ascontiguousarray(db_beam & ceil_mask)

        q_total  = int(_popcount_u16(q).sum())
        db_total = int(_popcount_u16(db).sum())

        if q_total + db_total == 0:
            return 0.0, 0

        ## All shifted copies via fancy indexing
        col_idx = np.arange(Na, dtype=np.int32)
        shift_offsets = np.arange(Na, dtype=np.int32)[:, None]
        src_cols = (col_idx[None, :] - shift_offsets) % Na
        db_shifts = db[:, src_cols].transpose(1, 0, 2)

        ## Intersection popcount for all shifts
        and_all = (q[None, :, :] & db_shifts)
        and_flat = np.ascontiguousarray(and_all.reshape(Na, Nr * Na))
        inter_per_shift = _popcount_sum_rows(and_flat)

        ## Union from identity
        union_per_shift = q_total + db_total - inter_per_shift

        ## Azimuth column validity
        q_col_any  = (q > 0).any(axis=0)
        db_col_any = (db > 0).any(axis=0)
        db_col_shifted = db_col_any[src_cols]
        az_union_cols = (q_col_any[None, :] | db_col_shifted).sum(axis=1)

        valid = ((union_per_shift >= cfg.min_shared_bins)
                 & (az_union_cols >= cfg.min_shared_az_cols))

        scores = np.where(
            valid,
            inter_per_shift.astype(np.float64) / np.maximum(union_per_shift, 1).astype(np.float64),
            0.0,
        )

        best_n = int(np.argmax(scores))
        return float(scores[best_n]), best_n

    ## ---- private — 4-D histogram scoring (rerank) ----

    @staticmethod
    def _build_4d_hist(
        token_id: np.ndarray,
        Nh: int, Nr: int, Ns: int, Na: int,
    ) -> np.ndarray:
        """Build full 4-D histogram (Nh, Nr*Ns, Na) from token_ids."""
        V = Nh * Nr * Ns * Na
        if token_id.shape[0] == 0:
            return np.zeros((Nh, Nr * Ns, Na), dtype=np.float32)
        ids = token_id.astype(np.int64)
        counts = np.bincount(ids, minlength=V)
        return counts.reshape(Nh, Nr * Ns, Na).astype(np.float32)

    @staticmethod
    def _score_jaccard4d(
        q_hist: np.ndarray,
        q_max_hb: int,
        db_hist: np.ndarray,
        db_max_hb: int,
        Nh: int,
    ) -> Tuple[float, int]:
        """Boolean Jaccard on the full 4-D occupancy (at fixed shift)."""
        ceiling = min(q_max_hb, db_max_hb)
        if ceiling < 0:
            return 0.0, 0

        hmask = np.zeros(Nh, dtype=bool)
        hmask[:ceiling + 1] = True
        q_occ  = (q_hist > 0)  & hmask[:, None, None]
        db_occ = (db_hist > 0) & hmask[:, None, None]

        both  = q_occ & db_occ
        union = q_occ | db_occ
        n_union = int(union.sum())
        if n_union == 0:
            return 0.0, 0
        return float(int(both.sum())) / n_union, 0

    @staticmethod
    def _score_cosine4d(
        q_hist: np.ndarray,
        q_max_hb: int,
        db_hist: np.ndarray,
        db_max_hb: int,
        Nh: int,
        eps: float = 1e-9,
    ) -> Tuple[float, int]:
        """Column-wise cosine similarity (Scan Context style, at fixed shift)."""
        Rs = q_hist.shape[1]
        Na = q_hist.shape[2]
        ceiling = min(q_max_hb, db_max_hb)
        if ceiling < 0:
            return 0.0, 0

        hmask = np.zeros(Nh, dtype=np.float32)
        hmask[:ceiling + 1] = 1.0
        q_c  = q_hist  * hmask[:, None, None]
        db_c = db_hist * hmask[:, None, None]

        q_cols  = q_c.reshape(Nh * Rs, Na)
        db_cols = db_c.reshape(Nh * Rs, Na)

        q_col_norms  = np.sqrt((q_cols ** 2).sum(axis=0)) + eps
        db_col_norms = np.sqrt((db_cols ** 2).sum(axis=0)) + eps

        dot = (q_cols * db_cols).sum(axis=0)
        cos_sim = dot / (q_col_norms * db_col_norms)
        active = (q_col_norms > 2 * eps) | (db_col_norms > 2 * eps)
        n_active = int(active.sum())
        if n_active == 0:
            return 0.0, 0
        return float(cos_sim[active].sum()) / n_active, 0

    ## ---- private — correspondence finding + RANSAC (verify) ----

    @staticmethod
    def _count_keypoint_inliers(
        q_pts: np.ndarray,
        db_pts: np.ndarray,
        yaw: float, tx: float, ty: float, tz: float,
        dist_thresh: float,
    ) -> Tuple[int, float]:
        """Transform all query keypoints and count those with a close DB neighbour.

        Returns (n_keypoint_inliers, inlier_rmse).
        """
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        q_x = cos_y * q_pts[:, 0] - sin_y * q_pts[:, 1] + tx
        q_y = sin_y * q_pts[:, 0] + cos_y * q_pts[:, 1] + ty
        q_z = q_pts[:, 2] + tz

        ## Nearest-neighbour via brute-force (typically <1k keypoints)
        ## Compute pairwise squared distances: (Nq, Ndb)
        dx = q_x[:, None] - db_pts[None, :, 0]
        dy = q_y[:, None] - db_pts[None, :, 1]
        dz = q_z[:, None] - db_pts[None, :, 2]
        dist_sq = dx * dx + dy * dy + dz * dz

        min_dist_sq = dist_sq.min(axis=1)
        thresh_sq = dist_thresh ** 2
        inlier_mask = min_dist_sq < thresh_sq

        n_inliers = int(inlier_mask.sum())
        if n_inliers == 0:
            return 0, float('inf')
        rmse = float(np.sqrt(np.mean(min_dist_sq[inlier_mask])))
        return n_inliers, rmse

    @staticmethod
    def _verify_tokens_vectorized(
        q_hb: np.ndarray, q_rb: np.ndarray, q_sb: np.ndarray, q_ab: np.ndarray,
        db_hb: np.ndarray, db_rb: np.ndarray, db_sb: np.ndarray, db_ab: np.ndarray,
        Nh: int, Nr: int, Na: int, tol: int,
    ) -> int:
        """Count query tokens with a matching DB neighbour.

        For each query token, generates all (2×tol+1)³ spatial neighbours
        (exact shape class), checks membership in the DB key set.
        """
        Kq = len(q_hb)
        if Kq == 0 or len(db_hb) == 0:
            return 0

        ## Build DB key set as sorted flat int64
        db_keys_arr = (
            db_sb.astype(np.int64) * (Nh * Nr * Na)
            + db_hb.astype(np.int64) * (Nr * Na)
            + db_rb.astype(np.int64) * Na
            + db_ab.astype(np.int64)
        )
        db_key_sorted = np.unique(db_keys_arr)

        if tol == 0:
            ## Exact match — fast path
            q_keys = (
                q_sb.astype(np.int64) * (Nh * Nr * Na)
                + q_hb.astype(np.int64) * (Nr * Na)
                + q_rb.astype(np.int64) * Na
                + q_ab.astype(np.int64)
            )
            idx = np.searchsorted(db_key_sorted, q_keys)
            idx = np.clip(idx, 0, len(db_key_sorted) - 1)
            return int((db_key_sorted[idx] == q_keys).sum())

        ## Soft spatial match with ±tol
        offsets = np.mgrid[-tol:tol+1, -tol:tol+1, -tol:tol+1].reshape(3, -1).T
        dh_arr, dr_arr, da_arr = offsets[:, 0], offsets[:, 1], offsets[:, 2]
        M = len(dh_arr)

        hb_exp = q_hb[:, None] + dh_arr[None, :]
        rb_exp = q_rb[:, None] + dr_arr[None, :]
        ab_exp = (q_ab[:, None] + da_arr[None, :]) % Na
        sb_exp = np.broadcast_to(q_sb[:, None], (Kq, M))

        valid = (hb_exp >= 0) & (hb_exp < Nh) & (rb_exp >= 0) & (rb_exp < Nr)

        keys = (
            sb_exp.astype(np.int64) * (Nh * Nr * Na)
            + hb_exp.astype(np.int64) * (Nr * Na)
            + rb_exp.astype(np.int64) * Na
            + ab_exp.astype(np.int64)
        )
        keys[~valid] = -1

        flat_keys = keys.ravel()
        idx = np.searchsorted(db_key_sorted, flat_keys)
        idx = np.clip(idx, 0, len(db_key_sorted) - 1)
        membership = (db_key_sorted[idx] == flat_keys).reshape(Kq, M)

        return int(membership.any(axis=1).sum())

    @staticmethod
    def _find_correspondences(
        q_hb: np.ndarray, q_rb: np.ndarray, q_ab: np.ndarray,
        db_hb: np.ndarray, db_rb: np.ndarray, db_ab: np.ndarray,
        azimuth_shift: int,
        cfg: VerifyConfig,
        Nh: int, Nr: int, Na: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find putative 3-D correspondences via geometric token matching.

        Matches on spatial bins (hb, rb, ab) only — shape class (sb) is
        excluded from the matching key to improve robustness across
        heterogeneous LiDAR sensors where PCA-based shape classification
        is not transferable.

        Tokens are 1:1 with keypoints, so array indices ARE keypoint
        indices (no separate kp_index needed).

        Returns (q_indices, db_indices) into the respective keypoint
        arrays.  Applies ±spatial_tol on (hb, rb, ab).
        """
        tol = cfg.spatial_tol

        q_hb  = q_hb.astype(np.int32)
        q_rb  = q_rb.astype(np.int32)
        q_ab  = q_ab.astype(np.int32)

        db_hb  = db_hb.astype(np.int32)
        db_rb  = db_rb.astype(np.int32)
        db_ab  = (db_ab.astype(np.int32) + azimuth_shift) % Na

        ## Build DB lookup: key -> first index (== keypoint index)
        ## Key encodes only (hb, rb, ab) — no shape class
        stride_h  = Nr * Na
        stride_r  = Na

        db_keys = (db_hb * stride_h + db_rb * stride_r + db_ab)

        db_dict = {}
        for j in range(len(db_hb)):
            k = int(db_keys[j])
            if k not in db_dict:
                db_dict[k] = j

        q_matched = []
        db_matched = []

        if tol == 0:
            q_keys = (q_hb * stride_h + q_rb * stride_r + q_ab)
            for qi in range(len(q_hb)):
                db_idx = db_dict.get(int(q_keys[qi]))
                if db_idx is not None:
                    q_matched.append(qi)
                    db_matched.append(db_idx)
        else:
            for qi in range(len(q_hb)):
                hb_q, rb_q, ab_q = (
                    int(q_hb[qi]), int(q_rb[qi]), int(q_ab[qi])
                )
                found = False
                for dh in range(-tol, tol + 1):
                    hb2 = hb_q + dh
                    if hb2 < 0 or hb2 >= Nh:
                        continue
                    for dr in range(-tol, tol + 1):
                        rb2 = rb_q + dr
                        if rb2 < 0 or rb2 >= Nr:
                            continue
                        for da in range(-tol, tol + 1):
                            ab2 = (ab_q + da) % Na
                            key = hb2 * stride_h + rb2 * stride_r + ab2
                            db_idx = db_dict.get(key)
                            if db_idx is not None:
                                q_matched.append(qi)
                                db_matched.append(db_idx)
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

        if len(q_matched) == 0:
            return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
        return np.array(q_matched, dtype=np.int32), np.array(db_matched, dtype=np.int32)

    @staticmethod
    def _ransac_2d_rigid(
        q_pts: np.ndarray,
        db_pts: np.ndarray,
        cfg: VerifyConfig,
    ) -> Tuple[Optional[np.ndarray], float, float, float]:
        """RANSAC for 2-D rigid transform (yaw + tx, ty).

        Each sample uses 2 point pairs.  Inlier check uses full 3-D
        distance.

        Returns (inlier_mask, yaw, tx, ty) or (None, 0, 0, 0) on failure.
        """
        N = q_pts.shape[0]
        if N < 2:
            return None, 0.0, 0.0, 0.0

        rng = np.random.default_rng(cfg.seed)
        thresh_sq = cfg.inlier_dist_thresh ** 2

        q_xy  = q_pts[:, :2]
        db_xy = db_pts[:, :2]

        best_count = 0
        best_mask  = None
        best_yaw   = 0.0
        best_tx    = 0.0
        best_ty    = 0.0

        for _ in range(cfg.ransac_iters):
            idx = rng.choice(N, size=2, replace=False)
            p1_q, p2_q = q_xy[idx[0]], q_xy[idx[1]]
            p1_d, p2_d = db_xy[idx[0]], db_xy[idx[1]]

            dq = p2_q - p1_q
            dd = p2_d - p1_d
            dq_norm = float(np.linalg.norm(dq))
            dd_norm = float(np.linalg.norm(dd))
            if dq_norm < 1e-6 or dd_norm < 1e-6:
                continue

            cos_a = float(np.dot(dq, dd)) / (dq_norm * dd_norm)
            sin_a = float(dq[0] * dd[1] - dq[1] * dd[0]) / (dq_norm * dd_norm)
            yaw = math.atan2(sin_a, cos_a)

            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            r_q1 = np.array([cos_y * p1_q[0] - sin_y * p1_q[1],
                              sin_y * p1_q[0] + cos_y * p1_q[1]])
            tx = p1_d[0] - r_q1[0]
            ty = p1_d[1] - r_q1[1]

            rot_x = cos_y * q_xy[:, 0] - sin_y * q_xy[:, 1] + tx
            rot_y = sin_y * q_xy[:, 0] + cos_y * q_xy[:, 1] + ty

            dx = rot_x - db_xy[:, 0]
            dy = rot_y - db_xy[:, 1]
            dz = q_pts[:, 2] - db_pts[:, 2]
            dist_sq = dx * dx + dy * dy + dz * dz

            mask  = dist_sq < thresh_sq
            count = int(mask.sum())
            if count > best_count:
                best_count = count
                best_mask  = mask.copy()
                best_yaw   = yaw
                best_tx    = tx
                best_ty    = ty

        return best_mask, best_yaw, best_tx, best_ty

    @staticmethod
    def _refine_2d_rigid(
        q_pts: np.ndarray,
        db_pts: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Least-squares 2-D rigid transform via SVD on inlier centroids.

        Returns (yaw, tx, ty).
        """
        q_xy  = q_pts[:, :2]
        db_xy = db_pts[:, :2]

        q_c  = q_xy.mean(axis=0)
        db_c = db_xy.mean(axis=0)

        H = (q_xy - q_c).T @ (db_xy - db_c)
        U, _, Vt = np.linalg.svd(H)

        d = np.linalg.det(Vt.T @ U.T)
        S = np.diag([1.0, float(np.sign(d))])
        R = Vt.T @ S @ U.T

        t = db_c - R @ q_c
        yaw = float(math.atan2(R[1, 0], R[0, 0]))
        return yaw, float(t[0]), float(t[1])

    ## ---- shared utilities ----

    @staticmethod
    def _resolve_topk(
        topk: Optional[int],
        topk_pct: Optional[float],
        cfg_topk: int,
        cfg_topk_pct: Optional[float],
        n_total: int,
    ) -> int:
        """Resolve the effective top-k from call-time and config-level
        overrides.  Call-time arguments take priority over config."""
        if topk is not None:
            return int(np.clip(topk, 1, n_total))
        if topk_pct is not None:
            return int(np.clip(max(1, round(n_total * topk_pct)), 1, n_total))
        if cfg_topk_pct is not None:
            return int(np.clip(max(1, round(n_total * cfg_topk_pct)), 1, n_total))
        return int(np.clip(cfg_topk, 1, n_total))

    def _topk_sorted(
        self,
        scores: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (db_ids, scores) for top-k, sorted descending."""
        assert self.db_ids is not None
        N = scores.shape[0]
        if k >= N:
            idx = np.arange(N, dtype=np.int64)
        else:
            idx = np.argpartition(scores, -k)[-k:].astype(np.int64)
        order = np.argsort(scores[idx])[::-1]
        idx = idx[order]
        return self.db_ids[idx], scores[idx]

    @staticmethod
    def _topk_order(scores: np.ndarray, k: int) -> np.ndarray:
        """Return indices into scores for the top-k, sorted descending."""
        n = scores.shape[0]
        k = min(k, n)
        if k >= n:
            order = np.argsort(scores)[::-1]
        else:
            order = np.argpartition(scores, -k)[-k:]
            order = order[np.argsort(scores[order])[::-1]]
        return order

    @staticmethod
    def _shifts_to_yaw(shifts: np.ndarray, Na: int) -> np.ndarray:
        """Convert azimuth bin shifts to yaw angles in (-π, π]."""
        raw = ((-shifts) % Na) * (2.0 * math.pi / Na)
        return np.where(raw <= math.pi, raw, raw - 2.0 * math.pi)