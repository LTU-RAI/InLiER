"""InLiER: An Intermediate LiDAR Encoding for Robust Heterogeneous Place Recognition

Responsible for creating the intermediate representation of keypoints
and token IDs from a raw 3-D point cloud.

Public API
----------
extract_keypoints(points_xyz, ...)      →   InLiER_Keypoints
tokenize(points_xyz, keypoints, ...)    →   InLiER_Tokens
encode(points_xyz, ...)                 →  (InLiER_Keypoints, InLiER_Tokens)

Token definition
----------------
Each keypoint is assigned a ``uint32`` (``uint64`` when the
mixed-radix product exceeds 2³²) token via mixed-radix encoding:

    token_id = ((hb × N_r + rb) × N_s + sb) × N_a + ab

Fully invertible given (N_h, N_r, N_s, N_a).
"""

from __future__ import annotations

import math
import time as _time
from typing import Dict, Optional, Tuple
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from inlier.core.Dataclasses import InLiER_Config, InLiER_Keypoints, InLiER_Tokens
from inlier.core.banner import print_banner, print_config_banner


class InLiER:
    """InLiER encoder: keypoint extraction + token_id assignment."""

    def __init__(self, config: InLiER_Config = InLiER_Config()) -> None:
        print_banner(version="1.0.0")
        self._config = config
        print_config_banner(config)

    @property
    def config(self) -> InLiER_Config:
        return self._config

    ## ---- public API ----

    def extract_keypoints(
            self, 
            points_xyz: np.ndarray,
            plane: Optional[Dict[str, np.ndarray]] = None,
            verbose: Optional[bool] = True,
        ) -> InLiER_Keypoints:
        """Extract z-height-sliced keypoints from a point cloud.

        Parameters
        ----------
        points_xyz : (N, 3) array — raw point cloud in sensor / world frame.
        plane      : Pre-computed ground-plane dict. If *None* the plane is estimated via RANSAC.

        Returns
        -------
        ``InLiER_Keypoints``
        """
        t0 = _time.perf_counter()
        cfg = self._config
        P = np.asarray(points_xyz, dtype=np.float64)

        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError("points_xyz must have shape (N, 3)")

        if P.shape[0] == 0:
            return self._empty_keypoints()

        ## ground plane + alignment
        if plane is None:
            plane = self._ransac_plane(P, verbose=verbose)
        Pa, R, t = self._align_ground(P, plane["normal"], plane["point"])
        Rinv, tinv = R.T, -(R.T @ t)

        ## Build T_ground (4×4 sensor -> aligned)
        T_ground = np.eye(4, dtype=np.float64)
        T_ground[:3, :3] = R
        T_ground[:3, 3] = t

        ## XY ROI 
        keep_xy = (np.abs(Pa[:, 0]) <= cfg.xy_max) & (np.abs(Pa[:, 1]) <= cfg.xy_max)
        Pa = Pa[keep_xy]
        if Pa.shape[0] == 0:
            return self._empty_keypoints()

        ## z filtering & slice edges
        keep_z = (Pa[:, 2] >= cfg.z_min) & (Pa[:, 2] <= cfg.z_max)
        Pa = Pa[keep_z]
        if Pa.shape[0] == 0:
            return self._empty_keypoints()
        z = Pa[:, 2]
        z_edges = np.linspace(cfg.z_min, cfg.z_max, cfg.N_h + 1)

        ## BEV grid dimensions
        W = int(math.ceil(2.0 * cfg.xy_max / cfg.cell_size))
        H = W
        x_min = -cfg.xy_max
        y_min = -cfg.xy_max

        out = self._process_slices(Pa, z, z_edges, W, H, x_min, y_min, Rinv, tinv)

        ## cap total keypoints
        n_kp = out["score"].shape[0]
        if n_kp > cfg.max_kp_total:
            order = np.argsort(out["score"])[::-1][: cfg.max_kp_total]
            out = {k: v[order] for k, v in out.items()}
            n_kp = cfg.max_kp_total

        kp = InLiER_Keypoints(
            p=out["p"],
            T_ground=T_ground,
        )

        if verbose:
            dt = _time.perf_counter() - t0
            print(
                f"[InLiER] extract_keypoints: {n_kp} keypoints "
                f"from {P.shape[0]} points in {dt * 1000:.0f}ms"
            )
        return kp

    def tokenize(
            self,
            points_xyz: np.ndarray,
            keypoints: InLiER_Keypoints,
            plane: Optional[Dict[str, np.ndarray]] = None,
            verbose: Optional[bool] = True,
        ) -> InLiER_Tokens:
        """Assign token IDs to keypoints (or all points).

        Parameters
        ----------
        points_xyz : (N, 3) raw point cloud (needed for PCA shape queries
                     and for ``"all_points"`` mode).
        keypoints  : Pre-extracted ``InLiER_Keypoints``.
        plane      : Pre-computed ground plane.  Required for
                     ``"all_points"`` mode; ignored in ``"keypoints"`` mode.

        Returns
        -------
        ``InLiER_Tokens``
        """
        t0 = _time.perf_counter()
        cfg = self._config
        pmode = cfg.point_mode.lower()

        Nh = int(cfg.N_h)
        Nr = int(cfg.N_r)
        Ns = int(max(1, cfg.N_s))
        Na = int(max(1, cfg.N_a))

        ## choose point set for tokenization
        z_edges = np.linspace(cfg.z_min, cfg.z_max, Nh + 1)

        if pmode == "keypoints":
            pts_world = keypoints.p
            pts_aligned = keypoints.p_aligned   # derived from T_ground on the fly
            z = np.clip(pts_aligned[:, 2], cfg.z_min, cfg.z_max)
            hb_arr = np.clip(
                np.searchsorted(z_edges[1:], z, side="right").astype(np.int16),
                0, Nh - 1,
            )

        elif pmode == "all_points":
            if plane is None:
                plane = self._ransac_plane(np.asarray(points_xyz, dtype=np.float64), verbose=verbose)
            pts_world, pts_aligned, hb_arr = (
                self._all_points_arrays(points_xyz, plane)
            )
        else:
            raise ValueError(
                'point_mode must be "keypoints" | "all_points"'
            )

        K = pts_aligned.shape[0]

        if K == 0:
            tokens = self._empty_tokens()
            if verbose:
                print("[InLiER] tokenize: 0 tokens (empty input)")
            return tokens

        ## coordinate bins
        xa, ya = pts_aligned[:, 0], pts_aligned[:, 1]
        r = np.sqrt(xa * xa + ya * ya).astype(np.float32)
        theta = np.arctan2(ya, xa)

        r_max = float(min(cfg.r_max, cfg.xy_max))
        rb = self._bin_radial(r, r_max, Nr).astype(np.int16)
        ab = self._bin_azimuth(theta, Na).astype(np.int16)
        hb = hb_arr

        ## shape class (PCA neighbourhood)
        if Ns > 1:
            shape_class, _ = self._compute_shape_pca(
                points_xyz, pts_world,
                radius=cfg.shape_radius,
                min_neighbors=cfg.shape_min_neighbors,
                n_classes=Ns,
            )
            sb = np.clip(shape_class.astype(np.int16), 0, Ns - 1)
        else:
            sb = np.zeros(K, dtype=np.int16)

        ## mixed-radix token_id — uint32 when product fits, uint64 otherwise
        _pack_dtype = (
            np.uint64 if int(Nh) * int(Nr) * int(Ns) * int(Na) > np.iinfo(np.uint32).max
            else np.uint32
        )
        token_id = self.pack_token_ids(hb, rb, sb, ab, Nr, Ns, Na, dtype=_pack_dtype)

        tokens = InLiER_Tokens(token_id=token_id)

        if verbose:
            dt = _time.perf_counter() - t0
            n_unique = int(np.unique(token_id).shape[0])
            print(
                f"[InLiER] tokenize: {K} tokens ({n_unique} unique IDs), "
                f"mode='{pmode}' in {dt * 1000:.0f}ms"
            )
        return tokens

    def tokenize_keypoints(
            self,
            points_xyz: np.ndarray,
            keypoints: InLiER_Keypoints,
            plane: Optional[Dict[str, np.ndarray]] = None,
            verbose: Optional[bool] = True,
        ) -> InLiER_Tokens:
        """Tokenize using the keypoint set, regardless of cfg.point_mode.

        Used in dual-mode (all_points stage1/2 + keypoint verify) so that
        the matcher's verify path has a token set whose indices line up
        1:1 with the keypoints array.
        """
        cfg = self._config
        saved = cfg.point_mode
        try:
            cfg.point_mode = "keypoints"
            return self.tokenize(points_xyz, keypoints, plane=plane, verbose=verbose)
        finally:
            cfg.point_mode = saved

    def encode(
            self,
            points_xyz: np.ndarray,
            plane: Optional[Dict[str, np.ndarray]] = None,
            verbose: Optional[bool] = True,
        ) -> Tuple[InLiER_Keypoints, InLiER_Tokens]:
        """Full pipeline: extract keypoints → tokenize.

        Returns
        -------
        (``InLiER_Keypoints``, ``InLiER_Tokens``)
        """
        t0 = _time.perf_counter()

        P = np.asarray(points_xyz, dtype=np.float64)
        if plane is None:
            plane = self._ransac_plane(P, verbose=verbose)

        keypoints = self.extract_keypoints(P, plane=plane, verbose=verbose)
        tokens = self.tokenize(P, keypoints, plane=plane, verbose=verbose)

        if verbose:
            dt = _time.perf_counter() - t0
            print(f"[InLiER] encode: total pipeline in {dt * 1000:.0f}ms")

        return keypoints, tokens

    ## ---- token packing / unpacking ----

    @staticmethod
    def pack_token_ids(
            hb: np.ndarray,
            rb: np.ndarray,
            sb: np.ndarray,
            ab: np.ndarray,
            N_r: int,
            N_s: int,
            N_a: int,
            dtype: type = np.uint32,
        ) -> np.ndarray:
        """Pack (hb, rb, sb, ab) -> token_id.

        ``token_id = ((hb × N_r + rb) × N_s + sb) × N_a + ab``

        *dtype* is ``np.uint32`` by default; pass ``np.uint64`` when the
        mixed-radix product ``N_h × N_r × N_s × N_a`` exceeds 2³²-1.
        """
        N_r, N_s, N_a = dtype(N_r), dtype(N_s), dtype(N_a)
        return (
            (hb.astype(dtype) * N_r + rb.astype(dtype))
            * N_s
            + sb.astype(dtype)
        ) * N_a + ab.astype(dtype)

    @staticmethod
    def unpack_token_ids(
            token_id: np.ndarray,
            N_r: int,
            N_s: int,
            N_a: int,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Unpack token_id -> (hb, rb, sb, ab).

        Returns
        -------
        (hb, rb, sb, ab) — each (K,) int64
        """
        token_id = np.asarray(token_id, dtype=np.int64)
        N_r, N_s, N_a = int(N_r), int(N_s), int(N_a)
        ab = token_id % N_a
        base_id = token_id // N_a
        sb = base_id % N_s
        rb = (base_id // N_s) % N_r
        hb = (base_id // N_s) // N_r
        return hb, rb, sb, ab

    ## ---- geometry helpers ----

    @staticmethod
    def _rodrigues(
            axis: np.ndarray, 
            angle: float
        ) -> np.ndarray:
        """Axis-angle -> 3 × 3 rotation matrix (Rodrigues' formula)."""
        axis = np.asarray(axis, dtype=np.float64)
        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            return np.eye(3)
        axis = axis / n
        x, y, z = axis
        K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
        return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)

    def _ransac_plane(self, 
            points: np.ndarray, 
            verbose: bool = True
        ) -> Dict[str, np.ndarray]:
        """RANSAC ground-plane estimation.

        Returns dict with keys ``normal``, ``d``, ``point``, ``inliers``.
        """
        tic = _time.perf_counter()
        cfg = self._config
        rng = np.random.default_rng(0)
        P = np.asarray(points, dtype=np.float64)
        N = P.shape[0]
        if N < 3:
            raise ValueError("Need at least 3 points for plane estimation")

        best_cnt, best_n, best_d, best_inl = -1, None, None, None

        for _ in range(cfg.ransac_iters):
            idx = rng.choice(N, size=3, replace=False)
            p1, p2, p3 = P[idx]
            n = np.cross(p2 - p1, p3 - p1)
            nn = float(np.linalg.norm(n))
            if nn < 1e-9:
                continue
            n = n / nn
            d = -float(n @ p1)
            dist = np.abs(P @ n + d)
            inliers = dist < cfg.ransac_dist_thresh
            cnt = int(inliers.sum())
            if cnt > best_cnt and cnt >= cfg.ransac_min_inliers:
                best_cnt, best_n, best_d, best_inl = cnt, n, d, inliers

        if best_n is None:
            ## fallback: PCA plane
            mu = P.mean(axis=0)
            _, evecs = np.linalg.eigh(np.cov((P - mu).T))
            best_n = evecs[:, 0]
            best_d = -float(best_n @ mu)
            best_inl = np.ones(N, dtype=bool)

        ## ensure normal is "up-ish"
        if best_n[2] < 0:
            best_n, best_d = -best_n, -best_d

        point = -best_d * best_n

        toc = _time.perf_counter()
        if verbose:
            print(
                f"[InLiER] RANSAC plane: {best_cnt} inliers, "
                f"normal={best_n[0]:.3f}, {best_n[1]:.3f}, {best_n[2]:.3f}, d={best_d:.2f} in {(toc - tic) * 1000:.0f}ms"
            )

        return {
            "normal": best_n,
            "d": np.array([best_d], dtype=np.float64),
            "point": point,
            "inliers": best_inl,
        }

    def _align_ground(
            self,
            points: np.ndarray,
            plane_normal: np.ndarray,
            plane_point: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rotate so that ``plane_normal -> +Z`` and translate ground to z=0.

        Returns ``(points_aligned, R, t)`` where ``x' = R x + t``.
        """
        pts = np.asarray(points, dtype=np.float64)
        n = np.asarray(plane_normal, dtype=np.float64)
        n = n / (np.linalg.norm(n) + 1e-12)

        z_axis = np.array([0.0, 0.0, 1.0])
        dot = float(np.clip(n @ z_axis, -1.0, 1.0))

        if abs(dot - 1.0) < 1e-9:
            R = np.eye(3)
        elif abs(dot + 1.0) < 1e-9:
            axis = np.cross(n, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(n, np.array([0.0, 1.0, 0.0]))
            R = self._rodrigues(axis, math.pi)
        else:
            axis = np.cross(n, z_axis)
            angle = math.atan2(np.linalg.norm(axis), dot)
            R = self._rodrigues(axis, angle)

        p0r = R @ np.asarray(plane_point, dtype=np.float64)
        t = np.array([0.0, 0.0, -p0r[2]], dtype=np.float64)

        pts_aligned = (R @ pts.T).T + t
        return pts_aligned, R, t

    ## ---- binning helpers ----

    @staticmethod
    def _bin_radial(r: np.ndarray, r_max: float, n_bins: int) -> np.ndarray:
        """Quantize radial distance into ``n_bins`` equal-width bins."""
        r = np.asarray(r, dtype=np.float64)
        step = max(1e-6, float(r_max)) / int(n_bins)
        return np.clip(np.floor(r / step).astype(np.int32), 0, n_bins - 1)

    @staticmethod
    def _bin_azimuth(theta: np.ndarray, n_bins: int) -> np.ndarray:
        """Quantize azimuth angle into ``n_bins`` bins centred on cardinal
        directions."""
        theta = np.asarray(theta, dtype=np.float64)
        n_bins = int(max(1, n_bins))
        step = (2.0 * math.pi) / n_bins
        t = (theta + math.pi) % (2.0 * math.pi)          # [0, 2π)
        t = (t + step / 2.0) % (2.0 * math.pi)           # centre on cardinals
        return np.clip(np.floor(t / step).astype(np.int32), 0, n_bins - 1)

    @staticmethod
    def _local_maxima_2d(img: np.ndarray, window: int, threshold: float) -> np.ndarray:
        """Boolean mask of local maxima in a 2-D image."""
        if window % 2 == 0:
            raise ValueError("window must be odd")
        pad = window // 2
        v = np.asarray(img)
        fill = float(np.nanmin(v)) if np.isfinite(v).any() else 0.0
        padded = np.pad(v, pad, mode="constant", constant_values=fill)
        win = sliding_window_view(padded, (window, window))
        local_max = win.max(axis=(-1, -2))
        return (v == local_max) & (v >= float(threshold))

    ## ---- shape PCA ----

    @staticmethod
    def _compute_shape_pca(
            points_xyz: np.ndarray,
            centers_xyz: np.ndarray,
            radius: float = 2.0,
            min_neighbors: int = 12,
            n_classes: int = 3,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised PCA shape classification (linear / planar / scatter).

        Parameters
        ----------
        points_xyz  : (N, 3) full cloud for neighbourhood queries.
        centers_xyz : (K, 3) query centres.
        radius      : neighbourhood radius (metres).
        min_neighbors : minimum neighbours for a valid PCA.
        n_classes   : 3, 5, or 7 — controls inclination sub-bins.

        Returns
        -------
        shape_class : (K,) int16
        lps         : (K, 3) float32  [linearity, planarity, scatter]
        """
        P = np.asarray(points_xyz, dtype=np.float64)
        C = np.asarray(centers_xyz, dtype=np.float64)
        K = C.shape[0]

        n_classes = int(n_classes)
        if n_classes <= 3:
            n_incl_bins, scatter_id = 0, 2
        elif n_classes <= 5:
            n_incl_bins, scatter_id = 2, 4
        else:
            n_incl_bins, scatter_id = 3, 6

        shape_class = np.full(K, scatter_id, dtype=np.int16)
        lps = np.zeros((K, 3), dtype=np.float32)

        if K == 0 or P.shape[0] == 0:
            return shape_class, lps

        radius = float(radius)
        min_neighbors = int(min_neighbors)

        ## batched neighbourhood query
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(P)
            all_idx = tree.query_ball_point(C, r=radius, workers=-1)
        except Exception:
            r2 = radius * radius
            all_idx = []
            for i in range(K):
                d2 = np.sum((P - C[i]) ** 2, axis=1)
                all_idx.append(np.where(d2 <= r2)[0].tolist())

        lengths = np.array([len(x) for x in all_idx], dtype=np.int64)
        valid = lengths >= min_neighbors

        if not valid.any():
            lps[:, 2] = 1.0
            return shape_class, lps

        ## flat neighbour array + segment index
        total = int(lengths.sum())
        starts = np.empty(K, dtype=np.int64)
        starts[0] = 0
        np.cumsum(lengths[:-1], out=starts[1:])

        flat_neigh = np.empty((total, 3), dtype=np.float64)
        for i in range(K):
            s, n = int(starts[i]), int(lengths[i])
            if n > 0:
                flat_neigh[s : s + n] = P[all_idx[i]]

        centre_idx = np.repeat(np.arange(K, dtype=np.int64), lengths)

        ## per-centre mean (scatter-add)
        sums = np.zeros((K, 3), dtype=np.float64)
        np.add.at(sums, centre_idx, flat_neigh)
        count_safe = np.maximum(lengths, 1).astype(np.float64)
        means = sums / count_safe[:, None]

        ## covariance matrices (scatter outer-product)
        flat_c = flat_neigh - means[centre_idx]
        covs = np.zeros((K, 3, 3), dtype=np.float64)
        np.add.at(covs, centre_idx, flat_c[:, :, None] * flat_c[:, None, :])
        denom = np.maximum(lengths - 1, 1).astype(np.float64)
        covs /= denom[:, None, None]
        covs[~valid] = 0.0

        ## batched eigendecomposition
        evals_asc, evecs_asc = np.linalg.eigh(covs)
        evals = evals_asc[:, ::-1]
        evecs = evecs_asc[:, :, ::-1]

        ## LPS (linearity, planarity, scatter)
        l1 = np.maximum(evals[:, 0], 1e-12)
        l2 = np.maximum(evals[:, 1], 0.0)
        l3 = np.maximum(evals[:, 2], 0.0)

        lps[:, 0] = ((l1 - l2) / l1).astype(np.float32)
        lps[:, 1] = ((l2 - l3) / l1).astype(np.float32)
        lps[:, 2] = (l3 / l1).astype(np.float32)
        lps[~valid] = [0.0, 0.0, 1.0]

        ## shape classification (vectorised)
        base_type = np.argmax(lps, axis=1).astype(np.int32)

        if n_incl_bins == 0:
            shape_class[valid] = base_type[valid].astype(np.int16)
        else:
            direction = evecs[:, :, 0]                     # largest eigvec
            normal = evecs[:, :, 2]                        # smallest eigvec

            norm_d = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
            norm_n = np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
            cos_d = np.clip(np.abs(direction[:, 2:3]) / norm_d, 0.0, 1.0)
            cos_n = np.clip(np.abs(normal[:, 2:3]) / norm_n, 0.0, 1.0)

            incl_d = np.degrees(np.arccos(cos_d[:, 0]))
            incl_n = np.degrees(np.arccos(cos_n[:, 0]))

            step = 90.0 / n_incl_bins
            bin_d = np.clip((incl_d / step).astype(np.int32), 0, n_incl_bins - 1)
            bin_n = np.clip((incl_n / step).astype(np.int32), 0, n_incl_bins - 1)

            is_lin = valid & (base_type == 0)
            is_plan = valid & (base_type == 1)
            shape_class[is_lin] = bin_d[is_lin].astype(np.int16)
            shape_class[is_plan] = (n_incl_bins + bin_n[is_plan]).astype(np.int16)

        return shape_class, lps

    def _process_slices(
            self,
            Pa: np.ndarray,
            z_vals: np.ndarray,
            z_edges: np.ndarray,
            W: int,
            H: int,
            x_min: float,
            y_min: float,
            Rinv: np.ndarray,
            tinv: np.ndarray,
        ) -> Dict[str, np.ndarray]:
        """Process each height slice: build BEV intensity image, detect
        local maxima, and return cell centroids (sensor frame) with scores.

        Returns dict with keys ``"p"`` (K, 3) and ``"score"`` (K,).
        """
        cfg = self._config
        pmode = cfg.point_mode.lower()

        p_list: list = []
        score_list: list = []

        for s in range(cfg.N_h):
            lo, hi = float(z_edges[s]), float(z_edges[s + 1])
            sel = (z_vals >= lo) & (z_vals < hi)
            if not np.any(sel):
                continue

            pts = Pa[sel]
            xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

            ix = np.floor((xs - x_min) / cfg.cell_size).astype(np.int32)
            iy = np.floor((ys - y_min) / cfg.cell_size).astype(np.int32)
            valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            if not np.any(valid):
                continue

            ix, iy = ix[valid], iy[valid]
            xs, ys, zs = xs[valid], ys[valid], zs[valid]
            flat = iy * W + ix

            count = np.bincount(flat, minlength=H * W).astype(np.int32)
            sum_x = np.bincount(flat, weights=xs, minlength=H * W)
            sum_y = np.bincount(flat, weights=ys, minlength=H * W)
            sum_z = np.bincount(flat, weights=zs, minlength=H * W)

            min_z = np.full(H * W, np.inf)
            max_z = np.full(H * W, -np.inf)
            np.minimum.at(min_z, flat, zs)
            np.maximum.at(max_z, flat, zs)

            ## intensity image
            intensity = np.where(count > 0, max_z - min_z, 0.0)
            img = intensity.reshape(H, W)
            thr = float(0.5 * (cfg.z_max - cfg.z_min) / cfg.N_h)

            ## detect cells of interest
            if pmode == "keypoints":
                kp_mask = self._local_maxima_2d(img, cfg.window, thr)
                coords = np.argwhere(kp_mask)
                if coords.shape[0] == 0:
                    continue
                scores = img[coords[:, 0], coords[:, 1]]
                if coords.shape[0] > cfg.max_kp_per_slice:
                    top = np.argsort(scores)[-cfg.max_kp_per_slice :]
                    coords, scores = coords[top], scores[top]

            elif pmode == "all_points":
                coords = np.argwhere(img >= thr)
                if coords.shape[0] == 0:
                    continue
                scores = img[coords[:, 0], coords[:, 1]]
            else:
                raise ValueError(
                    'point_mode must be "keypoints" | "all_points"'
                )

            ## compute cell centroids (vectorised)
            nz = count > 0
            mean_x = np.where(nz, sum_x / np.maximum(count, 1), 0.0)
            mean_y = np.where(nz, sum_y / np.maximum(count, 1), 0.0)
            mean_z = np.where(nz, sum_z / np.maximum(count, 1), 0.0)

            flat_coords = coords[:, 0] * W + coords[:, 1]
            occupied = count[flat_coords] > 0

            ## filter to occupied cells only
            scores = scores[occupied]
            flat_coords = flat_coords[occupied]

            if flat_coords.shape[0] == 0:
                continue

            p_aligned_batch = np.stack(
                [mean_x[flat_coords], mean_y[flat_coords], mean_z[flat_coords]],
                axis=1,
            )
            p_world_batch = (Rinv @ p_aligned_batch.T).T + tinv

            p_list.append(p_world_batch)
            score_list.append(scores)

        if not p_list:
            return self._empty_slice_dict()

        return {
            "p": np.concatenate(p_list),
            "score": np.concatenate(score_list),
        }

    def _all_points_arrays(
            self,
            points_xyz: np.ndarray,
            plane: Dict[str, np.ndarray],
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Produce per-raw-point arrays for ``"all_points"`` mode,
        reusing the ground plane so RANSAC runs only once."""
        cfg = self._config
        P = np.asarray(points_xyz, dtype=np.float64)
        Pa, R, t = self._align_ground(P, plane["normal"], plane["point"])

        ## XY ROI
        keep_xy = (np.abs(Pa[:, 0]) <= cfg.xy_max) & (np.abs(Pa[:, 1]) <= cfg.xy_max)
        Pa, P_world = Pa[keep_xy], P[keep_xy]

        if Pa.shape[0] == 0:
            return self._empty_all_points_tuple()

        ## z clipping
        z = Pa[:, 2]
        keep_z = (z >= cfg.z_min) & (z <= cfg.z_max)
        Pa, P_world, z = Pa[keep_z], P_world[keep_z], z[keep_z]

        if Pa.shape[0] == 0:
            return self._empty_all_points_tuple()

        ## height-bin assignment
        z_edges = np.linspace(cfg.z_min, cfg.z_max, cfg.N_h + 1)
        hb_arr = np.clip(
            np.searchsorted(z_edges[1:], z, side="right").astype(np.int16),
            0,
            cfg.N_h - 1,
        )

        return P_world, Pa, hb_arr

    ## ---- empty outputs for edge cases (no points, no keypoints, etc.) ----

    @staticmethod
    def _empty_keypoints() -> InLiER_Keypoints:
        return InLiER_Keypoints(
            p=np.zeros((0, 3), dtype=np.float64),
            T_ground=np.eye(4, dtype=np.float64),
        )

    @staticmethod
    def _empty_tokens() -> InLiER_Tokens:
        return InLiER_Tokens(
            token_id=np.zeros((0,), dtype=np.uint32),
        )

    @staticmethod
    def _empty_slice_dict() -> Dict[str, np.ndarray]:
        return {
            "p": np.zeros((0, 3), dtype=np.float64),
            "score": np.zeros((0,), dtype=np.float64),
        }

    @staticmethod
    def _empty_all_points_tuple():
        empty_h = np.zeros((0,), dtype=np.int16)
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            empty_h,
        )