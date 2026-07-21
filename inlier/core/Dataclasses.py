"""Dataclasses used by InLiER."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class InLiER_Config:

    ## Height Slicing 
    ## Height resolution: dz = (z_max - z_min) / N_h (metres per slice)  
    N_h: int = 10                       # number of height slices
    z_min: float = 0.0                  # metres above ground (lower slice edge)
    z_max: float = 20.0                 # metres above ground (upper slice edge)

    ## Token encoding — token_id = ((hb * N_r + rb) * N_s + sb) * N_a + ab
    ## Radial resolution: dr = r_max / N_r (metres per bin), with r_max <=  xy_max
    r_max: float = 100.0                # max radial distance (metres)
    N_r: int = 20                       # radial bins
    ## Azimuth resolution: da = 360° / N_a (degrees per bin)
    N_a: int = 60                       # azimuth bins
    ## Plane and linear get subdivision of the shape space, scatter remains the same 
    N_s: int = 7                        # shape classes (1,3,5,7), should be odd number

    ## Keypoint extraction
    cell_size: float = 1.0              # metres per BEV cell (recommended: 2.0 * voxel_size)
    xy_max: float = 100.0               # metres of ROI in X and Y (half-extent)
    window: int = 3                     # local-maxima NMS window (odd)
    max_kp_per_slice: int = 256          
    max_kp_total: int = 256 * N_h // 2    # integer division

    ## Ground alignment
    ransac_iters: int = 250
    ransac_dist_thresh: float = 1.0     # metres; should be tuned according to your voxel size
    ransac_min_inliers: int = 200

    ## Point selection mode:
    #   "keypoints"  — local-maxima NMS only
    #   "all_points" — every occupied BEV cell above threshold
    point_mode: str = "keypoints"

    ## Shape classification (PCA on local neighbourhood)
    shape_radius: float = 1.5           # metres of neighbourhood radius
    shape_min_neighbors: int = 8


@dataclass
class InLiER_Keypoints:
    """Minimal keypoint representation for communication / matching.

    Only the 3-D coordinates and the ground-alignment transform are
    stored.  Everything else (``slice_id``, ``cell``, ``score``, shape
    info) is encoder-internal and can be recovered via ``InLiER``
    utility methods when needed.

    ``T_ground`` is the 4×4 rigid transform from sensor frame to the
    ground-aligned frame: ``p_aligned = (T_ground @ [p; 1])[:3]``.
    """

    p: np.ndarray           # (K, 3) sensor / world frame
    T_ground: np.ndarray    # (4, 4) sensor → ground-aligned rigid transform

    @property
    def p_aligned(self) -> np.ndarray:
        """(K, 3) keypoints in the ground-aligned frame, derived on the fly."""
        R = self.T_ground[:3, :3]
        t = self.T_ground[:3, 3]
        return (R @ self.p.T).T + t

    @property
    def T_ground_inv(self) -> np.ndarray:
        """(4, 4) ground-aligned → sensor frame (inverse of T_ground)."""
        T_inv = np.eye(4, dtype=self.T_ground.dtype)
        R = self.T_ground[:3, :3]
        t = self.T_ground[:3, 3]
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -(R.T @ t)
        return T_inv

    def __len__(self) -> int:
        return int(self.p.shape[0])


@dataclass
class InLiER_Tokens:
    """Minimal per-keypoint token representation for communication.

    A single ``token_id`` array is the only field.  The token_id is a
    mixed-radix code fully invertible given (N_r, N_s, N_a) from
    ``InLiER_Config``:

        token_id = ((hb × N_r + rb) × N_s + sb) × N_a + ab

    dtype is uint32 when N_h*N_r*N_s*N_a ≤ 2³²−1 (typical), uint64
    otherwise.  All derived quantities (hb, rb, sb, ab) are recovered
    by ``InLiER.unpack_token_ids()`` at the receiver side.
    """

    token_id: np.ndarray       # (K,) uint32 or uint64

    def __len__(self) -> int:
        return int(self.token_id.shape[0])


@dataclass
class InLiER_Output:
    """Optional container returning both keypoints and tokens."""
    keypoints: InLiER_Keypoints
    tokens:    InLiER_Tokens


@dataclass
class ShortlistConfig:
    """Config for shortlist stage (rotation-invariant MINT retrieval).

    Reconstructs hist_matrix (Nh, Nr*Ns) from token_ids at finalize()
    time.  At query time applies pairwise height-ceiling pruning and
    L1-normalised Minimum-ceiling INTersection (MINT) histogram scoring.
    """

    topk:            int             = 100
    topk_pct:        Optional[float] = None   # overrides topk if set
    min_shared_rows: int             = 3
    mint_mode:       str             = "compact"  # "compact" (collapse height) | "full" (preserve height)
    mint_scoring:    str             = "l1_intersection"   # "cosine" | "raw_intersection" | "l1_intersection"

    eps: float = 1e-9


@dataclass
class ShortlistOutput:
    """Return type for shortlist stage."""
    ids:    List[int]
    scores: List[float]


@dataclass
class BEAMScoreConfig:
    """Config for beam_score stage.

    Compresses the 3-D (N_h, N_r, N_a) occupancy into a compact 2-D
    (N_r, N_a) matrix of uint16 binary elevation codes.  Scoring is
    bit-level Jaccard with circular azimuth shift search.
    """

    topk:               int             = 20
    topk_pct:           Optional[float] = None
    min_shared_bins:    int             = 5
    min_shared_az_cols: int             = 3
    score_threshold:    float           = 0.0


@dataclass
class BEAMScoreOutput:
    """Return type for beam_score stage.

    All lists aligned by index, sorted descending by scores.
    """
    ids:           List[int]
    scores:        List[float]       # Jaccard in [0, 1]
    yaw_estimates: List[float]       # radians
    best_shifts:   List[int]         # azimuth bin shift per candidate


@dataclass
class RerankConfig:
    """Config for rerank stage (full 4-D histogram scoring).

    Operates on the full (N_h, N_r, N_s, N_a) token histogram for
    maximum discriminative power on a small candidate set.

    scoring_mode = "jaccard4d"
        Boolean Jaccard on the full 4-D occupancy tensor.

    scoring_mode = "cosine4d"
        Column-wise cosine similarity (Scan Context style).
    """

    topk:            int   = 1
    topk_pct:        Optional[float] = None
    scoring_mode:    str   = "jaccard4d"   # "jaccard4d" | "cosine4d"
    min_shared_rows: int   = 3
    spatial_tol:     int   = 0             # ±bins for token inlier matching (0 = exact)
    score_threshold: float = 0.0

    eps: float = 1e-9


@dataclass
class RerankOutput:
    """Return type for rerank stage.

    ``scores`` is the product: ``hist_scores × inlier_ratios``.
    All lists aligned by index, sorted descending by scores.
    """
    ids:            List[int]
    scores:         List[float]       # combined product score
    hist_scores:    List[float]       # 4-D histogram component
    inlier_ratios:  List[float]       # token inlier component
    inlier_counts:  List[int]         # absolute matched tokens
    yaw_estimates:  List[float]
    best_shifts:    List[int]


@dataclass
class VerifyConfig:
    """Config for geometric verification / pose estimation.

    Uses token correspondences to establish 3-D point pairs, then RANSAC
    to estimate a 2-D rigid transform (yaw + tx, ty) in the
    ground-aligned frame.  tz is estimated from median inlier height
    residual.  The final output ``T_sensor`` is converted back to the
    original sensor frame via the per-scan ``T_ground`` transforms.
    """

    topk:                 int   = 1
    topk_pct:             Optional[float] = None
    ransac_iters:         int   = 500
    inlier_dist_thresh:   float = 3.0     # metres
    min_correspondences:  int   = 32       # early-exit floor on token correspondences
    min_ransac_inliers:   int   = 16       # floor on RANSAC inlier count
    min_keypoint_inliers: int   = 8       # floor on keypoint inliers after pose refinement
    spatial_tol:          int   = 0       # ±bins for token matching (0 = exact)
    seed:                 int   = 0


@dataclass
class VerifyOutput:
    """Return type for geometric verification.

    ``T_sensor`` is the 4×4 SE(3) transform in the **original sensor
    frame** that maps query points into the DB frame:

        p_db = T_sensor @ p_query   (homogeneous)

    It is derived from the ground-aligned 4-DOF estimate
    (yaw, tx, ty, tz) via:

        T_sensor = T_ground_db⁻¹ @ T_aligned @ T_ground_query

    The legacy scalar fields (yaw, tx, ty, tz) are retained for the
    ground-aligned frame for backwards compatibility and diagnostics.
    """
    success:              bool
    T_sensor:             np.ndarray    # (4, 4) SE(3) query→DB in sensor frame
    yaw:                  float         # ground-aligned yaw (rad)
    tx:                   float         # ground-aligned tx (m)
    ty:                   float         # ground-aligned ty (m)
    tz:                   float         # ground-aligned tz (m)
    n_correspondences:    int           # token-matched putative pairs
    n_ransac_inliers:     int           # RANSAC inlier count
    n_keypoint_inliers:   int           # keypoint inliers after refined pose
    n_total_keypoints:    int           # total query keypoints used for ratio
    ransac_inlier_ratio:  float         # n_ransac_inliers / n_correspondences
    keypoint_inlier_ratio: float        # n_keypoint_inliers / n_total_keypoints
    inlier_rmse:          float


@dataclass
class GICPRefineConfig:
    """Config for GICP-based 6-DOF pose refinement.

    Seeds ``small_gicp.align`` with the initial pose from the verify
    stage and returns the refined SE(3) transform in the original
    sensor frame.

    Parameters
    ----------
    registration_type : str
        ``'GICP'`` (default, recommended), ``'VGICP'``, ``'ICP'``, or ``'PLANE_ICP'``.
    max_correspondence_distance : float
        Maximum distance (m) for point-pair correspondences.
    downsampling_resolution : float
        Voxel size (m) for internal downsampling.  Set ≤0 to skip.
    voxel_resolution : float
        Voxel resolution for VGICP only.
    num_threads : int
        Parallelism for small_gicp.
    use_raw_clouds : bool
        If *True* the user must supply the full dense point clouds.
        If *False* the keypoint clouds (``InLiER_Keypoints.p``) are
        used instead.
    max_iterations : int
        Maximum number of ICP/GICP iterations.
    """
    registration_type:            str   = "GICP"
    max_correspondence_distance:  float = 5.0
    downsampling_resolution:      float = 0.5
    voxel_resolution:             float = 1.0
    num_threads:                  int   = 4
    use_raw_clouds:               bool  = True
    max_iterations:               int   = 64


@dataclass
class GICPRefineOutput:
    """Return type for GICP refinement stage.

    ``T_sensor`` is the refined 6-DOF SE(3) transform in the
    **original sensor frame** mapping query→DB:

        p_db = T_sensor @ p_query   (homogeneous)
    """
    success:       bool
    T_sensor:      np.ndarray      # (4, 4) refined SE(3) in sensor frame
    converged:     bool             # small_gicp convergence flag
    n_iterations:  int              # small_gicp iterations used
    final_error:   float            # small_gicp final residual
    n_inliers:     int              # small_gicp inlier count