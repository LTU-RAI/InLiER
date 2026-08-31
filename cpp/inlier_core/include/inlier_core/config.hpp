// Encoder configuration, mirroring inlier/core/Dataclasses.py InLiER_Config.
#pragma once

#include <cstdint>
#include <string>

namespace inlier {

/// Encoder configuration (Python: InLiER_Config).
struct InLiERConfig {
  // Height slicing
  int N_h = 10;
  double z_min = 0.0;
  double z_max = 20.0;

  // Token bins
  double r_max = 100.0;
  int N_r = 20;
  int N_a = 60;
  int N_s = 7;

  // BEV keypoint grid
  double cell_size = 1.0;
  double xy_max = 100.0;
  int window = 3;  // odd NMS window
  int max_kp_per_slice = 256;
  int max_kp_total = 1280;  // Python default: 256 * N_h / 2

  // Ground-plane RANSAC
  int ransac_iters = 250;
  double ransac_dist_thresh = 1.0;
  int ransac_min_inliers = 100;

  // "keypoints" | "all_points"
  std::string point_mode = "keypoints";

  // Shape PCA
  double shape_radius = 1.5;
  int shape_min_neighbors = 8;
};

/// Stage-1 MINT retrieval configuration (Python: ShortlistConfig).
struct ShortlistConfig {
  int topk = 100;
  double topk_pct = -1.0;  // < 0 means "None" (use topk)
  int min_shared_rows = 3;
  std::string mint_mode = "compact";            // "compact" | "full"
  std::string mint_scoring = "l1_intersection";  // "cosine" | "raw_intersection" | "l1_intersection"
  double eps = 1e-9;
};

/// Stage-2 BEAM configuration (Python: BEAMScoreConfig).
struct BEAMScoreConfig {
  int topk = 20;
  double topk_pct = -1.0;
  int min_shared_bins = 5;
  int min_shared_az_cols = 3;
  double score_threshold = 0.0;
};

/// Rerank-stage configuration (Python: RerankConfig).
struct RerankConfig {
  int topk = 1;
  double topk_pct = -1.0;
  std::string scoring_mode = "jaccard4d";  // "jaccard4d" | "cosine4d"
  int min_shared_rows = 3;
  int spatial_tol = 0;
  double score_threshold = 0.0;
  double eps = 1e-9;
};

/// Geometric-verification configuration (Python: VerifyConfig).
struct VerifyConfig {
  int topk = 1;
  double topk_pct = -1.0;
  int ransac_iters = 500;
  double inlier_dist_thresh = 3.0;
  int min_correspondences = 5;
  int min_ransac_inliers = 5;
  int min_keypoint_inliers = 3;
  int spatial_tol = 0;
  uint64_t seed = 0;
};

}  // namespace inlier
