// Data structures crossing the encoder/matcher API, mirroring the
// output dataclasses in inlier/core/Dataclasses.py.
#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <vector>

namespace inlier {

/// Ground plane (Python: the dict returned by InLiER._ransac_plane with
/// keys "normal", "d", "point", "inliers").
struct Plane {
  Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
  double d = 0.0;
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
  std::vector<uint8_t> inliers;  // per-input-point inlier mask
};

/// Keypoints (Python: InLiER_Keypoints). p is in the sensor/world frame;
/// T_ground is the 4x4 sensor -> ground-aligned rigid transform.
struct Keypoints {
  Eigen::Matrix<double, Eigen::Dynamic, 3> p;
  Eigen::Matrix4d T_ground = Eigen::Matrix4d::Identity();
};

/// Tokens (Python: InLiER_Tokens). Internally always uint64; the binding
/// layer downcasts to uint32 when N_h*N_r*N_s*N_a <= 2^32-1 to preserve
/// the public dtype contract.
struct Tokens {
  std::vector<uint64_t> token_id;
};

/// Stage-1 result (Python: ShortlistOutput).
struct ShortlistResult {
  std::vector<int64_t> ids;
  std::vector<double> scores;
};

/// Stage-2 result (Python: BEAMScoreOutput).
struct BeamResult {
  std::vector<int64_t> ids;
  std::vector<double> scores;
  std::vector<double> yaw_estimates;
  std::vector<int> best_shifts;
};

/// Rerank result (Python: RerankOutput).
struct RerankResult {
  std::vector<int64_t> ids;
  std::vector<double> scores;       // hist_score * inlier_ratio
  std::vector<double> hist_scores;
  std::vector<double> inlier_ratios;
  std::vector<int> inlier_counts;
  std::vector<double> yaw_estimates;
  std::vector<int> best_shifts;
};

/// Verification result (Python: VerifyOutput).
/// fail_stage / ransac_inliers_found are C++-side diagnostics for the
/// wrapper's verbose prints; the Python-visible failure output keeps
/// n_ransac_inliers = 0 exactly like the reference.
/// fail_stage: 0 = success, 1 = too few correspondences,
///             2 = too few RANSAC inliers, 3 = too few keypoint inliers.
struct VerifyResult {
  int fail_stage = 0;
  int ransac_inliers_found = 0;
  bool success = false;
  Eigen::Matrix4d T_sensor = Eigen::Matrix4d::Identity();
  double yaw = 0.0;
  double tx = 0.0;
  double ty = 0.0;
  double tz = 0.0;
  int n_correspondences = 0;
  int n_ransac_inliers = 0;
  int n_keypoint_inliers = 0;
  int n_total_keypoints = 0;
  double ransac_inlier_ratio = 0.0;
  double keypoint_inlier_ratio = 0.0;
  double inlier_rmse = 0.0;
};

}  // namespace inlier
