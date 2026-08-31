// Token-guided RANSAC geometric verification (Python: InLiER_Matcher.verify
// and helpers _find_correspondences / _ransac_2d_rigid / _refine_2d_rigid /
// _count_keypoint_inliers).
#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <vector>

#include "inlier_core/config.hpp"
#include "inlier_core/types.hpp"

namespace inlier {

/// Full verification of one query/DB candidate pair.
/// Keypoints are passed in their aligned frames together with T_ground;
/// T_sensor = T_ground_db^-1 * T_aligned * T_ground_query.
VerifyResult Verify(const std::vector<uint64_t> &query_token_id,
                    const Keypoints &query_keypoints,
                    const std::vector<uint64_t> &db_token_id,
                    const Keypoints &db_keypoints, int azimuth_shift,
                    const InLiERConfig &grid, const VerifyConfig &cfg);

}  // namespace inlier
