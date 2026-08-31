// Ground-plane estimation and alignment (Python: InLiER._ransac_plane,
// InLiER._align_ground, InLiER._rodrigues).
#pragma once

#include <Eigen/Core>

#include "inlier_core/types.hpp"

namespace inlier {

/// Axis-angle -> 3x3 rotation matrix (Rodrigues' formula).
Eigen::Matrix3d Rodrigues(const Eigen::Vector3d &axis, double angle);

/// RANSAC ground-plane estimation. Seeded mt19937_64(0) per call, PCA
/// fallback when RANSAC fails (statistically equivalent to the numpy
/// implementation, not bit-exact).
Plane RansacPlane(const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3>> &points,
                  int iters, double dist_thresh, int min_inliers);

/// Rigid transform aligning plane_normal -> +Z with the ground at z=0.
/// Returns T_ground (4x4, x' = R x + t).
Eigen::Matrix4d AlignGround(const Eigen::Vector3d &plane_normal,
                            const Eigen::Vector3d &plane_point);

}  // namespace inlier
