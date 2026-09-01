#include "inlier_core/plane.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <random>
#include <stdexcept>

namespace inlier {

Eigen::Matrix3d Rodrigues(const Eigen::Vector3d &axis, double angle) {
  const double n = axis.norm();
  if (n < 1e-12) {
    return Eigen::Matrix3d::Identity();
  }
  const Eigen::Vector3d a = axis / n;
  Eigen::Matrix3d K;
  K << 0.0, -a.z(), a.y(),  //
      a.z(), 0.0, -a.x(),   //
      -a.y(), a.x(), 0.0;
  return Eigen::Matrix3d::Identity() + std::sin(angle) * K +
         (1.0 - std::cos(angle)) * (K * K);
}

namespace {

/// Sample 3 distinct indices in [0, n) (statistical equivalent of
/// numpy rng.choice(n, size=3, replace=False)).
inline void Sample3Distinct(std::mt19937_64 &rng, int64_t n, int64_t idx[3]) {
  std::uniform_int_distribution<int64_t> dist(0, n - 1);
  idx[0] = dist(rng);
  do {
    idx[1] = dist(rng);
  } while (idx[1] == idx[0]);
  do {
    idx[2] = dist(rng);
  } while (idx[2] == idx[0] || idx[2] == idx[1]);
}

}  // namespace

Plane RansacPlane(
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3>> &points,
    int iters, double dist_thresh, int min_inliers) {
  const int64_t N = points.rows();
  if (N < 3) {
    throw std::invalid_argument("Need at least 3 points for plane estimation");
  }

  // Python: np.random.default_rng(0), re-seeded per call.
  std::mt19937_64 rng(0);

  int64_t best_cnt = -1;
  bool have_best = false;
  Eigen::Vector3d best_n = Eigen::Vector3d::Zero();
  double best_d = 0.0;
  std::vector<uint8_t> best_inl;

  std::vector<uint8_t> inl(static_cast<size_t>(N));
  for (int it = 0; it < iters; ++it) {
    int64_t idx[3];
    Sample3Distinct(rng, N, idx);
    const Eigen::Vector3d p1 = points.row(idx[0]);
    const Eigen::Vector3d p2 = points.row(idx[1]);
    const Eigen::Vector3d p3 = points.row(idx[2]);
    Eigen::Vector3d n = (p2 - p1).cross(p3 - p1);
    const double nn = n.norm();
    if (nn < 1e-9) {
      continue;
    }
    n /= nn;
    const double d = -n.dot(p1);

    int64_t cnt = 0;
    for (int64_t i = 0; i < N; ++i) {
      const bool is_inl =
          std::abs(points.row(i).dot(n) + d) < dist_thresh;
      inl[static_cast<size_t>(i)] = is_inl ? 1 : 0;
      cnt += is_inl;
    }
    if (cnt > best_cnt && cnt >= min_inliers) {
      best_cnt = cnt;
      best_n = n;
      best_d = d;
      best_inl = inl;
      have_best = true;
    }
  }

  if (!have_best) {
    // Fallback: PCA plane (smallest-eigenvalue direction of the covariance).
    const Eigen::Vector3d mu = points.colwise().mean();
    const Eigen::Matrix<double, Eigen::Dynamic, 3> centered =
        points.rowwise() - mu.transpose();
    const Eigen::Matrix3d cov =
        centered.transpose() * centered /
        static_cast<double>(std::max<int64_t>(N - 1, 1));
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    best_n = es.eigenvectors().col(0);  // ascending eigenvalues
    best_d = -best_n.dot(mu);
    best_inl.assign(static_cast<size_t>(N), 1);
    best_cnt = N;
  }

  // Ensure the normal is "up-ish".
  if (best_n.z() < 0.0) {
    best_n = -best_n;
    best_d = -best_d;
  }

  Plane plane;
  plane.normal = best_n;
  plane.d = best_d;
  plane.point = -best_d * best_n;
  plane.inliers = std::move(best_inl);
  return plane;
}

Eigen::Matrix4d AlignGround(const Eigen::Vector3d &plane_normal,
                            const Eigen::Vector3d &plane_point) {
  Eigen::Vector3d n = plane_normal / (plane_normal.norm() + 1e-12);

  const Eigen::Vector3d z_axis(0.0, 0.0, 1.0);
  double dot = n.dot(z_axis);
  dot = std::clamp(dot, -1.0, 1.0);

  Eigen::Matrix3d R;
  if (std::abs(dot - 1.0) < 1e-9) {
    R = Eigen::Matrix3d::Identity();
  } else if (std::abs(dot + 1.0) < 1e-9) {
    Eigen::Vector3d axis = n.cross(Eigen::Vector3d(1.0, 0.0, 0.0));
    if (axis.norm() < 1e-6) {
      axis = n.cross(Eigen::Vector3d(0.0, 1.0, 0.0));
    }
    R = Rodrigues(axis, M_PI);
  } else {
    const Eigen::Vector3d axis = n.cross(z_axis);
    const double angle = std::atan2(axis.norm(), dot);
    R = Rodrigues(axis, angle);
  }

  const Eigen::Vector3d p0r = R * plane_point;

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.topLeftCorner<3, 3>() = R;
  T(2, 3) = -p0r.z();  // t = [0, 0, -(R p0).z]
  return T;
}

}  // namespace inlier
