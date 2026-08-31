#include "inlier_core/verify.hpp"

#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>

#include "inlier_core/token.hpp"

namespace inlier {

namespace {

struct Correspondences {
  std::vector<int32_t> q_idx, db_idx;
};

/// InLiER_Matcher._find_correspondences: match on (hb, rb, ab) only —
/// shape class excluded. DB dict has first-index semantics; each query
/// token takes the FIRST neighbour found in (dh, dr, da) scan order.
Correspondences FindCorrespondences(
    const std::vector<int32_t> &q_hb, const std::vector<int32_t> &q_rb,
    const std::vector<int32_t> &q_ab, const std::vector<int32_t> &db_hb,
    const std::vector<int32_t> &db_rb, const std::vector<int32_t> &db_ab,
    int azimuth_shift, int tol, int Nh, int Nr, int Na) {
  std::unordered_map<int64_t, int32_t> db_dict;
  db_dict.reserve(db_hb.size() * 2);
  for (size_t j = 0; j < db_hb.size(); ++j) {
    const int ab = ((db_ab[j] + azimuth_shift) % Na + Na) % Na;
    db_dict.emplace(VerifyKey(db_hb[j], db_rb[j], ab, Nr, Na),
                    static_cast<int32_t>(j));  // first index wins
  }

  Correspondences out;
  for (size_t qi = 0; qi < q_hb.size(); ++qi) {
    if (tol == 0) {
      const auto it =
          db_dict.find(VerifyKey(q_hb[qi], q_rb[qi], q_ab[qi], Nr, Na));
      if (it != db_dict.end()) {
        out.q_idx.push_back(static_cast<int32_t>(qi));
        out.db_idx.push_back(it->second);
      }
      continue;
    }
    bool found = false;
    for (int dh = -tol; dh <= tol && !found; ++dh) {
      const int hb2 = q_hb[qi] + dh;
      if (hb2 < 0 || hb2 >= Nh) continue;
      for (int dr = -tol; dr <= tol && !found; ++dr) {
        const int rb2 = q_rb[qi] + dr;
        if (rb2 < 0 || rb2 >= Nr) continue;
        for (int da = -tol; da <= tol && !found; ++da) {
          const int ab2 = ((q_ab[qi] + da) % Na + Na) % Na;
          const auto it = db_dict.find(VerifyKey(hb2, rb2, ab2, Nr, Na));
          if (it != db_dict.end()) {
            out.q_idx.push_back(static_cast<int32_t>(qi));
            out.db_idx.push_back(it->second);
            found = true;
          }
        }
      }
    }
  }
  return out;
}

struct RansacResult {
  bool valid = false;
  std::vector<uint8_t> mask;
  int64_t count = 0;
  double yaw = 0.0, tx = 0.0, ty = 0.0;
};

/// InLiER_Matcher._ransac_2d_rigid: 2-point minimal samples, full 3-D
/// inlier test. mt19937_64(cfg.seed) — statistically equivalent to
/// numpy's default_rng(seed) stream, not bit-exact.
RansacResult Ransac2dRigid(
    const Eigen::Matrix<double, Eigen::Dynamic, 3> &q_pts,
    const Eigen::Matrix<double, Eigen::Dynamic, 3> &db_pts,
    const VerifyConfig &cfg) {
  RansacResult best;
  const int64_t N = q_pts.rows();
  if (N < 2) {
    return best;
  }
  std::mt19937_64 rng(cfg.seed);
  std::uniform_int_distribution<int64_t> dist(0, N - 1);
  const double thresh_sq = cfg.inlier_dist_thresh * cfg.inlier_dist_thresh;

  std::vector<uint8_t> mask(static_cast<size_t>(N));
  for (int it = 0; it < cfg.ransac_iters; ++it) {
    const int64_t i0 = dist(rng);
    int64_t i1;
    do {
      i1 = dist(rng);
    } while (i1 == i0);

    const Eigen::Vector2d p1_q = q_pts.row(i0).head<2>();
    const Eigen::Vector2d p2_q = q_pts.row(i1).head<2>();
    const Eigen::Vector2d p1_d = db_pts.row(i0).head<2>();
    const Eigen::Vector2d p2_d = db_pts.row(i1).head<2>();

    const Eigen::Vector2d dq = p2_q - p1_q;
    const Eigen::Vector2d dd = p2_d - p1_d;
    const double dq_norm = dq.norm();
    const double dd_norm = dd.norm();
    if (dq_norm < 1e-6 || dd_norm < 1e-6) {
      continue;
    }
    const double cos_a = dq.dot(dd) / (dq_norm * dd_norm);
    const double sin_a =
        (dq.x() * dd.y() - dq.y() * dd.x()) / (dq_norm * dd_norm);
    const double yaw = std::atan2(sin_a, cos_a);

    const double c = std::cos(yaw), s = std::sin(yaw);
    const double tx = p1_d.x() - (c * p1_q.x() - s * p1_q.y());
    const double ty = p1_d.y() - (s * p1_q.x() + c * p1_q.y());

    int64_t count = 0;
    for (int64_t i = 0; i < N; ++i) {
      const double rx = c * q_pts(i, 0) - s * q_pts(i, 1) + tx;
      const double ry = s * q_pts(i, 0) + c * q_pts(i, 1) + ty;
      const double dx = rx - db_pts(i, 0);
      const double dy = ry - db_pts(i, 1);
      const double dz = q_pts(i, 2) - db_pts(i, 2);
      const bool in = dx * dx + dy * dy + dz * dz < thresh_sq;
      mask[static_cast<size_t>(i)] = in;
      count += in;
    }
    if (count > best.count) {
      best.valid = true;
      best.count = count;
      best.mask = mask;
      best.yaw = yaw;
      best.tx = tx;
      best.ty = ty;
    }
  }
  return best;
}

/// InLiER_Matcher._refine_2d_rigid: Kabsch on the inlier centroids via
/// SVD with the numpy sign convention (np.sign(0) == 0 kept as-is).
void Refine2dRigid(const Eigen::Matrix<double, Eigen::Dynamic, 3> &q_pts,
                   const Eigen::Matrix<double, Eigen::Dynamic, 3> &db_pts,
                   double &yaw, double &tx, double &ty) {
  const Eigen::Vector2d q_c = q_pts.leftCols<2>().colwise().mean();
  const Eigen::Vector2d db_c = db_pts.leftCols<2>().colwise().mean();

  Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
  for (int64_t i = 0; i < q_pts.rows(); ++i) {
    const Eigen::Vector2d qc = q_pts.row(i).head<2>().transpose() - q_c;
    const Eigen::Vector2d dc = db_pts.row(i).head<2>().transpose() - db_c;
    H.noalias() += qc * dc.transpose();
  }

  Eigen::JacobiSVD<Eigen::Matrix2d> svd(
      H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Matrix2d U = svd.matrixU();
  const Eigen::Matrix2d V = svd.matrixV();
  const double d = (V * U.transpose()).determinant();
  const double sgn = (d > 0.0) ? 1.0 : (d < 0.0 ? -1.0 : 0.0);
  Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
  S(1, 1) = sgn;
  const Eigen::Matrix2d R = V * S * U.transpose();

  const Eigen::Vector2d t = db_c - R * q_c;
  yaw = std::atan2(R(1, 0), R(0, 0));
  tx = t.x();
  ty = t.y();
}

/// InLiER_Matcher._count_keypoint_inliers: brute-force NN after the
/// 4-DOF transform. Returns (n_inliers, rmse).
std::pair<int64_t, double> CountKeypointInliers(
    const Eigen::Matrix<double, Eigen::Dynamic, 3> &q_pts,
    const Eigen::Matrix<double, Eigen::Dynamic, 3> &db_pts, double yaw,
    double tx, double ty, double tz, double dist_thresh) {
  const int64_t Nq = q_pts.rows();
  const int64_t Ndb = db_pts.rows();
  if (Nq == 0 || Ndb == 0) {
    return {0, std::numeric_limits<double>::infinity()};
  }
  const double c = std::cos(yaw), s = std::sin(yaw);
  const double thresh_sq = dist_thresh * dist_thresh;

  int64_t n_inliers = 0;
  double sum_sq = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : n_inliers, sum_sq)
  for (int64_t i = 0; i < Nq; ++i) {
    const double qx = c * q_pts(i, 0) - s * q_pts(i, 1) + tx;
    const double qy = s * q_pts(i, 0) + c * q_pts(i, 1) + ty;
    const double qz = q_pts(i, 2) + tz;
    double min_sq = std::numeric_limits<double>::infinity();
    for (int64_t j = 0; j < Ndb; ++j) {
      const double dx = qx - db_pts(j, 0);
      const double dy = qy - db_pts(j, 1);
      const double dz = qz - db_pts(j, 2);
      min_sq = std::min(min_sq, dx * dx + dy * dy + dz * dz);
    }
    if (min_sq < thresh_sq) {
      n_inliers += 1;
      sum_sq += min_sq;
    }
  }
  if (n_inliers == 0) {
    return {0, std::numeric_limits<double>::infinity()};
  }
  return {n_inliers, std::sqrt(sum_sq / static_cast<double>(n_inliers))};
}

/// np.median (even-length averages the two middle values).
double Median(std::vector<double> v) {
  const size_t n = v.size();
  const size_t mid = n / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  const double hi = v[mid];
  if (n % 2 == 1) {
    return hi;
  }
  const double lo = *std::max_element(v.begin(), v.begin() + mid);
  return 0.5 * (lo + hi);
}

inline Eigen::Matrix<double, Eigen::Dynamic, 3> AlignedPoints(
    const Keypoints &kp) {
  const Eigen::Matrix3d R = kp.T_ground.topLeftCorner<3, 3>();
  const Eigen::Vector3d t = kp.T_ground.topRightCorner<3, 1>();
  Eigen::Matrix<double, Eigen::Dynamic, 3> out(kp.p.rows(), 3);
  for (int64_t i = 0; i < kp.p.rows(); ++i) {
    out.row(i) = (R * kp.p.row(i).transpose() + t).transpose();
  }
  return out;
}

}  // namespace

VerifyResult Verify(const std::vector<uint64_t> &query_token_id,
                    const Keypoints &query_keypoints,
                    const std::vector<uint64_t> &db_token_id,
                    const Keypoints &db_keypoints, int azimuth_shift,
                    const InLiERConfig &grid, const VerifyConfig &cfg) {
  const int Nh = grid.N_h, Nr = grid.N_r, Ns = std::max(1, grid.N_s),
            Na = std::max(1, grid.N_a);

  auto unpack = [&](const std::vector<uint64_t> &tids,
                    std::vector<int32_t> &hb, std::vector<int32_t> &rb,
                    std::vector<int32_t> &ab) {
    hb.reserve(tids.size());
    rb.reserve(tids.size());
    ab.reserve(tids.size());
    for (const uint64_t tid : tids) {
      const TokenBins b = UnpackToken(static_cast<int64_t>(tid), Nr, Ns, Na);
      hb.push_back(static_cast<int32_t>(b.hb));
      rb.push_back(static_cast<int32_t>(b.rb));
      ab.push_back(static_cast<int32_t>(b.ab));
    }
  };
  std::vector<int32_t> q_hb, q_rb, q_ab, db_hb, db_rb, db_ab;
  unpack(query_token_id, q_hb, q_rb, q_ab);
  unpack(db_token_id, db_hb, db_rb, db_ab);

  const Correspondences corr =
      FindCorrespondences(q_hb, q_rb, q_ab, db_hb, db_rb, db_ab,
                          azimuth_shift, cfg.spatial_tol, Nh, Nr, Na);
  const auto n_corr = static_cast<int>(corr.q_idx.size());

  // Pre-built failure output — every failure path returns exactly this
  // (n_ransac_inliers stays 0 even when RANSAC found some), as in Python.
  VerifyResult fail;
  fail.n_correspondences = n_corr;
  fail.n_total_keypoints = static_cast<int>(query_keypoints.p.rows());
  fail.inlier_rmse = std::numeric_limits<double>::infinity();

  if (n_corr < std::max(2, cfg.min_correspondences)) {
    fail.fail_stage = 1;
    return fail;
  }

  const auto q_aligned = AlignedPoints(query_keypoints);
  const auto db_aligned = AlignedPoints(db_keypoints);

  Eigen::Matrix<double, Eigen::Dynamic, 3> q_pts(n_corr, 3), db_pts(n_corr, 3);
  for (int i = 0; i < n_corr; ++i) {
    q_pts.row(i) = q_aligned.row(corr.q_idx[static_cast<size_t>(i)]);
    db_pts.row(i) = db_aligned.row(corr.db_idx[static_cast<size_t>(i)]);
  }

  const RansacResult ransac = Ransac2dRigid(q_pts, db_pts, cfg);
  const auto n_ransac_inliers = static_cast<int>(ransac.count);
  if (!ransac.valid || n_ransac_inliers < cfg.min_ransac_inliers) {
    fail.fail_stage = 2;
    fail.ransac_inliers_found = n_ransac_inliers;
    return fail;
  }

  // SVD refinement on the inliers; tz from the median height residual.
  Eigen::Matrix<double, Eigen::Dynamic, 3> inl_q(n_ransac_inliers, 3),
      inl_db(n_ransac_inliers, 3);
  std::vector<double> dz;
  dz.reserve(static_cast<size_t>(n_ransac_inliers));
  for (int i = 0, w = 0; i < n_corr; ++i) {
    if (ransac.mask[static_cast<size_t>(i)]) {
      inl_q.row(w) = q_pts.row(i);
      inl_db.row(w) = db_pts.row(i);
      dz.push_back(db_pts(i, 2) - q_pts(i, 2));
      ++w;
    }
  }
  double yaw, tx, ty;
  Refine2dRigid(inl_q, inl_db, yaw, tx, ty);
  const double tz = Median(std::move(dz));

  const auto [n_kp_inliers, rmse] = CountKeypointInliers(
      q_aligned, db_aligned, yaw, tx, ty, tz, cfg.inlier_dist_thresh);
  if (n_kp_inliers < cfg.min_keypoint_inliers) {
    fail.fail_stage = 3;
    fail.ransac_inliers_found = n_ransac_inliers;
    return fail;
  }

  // T_sensor = T_ground_db^-1 * T_aligned * T_ground_query
  Eigen::Matrix4d T_aligned = Eigen::Matrix4d::Identity();
  const double c = std::cos(yaw), s = std::sin(yaw);
  T_aligned(0, 0) = c;
  T_aligned(0, 1) = -s;
  T_aligned(0, 3) = tx;
  T_aligned(1, 0) = s;
  T_aligned(1, 1) = c;
  T_aligned(1, 3) = ty;
  T_aligned(2, 3) = tz;

  Eigen::Matrix4d T_db_inv = Eigen::Matrix4d::Identity();
  const Eigen::Matrix3d R_db = db_keypoints.T_ground.topLeftCorner<3, 3>();
  const Eigen::Vector3d t_db = db_keypoints.T_ground.topRightCorner<3, 1>();
  T_db_inv.topLeftCorner<3, 3>() = R_db.transpose();
  T_db_inv.topRightCorner<3, 1>() = -(R_db.transpose() * t_db);

  VerifyResult out;
  out.success = true;
  out.ransac_inliers_found = n_ransac_inliers;
  out.T_sensor = T_db_inv * T_aligned * query_keypoints.T_ground;
  out.yaw = yaw;
  out.tx = tx;
  out.ty = ty;
  out.tz = tz;
  out.n_correspondences = n_corr;
  out.n_ransac_inliers = n_ransac_inliers;
  out.n_keypoint_inliers = static_cast<int>(n_kp_inliers);
  out.n_total_keypoints = static_cast<int>(query_keypoints.p.rows());
  out.ransac_inlier_ratio =
      static_cast<double>(n_ransac_inliers) / std::max(n_corr, 1);
  out.keypoint_inlier_ratio =
      static_cast<double>(n_kp_inliers) /
      std::max<int64_t>(query_keypoints.p.rows(), 1);
  out.inlier_rmse = rmse;
  return out;
}

}  // namespace inlier
