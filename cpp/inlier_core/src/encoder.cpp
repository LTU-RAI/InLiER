#include "inlier_core/encoder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "inlier_core/plane.hpp"
#include "inlier_core/shape_pca.hpp"
#include "inlier_core/token.hpp"

namespace inlier {

namespace {

struct SliceAccum {
  // Dense per-cell buffers of size W*H, reset via the touched list.
  std::vector<int32_t> count;
  std::vector<double> sum_x, sum_y, sum_z;
  std::vector<double> min_z, max_z;
  std::vector<double> intensity;
  std::vector<int64_t> touched;

  explicit SliceAccum(size_t n)
      : count(n, 0),
        sum_x(n, 0.0),
        sum_y(n, 0.0),
        sum_z(n, 0.0),
        min_z(n, std::numeric_limits<double>::infinity()),
        max_z(n, -std::numeric_limits<double>::infinity()),
        intensity(n, 0.0) {}

  void Reset() {
    for (const int64_t f : touched) {
      const auto i = static_cast<size_t>(f);
      count[i] = 0;
      sum_x[i] = sum_y[i] = sum_z[i] = 0.0;
      min_z[i] = std::numeric_limits<double>::infinity();
      max_z[i] = -std::numeric_limits<double>::infinity();
      intensity[i] = 0.0;
    }
    touched.clear();
  }
};

/// v is a local maximum iff no cell in the odd `window` neighbourhood
/// exceeds it. Matches InLiER._local_maxima_2d: the pad value there is
/// the image minimum, which never exceeds v, so border clamping is
/// equivalent.
inline bool IsLocalMax(const std::vector<double> &img, int W, int H, int iy,
                       int ix, int window) {
  const int pad = window / 2;
  const double v = img[static_cast<size_t>(iy) * W + ix];
  const int y0 = std::max(0, iy - pad), y1 = std::min(H - 1, iy + pad);
  const int x0 = std::max(0, ix - pad), x1 = std::min(W - 1, ix + pad);
  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      if (img[static_cast<size_t>(y) * W + x] > v) {
        return false;
      }
    }
  }
  return true;
}

struct SliceResult {
  // Keypoints in the *sensor* frame, plus their scores (slice order).
  std::vector<Eigen::Vector3d> p;
  std::vector<double> score;
};

}  // namespace

Encoder::Encoder(InLiERConfig config) : config_(std::move(config)) {
  if (config_.window % 2 == 0) {
    throw std::invalid_argument("window must be odd");
  }
  if (config_.point_mode != "keypoints" && config_.point_mode != "all_points") {
    throw std::invalid_argument(
        "point_mode must be \"keypoints\" | \"all_points\"");
  }
  if (config_.N_h < 1 || config_.N_h > 64) {
    throw std::invalid_argument("N_h must be in [1, 64]");
  }
}

Keypoints Encoder::ExtractKeypoints(
    const Eigen::Ref<const PointMatrix> &points,
    const std::optional<Plane> &plane_in) const {
  const auto &cfg = config_;

  Keypoints empty;
  empty.p.resize(0, 3);
  if (points.rows() == 0) {
    return empty;
  }

  // Ground plane + alignment.
  const Plane plane =
      plane_in.has_value()
          ? *plane_in
          : RansacPlane(points, cfg.ransac_iters, cfg.ransac_dist_thresh,
                        cfg.ransac_min_inliers);
  const Eigen::Matrix4d T_ground = AlignGround(plane.normal, plane.point);
  const Eigen::Matrix3d R = T_ground.topLeftCorner<3, 3>();
  const Eigen::Vector3d t = T_ground.topRightCorner<3, 1>();
  const Eigen::Matrix3d Rinv = R.transpose();
  const Eigen::Vector3d tinv = -(R.transpose() * t);

  // XY ROI + z filter on aligned points, bucketed by height slice.
  // Slice s covers [z_edges[s], z_edges[s+1]) — a point exactly at
  // z_max passes the keep_z filter but falls in NO slice (Python
  // behaviour).
  const std::vector<double> z_edges =
      MakeZEdges(cfg.z_min, cfg.z_max, cfg.N_h);
  const int W = static_cast<int>(std::ceil(2.0 * cfg.xy_max / cfg.cell_size));
  const int H = W;
  const double x_min = -cfg.xy_max;
  const double y_min = -cfg.xy_max;

  std::vector<std::vector<Eigen::Vector3d>> slice_pts(
      static_cast<size_t>(cfg.N_h));
  bool any_kept = false;
  for (int64_t i = 0; i < points.rows(); ++i) {
    const Eigen::Vector3d pa = R * points.row(i).transpose() + t;
    if (std::abs(pa.x()) > cfg.xy_max || std::abs(pa.y()) > cfg.xy_max) {
      continue;
    }
    if (pa.z() < cfg.z_min || pa.z() > cfg.z_max) {
      continue;
    }
    any_kept = true;
    // slice index: edges[s] <= z < edges[s+1]
    const auto it = std::upper_bound(z_edges.begin(), z_edges.end(), pa.z());
    const int s = static_cast<int>(it - z_edges.begin()) - 1;
    if (s < 0 || s >= cfg.N_h) {
      continue;  // z == z_max
    }
    slice_pts[static_cast<size_t>(s)].push_back(pa);
  }
  if (!any_kept) {
    // Python _empty_keypoints(): T_ground stays identity here.
    return empty;
  }

  const bool keypoint_mode = (cfg.point_mode == "keypoints");
  const double thr = 0.5 * (cfg.z_max - cfg.z_min) / cfg.N_h;

  SliceAccum acc(static_cast<size_t>(W) * H);
  SliceResult out;

  for (int s = 0; s < cfg.N_h; ++s) {
    const auto &pts = slice_pts[static_cast<size_t>(s)];
    if (pts.empty()) {
      continue;
    }
    acc.Reset();

    for (const auto &p : pts) {
      const auto ix = static_cast<int64_t>(
          std::floor((p.x() - x_min) / cfg.cell_size));
      const auto iy = static_cast<int64_t>(
          std::floor((p.y() - y_min) / cfg.cell_size));
      if (ix < 0 || ix >= W || iy < 0 || iy >= H) {
        continue;
      }
      const int64_t flat = iy * W + ix;
      const auto f = static_cast<size_t>(flat);
      if (acc.count[f] == 0) {
        acc.touched.push_back(flat);
      }
      acc.count[f] += 1;
      acc.sum_x[f] += p.x();
      acc.sum_y[f] += p.y();
      acc.sum_z[f] += p.z();
      acc.min_z[f] = std::min(acc.min_z[f], p.z());
      acc.max_z[f] = std::max(acc.max_z[f], p.z());
    }
    if (acc.touched.empty()) {
      continue;
    }
    for (const int64_t f : acc.touched) {
      const auto i = static_cast<size_t>(f);
      acc.intensity[i] = acc.max_z[i] - acc.min_z[i];
    }

    // Candidate cells in row-major (iy, ix) order == np.argwhere order.
    // Only occupied cells can reach intensity >= thr when thr > 0; fall
    // back to a full scan for the degenerate thr <= 0 case.
    std::vector<int64_t> cand_flat;
    std::vector<double> cand_score;
    std::sort(acc.touched.begin(), acc.touched.end());
    auto consider = [&](int64_t flat) {
      const auto i = static_cast<size_t>(flat);
      const double v = acc.intensity[i];
      if (v < thr) {
        return;
      }
      if (keypoint_mode) {
        const int iy = static_cast<int>(flat / W);
        const int ix = static_cast<int>(flat % W);
        if (!IsLocalMax(acc.intensity, W, H, iy, ix, cfg.window)) {
          return;
        }
      }
      cand_flat.push_back(flat);
      cand_score.push_back(v);
    };
    if (thr > 0.0) {
      for (const int64_t f : acc.touched) {
        consider(f);
      }
    } else {
      for (int64_t f = 0; f < static_cast<int64_t>(W) * H; ++f) {
        consider(f);
      }
    }
    if (cand_flat.empty()) {
      continue;
    }

    // Per-slice cap: np.argsort(scores)[-max_kp_per_slice:] — stable
    // ascending sort, keep the last k, PRESERVING ascending order.
    std::vector<size_t> order(cand_flat.size());
    std::iota(order.begin(), order.end(), size_t{0});
    if (keypoint_mode &&
        cand_flat.size() > static_cast<size_t>(cfg.max_kp_per_slice)) {
      std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return cand_score[a] < cand_score[b];
      });
      order.erase(order.begin(),
                  order.end() - static_cast<size_t>(cfg.max_kp_per_slice));
    }

    for (const size_t k : order) {
      const auto f = static_cast<size_t>(cand_flat[k]);
      if (acc.count[f] <= 0) {
        continue;  // occupancy filter (mirrors the Python `occupied` mask)
      }
      const double cnt = std::max(acc.count[f], 1);
      const Eigen::Vector3d mean(acc.sum_x[f] / cnt, acc.sum_y[f] / cnt,
                                 acc.sum_z[f] / cnt);
      out.p.push_back(Rinv * mean + tinv);
      out.score.push_back(cand_score[k]);
    }
  }

  // Global cap: np.argsort(score)[::-1][:max_kp_total] — reversed stable
  // ascending == descending score with ties in DESCENDING original index.
  const auto n_kp = static_cast<int64_t>(out.p.size());
  std::vector<size_t> idx(static_cast<size_t>(n_kp));
  std::iota(idx.begin(), idx.end(), size_t{0});
  if (n_kp > cfg.max_kp_total) {
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
      return out.score[a] < out.score[b];
    });
    std::reverse(idx.begin(), idx.end());
    idx.resize(static_cast<size_t>(cfg.max_kp_total));
  }

  Keypoints kp;
  kp.T_ground = T_ground;
  kp.p.resize(static_cast<int64_t>(idx.size()), 3);
  for (size_t i = 0; i < idx.size(); ++i) {
    kp.p.row(static_cast<int64_t>(i)) = out.p[idx[i]].transpose();
  }
  return kp;
}

Tokens Encoder::Tokenize(const Eigen::Ref<const PointMatrix> &points,
                         const Keypoints &keypoints,
                         const std::optional<Plane> &plane_in,
                         TokenizeMode mode) const {
  const auto &cfg = config_;

  std::string pmode = cfg.point_mode;
  if (mode == TokenizeMode::kKeypoints) {
    pmode = "keypoints";
  } else if (mode == TokenizeMode::kAllPoints) {
    pmode = "all_points";
  }

  const int Nh = cfg.N_h;
  const int Nr = cfg.N_r;
  const int Ns = std::max(1, cfg.N_s);
  const int Na = std::max(1, cfg.N_a);
  const std::vector<double> z_edges = MakeZEdges(cfg.z_min, cfg.z_max, Nh);

  // Choose the point set: world coords (for PCA queries), aligned coords
  // (for binning), and height bins.
  Eigen::Matrix<double, Eigen::Dynamic, 3> pts_world, pts_aligned;
  std::vector<int16_t> hb;

  if (pmode == "keypoints") {
    const Eigen::Matrix3d R = keypoints.T_ground.topLeftCorner<3, 3>();
    const Eigen::Vector3d t = keypoints.T_ground.topRightCorner<3, 1>();
    pts_world = keypoints.p;
    pts_aligned.resize(keypoints.p.rows(), 3);
    hb.resize(static_cast<size_t>(keypoints.p.rows()));
    for (int64_t i = 0; i < keypoints.p.rows(); ++i) {
      const Eigen::Vector3d pa = R * keypoints.p.row(i).transpose() + t;
      pts_aligned.row(i) = pa.transpose();
      const double z = std::clamp(pa.z(), cfg.z_min, cfg.z_max);
      hb[static_cast<size_t>(i)] =
          static_cast<int16_t>(BinHeight(z, z_edges));
    }
  } else if (pmode == "all_points") {
    const Plane plane =
        plane_in.has_value()
            ? *plane_in
            : RansacPlane(points, cfg.ransac_iters, cfg.ransac_dist_thresh,
                          cfg.ransac_min_inliers);
    const Eigen::Matrix4d T = AlignGround(plane.normal, plane.point);
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d t = T.topRightCorner<3, 1>();

    std::vector<int64_t> keep;
    std::vector<Eigen::Vector3d> aligned;
    keep.reserve(static_cast<size_t>(points.rows()));
    aligned.reserve(static_cast<size_t>(points.rows()));
    for (int64_t i = 0; i < points.rows(); ++i) {
      const Eigen::Vector3d pa = R * points.row(i).transpose() + t;
      if (std::abs(pa.x()) > cfg.xy_max || std::abs(pa.y()) > cfg.xy_max) {
        continue;
      }
      if (pa.z() < cfg.z_min || pa.z() > cfg.z_max) {
        continue;
      }
      keep.push_back(i);
      aligned.push_back(pa);
    }
    pts_world.resize(static_cast<int64_t>(keep.size()), 3);
    pts_aligned.resize(static_cast<int64_t>(keep.size()), 3);
    hb.resize(keep.size());
    for (size_t k = 0; k < keep.size(); ++k) {
      pts_world.row(static_cast<int64_t>(k)) = points.row(keep[k]);
      pts_aligned.row(static_cast<int64_t>(k)) = aligned[k].transpose();
      hb[k] = static_cast<int16_t>(BinHeight(aligned[k].z(), z_edges));
    }
  } else {
    throw std::invalid_argument(
        "point_mode must be \"keypoints\" | \"all_points\"");
  }

  const int64_t K = pts_aligned.rows();
  Tokens tokens;
  if (K == 0) {
    return tokens;
  }

  // Coordinate bins. Python rounds r through float32 before binning.
  const double r_max = std::min(cfg.r_max, cfg.xy_max);
  std::vector<int> rb(static_cast<size_t>(K)), ab(static_cast<size_t>(K));
  for (int64_t i = 0; i < K; ++i) {
    const double xa = pts_aligned(i, 0);
    const double ya = pts_aligned(i, 1);
    const auto r32 = static_cast<float>(std::sqrt(xa * xa + ya * ya));
    rb[static_cast<size_t>(i)] =
        BinRadial(static_cast<double>(r32), r_max, Nr);
    ab[static_cast<size_t>(i)] = BinAzimuth(std::atan2(ya, xa), Na);
  }

  // Shape class (PCA neighbourhood in the raw cloud).
  std::vector<int16_t> sb(static_cast<size_t>(K), 0);
  if (Ns > 1) {
    const ShapePcaResult pca =
        ComputeShapePca(points, pts_world, cfg.shape_radius,
                        cfg.shape_min_neighbors, Ns);
    for (int64_t i = 0; i < K; ++i) {
      sb[static_cast<size_t>(i)] = std::clamp(
          pca.shape_class[static_cast<size_t>(i)], int16_t{0},
          static_cast<int16_t>(Ns - 1));
    }
  }

  tokens.token_id.resize(static_cast<size_t>(K));
  for (int64_t i = 0; i < K; ++i) {
    const auto s = static_cast<size_t>(i);
    tokens.token_id[s] =
        PackToken(static_cast<uint64_t>(hb[s]), static_cast<uint64_t>(rb[s]),
                  static_cast<uint64_t>(sb[s]), static_cast<uint64_t>(ab[s]),
                  Nr, Ns, Na);
  }
  return tokens;
}

std::pair<Keypoints, Tokens> Encoder::Encode(
    const Eigen::Ref<const PointMatrix> &points,
    const std::optional<Plane> &plane_in) const {
  const auto &cfg = config_;
  const Plane plane =
      plane_in.has_value()
          ? *plane_in
          : RansacPlane(points, cfg.ransac_iters, cfg.ransac_dist_thresh,
                        cfg.ransac_min_inliers);
  Keypoints kp = ExtractKeypoints(points, plane);
  Tokens tokens = Tokenize(points, kp, plane, TokenizeMode::kFromConfig);
  return {std::move(kp), std::move(tokens)};
}

}  // namespace inlier
