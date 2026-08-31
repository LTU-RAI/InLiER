// Private pybind11 backend for the `inlier` package.
//
// The public API lives in the Python wrappers (inlier/core/InLiER.py,
// inlier/core/InLiER_Matcher.py); this module only exposes the raw C++ core.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <string>

#include "inlier_core/config.hpp"
#include "inlier_core/encoder.hpp"
#include "inlier_core/matcher.hpp"
#include "inlier_core/plane.hpp"
#include "inlier_core/shape_pca.hpp"
#include "inlier_core/token.hpp"
#include "inlier_core/types.hpp"
#include "inlier_core/verify.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace {

template <typename T>
using Arr = py::array_t<T, py::array::c_style | py::array::forcecast>;

/// pack_token_ids(hb, rb, sb, ab, N_r, N_s, N_a) -> uint32 or uint64
/// array. dtype follows the Python rule: uint64 when the mixed-radix
/// product N_h*N_r*N_s*N_a exceeds 2^32-1 (N_h passed for that check).
py::array PackTokenIds(const Arr<int64_t> &hb, const Arr<int64_t> &rb,
                       const Arr<int64_t> &sb, const Arr<int64_t> &ab,
                       int64_t N_h, int64_t N_r, int64_t N_s, int64_t N_a) {
  const py::ssize_t n = hb.size();
  if (rb.size() != n || sb.size() != n || ab.size() != n) {
    throw py::value_error("hb, rb, sb, ab must have the same length");
  }
  const auto *hb_p = hb.data();
  const auto *rb_p = rb.data();
  const auto *sb_p = sb.data();
  const auto *ab_p = ab.data();

  const bool wide =
      static_cast<uint64_t>(N_h) * N_r * N_s * N_a > UINT64_C(0xFFFFFFFF);
  if (wide) {
    py::array_t<uint64_t> out(n);
    auto *o = out.mutable_data();
    for (py::ssize_t i = 0; i < n; ++i) {
      o[i] = inlier::PackToken(hb_p[i], rb_p[i], sb_p[i], ab_p[i], N_r, N_s,
                               N_a);
    }
    return out;
  }
  py::array_t<uint32_t> out(n);
  auto *o = out.mutable_data();
  for (py::ssize_t i = 0; i < n; ++i) {
    o[i] = static_cast<uint32_t>(inlier::PackToken(
        hb_p[i], rb_p[i], sb_p[i], ab_p[i], N_r, N_s, N_a));
  }
  return out;
}

/// unpack_token_ids(token_id, N_r, N_s, N_a) -> (hb, rb, sb, ab) int64.
py::tuple UnpackTokenIds(const Arr<int64_t> &token_id, int64_t N_r,
                         int64_t N_s, int64_t N_a) {
  const py::ssize_t n = token_id.size();
  py::array_t<int64_t> hb(n), rb(n), sb(n), ab(n);
  const auto *t = token_id.data();
  auto *hb_p = hb.mutable_data();
  auto *rb_p = rb.mutable_data();
  auto *sb_p = sb.mutable_data();
  auto *ab_p = ab.mutable_data();
  for (py::ssize_t i = 0; i < n; ++i) {
    const auto b = inlier::UnpackToken(t[i], N_r, N_s, N_a);
    hb_p[i] = b.hb;
    rb_p[i] = b.rb;
    sb_p[i] = b.sb;
    ab_p[i] = b.ab;
  }
  return py::make_tuple(hb, rb, sb, ab);
}

py::array_t<int32_t> BinRadialArr(const Arr<double> &r, double r_max,
                                  int n_bins) {
  const py::ssize_t n = r.size();
  py::array_t<int32_t> out(n);
  const auto *r_p = r.data();
  auto *o = out.mutable_data();
  for (py::ssize_t i = 0; i < n; ++i) {
    o[i] = inlier::BinRadial(r_p[i], r_max, n_bins);
  }
  return out;
}

py::array_t<int32_t> BinAzimuthArr(const Arr<double> &theta, int n_bins) {
  const py::ssize_t n = theta.size();
  py::array_t<int32_t> out(n);
  const auto *t_p = theta.data();
  auto *o = out.mutable_data();
  for (py::ssize_t i = 0; i < n; ++i) {
    o[i] = inlier::BinAzimuth(t_p[i], n_bins);
  }
  return out;
}

/// Height bin for pre-clipped z values (callers replicate the Python
/// clip/filter that precedes searchsorted).
py::array_t<int16_t> BinHeightArr(const Arr<double> &z, double z_min,
                                  double z_max, int N_h) {
  const auto edges = inlier::MakeZEdges(z_min, z_max, N_h);
  const py::ssize_t n = z.size();
  py::array_t<int16_t> out(n);
  const auto *z_p = z.data();
  auto *o = out.mutable_data();
  for (py::ssize_t i = 0; i < n; ++i) {
    o[i] = static_cast<int16_t>(inlier::BinHeight(z_p[i], edges));
  }
  return out;
}

py::array_t<int64_t> VerifyKeyArr(const Arr<int64_t> &hb,
                                  const Arr<int64_t> &rb,
                                  const Arr<int64_t> &ab, int64_t N_r,
                                  int64_t N_a) {
  const py::ssize_t n = hb.size();
  py::array_t<int64_t> out(n);
  const auto *hb_p = hb.data();
  const auto *rb_p = rb.data();
  const auto *ab_p = ab.data();
  auto *o = out.mutable_data();
  for (py::ssize_t i = 0; i < n; ++i) {
    o[i] = inlier::VerifyKey(hb_p[i], rb_p[i], ab_p[i], N_r, N_a);
  }
  return out;
}

py::array_t<int64_t> RerankKeyArr(const Arr<int64_t> &sb,
                                  const Arr<int64_t> &hb,
                                  const Arr<int64_t> &rb,
                                  const Arr<int64_t> &ab, int64_t N_h,
                                  int64_t N_r, int64_t N_a) {
  const py::ssize_t n = sb.size();
  py::array_t<int64_t> out(n);
  const auto *sb_p = sb.data();
  const auto *hb_p = hb.data();
  const auto *rb_p = rb.data();
  const auto *ab_p = ab.data();
  auto *o = out.mutable_data();
  for (py::ssize_t i = 0; i < n; ++i) {
    o[i] = inlier::RerankKey(sb_p[i], hb_p[i], rb_p[i], ab_p[i], N_h, N_r,
                             N_a);
  }
  return out;
}

/// Tokens -> numpy array with the public dtype contract: uint32 unless
/// the mixed-radix product N_h*N_r*N_s*N_a exceeds 2^32-1.
py::array TokensToArray(const inlier::Tokens &tokens,
                        const inlier::InLiERConfig &cfg) {
  const uint64_t product = static_cast<uint64_t>(cfg.N_h) * cfg.N_r *
                           std::max(1, cfg.N_s) * std::max(1, cfg.N_a);
  const auto n = static_cast<py::ssize_t>(tokens.token_id.size());
  if (product > UINT64_C(0xFFFFFFFF)) {
    py::array_t<uint64_t> out(n);
    std::copy(tokens.token_id.begin(), tokens.token_id.end(),
              out.mutable_data());
    return out;
  }
  py::array_t<uint32_t> out(n);
  auto *o = out.mutable_data();
  for (py::ssize_t i = 0; i < n; ++i) {
    o[i] = static_cast<uint32_t>(tokens.token_id[static_cast<size_t>(i)]);
  }
  return out;
}

inlier::TokenizeMode ParseTokenizeMode(const std::string &mode) {
  if (mode == "config") return inlier::TokenizeMode::kFromConfig;
  if (mode == "keypoints") return inlier::TokenizeMode::kKeypoints;
  if (mode == "all_points") return inlier::TokenizeMode::kAllPoints;
  throw py::value_error("mode must be 'config' | 'keypoints' | 'all_points'");
}

/// Thin binding wrapper: numpy (N,3) in, plain arrays out. The Python
/// wrapper layer (inlier/core/InLiER.py) rebuilds the dataclasses.
class PyEncoder {
 public:
  explicit PyEncoder(const inlier::InLiERConfig &cfg) : enc_(cfg) {}

  py::tuple Encode(const Arr<double> &points,
                   const std::optional<inlier::Plane> &plane) const {
    const auto pts = AsPoints(points);
    std::pair<inlier::Keypoints, inlier::Tokens> result;
    {
      py::gil_scoped_release release;
      result = enc_.Encode(pts, plane);
    }
    return py::make_tuple(result.first.p, result.first.T_ground,
                          TokensToArray(result.second, enc_.config()));
  }

  py::tuple ExtractKeypoints(const Arr<double> &points,
                             const std::optional<inlier::Plane> &plane) const {
    const auto pts = AsPoints(points);
    inlier::Keypoints kp;
    {
      py::gil_scoped_release release;
      kp = enc_.ExtractKeypoints(pts, plane);
    }
    return py::make_tuple(kp.p, kp.T_ground);
  }

  py::array Tokenize(const Arr<double> &points, const Arr<double> &kp_p,
                     const Eigen::Matrix4d &T_ground,
                     const std::optional<inlier::Plane> &plane,
                     const std::string &mode) const {
    const auto pts = AsPoints(points);
    inlier::Keypoints kp;
    kp.p = AsPoints(kp_p);
    kp.T_ground = T_ground;
    const auto m = ParseTokenizeMode(mode);
    inlier::Tokens tokens;
    {
      py::gil_scoped_release release;
      tokens = enc_.Tokenize(pts, kp, plane, m);
    }
    return TokensToArray(tokens, enc_.config());
  }

 private:
  static Eigen::Matrix<double, Eigen::Dynamic, 3> AsPoints(
      const Arr<double> &points) {
    if (points.ndim() != 2 || points.shape(1) != 3) {
      throw py::value_error("points_xyz must have shape (N, 3)");
    }
    const auto n = points.shape(0);
    Eigen::Matrix<double, Eigen::Dynamic, 3> out(n, 3);
    const auto *p = points.data();
    for (py::ssize_t i = 0; i < n; ++i) {
      out(i, 0) = p[3 * i];
      out(i, 1) = p[3 * i + 1];
      out(i, 2) = p[3 * i + 2];
    }
    return out;
  }

  inlier::Encoder enc_;
};

std::vector<uint64_t> TokenArrayToVector(const Arr<uint64_t> &token_id) {
  return {token_id.data(), token_id.data() + token_id.size()};
}

/// Thin matcher wrapper: numpy arrays in, result structs out. The
/// Python wrapper layer (inlier/core/InLiER_Matcher.py) rebuilds the
/// output dataclasses.
class PyMatcher {
 public:
  explicit PyMatcher(const inlier::InLiERConfig &cfg) : m_(cfg) {}

  void Add(int64_t database_id, const Arr<uint64_t> &token_id) {
    m_.Add(database_id, TokenArrayToVector(token_id));
  }
  void Reset() { m_.Reset(); }
  void Finalize() {
    py::gil_scoped_release release;
    m_.Finalize();
  }
  size_t Size() const { return m_.size(); }
  bool Finalized() const { return m_.finalized(); }

  py::array_t<int64_t> DbIds() const {
    const auto &ids = m_.db_ids();
    py::array_t<int64_t> out(static_cast<py::ssize_t>(ids.size()));
    std::copy(ids.begin(), ids.end(), out.mutable_data());
    return out;
  }

  py::tuple GetScanData(int64_t database_id) const {
    const inlier::ScanData s = m_.GetScanData(database_id);
    auto vec_i16 = [](const std::vector<int16_t> &v) {
      py::array_t<int16_t> a(static_cast<py::ssize_t>(v.size()));
      std::copy(v.begin(), v.end(), a.mutable_data());
      return a;
    };
    py::array_t<uint64_t> tid(static_cast<py::ssize_t>(s.token_id.size()));
    std::copy(s.token_id.begin(), s.token_id.end(), tid.mutable_data());
    return py::make_tuple(tid, vec_i16(s.hb), vec_i16(s.rb), vec_i16(s.sb),
                          vec_i16(s.ab), s.max_active_hb);
  }

  inlier::ShortlistResult Shortlist(const Arr<uint64_t> &q_tid,
                                    const inlier::ShortlistConfig &cfg,
                                    int topk, double topk_pct) {
    const auto q = TokenArrayToVector(q_tid);
    py::gil_scoped_release release;
    return m_.Shortlist(q, cfg, topk, topk_pct);
  }

  inlier::BeamResult BeamScore(const Arr<uint64_t> &q_tid,
                               const Arr<int64_t> &candidate_ids,
                               const inlier::BEAMScoreConfig &cfg, int topk,
                               double topk_pct) const {
    const auto q = TokenArrayToVector(q_tid);
    const std::vector<int64_t> cands(
        candidate_ids.data(), candidate_ids.data() + candidate_ids.size());
    py::gil_scoped_release release;
    return m_.BeamScore(q, cands, cfg, topk, topk_pct);
  }

  inlier::RerankResult Rerank(const Arr<uint64_t> &q_tid,
                              const Arr<int64_t> &candidate_ids,
                              const Arr<int32_t> &candidate_shifts,
                              const inlier::RerankConfig &cfg, int topk,
                              double topk_pct) const {
    const auto q = TokenArrayToVector(q_tid);
    const std::vector<int64_t> cands(
        candidate_ids.data(), candidate_ids.data() + candidate_ids.size());
    const std::vector<int> shifts(
        candidate_shifts.data(),
        candidate_shifts.data() + candidate_shifts.size());
    py::gil_scoped_release release;
    return m_.Rerank(q, cands, shifts, cfg, topk, topk_pct);
  }

 private:
  inlier::Matcher m_;
};

}  // namespace

PYBIND11_MODULE(_inlier_pybind, m) {
  m.doc() = "InLiER C++ core (private backend; use the `inlier` package API)";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  // --- token codec ---
  m.def("pack_token_ids", &PackTokenIds, py::arg("hb"), py::arg("rb"),
        py::arg("sb"), py::arg("ab"), py::arg("N_h"), py::arg("N_r"),
        py::arg("N_s"), py::arg("N_a"),
        "Mixed-radix token packing; uint32 unless N_h*N_r*N_s*N_a > 2^32-1.");
  m.def("unpack_token_ids", &UnpackTokenIds, py::arg("token_id"),
        py::arg("N_r"), py::arg("N_s"), py::arg("N_a"),
        "Inverse of pack_token_ids -> (hb, rb, sb, ab) int64 arrays.");
  m.def("bin_radial", &BinRadialArr, py::arg("r"), py::arg("r_max"),
        py::arg("n_bins"));
  m.def("bin_azimuth", &BinAzimuthArr, py::arg("theta"), py::arg("n_bins"));
  m.def("bin_height", &BinHeightArr, py::arg("z"), py::arg("z_min"),
        py::arg("z_max"), py::arg("N_h"));
  m.def("verify_key", &VerifyKeyArr, py::arg("hb"), py::arg("rb"),
        py::arg("ab"), py::arg("N_r"), py::arg("N_a"));
  m.def("rerank_key", &RerankKeyArr, py::arg("sb"), py::arg("hb"),
        py::arg("rb"), py::arg("ab"), py::arg("N_h"), py::arg("N_r"),
        py::arg("N_a"));

  // --- plane geometry ---
  py::class_<inlier::Plane>(m, "Plane")
      .def(py::init<>())
      .def_readwrite("normal", &inlier::Plane::normal)
      .def_readwrite("d", &inlier::Plane::d)
      .def_readwrite("point", &inlier::Plane::point)
      .def_readwrite("inliers", &inlier::Plane::inliers);

  m.def(
      "ransac_plane",
      [](const Arr<double> &points, int iters, double dist_thresh,
         int min_inliers) {
        if (points.ndim() != 2 || points.shape(1) != 3) {
          throw py::value_error("points must have shape (N, 3)");
        }
        Eigen::Matrix<double, Eigen::Dynamic, 3> pts(points.shape(0), 3);
        const auto *p = points.data();
        for (py::ssize_t i = 0; i < points.shape(0); ++i) {
          pts(i, 0) = p[3 * i];
          pts(i, 1) = p[3 * i + 1];
          pts(i, 2) = p[3 * i + 2];
        }
        py::gil_scoped_release release;
        return inlier::RansacPlane(pts, iters, dist_thresh, min_inliers);
      },
      py::arg("points"), py::arg("iters"), py::arg("dist_thresh"),
      py::arg("min_inliers"));
  m.def("align_ground", &inlier::AlignGround, py::arg("plane_normal"),
        py::arg("plane_point"),
        "T_ground (4x4) rotating plane_normal -> +Z with ground at z=0.");
  m.def("rodrigues", &inlier::Rodrigues, py::arg("axis"), py::arg("angle"));

  // --- config + encoder ---
  py::class_<inlier::InLiERConfig>(m, "InLiERConfig")
      .def(py::init<>())
      .def_readwrite("N_h", &inlier::InLiERConfig::N_h)
      .def_readwrite("z_min", &inlier::InLiERConfig::z_min)
      .def_readwrite("z_max", &inlier::InLiERConfig::z_max)
      .def_readwrite("r_max", &inlier::InLiERConfig::r_max)
      .def_readwrite("N_r", &inlier::InLiERConfig::N_r)
      .def_readwrite("N_a", &inlier::InLiERConfig::N_a)
      .def_readwrite("N_s", &inlier::InLiERConfig::N_s)
      .def_readwrite("cell_size", &inlier::InLiERConfig::cell_size)
      .def_readwrite("xy_max", &inlier::InLiERConfig::xy_max)
      .def_readwrite("window", &inlier::InLiERConfig::window)
      .def_readwrite("max_kp_per_slice", &inlier::InLiERConfig::max_kp_per_slice)
      .def_readwrite("max_kp_total", &inlier::InLiERConfig::max_kp_total)
      .def_readwrite("ransac_iters", &inlier::InLiERConfig::ransac_iters)
      .def_readwrite("ransac_dist_thresh", &inlier::InLiERConfig::ransac_dist_thresh)
      .def_readwrite("ransac_min_inliers", &inlier::InLiERConfig::ransac_min_inliers)
      .def_readwrite("point_mode", &inlier::InLiERConfig::point_mode)
      .def_readwrite("shape_radius", &inlier::InLiERConfig::shape_radius)
      .def_readwrite("shape_min_neighbors", &inlier::InLiERConfig::shape_min_neighbors);

  py::class_<PyEncoder>(m, "_Encoder")
      .def(py::init<const inlier::InLiERConfig &>(), py::arg("config"))
      .def("encode", &PyEncoder::Encode, py::arg("points"),
           py::arg("plane") = std::nullopt,
           "-> (p (K,3) f64, T_ground (4,4) f64, token_id (K,) u32|u64)")
      .def("extract_keypoints", &PyEncoder::ExtractKeypoints,
           py::arg("points"), py::arg("plane") = std::nullopt,
           "-> (p (K,3) f64, T_ground (4,4) f64)")
      .def("tokenize", &PyEncoder::Tokenize, py::arg("points"),
           py::arg("kp_p"), py::arg("T_ground"),
           py::arg("plane") = std::nullopt, py::arg("mode") = "config",
           "-> token_id (K,) u32|u64");

  // --- shape PCA ---
  m.def(
      "compute_shape_pca",
      [](const Arr<double> &points, const Arr<double> &centers, double radius,
         int min_neighbors, int n_classes) {
        auto as_pts = [](const Arr<double> &a) {
          if (a.ndim() != 2 || a.shape(1) != 3) {
            throw py::value_error("array must have shape (N, 3)");
          }
          Eigen::Matrix<double, Eigen::Dynamic, 3> out(a.shape(0), 3);
          const auto *p = a.data();
          for (py::ssize_t i = 0; i < a.shape(0); ++i) {
            out(i, 0) = p[3 * i];
            out(i, 1) = p[3 * i + 1];
            out(i, 2) = p[3 * i + 2];
          }
          return out;
        };
        const auto pts = as_pts(points);
        const auto ctr = as_pts(centers);
        inlier::ShapePcaResult r;
        {
          py::gil_scoped_release release;
          r = inlier::ComputeShapePca(pts, ctr, radius, min_neighbors,
                                      n_classes);
        }
        py::array_t<int16_t> cls(
            static_cast<py::ssize_t>(r.shape_class.size()));
        std::copy(r.shape_class.begin(), r.shape_class.end(),
                  cls.mutable_data());
        return py::make_tuple(cls, Eigen::MatrixXf(r.lps));
      },
      py::arg("points"), py::arg("centers"), py::arg("radius"),
      py::arg("min_neighbors"), py::arg("n_classes"),
      "-> (shape_class (K,) int16, lps (K,3) float32)");

  // --- matcher: DB bookkeeping, MINT shortlist, BEAM score, rerank ---
  py::class_<inlier::ShortlistConfig>(m, "ShortlistConfig")
      .def(py::init<>())
      .def_readwrite("topk", &inlier::ShortlistConfig::topk)
      .def_readwrite("topk_pct", &inlier::ShortlistConfig::topk_pct)
      .def_readwrite("min_shared_rows",
                     &inlier::ShortlistConfig::min_shared_rows)
      .def_readwrite("mint_mode", &inlier::ShortlistConfig::mint_mode)
      .def_readwrite("mint_scoring", &inlier::ShortlistConfig::mint_scoring)
      .def_readwrite("eps", &inlier::ShortlistConfig::eps);

  py::class_<inlier::BEAMScoreConfig>(m, "BEAMScoreConfig")
      .def(py::init<>())
      .def_readwrite("topk", &inlier::BEAMScoreConfig::topk)
      .def_readwrite("topk_pct", &inlier::BEAMScoreConfig::topk_pct)
      .def_readwrite("min_shared_bins",
                     &inlier::BEAMScoreConfig::min_shared_bins)
      .def_readwrite("min_shared_az_cols",
                     &inlier::BEAMScoreConfig::min_shared_az_cols)
      .def_readwrite("score_threshold",
                     &inlier::BEAMScoreConfig::score_threshold);

  py::class_<inlier::RerankConfig>(m, "RerankConfig")
      .def(py::init<>())
      .def_readwrite("topk", &inlier::RerankConfig::topk)
      .def_readwrite("topk_pct", &inlier::RerankConfig::topk_pct)
      .def_readwrite("scoring_mode", &inlier::RerankConfig::scoring_mode)
      .def_readwrite("min_shared_rows",
                     &inlier::RerankConfig::min_shared_rows)
      .def_readwrite("spatial_tol", &inlier::RerankConfig::spatial_tol)
      .def_readwrite("score_threshold",
                     &inlier::RerankConfig::score_threshold)
      .def_readwrite("eps", &inlier::RerankConfig::eps);

  py::class_<inlier::ShortlistResult>(m, "ShortlistResult")
      .def_readonly("ids", &inlier::ShortlistResult::ids)
      .def_readonly("scores", &inlier::ShortlistResult::scores);

  py::class_<inlier::BeamResult>(m, "BeamResult")
      .def_readonly("ids", &inlier::BeamResult::ids)
      .def_readonly("scores", &inlier::BeamResult::scores)
      .def_readonly("yaw_estimates", &inlier::BeamResult::yaw_estimates)
      .def_readonly("best_shifts", &inlier::BeamResult::best_shifts);

  py::class_<inlier::RerankResult>(m, "RerankResult")
      .def_readonly("ids", &inlier::RerankResult::ids)
      .def_readonly("scores", &inlier::RerankResult::scores)
      .def_readonly("hist_scores", &inlier::RerankResult::hist_scores)
      .def_readonly("inlier_ratios", &inlier::RerankResult::inlier_ratios)
      .def_readonly("inlier_counts", &inlier::RerankResult::inlier_counts)
      .def_readonly("yaw_estimates", &inlier::RerankResult::yaw_estimates)
      .def_readonly("best_shifts", &inlier::RerankResult::best_shifts);

  py::class_<PyMatcher>(m, "_Matcher")
      .def(py::init<const inlier::InLiERConfig &>(), py::arg("config"))
      .def("add", &PyMatcher::Add, py::arg("database_id"),
           py::arg("token_id"))
      .def("reset", &PyMatcher::Reset)
      .def("finalize", &PyMatcher::Finalize)
      .def("__len__", &PyMatcher::Size)
      .def_property_readonly("finalized", &PyMatcher::Finalized)
      .def("db_ids", &PyMatcher::DbIds)
      .def("get_scan_data", &PyMatcher::GetScanData, py::arg("database_id"),
           "-> (token_id, hb, rb, sb, ab, max_active_hb)")
      .def("shortlist", &PyMatcher::Shortlist, py::arg("query_token_id"),
           py::arg("config"), py::arg("topk") = -1,
           py::arg("topk_pct") = -1.0)
      .def("beam_score", &PyMatcher::BeamScore, py::arg("query_token_id"),
           py::arg("candidate_ids"), py::arg("config"), py::arg("topk") = -1,
           py::arg("topk_pct") = -1.0)
      .def("rerank", &PyMatcher::Rerank, py::arg("query_token_id"),
           py::arg("candidate_ids"), py::arg("candidate_shifts"),
           py::arg("config"), py::arg("topk") = -1,
           py::arg("topk_pct") = -1.0);

  // --- verify: token-guided RANSAC geometric verification ---
  py::class_<inlier::VerifyConfig>(m, "VerifyConfig")
      .def(py::init<>())
      .def_readwrite("topk", &inlier::VerifyConfig::topk)
      .def_readwrite("topk_pct", &inlier::VerifyConfig::topk_pct)
      .def_readwrite("ransac_iters", &inlier::VerifyConfig::ransac_iters)
      .def_readwrite("inlier_dist_thresh",
                     &inlier::VerifyConfig::inlier_dist_thresh)
      .def_readwrite("min_correspondences",
                     &inlier::VerifyConfig::min_correspondences)
      .def_readwrite("min_ransac_inliers",
                     &inlier::VerifyConfig::min_ransac_inliers)
      .def_readwrite("min_keypoint_inliers",
                     &inlier::VerifyConfig::min_keypoint_inliers)
      .def_readwrite("spatial_tol", &inlier::VerifyConfig::spatial_tol)
      .def_readwrite("seed", &inlier::VerifyConfig::seed);

  py::class_<inlier::VerifyResult>(m, "VerifyResult")
      .def_readonly("fail_stage", &inlier::VerifyResult::fail_stage)
      .def_readonly("ransac_inliers_found",
                    &inlier::VerifyResult::ransac_inliers_found)
      .def_readonly("success", &inlier::VerifyResult::success)
      .def_readonly("T_sensor", &inlier::VerifyResult::T_sensor)
      .def_readonly("yaw", &inlier::VerifyResult::yaw)
      .def_readonly("tx", &inlier::VerifyResult::tx)
      .def_readonly("ty", &inlier::VerifyResult::ty)
      .def_readonly("tz", &inlier::VerifyResult::tz)
      .def_readonly("n_correspondences",
                    &inlier::VerifyResult::n_correspondences)
      .def_readonly("n_ransac_inliers",
                    &inlier::VerifyResult::n_ransac_inliers)
      .def_readonly("n_keypoint_inliers",
                    &inlier::VerifyResult::n_keypoint_inliers)
      .def_readonly("n_total_keypoints",
                    &inlier::VerifyResult::n_total_keypoints)
      .def_readonly("ransac_inlier_ratio",
                    &inlier::VerifyResult::ransac_inlier_ratio)
      .def_readonly("keypoint_inlier_ratio",
                    &inlier::VerifyResult::keypoint_inlier_ratio)
      .def_readonly("inlier_rmse", &inlier::VerifyResult::inlier_rmse);

  m.def(
      "verify",
      [](const Arr<uint64_t> &q_tid, const Arr<double> &q_p,
         const Eigen::Matrix4d &q_T_ground, const Arr<uint64_t> &db_tid,
         const Arr<double> &db_p, const Eigen::Matrix4d &db_T_ground,
         int azimuth_shift, const inlier::InLiERConfig &grid,
         const inlier::VerifyConfig &cfg) {
        auto as_pts = [](const Arr<double> &a) {
          if (a.ndim() != 2 || a.shape(1) != 3) {
            throw py::value_error("keypoints must have shape (K, 3)");
          }
          Eigen::Matrix<double, Eigen::Dynamic, 3> out(a.shape(0), 3);
          const auto *p = a.data();
          for (py::ssize_t i = 0; i < a.shape(0); ++i) {
            out(i, 0) = p[3 * i];
            out(i, 1) = p[3 * i + 1];
            out(i, 2) = p[3 * i + 2];
          }
          return out;
        };
        inlier::Keypoints q_kp, db_kp;
        q_kp.p = as_pts(q_p);
        q_kp.T_ground = q_T_ground;
        db_kp.p = as_pts(db_p);
        db_kp.T_ground = db_T_ground;
        const auto q = TokenArrayToVector(q_tid);
        const auto db = TokenArrayToVector(db_tid);
        py::gil_scoped_release release;
        return inlier::Verify(q, q_kp, db, db_kp, azimuth_shift, grid, cfg);
      },
      py::arg("query_token_id"), py::arg("query_p"),
      py::arg("query_T_ground"), py::arg("db_token_id"), py::arg("db_p"),
      py::arg("db_T_ground"), py::arg("azimuth_shift"), py::arg("grid"),
      py::arg("config"),
      "Token-guided RANSAC verification -> VerifyResult");
}