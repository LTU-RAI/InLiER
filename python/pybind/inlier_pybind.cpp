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
#include "inlier_core/plane.hpp"
#include "inlier_core/shape_pca.hpp"
#include "inlier_core/token.hpp"
#include "inlier_core/types.hpp"

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
}