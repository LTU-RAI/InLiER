// Private pybind11 backend for the `inlier` package.
//
// The public API lives in the Python wrappers (inlier/core/InLiER.py,
// inlier/core/InLiER_Matcher.py); this module only exposes the raw C++ core.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>

#include "inlier_core/token.hpp"

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

}  // namespace

PYBIND11_MODULE(_inlier_pybind, m) {
  m.doc() = "InLiER C++ core (private backend; use the `inlier` package API)";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  // ---- M1: token codec ----
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
}
