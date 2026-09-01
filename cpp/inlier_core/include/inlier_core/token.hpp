// Token codec: mixed-radix pack/unpack, coordinate binning, and the two
// stage-specific flat-key encodings. Header-only.
//
// Bit-exactness notes (each function mirrors a specific numpy call):
//  * MakeZEdges       == np.linspace(z_min, z_max, N_h + 1)
//                        (edges[i] = i*step + z_min, last forced to z_max)
//  * BinHeight        == np.clip(np.searchsorted(z_edges[1:], z,
//                        side="right"), 0, N_h - 1); callers clip z first
//                        where the Python does.
//  * BinAzimuth       == InLiER._bin_azimuth — numpy's remainder is
//                        fmod + sign correction, replicated here.
//  * There are THREE distinct integer encodings in the pipeline; never
//    substitute one for another:
//      token_id   = ((hb*N_r + rb)*N_s + sb)*N_a + ab      (hb-major, sb in)
//      verify key = hb*(Nr*Na) + rb*Na + ab                 (sb excluded)
//      rerank key = sb*(Nh*Nr*Na) + hb*(Nr*Na) + rb*Na + ab (sb-major)
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace inlier {

inline constexpr double kTwoPi = 6.283185307179586476925286766559;

// --- mixed-radix token codec ---

/// token_id = ((hb * N_r + rb) * N_s + sb) * N_a + ab
inline uint64_t PackToken(uint64_t hb, uint64_t rb, uint64_t sb, uint64_t ab,
                          uint64_t N_r, uint64_t N_s, uint64_t N_a) {
  return ((hb * N_r + rb) * N_s + sb) * N_a + ab;
}

struct TokenBins {
  int64_t hb, rb, sb, ab;
};

/// Inverse of PackToken (Python: InLiER.unpack_token_ids).
inline TokenBins UnpackToken(int64_t token_id, int64_t N_r, int64_t N_s,
                             int64_t N_a) {
  TokenBins b;
  b.ab = token_id % N_a;
  const int64_t base = token_id / N_a;
  b.sb = base % N_s;
  b.rb = (base / N_s) % N_r;
  b.hb = (base / N_s) / N_r;
  return b;
}

// --- coordinate binning ---

/// z_edges = np.linspace(z_min, z_max, N_h + 1), replicated bit-exactly.
inline std::vector<double> MakeZEdges(double z_min, double z_max, int N_h) {
  const double step = (z_max - z_min) / static_cast<double>(N_h);
  std::vector<double> edges(static_cast<size_t>(N_h) + 1);
  for (int i = 0; i <= N_h; ++i) {
    edges[static_cast<size_t>(i)] = static_cast<double>(i) * step + z_min;
  }
  edges.back() = z_max;  // numpy linspace forces the endpoint
  return edges;
}

/// Height bin: searchsorted(z_edges[1:], z, side="right") clipped to
/// [0, N_h-1]. side="right" == count of edges[1..N_h] that are <= z
/// == std::upper_bound over that range.
inline int BinHeight(double z, const std::vector<double> &z_edges) {
  const int N_h = static_cast<int>(z_edges.size()) - 1;
  const auto first = z_edges.begin() + 1;
  const int idx =
      static_cast<int>(std::upper_bound(first, z_edges.end(), z) - first);
  return std::clamp(idx, 0, N_h - 1);
}

/// Radial bin: equal-width floor bins (Python: InLiER._bin_radial).
inline int BinRadial(double r, double r_max, int n_bins) {
  const double step = std::max(1e-6, r_max) / static_cast<double>(n_bins);
  const int64_t b = static_cast<int64_t>(std::floor(r / step));
  return static_cast<int>(
      std::clamp(b, int64_t{0}, static_cast<int64_t>(n_bins) - 1));
}

/// numpy remainder semantics: fmod + sign correction (non-negative for
/// a positive divisor).
inline double NumpyMod(double x, double y) {
  double r = std::fmod(x, y);
  if (r != 0.0 && ((r < 0.0) != (y < 0.0))) {
    r += y;
  }
  return r;
}

/// Azimuth bin, centred on cardinal directions (Python: InLiER._bin_azimuth).
inline int BinAzimuth(double theta, int n_bins) {
  n_bins = std::max(1, n_bins);
  const double step = kTwoPi / static_cast<double>(n_bins);
  double t = NumpyMod(theta + M_PI, kTwoPi);  // [0, 2π)
  t = NumpyMod(t + step / 2.0, kTwoPi);       // centre on cardinals
  const int64_t b = static_cast<int64_t>(std::floor(t / step));
  return static_cast<int>(
      std::clamp(b, int64_t{0}, static_cast<int64_t>(n_bins) - 1));
}

// --- stage-specific flat keys ---

/// Correspondence key (InLiER_Matcher._find_correspondences): matches on
/// spatial bins only — shape class sb is EXCLUDED.
inline int64_t VerifyKey(int64_t hb, int64_t rb, int64_t ab, int64_t N_r,
                         int64_t N_a) {
  return hb * (N_r * N_a) + rb * N_a + ab;
}

/// Token-membership key (InLiER_Matcher._verify_tokens_vectorized):
/// sb-major — a DIFFERENT field order than token_id.
inline int64_t RerankKey(int64_t sb, int64_t hb, int64_t rb, int64_t ab,
                         int64_t N_h, int64_t N_r, int64_t N_a) {
  return sb * (N_h * N_r * N_a) + hb * (N_r * N_a) + rb * N_a + ab;
}

}  // namespace inlier
