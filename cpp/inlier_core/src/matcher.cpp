#include "inlier_core/matcher.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "inlier_core/token.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define INLIER_POPCOUNT64(x) __builtin_popcountll(x)
#else
#include <bitset>
#define INLIER_POPCOUNT64(x) static_cast<int>(std::bitset<64>(x).count())
#endif

namespace inlier {

namespace {

/// Python round() (banker's rounding, round-half-even).
inline int64_t RoundHalfEven(double x) {
  return static_cast<int64_t>(std::nearbyint(x));
}

/// InLiER_Matcher._resolve_topk: call-time args take priority over config;
/// negative override / cfg values mean "None".
inline int ResolveTopK(int topk, double topk_pct, int cfg_topk,
                       double cfg_topk_pct, int64_t n_total) {
  auto clip = [n_total](int64_t v) {
    return static_cast<int>(std::clamp<int64_t>(v, 1, n_total));
  };
  if (topk >= 0) {
    return clip(topk);
  }
  if (topk_pct >= 0.0) {
    return clip(std::max<int64_t>(1, RoundHalfEven(n_total * topk_pct)));
  }
  if (cfg_topk_pct >= 0.0) {
    return clip(std::max<int64_t>(1, RoundHalfEven(n_total * cfg_topk_pct)));
  }
  return clip(cfg_topk);
}

/// Top-k indices sorted by score descending. Replicates
/// np.argsort(scores)[::-1] tie order (descending index) for the k >= n
/// path; below that numpy's argpartition boundary ties are arbitrary, so
/// the same comparator is the deterministic stand-in.
inline std::vector<size_t> TopKOrder(const std::vector<double> &scores,
                                     int k) {
  std::vector<size_t> order(scores.size());
  std::iota(order.begin(), order.end(), size_t{0});
  const auto cmp = [&scores](size_t a, size_t b) {
    if (scores[a] != scores[b]) {
      return scores[a] > scores[b];
    }
    return a > b;
  };
  const auto kk = std::min<size_t>(static_cast<size_t>(k), order.size());
  std::partial_sort(order.begin(), order.begin() + kk, order.end(), cmp);
  order.resize(kk);
  return order;
}

/// InLiER_Matcher._shifts_to_yaw: shift -> yaw in (-pi, pi].
inline double ShiftToYaw(int shift, int Na) {
  const int m = ((-shift) % Na + Na) % Na;
  const double raw = m * (2.0 * M_PI / Na);
  return raw <= M_PI ? raw : raw - 2.0 * M_PI;
}

/// (Nr, Na) BEAM bitmask matrix, flat row-major; bit h set iff a token
/// exists at (h, r, a). Always uint64 (Python narrows the dtype for
/// memory only; N_h <= 64 is enforced at config validation).
std::vector<uint64_t> BuildBeam(const std::vector<int16_t> &hb,
                                const std::vector<int16_t> &rb,
                                const std::vector<int16_t> &ab, int Nr,
                                int Na) {
  std::vector<uint64_t> beam(static_cast<size_t>(Nr) * Na, 0);
  for (size_t i = 0; i < hb.size(); ++i) {
    beam[static_cast<size_t>(rb[i]) * Na + ab[i]] |= uint64_t{1} << hb[i];
  }
  return beam;
}

/// InLiER_Matcher._score_beam_shifts: bit-level Jaccard over all Na
/// circular shifts; returns (best_score, best_shift) with np.argmax
/// first-max tie-breaking.
std::pair<double, int> ScoreBeamShifts(const std::vector<uint64_t> &q_beam,
                                       int q_max_hb,
                                       const std::vector<uint64_t> &db_beam,
                                       int db_max_hb, int Nr, int Na,
                                       const BEAMScoreConfig &cfg) {
  const int ceiling = std::min(q_max_hb, db_max_hb);
  if (ceiling < 0) {
    return {0.0, 0};
  }
  const uint64_t ceil_mask =
      (ceiling + 1 >= 64) ? ~uint64_t{0}
                          : ((uint64_t{1} << (ceiling + 1)) - 1);

  const size_t n_cells = q_beam.size();
  std::vector<uint64_t> q(n_cells), db(n_cells);
  int64_t q_total = 0, db_total = 0;
  for (size_t i = 0; i < n_cells; ++i) {
    q[i] = q_beam[i] & ceil_mask;
    db[i] = db_beam[i] & ceil_mask;
    q_total += INLIER_POPCOUNT64(q[i]);
    db_total += INLIER_POPCOUNT64(db[i]);
  }
  if (q_total + db_total == 0) {
    return {0.0, 0};
  }

  // Column occupancy for the azimuth-column gate.
  std::vector<uint8_t> q_col_any(static_cast<size_t>(Na), 0);
  std::vector<uint8_t> db_col_any(static_cast<size_t>(Na), 0);
  for (int r = 0; r < Nr; ++r) {
    const size_t base = static_cast<size_t>(r) * Na;
    for (int a = 0; a < Na; ++a) {
      q_col_any[static_cast<size_t>(a)] |= (q[base + a] != 0);
      db_col_any[static_cast<size_t>(a)] |= (db[base + a] != 0);
    }
  }

  double best_score = 0.0;
  int best_shift = 0;
  bool first = true;
  for (int shift = 0; shift < Na; ++shift) {
    // db column at output col a comes from source col (a - shift) mod Na.
    int64_t inter = 0;
    for (int r = 0; r < Nr; ++r) {
      const size_t base = static_cast<size_t>(r) * Na;
      for (int a = 0; a < Na; ++a) {
        const int src = (a - shift % Na + Na) % Na;
        inter += INLIER_POPCOUNT64(q[base + a] & db[base + src]);
      }
    }
    const int64_t uni = q_total + db_total - inter;

    int az_union_cols = 0;
    for (int a = 0; a < Na; ++a) {
      const int src = (a - shift % Na + Na) % Na;
      az_union_cols += (q_col_any[static_cast<size_t>(a)] ||
                        db_col_any[static_cast<size_t>(src)]);
    }

    double score = 0.0;
    if (uni >= cfg.min_shared_bins && az_union_cols >= cfg.min_shared_az_cols) {
      score = static_cast<double>(inter) /
              static_cast<double>(std::max<int64_t>(uni, 1));
    }
    if (first || score > best_score) {  // first-max (np.argmax)
      best_score = score;
      best_shift = shift;
      first = false;
    }
  }
  return {best_score, best_shift};
}

/// 4-D histogram (Nh, Nr*Ns, Na) as flat float32 counts.
std::vector<float> Build4dHist(const std::vector<uint64_t> &token_id, int Nh,
                               int Nr, int Ns, int Na) {
  std::vector<float> hist(
      static_cast<size_t>(Nh) * Nr * Ns * Na, 0.0f);
  for (const uint64_t tid : token_id) {
    hist[static_cast<size_t>(tid)] += 1.0f;
  }
  return hist;
}

}  // namespace

Matcher::Matcher(InLiERConfig config)
    : config_(std::move(config)),
      Nh_(config_.N_h),
      Nr_(config_.N_r),
      Ns_(config_.N_s),
      Na_(config_.N_a) {
  if (Nh_ < 1 || Nh_ > 64) {
    throw std::invalid_argument("N_h must be in [1, 64]");
  }
}

void Matcher::Add(int64_t database_id,
                  const std::vector<uint64_t> &token_id) {
  if (finalized_) {
    throw std::runtime_error(
        "Matcher is finalized; call reset() to rebuild.");
  }
  if (id_to_index_.count(database_id)) {
    throw std::invalid_argument("database_id " +
                                std::to_string(database_id) +
                                " already exists.");
  }
  id_to_index_.emplace(database_id, db_ids_.size());
  db_ids_.push_back(database_id);

  int32_t max_hb = -1;
  for (const uint64_t tid : token_id) {
    const TokenBins b =
        UnpackToken(static_cast<int64_t>(tid), Nr_, Ns_, Na_);
    hb_.push_back(static_cast<int16_t>(b.hb));
    rb_.push_back(static_cast<int16_t>(b.rb));
    sb_.push_back(static_cast<int16_t>(b.sb));
    ab_.push_back(static_cast<int16_t>(b.ab));
    token_ids_.push_back(tid);
    max_hb = std::max(max_hb, static_cast<int32_t>(b.hb));
  }
  max_hbs_.push_back(max_hb);
  offsets_.push_back(token_ids_.size());
}

void Matcher::Reset() {
  finalized_ = false;
  db_ids_.clear();
  id_to_index_.clear();
  offsets_.assign(1, 0);
  token_ids_.clear();
  hb_.clear();
  rb_.clear();
  sb_.clear();
  ab_.clear();
  max_hbs_.clear();
  hm_.clear();
}

void Matcher::Finalize() {
  if (finalized_) {
    return;
  }
  const size_t N = db_ids_.size();
  const size_t V = static_cast<size_t>(Nh_) * Nr_ * Ns_;
  hm_.assign(N * V, 0.0f);
  for (size_t i = 0; i < N; ++i) {
    float *row = hm_.data() + i * V;
    for (size_t j = offsets_[i]; j < offsets_[i + 1]; ++j) {
      // stage1_id strips azimuth
      row[token_ids_[j] / static_cast<uint64_t>(Na_)] += 1.0f;
    }
  }
  finalized_ = true;
}

size_t Matcher::IndexOf(int64_t database_id) const {
  const auto it = id_to_index_.find(database_id);
  if (it == id_to_index_.end()) {
    throw std::out_of_range("database_id " + std::to_string(database_id) +
                            " not found.");
  }
  return it->second;
}

ScanData Matcher::GetScanData(int64_t database_id) const {
  const size_t idx = IndexOf(database_id);
  const size_t lo = offsets_[idx], hi = offsets_[idx + 1];
  ScanData s;
  s.token_id.assign(token_ids_.begin() + lo, token_ids_.begin() + hi);
  s.hb.assign(hb_.begin() + lo, hb_.begin() + hi);
  s.rb.assign(rb_.begin() + lo, rb_.begin() + hi);
  s.sb.assign(sb_.begin() + lo, sb_.begin() + hi);
  s.ab.assign(ab_.begin() + lo, ab_.begin() + hi);
  s.max_active_hb = max_hbs_[idx];
  return s;
}

ShortlistResult Matcher::Shortlist(
    const std::vector<uint64_t> &query_token_id, const ShortlistConfig &cfg,
    int topk_override, double topk_pct_override) {
  Finalize();  // lazy, as in Python

  const auto N = static_cast<int64_t>(db_ids_.size());
  ShortlistResult out;
  if (N == 0) {
    return out;
  }
  const int k = ResolveTopK(topk_override, topk_pct_override, cfg.topk,
                            cfg.topk_pct, N);

  const size_t Rw = static_cast<size_t>(Nr_) * Ns_;
  const size_t V = static_cast<size_t>(Nh_) * Rw;

  // Query histogram + max active height bin.
  std::vector<float> q_hm(V, 0.0f);
  int q_max_hb = -1;
  for (const uint64_t tid : query_token_id) {
    q_hm[tid / static_cast<uint64_t>(Na_)] += 1.0f;
    const TokenBins b = UnpackToken(static_cast<int64_t>(tid), Nr_, Ns_, Na_);
    q_max_hb = std::max(q_max_hb, static_cast<int>(b.hb));
  }

  std::vector<double> scores(static_cast<size_t>(N), 0.0);

  if (q_max_hb >= 0) {
    // q_row_active[h] = (row sum >= 0.5)
    std::vector<uint8_t> q_row_active(static_cast<size_t>(Nh_), 0);
    for (int h = 0; h < Nh_; ++h) {
      double s = 0.0;
      for (size_t c = 0; c < Rw; ++c) {
        s += q_hm[static_cast<size_t>(h) * Rw + c];
      }
      q_row_active[static_cast<size_t>(h)] = (s >= 0.5);
    }

    const bool compact = (cfg.mint_mode == "compact");
    if (!compact && cfg.mint_mode != "full") {
      throw std::invalid_argument("mint_mode must be \"compact\" | \"full\"");
    }
    enum class Scoring { kCosine, kRaw, kL1 };
    Scoring scoring;
    if (cfg.mint_scoring == "cosine") {
      scoring = Scoring::kCosine;
    } else if (cfg.mint_scoring == "raw_intersection") {
      scoring = Scoring::kRaw;
    } else if (cfg.mint_scoring == "l1_intersection") {
      scoring = Scoring::kL1;
    } else {
      throw std::invalid_argument(
          "mint_scoring must be \"cosine\" | \"raw_intersection\" | "
          "\"l1_intersection\"");
    }
    const double eps = cfg.eps;

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
      const int ceiling = std::min(q_max_hb, static_cast<int>(max_hbs_[i]));
      const int h_top = std::min(ceiling, Nh_ - 1);  // rows h <= ceiling

      int n_valid = 0;
      for (int h = 0; h <= h_top; ++h) {
        n_valid += q_row_active[static_cast<size_t>(h)];
      }
      if (n_valid < cfg.min_shared_rows) {
        continue;  // score stays 0
      }

      const float *db_row = hm_.data() + static_cast<size_t>(i) * V;

      // Scoring accumulators over the masked descriptor vectors. For
      // "compact" the vectors are height-collapsed sums; for "full" the
      // masked rows are compared element-wise. Both reduce to the same
      // running sums except for min(), which does not commute with the
      // height collapse — handle the two modes separately.
      double dot = 0.0, qq = 0.0, dd = 0.0;
      double q_sum = 0.0, d_sum = 0.0, inter = 0.0;

      if (compact) {
        for (size_t c = 0; c < Rw; ++c) {
          double qv = 0.0, dv = 0.0;
          for (int h = 0; h <= h_top; ++h) {
            qv += q_hm[static_cast<size_t>(h) * Rw + c];
            dv += db_row[static_cast<size_t>(h) * Rw + c];
          }
          dot += qv * dv;
          qq += qv * qv;
          dd += dv * dv;
          q_sum += qv;
          d_sum += dv;
          inter += std::min(qv, dv);
        }
      } else {
        for (int h = 0; h <= h_top; ++h) {
          for (size_t c = 0; c < Rw; ++c) {
            const double qv = q_hm[static_cast<size_t>(h) * Rw + c];
            const double dv = db_row[static_cast<size_t>(h) * Rw + c];
            dot += qv * dv;
            qq += qv * qv;
            dd += dv * dv;
            q_sum += qv;
            d_sum += dv;
            inter += std::min(qv, dv);
          }
        }
      }

      double score = 0.0;
      switch (scoring) {
        case Scoring::kCosine:
          score = dot / ((std::sqrt(qq) + eps) * (std::sqrt(dd) + eps));
          break;
        case Scoring::kRaw:
          score = inter / (q_sum + eps);
          break;
        case Scoring::kL1: {
          // sum(min(q/(q_sum+eps), d/(d_sum+eps))): with all-nonnegative
          // entries, min(q/Q, d/D) must be re-evaluated per element.
          const double Q = q_sum + eps, D = d_sum + eps;
          double s = 0.0;
          if (compact) {
            for (size_t c = 0; c < Rw; ++c) {
              double qv = 0.0, dv = 0.0;
              for (int h = 0; h <= h_top; ++h) {
                qv += q_hm[static_cast<size_t>(h) * Rw + c];
                dv += db_row[static_cast<size_t>(h) * Rw + c];
              }
              s += std::min(qv / Q, dv / D);
            }
          } else {
            for (int h = 0; h <= h_top; ++h) {
              for (size_t c = 0; c < Rw; ++c) {
                s += std::min(
                    q_hm[static_cast<size_t>(h) * Rw + c] / Q,
                    db_row[static_cast<size_t>(h) * Rw + c] / D);
              }
            }
          }
          score = s;
          break;
        }
      }
      // Python stores the gated result as float32.
      scores[static_cast<size_t>(i)] = static_cast<float>(score);
    }
  }

  const std::vector<size_t> order = TopKOrder(scores, k);
  out.ids.reserve(order.size());
  out.scores.reserve(order.size());
  for (const size_t j : order) {
    out.ids.push_back(db_ids_[j]);
    out.scores.push_back(scores[j]);
  }
  return out;
}

BeamResult Matcher::BeamScore(const std::vector<uint64_t> &query_token_id,
                              const std::vector<int64_t> &candidate_ids,
                              const BEAMScoreConfig &cfg, int topk_override,
                              double topk_pct_override) const {
  const auto n_cands = static_cast<int64_t>(candidate_ids.size());
  const int k = ResolveTopK(topk_override, topk_pct_override, cfg.topk,
                            cfg.topk_pct, n_cands);

  // Unpack query once, build its BEAM.
  std::vector<int16_t> q_hb, q_rb, q_ab;
  int q_max_hb = -1;
  q_hb.reserve(query_token_id.size());
  q_rb.reserve(query_token_id.size());
  q_ab.reserve(query_token_id.size());
  for (const uint64_t tid : query_token_id) {
    const TokenBins b = UnpackToken(static_cast<int64_t>(tid), Nr_, Ns_, Na_);
    q_hb.push_back(static_cast<int16_t>(b.hb));
    q_rb.push_back(static_cast<int16_t>(b.rb));
    q_ab.push_back(static_cast<int16_t>(b.ab));
    q_max_hb = std::max(q_max_hb, static_cast<int>(b.hb));
  }
  const std::vector<uint64_t> q_beam = BuildBeam(q_hb, q_rb, q_ab, Nr_, Na_);

  // Resolve candidate indices up-front (throws like Python's KeyError).
  std::vector<size_t> cand_idx(candidate_ids.size());
  for (size_t i = 0; i < candidate_ids.size(); ++i) {
    cand_idx[i] = IndexOf(candidate_ids[i]);
  }

  std::vector<double> scores(candidate_ids.size(), 0.0);
  std::vector<int> shifts(candidate_ids.size(), 0);

#pragma omp parallel for schedule(dynamic, 4)
  for (int64_t i = 0; i < n_cands; ++i) {
    const size_t idx = cand_idx[static_cast<size_t>(i)];
    const size_t lo = offsets_[idx], hi = offsets_[idx + 1];
    const std::vector<int16_t> db_hb(hb_.begin() + lo, hb_.begin() + hi);
    const std::vector<int16_t> db_rb(rb_.begin() + lo, rb_.begin() + hi);
    const std::vector<int16_t> db_ab(ab_.begin() + lo, ab_.begin() + hi);
    const std::vector<uint64_t> db_beam =
        BuildBeam(db_hb, db_rb, db_ab, Nr_, Na_);
    const auto [score, shift] = ScoreBeamShifts(
        q_beam, q_max_hb, db_beam, max_hbs_[idx], Nr_, Na_, cfg);
    scores[static_cast<size_t>(i)] = score;
    shifts[static_cast<size_t>(i)] = shift;
  }

  // Score threshold, then top-k.
  for (auto &s : scores) {
    if (s < cfg.score_threshold) {
      s = 0.0;
    }
  }
  const std::vector<size_t> order = TopKOrder(scores, k);

  BeamResult out;
  for (const size_t j : order) {
    out.ids.push_back(candidate_ids[j]);
    out.scores.push_back(scores[j]);
    out.yaw_estimates.push_back(ShiftToYaw(shifts[j], Na_));
    out.best_shifts.push_back(shifts[j]);
  }
  return out;
}

RerankResult Matcher::Rerank(const std::vector<uint64_t> &query_token_id,
                             const std::vector<int64_t> &candidate_ids,
                             const std::vector<int> &candidate_shifts,
                             const RerankConfig &cfg, int topk_override,
                             double topk_pct_override) const {
  if (candidate_shifts.size() != candidate_ids.size()) {
    throw std::invalid_argument(
        "candidate_shifts must match candidate_ids length");
  }
  const bool jaccard = (cfg.scoring_mode == "jaccard4d");
  if (!jaccard && cfg.scoring_mode != "cosine4d") {
    throw std::invalid_argument(
        "scoring_mode must be \"jaccard4d\" | \"cosine4d\"");
  }
  const auto n_cands = static_cast<int64_t>(candidate_ids.size());
  const int k = ResolveTopK(topk_override, topk_pct_override, cfg.topk,
                            cfg.topk_pct, n_cands);
  const int tol = cfg.spatial_tol;

  // Query: 4-D hist + unpacked bins.
  const std::vector<float> q_hist =
      Build4dHist(query_token_id, Nh_, Nr_, Ns_, Na_);
  std::vector<int32_t> q_hb, q_rb, q_sb, q_ab;
  int q_max_hb = -1;
  for (const uint64_t tid : query_token_id) {
    const TokenBins b = UnpackToken(static_cast<int64_t>(tid), Nr_, Ns_, Na_);
    q_hb.push_back(static_cast<int32_t>(b.hb));
    q_rb.push_back(static_cast<int32_t>(b.rb));
    q_sb.push_back(static_cast<int32_t>(b.sb));
    q_ab.push_back(static_cast<int32_t>(b.ab));
    q_max_hb = std::max(q_max_hb, static_cast<int>(b.hb));
  }
  const auto Kq = static_cast<int64_t>(query_token_id.size());

  std::vector<size_t> cand_idx(candidate_ids.size());
  for (size_t i = 0; i < candidate_ids.size(); ++i) {
    cand_idx[i] = IndexOf(candidate_ids[i]);
  }

  std::vector<double> hist_scores(candidate_ids.size(), 0.0);
  std::vector<double> inlier_ratios(candidate_ids.size(), 0.0);
  std::vector<int> inlier_counts(candidate_ids.size(), 0);
  std::vector<double> combined(candidate_ids.size(), 0.0);

  const size_t Rs = static_cast<size_t>(Nr_) * Ns_;

#pragma omp parallel for schedule(dynamic, 2)
  for (int64_t i = 0; i < n_cands; ++i) {
    const size_t idx = cand_idx[static_cast<size_t>(i)];
    const size_t lo = offsets_[idx], hi = offsets_[idx + 1];
    const int shift = candidate_shifts[static_cast<size_t>(i)];
    const int db_max_hb = max_hbs_[idx];

    // --- Histogram score (db hist rolled by `shift` along azimuth:
    // output column a reads source column (a - shift) mod Na) ---
    const std::vector<uint64_t> db_tid(token_ids_.begin() + lo,
                                       token_ids_.begin() + hi);
    const std::vector<float> db_hist =
        Build4dHist(db_tid, Nh_, Nr_, Ns_, Na_);
    auto db_at = [&](size_t h, size_t rs, int a) -> float {
      const int src = ((a - shift) % Na_ + Na_) % Na_;
      return db_hist[(h * Rs + rs) * Na_ + static_cast<size_t>(src)];
    };

    const int ceiling = std::min(q_max_hb, db_max_hb);
    double h_score = 0.0;
    if (ceiling >= 0) {
      const auto h_top = static_cast<size_t>(std::min(ceiling, Nh_ - 1));
      if (jaccard) {
        int64_t n_both = 0, n_union = 0;
        for (size_t h = 0; h <= h_top; ++h) {
          for (size_t rs = 0; rs < Rs; ++rs) {
            for (int a = 0; a < Na_; ++a) {
              const bool qo =
                  q_hist[(h * Rs + rs) * Na_ + static_cast<size_t>(a)] > 0;
              const bool dbo = db_at(h, rs, a) > 0;
              n_both += (qo && dbo);
              n_union += (qo || dbo);
            }
          }
        }
        h_score = n_union == 0
                      ? 0.0
                      : static_cast<double>(n_both) / n_union;
      } else {
        // cosine4d: per-azimuth-column cosine over the (Nh*Rs) axis,
        // averaged over active columns.
        const double eps = cfg.eps;
        double total = 0.0;
        int n_active = 0;
        for (int a = 0; a < Na_; ++a) {
          double dot = 0.0, qq = 0.0, dd = 0.0;
          for (size_t h = 0; h <= h_top; ++h) {
            for (size_t rs = 0; rs < Rs; ++rs) {
              const double qv =
                  q_hist[(h * Rs + rs) * Na_ + static_cast<size_t>(a)];
              const double dv = db_at(h, rs, a);
              dot += qv * dv;
              qq += qv * qv;
              dd += dv * dv;
            }
          }
          const double qn = std::sqrt(qq) + eps;
          const double dn = std::sqrt(dd) + eps;
          if (qn > 2 * eps || dn > 2 * eps) {
            total += dot / (qn * dn);
            ++n_active;
          }
        }
        h_score = n_active == 0 ? 0.0 : total / n_active;
      }
    }
    hist_scores[static_cast<size_t>(i)] = h_score;

    // --- Token inlier score (rerank_key membership, shifted db_ab) ---
    std::vector<int64_t> db_keys;
    db_keys.reserve(hi - lo);
    for (size_t j = lo; j < hi; ++j) {
      const int ab_shifted = (static_cast<int>(ab_[j]) + shift) % Na_;
      db_keys.push_back(RerankKey(sb_[j], hb_[j], rb_[j],
                                  ((ab_shifted % Na_) + Na_) % Na_, Nh_, Nr_,
                                  Na_));
    }
    std::sort(db_keys.begin(), db_keys.end());
    db_keys.erase(std::unique(db_keys.begin(), db_keys.end()),
                  db_keys.end());

    int count = 0;
    if (!db_keys.empty()) {
      for (int64_t qi = 0; qi < Kq; ++qi) {
        const auto s = static_cast<size_t>(qi);
        bool found = false;
        for (int dh = -tol; dh <= tol && !found; ++dh) {
          const int hb2 = q_hb[s] + dh;
          if (hb2 < 0 || hb2 >= Nh_) continue;
          for (int dr = -tol; dr <= tol && !found; ++dr) {
            const int rb2 = q_rb[s] + dr;
            if (rb2 < 0 || rb2 >= Nr_) continue;
            for (int da = -tol; da <= tol && !found; ++da) {
              const int ab2 = ((q_ab[s] + da) % Na_ + Na_) % Na_;
              found = std::binary_search(
                  db_keys.begin(), db_keys.end(),
                  RerankKey(q_sb[s], hb2, rb2, ab2, Nh_, Nr_, Na_));
            }
          }
        }
        count += found;
      }
    }
    inlier_counts[static_cast<size_t>(i)] = count;
    inlier_ratios[static_cast<size_t>(i)] =
        static_cast<double>(count) / std::max<int64_t>(Kq, 1);
    combined[static_cast<size_t>(i)] =
        h_score * inlier_ratios[static_cast<size_t>(i)];
  }

  for (auto &s : combined) {
    if (s < cfg.score_threshold) {
      s = 0.0;
    }
  }
  const std::vector<size_t> order = TopKOrder(combined, k);

  RerankResult out;
  for (const size_t j : order) {
    out.ids.push_back(candidate_ids[j]);
    out.scores.push_back(combined[j]);
    out.hist_scores.push_back(hist_scores[j]);
    out.inlier_ratios.push_back(inlier_ratios[j]);
    out.inlier_counts.push_back(inlier_counts[j]);
    out.yaw_estimates.push_back(
        ShiftToYaw(candidate_shifts[j], Na_));
    out.best_shifts.push_back(candidate_shifts[j]);
  }
  return out;
}

}  // namespace inlier
