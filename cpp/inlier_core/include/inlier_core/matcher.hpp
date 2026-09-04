// Multi-stage token matcher: DB bookkeeping + MINT / BEAM / rerank
// (Python: InLiER_Matcher minus verify/GICP/KISS).
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "inlier_core/config.hpp"
#include "inlier_core/types.hpp"

namespace inlier {

/// Per-scan token views into the CSR storage (Python: get_scan_data).
struct ScanData {
  std::vector<uint64_t> token_id;
  std::vector<int16_t> hb, rb, sb, ab;
  int32_t max_active_hb = -1;
};

class Matcher {
 public:
  explicit Matcher(InLiERConfig config);

  /// Register one database scan (unpacks token_id once).
  void Add(int64_t database_id, const std::vector<uint64_t> &token_id);
  void Reset();
  /// Pre-allocate room for `n` scans so per-frame latency stays flat
  /// instead of spiking on a vector realloc (online protocols).
  void Reserve(size_t n);
  /// Stack the per-scan histograms into the dense HM matrix. Append-only
  /// and idempotent: rows already built are kept, only scans added since
  /// the last call are filled in.
  void Finalize();
  size_t size() const { return db_ids_.size(); }
  bool finalized() const { return finalized_; }
  const std::vector<int64_t> &db_ids() const { return db_ids_; }
  ScanData GetScanData(int64_t database_id) const;

  /// topk_override >= 0 / topk_pct_override >= 0 replicate the call-time
  /// arguments of the Python methods (which take precedence over config).
  /// max_db_index >= 0 restricts the search to the first max_db_index
  /// scans in insertion order (exclusive bound), so causal exclusion
  /// happens inside the scoring loop rather than by discarding results
  /// afterwards -- which would silently cost recall whenever the excluded
  /// frames dominate the top-k.
  ShortlistResult Shortlist(const std::vector<uint64_t> &query_token_id,
                            const ShortlistConfig &cfg, int topk_override,
                            double topk_pct_override,
                            int64_t max_db_index = -1);

  BeamResult BeamScore(const std::vector<uint64_t> &query_token_id,
                       const std::vector<int64_t> &candidate_ids,
                       const BEAMScoreConfig &cfg, int topk_override,
                       double topk_pct_override) const;

  RerankResult Rerank(const std::vector<uint64_t> &query_token_id,
                      const std::vector<int64_t> &candidate_ids,
                      const std::vector<int> &candidate_shifts,
                      const RerankConfig &cfg, int topk_override,
                      double topk_pct_override) const;

 private:
  size_t IndexOf(int64_t database_id) const;

  InLiERConfig config_;
  int Nh_, Nr_, Ns_, Na_;
  bool finalized_ = false;

  // CSR per-scan token storage (unpacked once in Add).
  std::vector<int64_t> db_ids_;
  std::unordered_map<int64_t, size_t> id_to_index_;
  std::vector<size_t> offsets_{0};
  std::vector<uint64_t> token_ids_;
  std::vector<int16_t> hb_, rb_, sb_, ab_;
  std::vector<int32_t> max_hbs_;

  // Finalized dense histogram matrix (N, N_h, N_r*N_s), row-major flat.
  std::vector<float> hm_;
};

}  // namespace inlier
