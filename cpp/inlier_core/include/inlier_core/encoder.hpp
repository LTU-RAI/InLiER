// InLiER encoder: keypoint extraction + tokenization (Python: InLiER).
#pragma once

#include <Eigen/Core>
#include <optional>
#include <utility>

#include "inlier_core/config.hpp"
#include "inlier_core/types.hpp"

namespace inlier {

/// Point set used for tokenization. Replaces the Python
/// tokenize_keypoints() config-mutation hack with an explicit argument.
enum class TokenizeMode {
  kFromConfig,  // follow cfg.point_mode
  kKeypoints,
  kAllPoints,
};

class Encoder {
 public:
  using PointMatrix = Eigen::Matrix<double, Eigen::Dynamic, 3>;

  explicit Encoder(InLiERConfig config);

  const InLiERConfig &config() const { return config_; }

  /// Full pipeline: extract keypoints then tokenize, sharing one plane.
  std::pair<Keypoints, Tokens> Encode(
      const Eigen::Ref<const PointMatrix> &points,
      const std::optional<Plane> &plane = std::nullopt) const;

  Keypoints ExtractKeypoints(
      const Eigen::Ref<const PointMatrix> &points,
      const std::optional<Plane> &plane = std::nullopt) const;

  Tokens Tokenize(const Eigen::Ref<const PointMatrix> &points,
                  const Keypoints &keypoints,
                  const std::optional<Plane> &plane = std::nullopt,
                  TokenizeMode mode = TokenizeMode::kFromConfig) const;

 private:
  InLiERConfig config_;
};

}  // namespace inlier
