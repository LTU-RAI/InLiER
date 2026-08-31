 // PCA shape classification (Python: InLiER._compute_shape_pca).
  #pragma once

  #include <Eigen/Core>
  #include <cstdint>
  #include <vector>

  namespace inlier {
  
  struct ShapePcaResult {
    std::vector<int16_t> shape_class;               // (K,)
    Eigen::Matrix<float, Eigen::Dynamic, 3> lps;    // (K,3) linearity/planarity/scatter
  };

  /// Classify each center by the PCA of its radius-neighbourhood in the
  /// full cloud. n_classes in {3, 5, 7} controls inclination sub-bins.
  ShapePcaResult ComputeShapePca(
      const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3>> &points,
      const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3>> &centers,
      double radius, int min_neighbors, int n_classes);
 
  }  // namespace inlier