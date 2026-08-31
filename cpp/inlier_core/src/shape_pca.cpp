#include "inlier_core/shape_pca.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <nanoflann.hpp>
#include <vector>

namespace inlier {

namespace {

/// nanoflann adaptor over an Eigen (N,3) matrix.
struct CloudAdaptor {
  const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3>> &pts;

  size_t kdtree_get_point_count() const {
    return static_cast<size_t>(pts.rows());
  }
  double kdtree_get_pt(size_t idx, size_t dim) const {
    return pts(static_cast<int64_t>(idx), static_cast<int64_t>(dim));
  }
  template <class BBox>
  bool kdtree_get_bbox(BBox &) const {
    return false;
  }
};

using KdTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, CloudAdaptor>, CloudAdaptor, 3>;

}  // namespace

ShapePcaResult ComputeShapePca(
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3>> &points,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3>> &centers,
    double radius, int min_neighbors, int n_classes) {
  const int64_t N = points.rows();
  const int64_t K = centers.rows();

  // n_classes 3/5/7 -> inclination sub-bins 0/2/3, scatter id 2/4/6.
  int n_incl_bins, scatter_id;
  if (n_classes <= 3) {
    n_incl_bins = 0;
    scatter_id = 2;
  } else if (n_classes <= 5) {
    n_incl_bins = 2;
    scatter_id = 4;
  } else {
    n_incl_bins = 3;
    scatter_id = 6;
  }

  ShapePcaResult res;
  res.shape_class.assign(static_cast<size_t>(K),
                         static_cast<int16_t>(scatter_id));
  res.lps = Eigen::Matrix<float, Eigen::Dynamic, 3>::Zero(K, 3);
  if (K == 0 || N == 0) {
    return res;
  }

  CloudAdaptor adaptor{points};
  KdTree tree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));

  // scipy query_ball_point is d <= r (inclusive); nanoflann's radius
  // test is strict, so nudge the squared radius up one ulp.
  const double r_sq = std::nextafter(
      radius * radius, std::numeric_limits<double>::infinity());

#pragma omp parallel
  {
    std::vector<nanoflann::ResultItem<uint32_t, double>> matches;
    nanoflann::SearchParameters params;
    params.sorted = false;

#pragma omp for schedule(dynamic, 64)
    for (int64_t k = 0; k < K; ++k) {
      const double query[3] = {centers(k, 0), centers(k, 1), centers(k, 2)};
      matches.clear();
      const size_t n_found = tree.radiusSearch(query, r_sq, matches, params);
      const auto n = static_cast<int64_t>(n_found);

      if (n < min_neighbors) {
        // invalid: scatter class, lps = [0, 0, 1]
        res.lps(k, 2) = 1.0f;
        continue;
      }

      // Mean and covariance (normalised by max(n-1, 1), matching numpy).
      Eigen::Vector3d mean = Eigen::Vector3d::Zero();
      for (const auto &mt : matches) {
        mean += points.row(static_cast<int64_t>(mt.first)).transpose();
      }
      mean /= static_cast<double>(n);

      Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
      for (const auto &mt : matches) {
        const Eigen::Vector3d c =
            points.row(static_cast<int64_t>(mt.first)).transpose() - mean;
        cov.noalias() += c * c.transpose();
      }
      cov /= static_cast<double>(std::max<int64_t>(n - 1, 1));

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
      // Eigen: ascending -> descending lambda1 >= lambda2 >= lambda3.
      const double l1 = std::max(es.eigenvalues()(2), 1e-12);
      const double l2 = std::max(es.eigenvalues()(1), 0.0);
      const double l3 = std::max(es.eigenvalues()(0), 0.0);

      const auto lin = static_cast<float>((l1 - l2) / l1);
      const auto pla = static_cast<float>((l2 - l3) / l1);
      const auto sca = static_cast<float>(l3 / l1);
      res.lps(k, 0) = lin;
      res.lps(k, 1) = pla;
      res.lps(k, 2) = sca;

      // argmax with first-max tie-break (np.argmax semantics), on float32.
      int base_type = 0;
      float best = lin;
      if (pla > best) {
        base_type = 1;
        best = pla;
      }
      if (sca > best) {
        base_type = 2;
      }

      if (n_incl_bins == 0) {
        res.shape_class[static_cast<size_t>(k)] =
            static_cast<int16_t>(base_type);
        continue;
      }

      if (base_type == 2) {
        continue;  // stays scatter_id
      }

      // Inclination of the relevant eigenvector w.r.t. +Z:
      // direction = largest-eigenvalue vector (linear), normal =
      // smallest (planar). Sign-invariant via |z|.
      const Eigen::Vector3d evec =
          (base_type == 0) ? es.eigenvectors().col(2)   // direction
                           : es.eigenvectors().col(0);  // normal
      const double norm = evec.norm() + 1e-12;
      const double cosang = std::clamp(std::abs(evec.z()) / norm, 0.0, 1.0);
      const double incl_deg = std::acos(cosang) * (180.0 / M_PI);
      const double step = 90.0 / n_incl_bins;
      const int bin = std::clamp(static_cast<int>(incl_deg / step), 0,
                                 n_incl_bins - 1);
      res.shape_class[static_cast<size_t>(k)] = static_cast<int16_t>(
          base_type == 0 ? bin : n_incl_bins + bin);
    }
  }

  return res;
}

}  // namespace inlier
