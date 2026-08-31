#include "inlier_core/shape_pca.hpp"

#include <stdexcept>

namespace inlier {

ShapePcaResult ComputeShapePca(
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3>> &,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3>> &,
    double, int, int) {
// M2 stub: only reachable when N_s > 1, which M2's own tests never
// exercise. Real nanoflann+Eigen implementation lands in M3.
throw std::logic_error(
    "ComputeShapePca not yet implemented (M3) — use N_s=1 for now");
}

}  // namespace inlier