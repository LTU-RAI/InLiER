# Prefer a system Eigen3; fall back to FetchContent (header-only).
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  message(STATUS "inlier_core: system Eigen3 not found, fetching 3.4.0")
  include(FetchContent)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0)
  FetchContent_GetProperties(eigen)
  if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
  endif()
  add_library(eigen INTERFACE)
  add_library(Eigen3::Eigen ALIAS eigen)
  target_include_directories(eigen INTERFACE $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>)
endif()
