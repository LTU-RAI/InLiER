# inlier_core — pure C++ static library (no Python includes).
include(${CMAKE_CURRENT_LIST_DIR}/cmake/eigen.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/nanoflann.cmake)

add_library(inlier_core STATIC
  src/plane.cpp
  src/encoder.cpp
  src/shape_pca.cpp
  src/matcher.cpp
  src/verify.cpp)

target_include_directories(inlier_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(inlier_core PUBLIC Eigen3::Eigen nanoflann::nanoflann)

target_compile_options(inlier_core PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -Wall -Wextra>)

# OpenMP is optional: the hot loops fall back to serial execution.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(inlier_core PUBLIC OpenMP::OpenMP_CXX)
  message(STATUS "inlier_core: OpenMP enabled")
else()
  message(STATUS "inlier_core: OpenMP not found, building serial")
endif()
