# Prefer a system nanoflann; fall back to FetchContent (single header).
find_package(nanoflann QUIET)
if(NOT TARGET nanoflann::nanoflann)
  message(STATUS "inlier_core: system nanoflann not found, fetching v1.6.2")
  include(FetchContent)
  FetchContent_Declare(
    nanoflann
    GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
    GIT_TAG v1.6.2)
  set(NANOFLANN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(NANOFLANN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(nanoflann)
endif()
