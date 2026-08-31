cmake_minimum_required(VERSION 3.16...3.31)

project(inlier_pybind VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(DEFINED SKBUILD)
  message(STATUS "Building with scikit-build-core")
endif()

# --- pybind11 (prefer the pip-installed package, fall back to FetchContent) ---
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE _pybind11_cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(_pybind11_cmakedir)
  list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmakedir}")
endif()
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.13.6)
  FetchContent_MakeAvailable(pybind11)
endif()


# --- Private binding module, installed inside the `inlier_legacy` package ---
pybind11_add_module(_inlier_pybind python/pybind/inlier_pybind.cpp)
target_compile_definitions(_inlier_pybind
                           PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _inlier_pybind DESTINATION inlier)
