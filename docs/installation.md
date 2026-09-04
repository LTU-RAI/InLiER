# 🚀 Installation

← back to the [README](../README.md)

## Prerequisites

- Python ≥ 3.10
- A **C++17 compiler** and **CMake ≥ 3.16** — the core is a C++ library with pybind11 bindings and is compiled during `pip install` (see [C++ Core](cpp-core.md)). On Ubuntu: 
  ```bash
  sudo apt install build-essential cmake
  ```
- The core library depends on **NumPy** and [`small_gicp`](https://github.com/koide3/small_gicp) (used for the GICP-based 6-DoF pose refinement). The evaluation and visualization tools additionally use `open3d`, `scipy`, `pyyaml`, `tqdm`, `matplotlib`, and `pandas` (installed via extras below). The live viewer for `inlier run --live` uses [`pyridescence`](https://github.com/koide3/iridescence), which has its own extra because it needs an OpenGL display.
- The C++ build depends on [Eigen](https://gitlab.com/libeigen/eigen) ≥ 3.3 and [nanoflann](https://github.com/jlblancoc/nanoflann), both header-only. CMake uses the system packages when they are installed, and otherwise clones them on the first build via `FetchContent` — which needs **git** and network access. Installing them up front keeps the build offline and a little faster:

  ```bash
  sudo apt install libeigen3-dev libnanoflann-dev
  ```

  **OpenMP** is optional — it is used for the parallel hot loops when found, and the same code path runs serially when it isn't.

## Getting the Source

```bash
git clone https://github.com/LTU-RAI/InLiER.git
cd InLiER
```

## Environment Setup

### Using Conda (Recommended)

```bash
conda create -n inlier python=3.10
conda activate inlier
pip install -e ".[eval]"
```

### Using pip with venv

```bash
python -m venv inlier-env
source inlier-env/bin/activate
pip install -e ".[eval]"
```

The `[eval]` extra installs everything the evaluation workflow needs — overlap-GT building, the HeLiPR evaluation, and the playback visualization. Install just the core library (no evaluation scripts) with `pip install -e .`, and add `[test]` (`pip install -e ".[eval,test]"`) for the pytest suite.

`[viz]` is separate: it installs `pyridescence` for the live 3D viewer behind [`inlier run --live`](cli.md#watching-it-run----live), and nothing else needs it. For a headless evaluation you do not need to pull in the OpenGL stack. Add it when you want the viewer:

```bash
pip install -e ".[eval,viz]"
```

`inlier doctor` reports it as a warning rather than a failure when it is absent.

The install builds the C++ core: the project uses [`scikit-build-core`](https://github.com/scikit-build/scikit-build-core) as its build backend, which drives CMake and puts the compiled `inlier._inlier_pybind` module inside the package. The first build takes a couple of minutes (CMake configure, plus fetching Eigen/nanoflann if they are not installed system-wide); build artifacts land in `build/`. Editable installs are configured with `editable.rebuild = true`, so edits under [`cpp/`](../cpp) or [`python/pybind/`](../python/pybind) are recompiled automatically the next time `inlier` is imported — no reinstall needed (importing then prints a short CMake rebuild line).

## Verifying the Build

```bash
python3 -c "import inlier; from inlier.core.InLiER import _BACKEND; print(inlier.__version__, _BACKEND)"
# 1.1.0 cpp
```

`cpp` means the compiled extension loaded. `python` means it could not be imported and the pure-numpy reference implementation is being used instead — a warning is printed at import time in that case, with the underlying `ImportError`.

## Next Steps

- [Python API](python-api.md) — use InLiER as a library.
- [Command Line](cli.md) — the `inlier` command.
- [HeLiPR Benchmark](helipr-benchmark.md) — reproduce the paper's results.
