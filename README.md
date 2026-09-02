<div align="center">

# InLiER
## Learning-Free Heterogeneous LiDAR Place Recognition via Intermediate Mixed-Radix Structural Keypoint Tokenization

[![arXiv](https://img.shields.io/badge/Arxiv-2607.16862-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2607.16862)
[![DOI:10.1109/LRA.2026.3723737](https://img.shields.io/badge/IEEE-10.1109/LRA.2026.3723737-00629B.svg)](https://doi.org/10.1109/LRA.2026.3723737)
  <a href="https://www.youtube.com/watch?v=f73wsWx8vxg/"><img src="https://badges.aleen42.com/src/youtube.svg" alt="YouTube" /></a>

![Python](https://img.shields.io/badge/Python-00599C?logo=python&logoColor=ffdd54)
![C++](https://img.shields.io/badge/C++-00599C?logo=cplusplus&logoColor=white)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)

[**Nikolaos Stathoulopoulos**](https://github.com/nstathou) · [**George Nikolakopoulos**](https://github.com/geonikolak)

</div>

<p align=center>
  <img src="figures/InLiER-pipeline_git.png" alt="InLiER pipeline" width="65%"/>
</p>

## 💡 Introduction

**InLiER** (**In**termediate **Li**DAR **E**ncoding for **R**etrieval) is a **learning-free** place recognition pipeline for **heterogeneous** LiDAR sensors — spinning, solid-state, and FMCW units with different fields of view, resolutions, and scanning patterns. Instead of encoding a scan directly into a descriptor, InLiER inserts an **intermediate representation**: height-sliced structural keypoints are each mapped to a compact **mixed-radix token** that encodes height, radial distance, local shape, and azimuth. The same vocabulary is then re-organized on the fly across three retrieval stages, yielding fast rotation-invariant retrieval, yaw estimation and reranking, and full 6-DoF pose verification from a single sub-2KB representation.

<p align=center>
  <img src="figures/InLiER-keypoint-extraction_git.png" alt="Intermediate keypoint and token extraction" width="49.5%"/>
  <img src="figures/InLiER-keys_git.png" alt="Intermediate keypoint and token extraction" width="49.5%"/>
</p>

The pipeline re-organizes one token vocabulary across three stages:
- 🌿 <span style="color:green">**MINT**</span> *(Minimum-ceiling INTersection)* — height-ceiling histogram intersection for fast, rotation-invariant shortlisting.
- 💥 <span style="color:red">**BEAM**</span> *(Binary Elevation-Azimuth Matching)* — bitmask alignment for yaw estimation and reranking.
- ✔️ **Verify** — token-guided geometric verification for 6-DoF pose estimation.

## 📰 Latest News

- **[2026-09-01]** ⚡ The **C++ core** is out — the encoder, MINT/BEAM matcher, and token-guided verification are now C++17 with pybind11 bindings, behind the same Python API. Up to **39× faster verification** and **2.1× end-to-end** on the HeLiPR benchmark; see [C++ Core](#-c-core).
- **[2026-08-13]** 📄 The **published RA-L version** is out — IEEE Robotics and Automation Letters, vol. 11, no. 10, pp. 11275–11282, [10.1109/LRA.2026.3723737](https://doi.org/10.1109/LRA.2026.3723737).
- **[2026-07-21]** 🎉 **Preprint and code released** — the paper is on [arXiv](https://arxiv.org/abs/2607.16862) and the Python implementation is public.
- **[2026-07-18]** ✅ The paper is **accepted** to IEEE Robotics and Automation Letters (RA-L).

## 📋 Table of Contents

- [💡 Introduction](#-introduction)
- [🚀 Setup](#-setup) — [Prerequisites](#prerequisites) • [Installation](#installation) • [Environment Setup](#environment-setup) • [Verifying the Build](#verifying-the-build)
- [⚡ C++ Core](#-c-core) — [Backend Selection](#backend-selection) • [Benchmarks](#benchmarks) • [Equivalence Tests](#equivalence-tests)
- [🐍 Python API](#-python-api)
- [🖥️ Command Line](#️-command-line)
- [🕹️ Run the Example](#️-run-the-example) — [Dataset Setup](#dataset-setup) • [Building Overlap GT](#building-overlap-ground-truth) • [Validating Overlaps](#validate-calculated-overlaps) • [Configuration](#configuration) • [Evaluation](#running-the-evaluation) • [Visualization](#visualization)
- [🗃️ Test Your Own Data](#️-test-your-own-data) — [Dataset Layout](#dataset-layout) • [Overlap Ground Truth](#overlap-ground-truth) • [Evaluation](#evaluation)
- [🔜 Coming Soon](#-coming-soon) — [More Evaluation Protocols](#more-evaluation-protocols) • [ROS2 Support](#ros2-support)
- [🙏 Acknowledgements](#-acknowledgements)
- [📝 Citation](#-citation)
- [📬 Contact](#-contact)

## 🚀 Setup

### Prerequisites

- Python ≥ 3.10
- A **C++17 compiler** and **CMake ≥ 3.16** — the core is a C++ library with pybind11 bindings and is compiled during `pip install` (see [C++ Core](#-c-core)). On Ubuntu: 
  ```bash
  sudo apt install build-essential cmake
  ```
- The core library depends on **NumPy** and [`small_gicp`](https://github.com/koide3/small_gicp) (used for the GICP-based 6-DoF pose refinement). The evaluation and visualization tools additionally use `open3d`, `scipy`, `pyyaml`, `tqdm`, `matplotlib`, and `pandas` (installed via extras below).
- The C++ build depends on [Eigen](https://gitlab.com/libeigen/eigen) ≥ 3.3 and [nanoflann](https://github.com/jlblancoc/nanoflann), both header-only. CMake uses the system packages when they are installed, and otherwise clones them on the first build via `FetchContent` — which needs **git** and network access. Installing them up front keeps the build offline and a little faster:

  ```bash
  sudo apt install libeigen3-dev libnanoflann-dev
  ```

  **OpenMP** is optional — it is used for the parallel hot loops when found, and the same code path runs serially when it isn't.

### Installation

```bash
git clone https://github.com/LTU-RAI/InLiER.git
cd InLiER
```

### Environment Setup

#### Using Conda (Recommended)

```bash
conda create -n inlier python=3.10
conda activate inlier
pip install -e ".[eval]"
```

#### Using pip with venv

```bash
python -m venv inlier-env
source inlier-env/bin/activate
pip install -e ".[eval]"
```

The `[eval]` extra installs everything the evaluation workflow needs — overlap-GT building, the HeLiPR evaluation, and the playback visualization. Install just the core library (no evaluation scripts) with `pip install -e .`, and add `[test]` (`pip install -e ".[eval,test]"`) for the pytest suite.

The install builds the C++ core: the project uses [`scikit-build-core`](https://github.com/scikit-build/scikit-build-core) as its build backend, which drives CMake and puts the compiled `inlier._inlier_pybind` module inside the package. The first build takes a couple of minutes (CMake configure, plus fetching Eigen/nanoflann if they are not installed system-wide); build artifacts land in `build/`. Editable installs are configured with `editable.rebuild = true`, so edits under [`cpp/`](cpp) or [`python/pybind/`](python/pybind) are recompiled automatically the next time `inlier` is imported — no reinstall needed (importing then prints a short CMake rebuild line).

### Verifying the Build

```bash
python3 -c "import inlier; from inlier.core.InLiER import _BACKEND; print(inlier.__version__, _BACKEND)"
# 0.2.0 cpp
```

`cpp` means the compiled extension loaded. `python` means it could not be imported and the pure-numpy reference implementation is being used instead — a warning is printed at import time in that case, with the underlying `ImportError`.

## ⚡ C++ Core

The encoder, matcher, and verifier are implemented in C++17 under [`cpp/inlier_core/`](cpp/inlier_core) and exposed through pybind11 bindings in [`python/pybind/`](python/pybind). The Python `InLiER` and `InLiER_Matcher` classes are thin wrappers over that core — identical public API, dataclasses, and verbose output; only the hot loops moved. The original pure-numpy implementation is kept verbatim under [`inlier/core/reference/`](inlier/core/reference), where it serves as both the ground truth for the equivalence test-suite and the automatic fallback when the extension is unavailable.

### Backend Selection

The C++ backend is used whenever it is importable. Set `INLIER_FORCE_PYTHON=1` to force the pure-numpy reference instead — useful for debugging, for A/B checks, and for running without a compiler:

```bash
inlier eval cross-session --backend python ...
```

### Benchmarks

Full HeLiPR evaluation, Roundabout01 (Ouster, DB = 2705) ← Roundabout03 (Aeva, Q = 2774), encoding from scratch with the descriptor cache disabled, `config/default.yaml`:

| stage | c++ | python | speedup |
|---|---:|---:|---:|
| encoding / frame | 44.91 ms | 74.73 ms | 1.7× |
| MINT / query | 2.89 ms | 4.78 ms | 1.7× |
| BEAM / query | 34.67 ms | 86.58 ms | 2.5× |
| verify / query | 3.29 ms | 130.40 ms | 39.6× |
| **wall total** | **588.1 s** | **1252.7 s** | **2.1×** |

The full table lives in [`results/bench_cpp_vs_py/comparison_cpp_vs_py.md`](results/bench_cpp_vs_py/comparison_cpp_vs_py.md) (single core) and is regenerated by running the whole evaluation twice, once per backend:

```bash
inlier bench cpp-vs-py \
    --config config/default.yaml \
    --dataset /path/to/HeLiPR \
    --db_sequence Roundabout01 --q_sequence Roundabout03 --pair O-Aeva \
    --overlap_threshold 0.2 --max_pose_dist 10.0
```

### Equivalence Tests

[`tests/`](tests) pins the C++ core against the numpy reference stage by stage — plane fitting, keypoints, shape PCA, tokens, the MINT/BEAM/verify stages, and an end-to-end pass:

```bash
pip install -e ".[eval,test]"
pytest tests/ -v
```

A few matcher tests use a real cached descriptor set from `cache_inlier/` and are skipped if none is present — run the [evaluation](#running-the-evaluation) once to populate it.

## 🐍 Python API

```python
import numpy as np
from inlier import InLiER, InLiER_Matcher, InLiER_Config, VerifyConfig

encoder = InLiER(InLiER_Config())          # defaults match config/default.yaml
matcher = InLiER_Matcher(verify_config=VerifyConfig())

# Build a database: one entry per scan (points are (N, 3) float32, sensor frame).
# Verification needs the keypoints too, so keep them alongside the matcher.
db_keypoints, db_tokens = [], []
for i, scan in enumerate(database_scans):
    keypoints, tokens = encoder.encode(scan, verbose=False)
    db_keypoints.append(keypoints)
    db_tokens.append(tokens)
    matcher.add(i, tokens)
matcher.finalize()

# Query it
q_keypoints, q_tokens = encoder.encode(query_scan, verbose=False)

s1 = matcher.shortlist(q_tokens, topk=100)          # MINT  — rotation-invariant
s2 = matcher.beam_score(q_tokens, s1.ids, topk=20)  # BEAM  — yaw + reranking

best, shift = s2.ids[0], s2.best_shifts[0]
result = matcher.verify(                            # token-guided 6-DoF pose
    q_tokens, q_keypoints,
    db_tokens[best], db_keypoints[best],
    azimuth_shift=shift,                            # or config=VerifyConfig(...)
)
if result.success:
    print(result.T_sensor)   # p_db = T_sensor @ p_query
```

To load a configuration file instead of the dataclass defaults:

```python
from inlier.config import load, resolve

cfg = resolve(load("config/default.yaml"))
encoder = InLiER(cfg.inlier)
matcher = InLiER_Matcher(cfg.inlier, cfg.shortlist, cfg.beam)
```

Imports are lazy: `import inlier` alone loads neither the compiled extension nor
`small_gicp`, so setting `INLIER_FORCE_PYTHON=1` before the first attribute
access still selects the backend.

## 🖥️ Command Line

Installing the package puts an `inlier` command on your PATH:

```bash
inlier --help
```

| command | what it does |
|---|---|
| `inlier doctor` | check the backend, dependencies, dataset layout, and ground-truth consistency |
| `inlier config show` \| `dump` | print the effective configuration — after merging defaults, `--config`, and `--set` |
| `inlier encode` | run just the encoder on a scan or a directory, writing keypoints + tokens, and optionally plotting them |
| `inlier gt build` \| `validate` | build or sanity-check the overlap ground truth |
| `inlier eval cross-session` | run the evaluation protocol |
| `inlier play` | replay a finished run as an animation |
| `inlier bench cpp-vs-py` | time the C++ core against the numpy reference |

Common flags work on either side of the command name:

- `-c/--config FILE` — a YAML file, merged onto the packaged defaults
- `--set KEY=VALUE` — override one key, e.g. `--set stage1.topk=50` (repeatable)
- `--backend {auto,cpp,python}` — force a backend
- `-q/--quiet` — suppress the banner and per-stage progress

Configuration is layered: the packaged [`default.yaml`](inlier/config/default.yaml)
is always the base, your `--config` is merged on top, and `--set` overrides come
last. Unknown keys are rejected rather than silently ignored. `inlier config show`
prints exactly what will run, including the values derived from `voxel_size`:

```bash
inlier config show --set stage1.topk=50
```

`inlier doctor` checks `--dataset` against the layout `--dataset-type` names —
the same flag `inlier eval` takes, defaulting to `helipr`. The two layouts have
nothing in common, so point it at the right one:

```bash
inlier doctor --dataset /data/HeLiPR                          # <seq>/Undistorted/<sensor>/
inlier doctor --dataset /data/campus --dataset-type generic   # scans/ + poses_kitti.txt
```

A dataset checked against the wrong layout is reported as a layout mismatch
rather than as empty.

### Inspecting a descriptor

`inlier encode --viz` draws one page per scan: the cloud and its keypoints in
the ground-aligned frame the encoder bins in, then the three matrices the
matcher actually scores on — the token histogram `H`, the MINT row `R` that
stage 1 compares, and the BEAM elevation codes `A` that stage 2 shifts and
scores — plus the shape-class and height-slice distributions.

```bash
inlier encode scan.bin --viz                       # open a window
inlier encode scan.bin --viz-save scan.png         # write it instead
inlier encode scans/ -o tokens/ --viz-save figs/   # one figure per scan
```

`--viz-save` forces a headless backend, so it works over ssh and in CI. It is
required when the input is a directory, since `--viz` alone would open a window
per scan. `-o/--output` is optional when you only want the figure.

This is the quickest way to tell whether a token grid suits a new sensor: if
the top height slices are empty, `z_max` is too high for the platform; if the
BEAM panel is nearly all one colour, `N_r`/`N_a` are too coarse to separate
structure.

### Encoding submaps

A bare scan path encodes one scan. The evaluation encodes accumulated
**submaps** whenever `--n-db`/`--n-q` are above 1, so what you inspect should
be built the same way — `--dataset` does that, merging each window of scans
into its keyframe's frame using the poses:

```bash
inlier encode --dataset /data/campus --n-scans 10 --index 0 --viz
inlier encode --dataset /data/campus --n-scans 10 --stride 5 --range 0:20 -o submaps/
```

Only the scans the selected submaps need are read, so inspecting one submap out
of several hundred is cheap. `--index` accepts negatives (`-1` is the last
submap) and resolves them before they reach filenames or provenance.

`--n-scans` and `--stride` **must match what the overlap ground truth was built
with** — the matrix is indexed by submap. The window rule is shared with the
overlap builder ([`inlier/eval/submaps.py`](inlier/eval/submaps.py)) so the two
cannot drift, and the values are written into the `.npz` alongside the token
radices.

> HeLiPR has no submap accumulation: it is evaluated scan by scan
> (`HeLiPRSource` carries no `n_scans`/`stride`, and the published results are
> one submap per scan). Point `inlier encode` straight at a `.bin`.

> **Migrating from 0.2.x** — the scripts under `evaluation/` are now thin shims
> that print the equivalent `inlier` command and forward. They are removed in
> 0.4.0. Flag names are unchanged; both `--snake_case` and `--kebab-case`
> spellings are accepted.

## 🕹️ Run the Example

This walks through reproducing our HeLiPR results on the Roundabout01 (Ouster OS2-128, database) ← Roundabout03 (Aeva Aeries II, query) pair — a heterogeneous, spinning-vs-solid-state setup. The full pipeline runs in three steps: build the overlap ground truth, run the evaluation, then (optionally) replay the results. We provide the precomputed overlap matrix for this pair under [`overlap_matrices/`](overlap_matrices), so you can skip straight to [Running the Evaluation](#running-the-evaluation) if you don't want to regenerate it.

### Dataset Setup

InLiER is evaluated on the [**HeLiPR**](https://sites.google.com/view/heliprdataset) benchmark. The loader expects the following layout under the dataset root:

```text
HeLiPR/
└── <Sequence>/                          # e.g. Roundabout01, Roundabout03, ...
    ├── LiDAR/                            # raw scans, as distributed by HeLiPR
    │   └── <Sensor>/                     # Aeva | Avia | Ouster | Velodyne
    │       └── *.bin
    ├── LiDAR_GT/
    │   └── global_<Sensor>_gt.txt        # ground-truth poses (t x y z qx qy qz qw)
    └── Undistorted/                      # generated — see below
        └── <Sensor>/
            └── *.bin
```

The raw HeLiPR scans are motion-distorted. All evaluation scripts ([`inlier gt build`](inlier/eval/overlap_build.py), [`inlier eval cross-session`](inlier/eval/protocols/cross_session.py)) read from `Undistorted/`, not `LiDAR/`, so it must be populated before running anything. Undistort and accumulate the raw scans with the [**HeLiPR-Pointcloud-Toolbox**](https://github.com/minwoo0611/HeLiPR-Pointcloud-Toolbox), using [`config/HeLiPR-Toolbox/config.yaml`](config/HeLiPR-Toolbox/config.yaml) as a starting point — it holds the exact settings we used (frame accumulation, downsampling, cropping) to generate the scans in our results. Adjust `Path.binPath` / `Path.trajPath` / `Path.savePath` per sequence and sensor, then point `savePath` at the corresponding `<Sequence>/Undistorted/<Sensor>/` folder above.

### Building Overlap Ground Truth

Precompute the pairwise scan-overlap matrices used to label true/false positives:

```bash
inlier gt build \
    --dataset_type helipr \
    --dataset /path/to/HeLiPR \
    --db_sequence Roundabout01 --q_sequence Roundabout03 \
    --pairs O-Aeva \
    --output_dir overlap_matrices \
    --voxel_size 0.5 --distance_threshold 100
```

Alongside the matrix this writes a small `overlap_*.json` sidecar recording the
parameters it was built with (`n_db`, `n_q`, `stride_db`, `stride_q`, voxel size,
thresholds). The evaluation reads it back and **refuses to run** if they disagree
with what it is about to assume — see [Overlap Ground Truth](#overlap-ground-truth).

- `--pairs` is `<DB sensor>-<Q sensor>`; here `O-Aeva` means the Roundabout01 **Ouster** scans are the database and the Roundabout03 **Aeva** scans are the query.
- `--voxel_size` is the voxel size δ (m) used when computing per-voxel overlap between a DB/Q scan pair — smaller values are stricter (more voxels must actually coincide).
- `--distance_threshold` caps the pose-to-pose distance (m) beyond which a DB/Q pair is assumed non-overlapping and skipped, without spending time voxelizing it.

### Validate Calculated Overlaps

Sanity-check a precomputed overlap matrix before trusting it as GT with [`inlier gt validate`](inlier/eval/overlap_validate.py):

```bash
inlier gt validate \
    --dataset_type helipr \
    --dataset /path/to/HeLiPR \
    --db_sequence Roundabout01 --q_sequence Roundabout03 \
    --pair O-Aeva \
    --overlap_dir overlap_matrices \
    --voxel_size 0.5 --pose_dist_threshold 10.0 --overlap_threshold 0.2
```

It re-loads the DB/Q poses and scans (same range-filter → global-frame → voxelize pipeline as [`inlier gt build`](inlier/eval/overlap_build.py)) and opens a figure with: (a) DB/Q trajectories in 3D with edges drawn between scan pairs above `--overlap_threshold` and within `--pose_dist_threshold`, colored by overlap value; (b) summary statistics (non-zero entries, entries above threshold, max/mean overlap); (c) a histogram of non-zero overlap values; and (d) top-down aligned views of one randomly picked high-overlap and one low-overlap scan pair. Pass `--seed` to fix which example pair is shown.

<p align=center>
  <img src="figures/overlaps_example.png" alt="Overlap validation figure for Roundabout01 (Ouster) vs Roundabout03 (Aeva)" width="90%"/>
</p>

### Configuration

Encoder, retrieval, verification, and refinement parameters live in YAML files under [`config/`](config). Start from [`config/default.yaml`](config/default.yaml) and pass it with `--config`.

The same file ships inside the package as [`inlier/config/default.yaml`](inlier/config/default.yaml) and is **always** the base: your `--config` is deep-merged onto it, so a partial file only needs the keys you actually want to change. Unknown keys are rejected rather than silently ignored, and `inlier config show` prints the merged result.

The `encoder:` block defines the descriptor itself — every point that survives the crop is reduced to a token `((hb·N_r + rb)·N_s + sb)·N_a + ab`, so the four `N_*` values set the descriptor's resolution and its vocabulary size (`N_h · N_r · N_s · N_a`).

| Parameter | Description | Default |
|-----------|-------------|---------|
| `voxel_size` *(top-level)* | Voxel size (m) used to downsample the submap **before** encoding. Only affects the encoder input — GICP does its own downsampling (`gicp.downsampling_resolution`). | 0.5 |
| `N_h` | Number of height slices between `z_min` and `z_max`. Sets the `hb` token field. | 10 |
| `z_min` / `z_max` | Height band (m, above the estimated ground plane) kept for encoding; points outside are dropped. Slice thickness is `(z_max − z_min) / N_h`. | 0.0 / 20.0 |
| `N_r` | Radial bins over `[0, min(r_max, xy_max)]`. Sets the `rb` field. | 20 |
| `N_a` | Azimuth bins over the full 360°, i.e. 6° per bin at the default. Sets the `ab` field and the shift resolution BEAM searches over for yaw. | 60 |
| `N_s` | Number of PCA shape classes (linear / planar / scattered mixtures from the local eigenvalue spread). Sets the `sb` field; `N_s: 1` disables shape and collapses the field. | 7 |
| `r_max` | Max radius (m) for the radial/azimuth binning. Effectively clamped to `xy_max`. | 100.0 |
| `xy_max` | XY half-extent (m) of the crop and of the BEV height-cell grid — points with \|x\| or \|y\| beyond it are discarded. | 100.0 |
| `cell_size` | BEV cell size (m) of the per-slice height image used for keypoint extraction. Keep it ≈ `2 × voxel_size` so cells are populated but not oversmoothed. | 1.0 |
| `window` | Side length (odd) of the non-maximum-suppression window applied to each slice's height image when picking local maxima as keypoints. Larger = fewer, more spread-out keypoints. | 3 |


### Running the Evaluation

```bash
inlier eval cross-session \
    --config config/default.yaml \
    --dataset /path/to/HeLiPR \
    --db_sequence Roundabout01 --q_sequence Roundabout03 \
    --pair O-Aeva \
    --overlap_dir overlap_matrices \
    --output_dir results/HeLiPR \
    --overlap_threshold 0.2 --max_pose_dist 10.0
```

This writes the Recall/PR-AUC metrics (`results_*.json`), the loop-closure candidates (`candidates_*.csv`), per-pair verify poses (`per_pair_verify_*.csv`), the descriptor caches (`cache_inlier/desc_*.npz`), and a trajectory plot (`trajectory_*.png`) to the output folder. For datasets other than HeLiPR, see [Test Your Own Data](#️-test-your-own-data) below.

### Visualization

Replay a `DB←Q` run as an animation of the growing query trajectory, loop closures, matched keypoints, and the MINT/BEAM descriptors — driven entirely by the evaluation artifacts above:

```bash
inlier play \
    --dataset /path/to/HeLiPR \
    --output_dir results/HeLiPR/dbR01-O-qR03-Aeva_vs0.5_cs1_nh10_nr20_na60_ns7 \
    --cache_dir cache_inlier
```

The run's identity (sequences, sensors, GT thresholds, score threshold) is read from the `results_*.json` in `--output_dir` — no need to repeat it on the command line.

Controls: `SPACE` play/pause, `←` / `→` step. Pass `--record out.mp4` to render headlessly.

<p align=center>
  <img src="figures/helipr.gif" alt="InLiER loop-closure playback" width="90%"/>
</p>

## 🗃️ Test Your Own Data

InLiER isn't tied to the HeLiPR loader — [`inlier eval cross-session --dataset_type generic`](inlier/eval/protocols/cross_session.py) runs the identical retrieval/evaluation pipeline on any folder-based dataset, via [`datasets/generic.py`](inlier/eval/datasets/generic.py).

### Dataset Layout

Each of your database and query sequences is its own folder:

```text
<db_path>/ (and <q_path>/, same layout)
├── scans/
│   ├── 000000.pcd
│   ├── 000001.pcd
│   └── ...
├── poses_kitti.txt          # preferred: 12 floats per line (row-major 3x4), 1:1 with scans
├── poses_tum.txt            # alternative: "#timestamp x y z qx qy qz qw"
└── transform.txt            # optional: 4x4, maps DB world frame → Q world frame
```

**`transform.txt` — when you need it.** InLiER compares database and query poses directly (to build the overlap GT and to filter candidates by `--max_pose_dist`), so both sequences must live in the *same* world frame. If they already do — e.g. both were mapped in one session, or registered to a shared global frame — no transform is needed. If they were mapped independently, each sequence's poses start at its own arbitrary origin, and the DB/Q pose distances would be meaningless. `transform.txt` is the 4×4 matrix that maps the **DB world frame into the Q world frame**, and it's applied to the DB keyframe poses before any distance or overlap computation.

The evaluation auto-detects `<db_path>/transform.txt` if it exists; pass `--transform <path>` to point elsewhere, or `--no_transform` to force the shared-frame assumption even when the file is present. Use the same choice for both [`inlier gt build`](inlier/eval/overlap_build.py) and [`inlier eval cross-session --dataset_type generic`](inlier/eval/protocols/cross_session.py) — a mismatch silently produces a wrong GT matrix.

**Pre-accumulated vs. single scans.** `scans/` can hold either. If your `.pcd` files are already accumulated submaps (as with the HeLiPR toolbox output), run with `--n_db 1 --n_q 1` and each file is used as-is. If they're single sensor scans, let the evaluation accumulate them: `--n_db` / `--n_q` set how many consecutive scans form one submap, each window anchored at its first scan (the keyframe) with the rest transformed into that keyframe's pose via `inv(T_i) @ T_k`. `--stride_db` / `--stride_q` set the step between consecutive submaps and default to `n_db` / `n_q`, i.e. non-overlapping submaps; a stride of 1 gives maximally overlapping ones. Either way there must be exactly one pose per `.pcd` file — the loader errors out on a count mismatch.

Sparse single scans (solid-state, or low-resolution spinning units) generally need accumulation for the height-slice keypoints to be stable.

### Overlap Ground Truth

[`inlier gt build`](inlier/eval/overlap_build.py) also supports generic data — pass `--dataset_type generic --db_path ... --q_path ...` (same `.pcd` + `poses_kitti.txt` layout, plus the DB→Q `--transform`) to compute the pairwise overlap matrix, just as in [Building Overlap Ground Truth](#building-overlap-ground-truth) for HeLiPR.

With pre-accumulated scans, the defaults (`--n_db 1 --n_q 1`) use each `.pcd` as-is:

```bash
inlier gt build \
    --dataset_type generic \
    --db_path /path/to/database \
    --q_path  /path/to/query \
    --transform /path/to/transform.txt \
    --output_dir overlap_matrices \
    --voxel_size 0.5 --distance_threshold 100
```

With single scans, accumulate them into submaps — e.g. 10 scans per submap, stepping one scan at a time:

```bash
inlier gt build \
    --dataset_type generic \
    --db_path /path/to/database \
    --q_path  /path/to/query \
    --transform /path/to/transform.txt \
    --output_dir overlap_matrices \
    --n_db 10 --n_q 10 --stride_db 1 --stride_q 1 \
    --voxel_size 0.5 --distance_threshold 100
```

A stride of 1 keeps one submap per scan, so the overlap matrix stays at full resolution (`M_db × M_q` with `M ≈ number of scans`) at the cost of a longer build. Leaving `--stride_*` out defaults it to `n`, giving non-overlapping submaps and a ~10× smaller matrix. `--n_db` / `--n_q` / `--stride_db` / `--stride_q` **must match what you later pass to** [`inlier eval cross-session --dataset_type generic`](inlier/eval/protocols/cross_session.py) — the matrix is indexed by submap, so any difference misaligns the GT against the retrieval results.

### Evaluation

```bash
inlier eval cross-session --dataset_type generic \
    --config config/default.yaml \
    --db_path /path/to/database \
    --q_path  /path/to/query \
    --transform /path/to/transform.txt \
    --overlap_file /path/to/overlap.txt \
    --overlap_threshold 0.2 --max_pose_dist 25.0 \
    --n_db 10 --n_q 10 --stride_db 1 --stride_q 1 \
    --output_dir results/generic_dataset
```

`--transform` defaults to `<db_path>/transform.txt` if present, and `--no_transform` disables it when both sequences already share a world frame. Outputs match the HeLiPR driver (`results_*.json`, `candidates_*.csv`, descriptor caches, trajectory plot) under `--output_dir`.

## 🔜 Coming Soon

### More Evaluation Protocols

Place recognition is evaluated under several incompatible protocols, and `inlier eval` currently implements one of them. Coming next:

- 🔁 **`online-lcd`** — single-session online loop closure detection. The database grows as the query streams and candidates are restricted to frames older than an exclusion window, reported with the SLAM convention (F1max, max-recall at 100% precision) rather than Recall@N.
- 📍 **`online-global`** — online localization against a fixed prior map, deciding at a *fixed* threshold with no post-hoc selection, and reporting first-fix latency and per-frame cost.
- 🗺️ **`multi-session`** — N sessions all-vs-all, aggregated into one benchmark table.
- ▶️ **`inlier run`** — produce loop closures and 6-DoF poses on data with **no ground truth**, which is what a deployment actually has.

### ROS2 Support

- 🤖 We are also planning to release ROS2 nodes to support front-end agnostic loop closures, including a GTSAM based back-end optimization.

## 🙏 Acknowledgements

We thank the authors of [**HeLiPR**](https://sites.google.com/view/heliprdataset) and [**HeLiOS**](https://github.com/minwoo0611/HeLiOS) for their open dataset and tooling, which form the base of our benchmarking ( [**HeLiPR-Pointcloud-Toolbox**](https://github.com/minwoo0611/HeLiPR-Pointcloud-Toolbox), [**HeLiPR-Place-Recognition**](https://github.com/minwoo0611/HeLiPR-Place-Recognition)).

We also thank Koide *et al.* ([@koide3](https://github.com/koide3)) for [`small_gicp`](https://github.com/koide3/small_gicp), which is used throughout our evaluation pipeline and the 6-DoF refinement step.

## 📝 Citation

If you find **InLiER** useful in your research, please consider citing:

```bibtex
@article{stathoulopoulos2026inlier,
  author={Stathoulopoulos, Nikolaos and Nikolakopoulos, George},
  journal={IEEE Robotics and Automation Letters}, 
  title={{InLiER: Learning-Free Heterogeneous LiDAR Place Recognition via Intermediate Mixed-Radix Structural Keypoint Tokenization}}, 
  year={2026},
  volume={11},
  number={10},
  pages={11275-11282},
  doi={10.1109/LRA.2026.3723737}}
}
```

## 📬 Contact

For questions, issues, or collaboration inquiries, feel free to reach out to [niksta@ltu.se](mailto:niksta@ltu.se) or open a git issue [here](https://github.com/LTU-RAI/InLiER/issues).
