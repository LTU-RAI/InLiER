# 🕹️ HeLiPR Benchmark

← back to the [README](../README.md)

This walks through reproducing our HeLiPR results on the Roundabout01 (Ouster OS2-128, database) ← Roundabout03 (Aeva Aeries II, query) pair — a heterogeneous, spinning-vs-solid-state setup. The full pipeline runs in three steps: build the overlap ground truth, run the evaluation, then (optionally) replay the results. We provide the precomputed overlap matrix for this pair under [`overlap_matrices/`](../overlap_matrices), so you can skip straight to [Running the Evaluation](#running-the-evaluation) if you don't want to regenerate it.

## Dataset Setup

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

The raw HeLiPR scans are motion-distorted. All evaluation scripts ([`inlier gt build`](../inlier/eval/overlap_build.py), [`inlier eval cross-session`](../inlier/eval/protocols/cross_session.py)) read from `Undistorted/`, not `LiDAR/`, so it must be populated before running anything. Undistort and accumulate the raw scans with the [**HeLiPR-Pointcloud-Toolbox**](https://github.com/minwoo0611/HeLiPR-Pointcloud-Toolbox), using [`config/HeLiPR-Toolbox/config.yaml`](../config/HeLiPR-Toolbox/config.yaml) as a starting point — it holds the exact settings we used (frame accumulation, downsampling, cropping) to generate the scans in our results. Adjust `Path.binPath` / `Path.trajPath` / `Path.savePath` per sequence and sensor, then point `savePath` at the corresponding `<Sequence>/Undistorted/<Sensor>/` folder above.

`inlier doctor --dataset /path/to/HeLiPR` checks this layout before you spend time on a run.

## Building Overlap Ground Truth

Precompute the pairwise scan-overlap matrices used to label true/false positives:

```bash
inlier gt build \
    --dataset-type helipr \
    --dataset /path/to/HeLiPR \
    --db-sequence Roundabout01 --q-sequence Roundabout03 \
    --pairs O-Aeva \
    --output-dir overlap_matrices \
    --voxel-size 0.5 --distance-threshold 100
```

Alongside the matrix this writes a small `overlap_*.json` sidecar recording the parameters it was built with (`n_db`, `n_q`, `stride_db`, `stride_q`, voxel size, thresholds), and the evaluation **refuses to run** if they disagree with what it is about to assume — see [Overlap Ground Truth](custom-data.md#overlap-ground-truth).

- `--pairs` is `<DB sensor>-<Q sensor>`; here `O-Aeva` means the Roundabout01 **Ouster** scans are the database and the Roundabout03 **Aeva** scans are the query.
- `--voxel-size` is the voxel size δ (m) used when computing per-voxel overlap between a DB/Q scan pair — smaller values are stricter (more voxels must actually coincide).
- `--distance-threshold` caps the pose-to-pose distance (m) beyond which a DB/Q pair is assumed non-overlapping and skipped, without spending time voxelizing it.

## Validate Calculated Overlaps

Sanity-check a precomputed overlap matrix before trusting it as GT with [`inlier gt validate`](../inlier/eval/overlap_validate.py):

```bash
inlier gt validate \
    --dataset-type helipr \
    --dataset /path/to/HeLiPR \
    --db-sequence Roundabout01 --q-sequence Roundabout03 \
    --pair O-Aeva \
    --overlap-dir overlap_matrices \
    --voxel-size 0.5 --pose-dist-threshold 10.0 --overlap-threshold 0.2
```

It re-loads the DB/Q poses and scans (same range-filter → global-frame → voxelize pipeline as [`inlier gt build`](../inlier/eval/overlap_build.py)) and opens a figure with: (a) DB/Q trajectories in 3D with edges drawn between scan pairs above `--overlap-threshold` and within `--pose-dist-threshold`, colored by overlap value; (b) summary statistics (non-zero entries, entries above threshold, max/mean overlap); (c) a histogram of non-zero overlap values; and (d) top-down aligned views of one randomly picked high-overlap and one low-overlap scan pair. Pass `--seed` to fix which example pair is shown.

<p align=center>
  <img src="../figures/overlaps_example.png" alt="Overlap validation figure for Roundabout01 (Ouster) vs Roundabout03 (Aeva)" width="90%"/>
</p>

## Running the Evaluation

```bash
inlier eval cross-session \
    --config config/default.yaml \
    --dataset /path/to/HeLiPR \
    --db-sequence Roundabout01 --q-sequence Roundabout03 \
    --pair O-Aeva \
    --overlap-dir overlap_matrices \
    --output-dir results/HeLiPR \
    --overlap-threshold 0.2 --max-pose-dist 10.0
```

This writes the Recall/PR-AUC metrics (`results_*.json`), the loop-closure candidates (`candidates_*.csv`), per-pair verify poses (`per_pair_verify_*.csv`), the descriptor caches (`cache_inlier/desc_*.npz`), and a trajectory plot (`trajectory_*.png`) to the output folder.

The encoder and retrieval parameters come from `--config`; see [Configuration](configuration.md). For datasets other than HeLiPR, see [Your Own Data](custom-data.md).

## Visualization

Replay a `DB←Q` run as an animation of the growing query trajectory, loop closures, matched keypoints, and the MINT/BEAM descriptors — driven entirely by the evaluation artifacts above:

```bash
inlier play \
    --run-dir results/HeLiPR/dbR01-O-qR03-Aeva_vs0.5_cs1_nh10_nr20_na60_ns7 \
    --cache-dir cache_inlier
```

The run's identity — sequences, sensors, dataset root, GT thresholds, score threshold, and the tag its filenames are built from — is read from the `results_*.json` in `--output-dir`, so none of it is repeated on the command line. Pass `--dataset` only if the dataset has moved since the run.

Controls: `SPACE` play/pause, `←` / `→` step. Pass `--record out.mp4` to render headlessly.

<p align=center>
  <img src="../figures/helipr.gif" alt="InLiER loop-closure playback" width="90%"/>
</p>
