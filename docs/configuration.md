# ⚙️ Configuration

← back to the [README](../README.md)

Encoder, retrieval, verification, and refinement parameters live in YAML files under [`config/`](../config). Start from [`config/default.yaml`](../config/default.yaml) and pass it with `--config`.

## How the Layers Merge

Configuration is layered:

1. The packaged [`inlier/config/default.yaml`](../inlier/config/default.yaml) is **always** the base.
2. Your `-c/--config FILE` is deep-merged on top, so a partial file only needs the keys you actually want to change.
3. `--set KEY=VALUE` overrides come last, e.g. `--set stage1.topk=50` (repeatable).

Unknown keys are rejected rather than silently ignored. `inlier config show` prints exactly what will run, including the values derived from `voxel_size`:

```bash
inlier config show --set stage1.topk=50
```

## Encoder Parameters

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

## Checking a Grid Against a New Sensor

`inlier encode --viz` is the quickest way to tell whether these values suit a
new platform — see [Inspecting a descriptor](cli.md#inspecting-a-descriptor).
