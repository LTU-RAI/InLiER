# ⚙️ Configuration

← back to the [README](../README.md)

Encoder, retrieval, verification, and refinement parameters live in YAML files under [`config/`](../config). Start from [`config/default.yaml`](../config/default.yaml) and pass it with `--config`.

## How the Layers Merge

Configuration is layered:

1. The packaged [`inlier/config/default.yaml`](../inlier/config/default.yaml) is **always** the base.
2. Your `-c/--config FILE` is deep-merged on top, so a partial file only needs the keys you actually want to change.
3. `--set KEY=VALUE` overrides come last, e.g. `--set stage1.topk=50` (repeatable).

## What `--set` Takes

`KEY` is the dotted path of the key in the YAML — `<block>.<name>`, or just the
name for the one top-level key, `voxel_size`. `VALUE` is parsed as YAML, so
`true`/`false`, `null`, integers, floats, and quoted strings all mean what they
look like:

```bash
inlier config show --set voxel_size=0.3                 # top-level scalar
inlier config show --set encoder.N_a=120                # 3 deg azimuth bins
inlier config show --set gicp.skip=true                 # bool, not "refine=false"
inlier config show --set stage2.topk_pct=null           # null clears a value
inlier config show --set stage1.mint_mode="full"        # string
inlier config show --set stage1.topk=50 --set verify.topv=10   # repeatable
```

Unknown keys are rejected rather than silently ignored, and the error lists
every accepted key — so a wrong guess fails loudly instead of no-opping:

```console
$ inlier config show --set gicp.refine=false
ValueError: unknown config key(s): gicp.refine
accepted keys: voxel_size, encoder.{...}, stage1.{...}, ... gicp.{downsampling_resolution,
max_correspondence_distance, max_iterations, num_threads, registration_type, skip,
use_raw_clouds, voxel_resolution}
```

The tables below list every key by its `--set` name. `inlier config show`
prints what will actually run, including the values derived from `voxel_size`;
`inlier config dump` prints the whole merged YAML.

## Encoder Parameters

The `encoder:` block defines the descriptor itself — every point that survives the crop is reduced to a token `((hb·N_r + rb)·N_s + sb)·N_a + ab`, so the four `N_*` values set the descriptor's resolution and its vocabulary size (`N_h · N_r · N_s · N_a`).

| Parameter | Description | Default |
|-----------|-------------|---------|
| `voxel_size` *(top-level)* | Voxel size (m) used to downsample the submap **before** encoding. Only affects the encoder input — GICP does its own downsampling (`gicp.downsampling_resolution`). | 0.5 |
| `encoder.N_h` | Number of height slices between `z_min` and `z_max`. Sets the `hb` token field. | 10 |
| `encoder.z_min` / `encoder.z_max` | Height band (m, above the estimated ground plane) kept for encoding; points outside are dropped. Slice thickness is `(z_max − z_min) / N_h`. | 0.0 / 20.0 |
| `encoder.N_r` | Radial bins over `[0, min(r_max, xy_max)]`. Sets the `rb` field. | 20 |
| `encoder.N_a` | Azimuth bins over the full 360°, i.e. 6° per bin at the default. Sets the `ab` field and the shift resolution BEAM searches over for yaw. | 60 |
| `encoder.N_s` | Number of PCA shape classes (linear / planar / scattered mixtures from the local eigenvalue spread). Sets the `sb` field; `N_s: 1` disables shape and collapses the field. | 7 |
| `encoder.r_max` | Max radius (m) for the radial/azimuth binning. Effectively clamped to `xy_max`. | 100.0 |
| `encoder.xy_max` | XY half-extent (m) of the crop and of the BEV height-cell grid — points with \|x\| or \|y\| beyond it are discarded. | 100.0 |
| `encoder.cell_size` | BEV cell size (m) of the per-slice height image used for keypoint extraction. Keep it ≈ `2 × voxel_size` so cells are populated but not oversmoothed. | 1.0 |
| `encoder.window` | Side length (odd) of the non-maximum-suppression window applied to each slice's height image when picking local maxima as keypoints. Larger = fewer, more spread-out keypoints. | 3 |

| `encoder.point_mode` | `keypoints` encodes the extracted keypoints; `all_points` tokenizes every surviving point. | `keypoints` |

`encoder` also accepts the keypoint and shape-estimation knobs that
`default.yaml` leaves out, so they stay derived from `voxel_size` unless you opt
in: `max_kp_per_slice`, `max_kp_total`, `ransac_iters`, `ransac_dist_thresh`,
`ransac_min_inliers`, `shape_radius`, `shape_min_neighbors`.

## Retrieval Parameters

`stage1:` is MINT, the rotation-invariant shortlist; `stage2:` is BEAM, which
searches azimuth shifts to rerank it and estimate yaw. `rerank:` is an optional
4-D histogram stage, off by default and not part of the paper.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `stage1.topk` | Shortlist size. Ignored when `topk_pct` is set. | 100 |
| `stage1.topk_pct` | Shortlist as a fraction of database size (`0.1` = top 10%). `null` to use `topk`. | 0.10 |
| `stage1.min_shared_rows` | Minimum shared height rows before a candidate is scored at all. | 3 |
| `stage1.mint_mode` | `compact` collapses height into the MINT row, `full` preserves it. | `compact` |
| `stage1.mint_scoring` | `l1_intersection`, `raw_intersection`, or `cosine`. | `l1_intersection` |
| `stage2.skip` | Skip BEAM entirely — no yaw estimate, no reranking. | false |
| `stage2.topk` | Candidates kept after BEAM. Ignored when `topk_pct` is set. | 20 |
| `stage2.topk_pct` | Same, as a fraction of the stage-1 shortlist. | null |
| `stage2.min_shared_bins` / `stage2.min_shared_az_cols` | Minimum shared spatial bins / azimuth columns for a valid BEAM score. | 4 / 3 |
| `stage2.score_threshold` | Deployment score threshold; `--mode eval` forces `-2.0` so the PR sweep sees every candidate. | 0.0 |
| `rerank.run` | Turn the 4-D histogram rerank on. | false |
| `rerank.topk` / `rerank.topk_pct` | Candidates taken from BEAM into the rerank. | 50 / null |
| `rerank.scoring_mode` | `jaccard4d` or `cosine4d`. | `jaccard4d` |
| `rerank.min_shared_rows`, `rerank.spatial_tol`, `rerank.score_threshold` | As their stage-1 / verify counterparts. | — / — / 0.0 |

## Verification and Refinement Parameters

`verify:` is the token-guided RANSAC that turns a candidate into a 6-DoF pose;
`gicp:` is the refinement applied afterwards.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `verify.skip` | Return the ranking without verifying anything. | false |
| `verify.topv` | Candidates verified per query. | 20 |
| `verify.ransac_iters` | RANSAC iterations. | 500 |
| `verify.inlier_dist` | Inlier distance (m), and the inlier-ratio threshold. | 1.0 |
| `verify.min_correspondences` | Token correspondences needed before RANSAC is attempted. | 32 |
| `verify.min_ransac_inliers` / `verify.min_keypoint_inliers` | Inliers a pose must reach to count as verified. | 16 / 8 |
| `verify.spatial_tol` | ±bins of slack when matching tokens into correspondences. | 0 |
| `verify.seed` | Fix RANSAC's seed for a reproducible run. | unset |
| `gicp.skip` | Skip refinement and keep the RANSAC pose. | false |
| `gicp.registration_type` | `GICP`, `VGICP`, `ICP`, or `PLANE_ICP`. | `GICP` |
| `gicp.max_correspondence_distance` | Correspondence cutoff (m). | 1.5 |
| `gicp.use_raw_clouds` | Refine on the raw clouds; `false` refines on keypoints. | true |
| `gicp.downsampling_resolution` | GICP's own registration voxel (m), independent of `voxel_size`; `≤0` skips it. | 0.2 |
| `gicp.voxel_resolution` | VGICP only. | 1.0 |
| `gicp.num_threads` / `gicp.max_iterations` | Threads and iteration cap. | 8 / 16 |

## Checking a Grid Against a New Sensor

`inlier encode --viz` is the quickest way to tell whether these values suit a
new platform — see [Inspecting a Descriptor](cli.md#inspecting-a-descriptor).
