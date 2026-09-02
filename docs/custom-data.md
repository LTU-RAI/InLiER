# 🗃️ Test Your Own Data

← back to the [README](../README.md)

InLiER isn't tied to the HeLiPR loader — [`inlier eval cross-session --dataset-type generic`](../inlier/eval/protocols/cross_session.py) runs the identical retrieval/evaluation pipeline on any folder-based dataset, via [`datasets/generic.py`](../inlier/eval/datasets/generic.py).

## Dataset Layout

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

`inlier doctor --dataset <path> --dataset-type generic` checks a folder against this layout.

**`transform.txt` — when you need it.** InLiER compares database and query poses directly (to build the overlap GT and to filter candidates by `--max-pose-dist`), so both sequences must live in the *same* world frame. If they already do — e.g. both were mapped in one session, or registered to a shared global frame — no transform is needed. If they were mapped independently, each sequence's poses start at its own arbitrary origin, and the DB/Q pose distances would be meaningless. `transform.txt` is the 4×4 matrix that maps the **DB world frame into the Q world frame**, and it's applied to the DB keyframe poses before any distance or overlap computation.

The evaluation auto-detects `<db_path>/transform.txt` if it exists; pass `--transform <path>` to point elsewhere, or `--no-transform` to force the shared-frame assumption even when the file is present. Use the same choice for both [`inlier gt build`](../inlier/eval/overlap_build.py) and [`inlier eval cross-session --dataset-type generic`](../inlier/eval/protocols/cross_session.py) — a mismatch silently produces a wrong GT matrix.

**Pre-accumulated vs. single scans.** `scans/` can hold either. If your `.pcd` files are already accumulated submaps (as with the HeLiPR toolbox output), run with `--n-db 1 --n-q 1` and each file is used as-is. If they're single sensor scans, let the evaluation accumulate them: `--n-db` / `--n-q` set how many consecutive scans form one submap, each window anchored at its first scan (the keyframe) with the rest transformed into that keyframe's pose via `inv(T_i) @ T_k`. `--stride-db` / `--stride-q` set the step between consecutive submaps and default to `n_db` / `n_q`, i.e. non-overlapping submaps; a stride of 1 gives maximally overlapping ones. Either way there must be exactly one pose per `.pcd` file — the loader errors out on a count mismatch.

Sparse single scans (solid-state, or low-resolution spinning units) generally need accumulation for the height-slice keypoints to be stable.

## Overlap Ground Truth

[`inlier gt build`](../inlier/eval/overlap_build.py) also supports generic data — pass `--dataset-type generic --db-path ... --q-path ...` (same `.pcd` + `poses_kitti.txt` layout, plus the DB→Q `--transform`) to compute the pairwise overlap matrix, just as in [Building Overlap Ground Truth](helipr-benchmark.md#building-overlap-ground-truth) for HeLiPR.

With pre-accumulated scans, the defaults (`--n-db 1 --n-q 1`) use each `.pcd` as-is:

```bash
inlier gt build \
    --dataset-type generic \
    --db-path /path/to/database \
    --q-path  /path/to/query \
    --transform /path/to/transform.txt \
    --output-dir overlap_matrices \
    --voxel-size 0.5 --distance-threshold 100
```

With single scans, accumulate them into submaps — e.g. 10 scans per submap, stepping one scan at a time:

```bash
inlier gt build \
    --dataset-type generic \
    --db-path /path/to/database \
    --q-path  /path/to/query \
    --transform /path/to/transform.txt \
    --output-dir overlap_matrices \
    --n-db 10 --n-q 10 --stride-db 1 --stride-q 1 \
    --voxel-size 0.5 --distance-threshold 100
```

A stride of 1 keeps one submap per scan, so the overlap matrix stays at full resolution (`M_db × M_q` with `M ≈ number of scans`) at the cost of a longer build. Leaving `--stride-*` out defaults it to `n`, giving non-overlapping submaps and a ~10× smaller matrix.

> ⚠️ `--n-db` / `--n-q` / `--stride-db` / `--stride-q` **must match what you later pass to** [`inlier eval cross-session`](../inlier/eval/protocols/cross_session.py) and to [`inlier encode --dataset`](cli.md#encoding-submaps) — the matrix is indexed by submap, so any difference misaligns the GT against the retrieval results. `inlier gt build` writes the values into an `overlap_*.json` sidecar and the evaluation refuses to run when they disagree.

## Evaluation

```bash
inlier eval cross-session --dataset-type generic \
    --config config/default.yaml \
    --db-path /path/to/database \
    --q-path  /path/to/query \
    --transform /path/to/transform.txt \
    --overlap-file /path/to/overlap.txt \
    --overlap-threshold 0.2 --max-pose-dist 25.0 \
    --n-db 10 --n-q 10 --stride-db 1 --stride-q 1 \
    --output-dir results/generic_dataset
```

`--transform` defaults to `<db_path>/transform.txt` if present, and `--no-transform` disables it when both sequences already share a world frame. Outputs match the HeLiPR driver: `results_*.json`, `candidates_*.csv`, `ranked_*.csv`, the per-pair verify poses and the trajectory plot land under `--output-dir`; the descriptor caches go to `--cache-dir`.
