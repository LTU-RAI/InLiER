# C++ vs Python — full HeLiPR eval (from scratch)

- DB/Q: `Roundabout01/Ouster` → `Roundabout03/Aeva`  (DB=2705, Q=2774, GT+=2378)
- overlap_threshold=0.2  max_pose_dist=10.0  voxel_size=0.5

## Accuracy  (should match — C++ is a port of the reference)

| stage | metric | c++ | python | Δ(py−cpp) |
|---|---|---:|---:|---:|
| MINT | Recall@100 |   0.6501 |   0.6501 |   0.0000 |
| BEAM | Recall@20 |   0.7140 |   0.7145 |   0.0004 |
| Verify | Recall@1 |   0.6472 |   0.6480 |   0.0008 |
| Verify | Recall@5 |   0.6812 |   0.6817 |   0.0004 |

## Processing time (seconds)  — from scratch

| stage | c++ (s) | python (s) | speedup |
|---|---:|---:|---:|
| encoding |    246.0 |    409.4 |    1.7x |
| encoding / frame (ms) |    44.91 |    74.73 |    1.7x |
| MINT |      8.0 |     13.2 |    1.7x |
| BEAM |     96.2 |    240.2 |    2.5x |
| verify |      9.1 |    361.7 |   39.6x |
| gicp |    109.7 |    109.3 |    1.0x |
| **wall total** |    588.1 |   1252.7 |    2.1x |

## Time per query (ms) — one query against the full DB

| stage | c++ (ms) | python (ms) | speedup |
|---|---:|---:|---:|
| MINT |     2.89 |     4.78 |    1.7x |
| BEAM |    34.67 |    86.58 |    2.5x |
| verify |     3.29 |   130.40 |   39.6x |
