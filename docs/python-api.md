# 🐍 Python API

← back to the [README](../README.md)

InLiER is usable as a plain library — no dataset loader, no CLI, nothing on
disk. You hand `InLiER.encode()` an `(N, 3)` float32 point cloud in the sensor
frame and get back keypoints and their tokens; `InLiER_Matcher` holds the
database and runs the three stages over those tokens: MINT for a
rotation-invariant shortlist, BEAM for yaw estimation and reranking, and
Verify for the 6-DoF pose. The example below walks through all of it once.

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

## A growing database

`add()` may be called after `finalize()`. The next `finalize()` appends only
the scans added since the last one, so a database can grow one scan at a time
without rebuilding — which is what streaming loop-closure detection needs:

```python
matcher.reserve(len(stream))            # optional: keeps per-frame cost flat

for t, scan in enumerate(stream):
    keypoints, tokens = encoder.encode(scan, verbose=False)
    if t > exclusion_window:
        # search only frames older than the window, never the recent ones
        s1 = matcher.shortlist(tokens, topk=100,
                               max_db_index=t - exclusion_window)
        ...
    matcher.add(t, tokens)
```

`max_db_index` is an **exclusive bound in insertion order**, not a database ID,
and it is applied inside the scoring loop. A bounded search returns exactly
what an unbounded search over a database built from that prefix would return —
including how `topk_pct` resolves, which is relative to the bounded count.
Filtering the results afterwards instead would silently cost recall whenever
the excluded frames crowd out the top-k.

To load a configuration file instead of the dataclass defaults:

```python
from inlier.config import load, resolve

cfg = resolve(load("config/default.yaml"))
encoder = InLiER(cfg.inlier)
matcher = InLiER_Matcher(cfg.inlier, cfg.shortlist, cfg.beam)
```

See [Configuration](configuration.md) for what the individual parameters do.

Imports are lazy: `import inlier` alone loads neither the compiled extension nor
`small_gicp`, so setting `INLIER_FORCE_PYTHON=1` before the first attribute
access still selects the backend — see [C++ Core](cpp-core.md).
