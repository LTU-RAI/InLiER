# 🐍 Python API

← back to the [README](../README.md)

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

See [Configuration](configuration.md) for what the individual parameters do.

Imports are lazy: `import inlier` alone loads neither the compiled extension nor
`small_gicp`, so setting `INLIER_FORCE_PYTHON=1` before the first attribute
access still selects the backend — see [C++ Core](cpp-core.md).
