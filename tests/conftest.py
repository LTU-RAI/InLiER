"""Shared fixtures for the C++ vs pure-numpy equivalence test-suite.

Run with the system interpreter (has numpy/scipy + the built extension):

    /usr/bin/python3 -m pytest tests/ -x
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "cache_inlier")

## Default-config cache hash for config/default.yaml (vs0.5, ns7).
DEFAULT_CACHE_HASH = "a6a8d4c7cbd5"


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _find_cache(pattern: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(CACHE_DIR, pattern)))
    return hits[0] if hits else None


@pytest.fixture(scope="session")
def cached_descriptors():
    """A real cached descriptor set (CSR layout) for matcher tests.

    keys: positions (N,3), poses (N,4,4), offsets (N+1,), token_ids (K,),
    kp_aligned (K,3), kp_sensor (K,3), T_grounds (N,4,4).
    """
    path = _find_cache(f"desc_*_{DEFAULT_CACHE_HASH}.npz")
    if path is None:
        pytest.skip("no cached descriptors with the default-config hash")
    return np.load(path)


@pytest.fixture(scope="session")
def synthetic_cloud(rng) -> np.ndarray:
    """Urban-ish synthetic scene: ground plane + poles + walls + scatter."""
    parts = []
    ## ground plane, slightly tilted, with noise
    n_ground = 20000
    gx = rng.uniform(-80, 80, n_ground)
    gy = rng.uniform(-80, 80, n_ground)
    gz = 0.02 * gx + 0.01 * gy + rng.normal(0, 0.05, n_ground)
    parts.append(np.stack([gx, gy, gz], axis=1))
    ## poles (linear)
    for _ in range(40):
        cx, cy = rng.uniform(-60, 60, 2)
        z = rng.uniform(0.2, 8.0, 200)
        x = cx + rng.normal(0, 0.05, 200)
        y = cy + rng.normal(0, 0.05, 200)
        parts.append(np.stack([x, y, z], axis=1))
    ## walls (planar)
    for _ in range(20):
        cx, cy = rng.uniform(-60, 60, 2)
        along = rng.uniform(-4, 4, 400)
        up = rng.uniform(0.2, 6.0, 400)
        ang = rng.uniform(0, np.pi)
        x = cx + along * np.cos(ang) + rng.normal(0, 0.03, 400)
        y = cy + along * np.sin(ang) + rng.normal(0, 0.03, 400)
        parts.append(np.stack([x, y, up], axis=1))
    ## scatter blobs (vegetation-ish)
    for _ in range(15):
        c = np.array([*rng.uniform(-60, 60, 2), rng.uniform(1.5, 5.0)])
        parts.append(c + rng.normal(0, 1.0, (300, 3)))
    return np.concatenate(parts).astype(np.float64)
