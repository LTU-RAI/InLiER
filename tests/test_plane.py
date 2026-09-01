""" Plane geometry — Rodrigues/align are bit-exact; RANSAC is only
  statistically equivalent (different RNG stream, see plane.hpp)."""

from __future__ import annotations

import numpy as np

from inlier import _inlier_pybind as ip
from inlier.core.InLiER import InLiER

CFG = InLiER().config  # default InLiER_Config, unused here but keeps parity


def test_rodrigues_matches_reference():
    rng = np.random.default_rng(0)
    for _ in range(50):
        axis = rng.normal(size=3)
        angle = rng.uniform(-np.pi, np.pi)
        got = ip.rodrigues(axis, angle)
        want = InLiER._rodrigues(axis, angle)
        np.testing.assert_allclose(got, want, atol=1e-12)
        
        
def test_align_ground_matches_reference():
    rng = np.random.default_rng(1)
    for _ in range(50):
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        point = rng.normal(size=3) * 5.0
        
        T_got = ip.align_ground(normal, point)
        pts = rng.normal(size=(10, 3)) * 10.0
        got_aligned = (T_got[:3, :3] @ pts.T).T + T_got[:3, 3]

        _, R_want, t_want = InLiER()._align_ground(pts, normal, point)
        want_aligned = (R_want @ pts.T).T + t_want

        np.testing.assert_allclose(got_aligned, want_aligned, atol=1e-9)

def test_ransac_plane_runs_and_is_reasonable():
      """Not bit-exact (different RNG) — just check it finds a sane plane
      for an obviously flat point set."""
      rng = np.random.default_rng(2)
      x = rng.uniform(-10, 10, 5000)
      y = rng.uniform(-10, 10, 5000)
      z = 0.01 * x + 0.02 * y + rng.normal(0, 0.01, 5000)
      pts = np.stack([x, y, z], axis=1)
      
      plane = ip.ransac_plane(pts, iters=250, dist_thresh=0.05, min_inliers=100)
      assert plane.normal[2] > 0.99  # nearly flat -> normal close to +Z
      assert abs(plane.d) < 0.1
