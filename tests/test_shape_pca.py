""" Shape-PCA equivalence — C++ (nanoflann + Eigen) vs numpy/scipy.

Eigendecompositions differ at the LSB between LAPACK and Eigen, so lps
uses allclose and class agreement allows rare flips at bin/argmax
boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest

from inlier import _inlier_pybind as ip
from inlier.core import _cfg_bridge as bridge
from inlier.core.Dataclasses import InLiER_Config
from inlier.core.InLiER import InLiER

RADIUS = 1.5
MIN_NEIGHBORS = 8


def make_labelled_cloud(rng):
    """Cloud with known-shape regions: poles, walls, blobs."""
    parts, centers = [], []
    for i in range(10):  # vertical poles -> linear, inclination ~0
        cx, cy = 10.0 * i, 0.0
        z = np.linspace(0.0, 6.0, 300)
        pts = np.stack([cx + rng.normal(0, 0.03, 300),
                        cy + rng.normal(0, 0.03, 300), z], axis=1)
        parts.append(pts)
        centers.append([cx, cy, 3.0])
    for i in range(10):  # horizontal rails -> linear, inclination ~90
        cy = 10.0 + 5.0 * i
        x = np.linspace(-3.0, 3.0, 300)
        pts = np.stack([x, cy + rng.normal(0, 0.03, 300),
                        2.0 + rng.normal(0, 0.03, 300)], axis=1)
        parts.append(pts)
        centers.append([0.0, cy, 2.0])
    for i in range(10):  # vertical walls -> planar, normal horizontal
        cx = -10.0 - 5.0 * i
        a = rng.uniform(-3, 3, 500)
        u = rng.uniform(0, 5, 500)
        pts = np.stack([cx + rng.normal(0, 0.02, 500), a, u], axis=1)
        parts.append(pts)
        centers.append([cx, 0.0, 2.5])
    for i in range(10):  # ground patches -> planar, normal vertical
        cy = -10.0 - 5.0 * i
        pts = np.stack([rng.uniform(-3, 3, 500),
                        cy + rng.uniform(-3, 3, 500),
                        rng.normal(0, 0.02, 500)], axis=1)
        parts.append(pts)
        centers.append([0.0, cy, 0.0])
    for i in range(10):  # blobs -> scatter
        c = np.array([50.0 + 8.0 * i, 50.0, 2.0])
        parts.append(c + rng.normal(0, 0.8, (400, 3)))
        centers.append(c.tolist())
    for i in range(5):  # sparse spots -> invalid (below min_neighbors)
        c = np.array([-50.0 - 8.0 * i, 50.0, 2.0])
        parts.append(c + rng.normal(0, 0.05, (3, 3)))
        centers.append(c.tolist())
    return np.concatenate(parts), np.asarray(centers, dtype=np.float64)


@pytest.fixture(scope="module")
def labelled(rng):
    return make_labelled_cloud(rng)


@pytest.mark.parametrize("n_classes", [3, 5, 7])
def test_shape_pca_vs_reference(labelled, n_classes):
    cloud, centers = labelled
    cls_py, lps_py = InLiER._compute_shape_pca(
        cloud, centers, radius=RADIUS, min_neighbors=MIN_NEIGHBORS,
        n_classes=n_classes)
    cls_cpp, lps_cpp = ip.compute_shape_pca(
        cloud, centers, RADIUS, MIN_NEIGHBORS, n_classes)

    np.testing.assert_allclose(lps_cpp, lps_py, rtol=1e-4, atol=1e-5)
    agree = float(np.mean(cls_cpp == cls_py))
    assert agree >= 0.995, f"class agreement {agree:.4f}"


def test_shape_pca_vs_reference_on_scene(synthetic_cloud):
    """Real-ish scene: centers = a keypoint-like subsample of the cloud."""
    rng = np.random.default_rng(7)
    centers = synthetic_cloud[
        rng.choice(synthetic_cloud.shape[0], 800, replace=False)]
    cls_py, lps_py = InLiER._compute_shape_pca(
        synthetic_cloud, centers, radius=RADIUS,
        min_neighbors=MIN_NEIGHBORS, n_classes=7)
    cls_cpp, lps_cpp = ip.compute_shape_pca(
        synthetic_cloud, centers, RADIUS, MIN_NEIGHBORS, 7)

    np.testing.assert_allclose(lps_cpp, lps_py, rtol=1e-4, atol=1e-5)
    agree = float(np.mean(cls_cpp == cls_py))
    assert agree >= 0.995, f"class agreement {agree:.4f}"


def test_shape_pca_expected_classes(labelled):
    """Semantic sanity for n_classes=7: poles->0, rails->2, ground->3,
    walls->5, blobs/sparse->6 (allowing minor noise)."""
    cloud, centers = labelled
    cls, _ = ip.compute_shape_pca(cloud, centers, RADIUS, MIN_NEIGHBORS, 7)
    cls = np.asarray(cls)
    assert np.mean(cls[0:10] == 0) >= 0.9
    assert np.mean(cls[10:20] == 2) >= 0.9
    assert np.mean(cls[20:30] == 5) >= 0.9
    assert np.mean(cls[30:40] == 3) >= 0.9
    assert np.mean(cls[40:50] == 6) >= 0.9
    assert np.all(cls[50:55] == 6)


def test_shape_pca_empty_inputs():
    cls, lps = ip.compute_shape_pca(
        np.zeros((0, 3)), np.zeros((0, 3)), RADIUS, MIN_NEIGHBORS, 7)
    assert len(cls) == 0 and lps.shape == (0, 3)

    centers = np.zeros((4, 3))
    cls_py, lps_py = InLiER._compute_shape_pca(
        np.zeros((0, 3)), centers, radius=RADIUS,
        min_neighbors=MIN_NEIGHBORS, n_classes=7)
    cls_cpp, lps_cpp = ip.compute_shape_pca(
        np.zeros((0, 3)), centers, RADIUS, MIN_NEIGHBORS, 7)
    np.testing.assert_array_equal(cls_cpp, cls_py)
    np.testing.assert_allclose(lps_cpp, lps_py, atol=0)


def test_full_encode_ns7_matches_reference(synthetic_cloud):
    """End-to-end encode with N_s=7 (the real default) and a fixed
    plane: tokens must agree except where PCA class flips at numeric
    boundaries (<=0.5%)."""
    py_cfg = InLiER_Config()  # N_s=7 default
    py_enc = InLiER(py_cfg)
    cpp_cfg = bridge.to_cpp_inlier_config(py_cfg)
    cpp_enc = ip._Encoder(cpp_cfg)

    plane = py_enc._ransac_plane(synthetic_cloud, verbose=False)
    kp_py, tok_py = py_enc.encode(synthetic_cloud, plane=plane, verbose=False)
    p_cpp, _, tok_cpp = cpp_enc.encode(
        synthetic_cloud, bridge.plane_dict_to_cpp(plane))

    np.testing.assert_allclose(p_cpp, kp_py.p, rtol=0, atol=1e-9)
    assert tok_cpp.shape == tok_py.token_id.shape
    mismatch = float(np.mean(tok_cpp != tok_py.token_id))
    assert mismatch <= 0.005, f"token mismatch rate {mismatch:.4f}"
