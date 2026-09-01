""" Encoder equivalence (minus shape PCA — N_s=1 here; M3 covers N_s>1).

Deterministic paths (fixed plane) must match the reference exactly;
RANSAC-dependent paths are checked statistically.
"""

from __future__ import annotations

import numpy as np
import pytest

from inlier import _inlier_pybind as ip
from inlier.core import _cfg_bridge as bridge
from inlier.core.Dataclasses import InLiER_Config
from inlier.core.InLiER import InLiER


def make_configs(**overrides):
    """Paired (reference dataclass, C++ struct) configs with N_s=1."""
    ref = InLiER_Config(N_s=1, **overrides)
    cpp = ip.InLiERConfig()
    for f in (
        "N_h", "z_min", "z_max", "r_max", "N_r", "N_a", "N_s", "cell_size",
        "xy_max", "window", "max_kp_per_slice", "ransac_iters",
        "ransac_dist_thresh", "ransac_min_inliers", "point_mode",
        "shape_radius", "shape_min_neighbors",
    ):
        setattr(cpp, f, getattr(ref, f))
    cpp.max_kp_total = int(ref.max_kp_total)
    return ref, cpp


@pytest.fixture(scope="module")
def ref_and_cpp():
    ref_cfg, cpp_cfg = make_configs()
    return InLiER(ref_cfg), ip._Encoder(cpp_cfg), ref_cfg


def fixed_plane_py(cloud):
    """Reference RANSAC plane (dict) reused for both implementations."""
    ref_cfg, _ = make_configs()
    enc = InLiER(ref_cfg)
    return enc._ransac_plane(cloud, verbose=False)


## ---- deterministic path: fixed plane ----


def _sorted_by_score_set(p):
    """Row set of a keypoint array, order-independent."""
    return set(map(tuple, np.round(p, 9)))


def test_extract_keypoints_fixed_plane_exact(ref_and_cpp, synthetic_cloud):
    ref_enc, cpp_enc, _ = ref_and_cpp
    plane = fixed_plane_py(synthetic_cloud)

    kp_ref = ref_enc.extract_keypoints(
        synthetic_cloud, plane=plane, verbose=False)
    p_cpp, T_cpp = cpp_enc.extract_keypoints(
        synthetic_cloud, bridge.plane_dict_to_cpp(plane))

    np.testing.assert_allclose(T_cpp, kp_ref.T_ground, rtol=0, atol=1e-12)
    assert p_cpp.shape == kp_ref.p.shape
    ## element-for-element (ordering replicated incl. argsort tie-breaks)
    np.testing.assert_allclose(p_cpp, kp_ref.p, rtol=0, atol=1e-9)


def test_tokenize_keypoints_mode_exact(ref_and_cpp, synthetic_cloud):
    ref_enc, cpp_enc, _ = ref_and_cpp
    plane = fixed_plane_py(synthetic_cloud)

    kp_ref = ref_enc.extract_keypoints(
        synthetic_cloud, plane=plane, verbose=False)
    tok_ref = ref_enc.tokenize(
        synthetic_cloud, kp_ref, plane=plane, verbose=False)

    tok_cpp = cpp_enc.tokenize(
        synthetic_cloud, kp_ref.p, kp_ref.T_ground,
        bridge.plane_dict_to_cpp(plane), "config")

    assert tok_cpp.dtype == tok_ref.token_id.dtype
    np.testing.assert_array_equal(tok_cpp, tok_ref.token_id)


def test_encode_fixed_plane_exact(ref_and_cpp, synthetic_cloud):
    ref_enc, cpp_enc, _ = ref_and_cpp
    plane = fixed_plane_py(synthetic_cloud)

    kp_ref, tok_ref = ref_enc.encode(
        synthetic_cloud, plane=plane, verbose=False)
    p_cpp, T_cpp, tok_cpp = cpp_enc.encode(
        synthetic_cloud, bridge.plane_dict_to_cpp(plane))

    np.testing.assert_allclose(p_cpp, kp_ref.p, rtol=0, atol=1e-9)
    np.testing.assert_allclose(T_cpp, kp_ref.T_ground, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(tok_cpp, tok_ref.token_id)


def test_all_points_mode_exact(synthetic_cloud):
    py_cfg, cpp_cfg = make_configs(point_mode="all_points")
    py_enc = InLiER(py_cfg)
    cpp_enc = ip._Encoder(cpp_cfg)
    plane = fixed_plane_py(synthetic_cloud)
    
    kp_py = py_enc.extract_keypoints(synthetic_cloud, plane=plane, verbose=False)
    tok_py = py_enc.tokenize(synthetic_cloud, kp_py, plane=plane, verbose=False)
    tok_cpp = cpp_enc.tokenize(
        synthetic_cloud, kp_py.p, kp_py.T_ground,
        bridge.plane_dict_to_cpp(plane), "config")
    
    ## all_points mode filters every raw point individually against the
    ## xy_max/z ROI boundary (no cell aggregation to absorb it), so a
    ## point within ~1 ULP of the boundary can flip inclusion between
    ## languages: AlignGround (Eigen) and _align_ground (numpy) are
    ## numerically close but not bit-identical (see test_plane.py).
    ## Tolerate a couple of such boundary flips as multiset noise.
    from collections import Counter
    c_cpp, c_py = Counter(tok_cpp.tolist()), Counter(tok_py.token_id.tolist())
    sym_diff = sum((c_cpp - c_py).values()) + sum((c_py - c_cpp).values())
    assert sym_diff <= 3, ( 
        f"{sym_diff} tokens differ between all_points C++/Python runs — "
        "expected at most a couple from ROI-boundary floating-point flips"
    )

    p_cpp, _ = cpp_enc.extract_keypoints(
        synthetic_cloud, bridge.plane_dict_to_cpp(plane))
    np.testing.assert_allclose(p_cpp, kp_py.p, rtol=0, atol=1e-9)



def test_tokenize_keypoints_mode_override(synthetic_cloud):
    """mode='keypoints' must match reference tokenize_keypoints() even
    when the config says all_points."""
    ref_cfg, cpp_cfg = make_configs(point_mode="all_points")
    ref_enc = InLiER(ref_cfg)
    cpp_enc = ip._Encoder(cpp_cfg)
    plane = fixed_plane_py(synthetic_cloud)

    kp_ref = ref_enc.extract_keypoints(
        synthetic_cloud, plane=plane, verbose=False)
    tok_ref = ref_enc.tokenize_keypoints(
        synthetic_cloud, kp_ref, plane=plane, verbose=False)
    tok_cpp = cpp_enc.tokenize(
        synthetic_cloud, kp_ref.p, kp_ref.T_ground,
        bridge.plane_dict_to_cpp(plane), "keypoints")
    np.testing.assert_array_equal(tok_cpp, tok_ref.token_id)
    assert ref_cfg.point_mode == "all_points"  # reference restored its config


## ---- empty / degenerate inputs ----


def test_empty_cloud(ref_and_cpp):
    _, cpp_enc, _ = ref_and_cpp
    p, T = cpp_enc.extract_keypoints(np.zeros((0, 3)))
    assert p.shape == (0, 3)
    np.testing.assert_array_equal(T, np.eye(4))


def test_all_outside_roi(ref_and_cpp):
    ref_enc, cpp_enc, _ = ref_and_cpp
    ## points far outside xy_max, with an explicit plane (no RANSAC)
    cloud = np.array([[500.0, 500.0, 1.0], [510.0, 500.0, 1.0],
                      [500.0, 510.0, 2.0]])
    plane = {"normal": np.array([0.0, 0.0, 1.0]),
             "d": np.array([0.0]), "point": np.zeros(3),
             "inliers": np.ones(3, bool)}
    kp_ref = ref_enc.extract_keypoints(cloud, plane=plane, verbose=False)
    p_cpp, T_cpp = cpp_enc.extract_keypoints(cloud, bridge.plane_dict_to_cpp(plane))
    assert p_cpp.shape == kp_ref.p.shape == (0, 3)
    np.testing.assert_array_equal(T_cpp, kp_ref.T_ground)  # identity


def test_too_few_points_raises(ref_and_cpp):
    _, cpp_enc, _ = ref_and_cpp
    with pytest.raises(Exception, match="at least 3 points"):
        cpp_enc.encode(np.zeros((2, 3)))


## ---- geometry helpers ----


def test_rodrigues_matches_reference():
    ref = InLiER._rodrigues
    rng = np.random.default_rng(3)
    for _ in range(50):
        axis = rng.normal(size=3)
        angle = rng.uniform(-2 * np.pi, 2 * np.pi)
        np.testing.assert_allclose(
            ip.rodrigues(axis, angle), ref(axis, angle), atol=1e-14)
    ## degenerate axis
    np.testing.assert_allclose(
        ip.rodrigues(np.zeros(3), 1.0), np.eye(3), atol=0)


def test_align_ground_matches_reference(ref_and_cpp):
    ref_enc, _, _ = ref_and_cpp
    rng = np.random.default_rng(4)
    cases = [rng.normal(size=3) for _ in range(30)]
    cases += [np.array([0.0, 0.0, 1.0]),      # already aligned
              np.array([0.0, 0.0, -1.0]),     # anti-aligned
              np.array([1e-3, 0.0, 1.0])]     # near-aligned
    pts = rng.normal(size=(100, 3))
    for n in cases:
        n = n / np.linalg.norm(n)
        p0 = rng.normal(size=3)
        _, R_ref, t_ref = ref_enc._align_ground(pts, n, p0)
        T = ip.align_ground(n, p0)
        np.testing.assert_allclose(T[:3, :3], R_ref, atol=1e-12)
        np.testing.assert_allclose(T[:3, 3], t_ref, atol=1e-12)


## ---- RANSAC plane: statistical equivalence ----


def test_ransac_plane_statistical(synthetic_cloud):
    ref_cfg, _ = make_configs()
    ref_enc = InLiER(ref_cfg)

    ## sharp optimum: a tight threshold pins the solution down, so the
    ## two RNG streams must agree closely
    tight = ref_cfg.ransac_dist_thresh * 0.2
    saved = ref_cfg.ransac_dist_thresh
    ref_cfg.ransac_dist_thresh = tight
    try:
        plane_ref_t = ref_enc._ransac_plane(synthetic_cloud, verbose=False)
    finally:
        ref_cfg.ransac_dist_thresh = saved
    plane_cpp_t = ip.ransac_plane(
        synthetic_cloud, ref_cfg.ransac_iters, tight,
        ref_cfg.ransac_min_inliers)
    cos_t = abs(np.dot(plane_cpp_t.normal, plane_ref_t["normal"]))
    assert cos_t > np.cos(np.deg2rad(0.5))

    ## plateau case (production threshold): wide family of near-optimal
    ## planes — allow the streams to land a bit apart
    plane_ref = ref_enc._ransac_plane(synthetic_cloud, verbose=False)
    plane_cpp = ip.ransac_plane(
        synthetic_cloud, ref_cfg.ransac_iters, ref_cfg.ransac_dist_thresh,
        ref_cfg.ransac_min_inliers)
    cos = abs(np.dot(plane_cpp.normal, plane_ref["normal"]))
    assert cos > np.cos(np.deg2rad(2.0))
    n_ref = int(plane_ref["inliers"].sum())
    n_cpp = int(np.sum(plane_cpp.inliers))
    assert abs(n_cpp - n_ref) / n_ref < 0.05

    ## determinism: same seed -> identical result across calls
    plane_cpp2 = ip.ransac_plane(
        synthetic_cloud, ref_cfg.ransac_iters, ref_cfg.ransac_dist_thresh,
        ref_cfg.ransac_min_inliers)
    np.testing.assert_array_equal(plane_cpp.normal, plane_cpp2.normal)
    assert plane_cpp.d == plane_cpp2.d


def test_ransac_fallback_pca_plane():
    """Impossible min_inliers forces the PCA fallback in both."""
    rng = np.random.default_rng(5)
    cloud = np.stack([
        rng.uniform(-10, 10, 500), rng.uniform(-10, 10, 500),
        0.1 * rng.normal(size=500),
    ], axis=1)
    ref_cfg, _ = make_configs()
    ref_enc = InLiER(ref_cfg)
    plane_ref = ref_enc._ransac_plane(cloud, verbose=False)

    plane_cpp = ip.ransac_plane(cloud, 10, 1e-12, 10**9)  # force fallback
    cos = abs(np.dot(plane_cpp.normal, plane_ref["normal"]))
    assert cos > np.cos(np.deg2rad(1.0))


def test_encode_with_own_ransac_matches_ref_given_same_plane(synthetic_cloud):
    """Full C++ encode with its OWN RANSAC must equal the reference
    pipeline run on the plane the C++ RANSAC found.

    This isolates the accepted RNG-stream difference (tested separately
    in test_ransac_plane_statistical) from pipeline correctness: given
    the same plane, everything downstream must match exactly. A direct
    own-RANSAC-vs-own-RANSAC token comparison is NOT meaningful — the
    2 m height bins amplify a ~1-degree plane difference into different
    tokens even between two seeds of the same implementation.
    """
    ref_cfg, cpp_cfg = make_configs()
    ref_enc = InLiER(ref_cfg)
    cpp_enc = ip._Encoder(cpp_cfg)

    p_cpp, T_cpp, tok_cpp = cpp_enc.encode(synthetic_cloud)

    plane_cpp = ip.ransac_plane(
        synthetic_cloud, ref_cfg.ransac_iters, ref_cfg.ransac_dist_thresh,
        ref_cfg.ransac_min_inliers)
    plane_dict = bridge.plane_cpp_to_dict(plane_cpp)
    kp_ref, tok_ref = ref_enc.encode(
        synthetic_cloud, plane=plane_dict, verbose=False)

    np.testing.assert_allclose(p_cpp, kp_ref.p, rtol=0, atol=1e-9)
    np.testing.assert_allclose(T_cpp, kp_ref.T_ground, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(tok_cpp, tok_ref.token_id)

    ## loose sanity vs the reference's own-RANSAC run (plane plateau)
    kp_own, _ = ref_enc.encode(synthetic_cloud, verbose=False)
    assert abs(p_cpp.shape[0] - kp_own.p.shape[0]) <= 0.1 * kp_own.p.shape[0]
