""" The public wrapper API (C++-backed) vs the reference, end to end.

Because the wrapper inherits the numpy `_ransac_plane` (same PCG64
stream), encoder outputs must be IDENTICAL to the reference apart from
shape-PCA LSB flips. The retrieval pipeline is compared stage by stage
on real cached descriptors.
"""

from __future__ import annotations

import numpy as np
import pytest

import inlier
from inlier import InLiER, InLiER_Config, InLiER_Matcher
from inlier.core.Dataclasses import (
    BEAMScoreConfig,
    InLiER_Tokens,
    ShortlistConfig,
    VerifyConfig,
)
from inlier.core.reference.InLiER import InLiER as RefInLiER
from inlier.core.reference.InLiER_Matcher import InLiER_Matcher as RefMatcher


def test_cpp_backend_active():
    from inlier.core import InLiER as enc_mod
    from inlier.core import InLiER_Matcher as m_mod
    assert enc_mod._BACKEND == "cpp"
    assert m_mod._BACKEND == "cpp"
    ## the wrappers subclass the reference -> isinstance stays valid
    assert issubclass(InLiER, RefInLiER)
    assert issubclass(InLiER_Matcher, RefMatcher)


def test_public_exports_unchanged():
    assert {"InLiER", "InLiER_Matcher", "InLiER_Config"} <= set(inlier.__all__)
    ## statics inherited from the reference implementation
    assert InLiER.pack_token_ids is RefInLiER.pack_token_ids
    assert InLiER_Matcher.refine_gicp is RefMatcher.refine_gicp


def test_encoder_wrapper_matches_reference(synthetic_cloud):
    """Same seeded numpy plane -> keypoints identical; tokens identical
    up to shape-PCA LSB flips."""
    cfg = InLiER_Config()
    enc = InLiER(cfg)
    ref = RefInLiER(cfg)

    kp, tok = enc.encode(synthetic_cloud, verbose=False)
    kp_ref, tok_ref = ref.encode(synthetic_cloud, verbose=False)

    np.testing.assert_allclose(kp.p, kp_ref.p, rtol=0, atol=1e-9)
    np.testing.assert_allclose(kp.T_ground, kp_ref.T_ground, atol=0)
    assert tok.token_id.dtype == tok_ref.token_id.dtype
    mismatch = float(np.mean(tok.token_id != tok_ref.token_id))
    assert mismatch <= 0.005, f"token mismatch rate {mismatch:.4f}"

    ## private helpers still exist and are the numpy implementations
    plane = enc._ransac_plane(synthetic_cloud, verbose=False)
    assert set(plane) == {"normal", "d", "point", "inliers"}
    pa, R, t = enc._align_ground(
        synthetic_cloud[:10], plane["normal"], plane["point"])
    assert pa.shape == (10, 3) and R.shape == (3, 3)

    ## tokenize_keypoints does not mutate the shared config
    enc.tokenize_keypoints(synthetic_cloud, kp, plane=plane, verbose=False)
    assert cfg.point_mode == "keypoints"


@pytest.fixture(scope="module")
def pipeline_data(cached_descriptors):
    d = cached_descriptors
    offsets, tids = d["offsets"], d["token_ids"]
    n = len(offsets) - 1
    step = n // 130
    scans = [tids[offsets[i * step]:offsets[i * step + 1]].astype(np.uint32)
             for i in range(130)]
    return scans[:100], scans[100:]


def test_full_retrieval_pipeline_parity(pipeline_data):
    db, queries = pipeline_data
    kw = dict(
        inlier_config=InLiER_Config(),
        shortlist_config=ShortlistConfig(topk_pct=0.15, min_shared_rows=3),
        beam_score_config=BEAMScoreConfig(
            topk=20, min_shared_bins=4, min_shared_az_cols=3,
            score_threshold=-2.0),
    )
    cpp_m = InLiER_Matcher(**kw)
    ref_m = RefMatcher(**kw)
    for i, tid in enumerate(db):
        cpp_m.add(i, InLiER_Tokens(token_id=tid))
        ref_m.add(i, InLiER_Tokens(token_id=tid))
    cpp_m.finalize(verbose=False)
    ref_m.finalize(verbose=False)
    assert len(cpp_m) == len(ref_m)
    np.testing.assert_array_equal(cpp_m.db_ids, ref_m.db_ids)

    top1_agree = 0
    for q in queries:
        qt = InLiER_Tokens(token_id=q)
        s1_c = cpp_m.shortlist(qt, verbose=False)
        s1_r = ref_m.shortlist(qt, verbose=False)
        assert len(s1_c.ids) == len(s1_r.ids)
        ## same shortlist scores per id
        map_c = dict(zip(s1_c.ids, s1_c.scores))
        map_r = dict(zip(s1_r.ids, s1_r.scores))
        common = set(map_c) & set(map_r)
        assert len(common) >= 0.95 * len(map_r)  # ties at the pct cut
        for i in common:
            np.testing.assert_allclose(map_c[i], map_r[i],
                                       rtol=1e-5, atol=1e-7)

        s2_c = cpp_m.beam_score(qt, s1_r.ids, topk=len(s1_r.ids),
                                verbose=False)
        s2_r = ref_m.beam_score(qt, s1_r.ids, topk=len(s1_r.ids),
                                verbose=False)
        bmap_c = dict(zip(s2_c.ids, zip(s2_c.scores, s2_c.best_shifts)))
        bmap_r = dict(zip(s2_r.ids, zip(s2_r.scores, s2_r.best_shifts)))
        assert set(bmap_c) == set(bmap_r)
        for i in bmap_r:
            assert bmap_c[i][0] == pytest.approx(bmap_r[i][0], abs=0)
            assert bmap_c[i][1] == bmap_r[i][1]

        top1_agree += (s2_c.ids[0] == s2_r.ids[0]
                       or bmap_r[s2_c.ids[0]][0] == bmap_r[s2_r.ids[0]][0])
    assert top1_agree == len(queries)


def test_verify_wrapper_smoke(cached_descriptors):
    """Wrapper verify runs against real data and returns a VerifyOutput
    with the exact same correspondence count as the reference."""
    d = cached_descriptors
    from inlier.core.Dataclasses import InLiER_Keypoints

    def scan(i):
        o = d["offsets"]
        s, e = o[i], o[i + 1]
        return (InLiER_Tokens(token_id=d["token_ids"][s:e].astype(np.uint32)),
                InLiER_Keypoints(p=d["kp_sensor"][s:e], T_ground=d["T_grounds"][i]))

    qt, qkp = scan(10)
    dt_, dkp = scan(11)  # adjacent scans -> should verify
    vcfg = VerifyConfig(ransac_iters=500, inlier_dist_thresh=1.0,
                        min_correspondences=32, min_ransac_inliers=16,
                        min_keypoint_inliers=8, spatial_tol=0)
    cpp_m = InLiER_Matcher()
    ref_m = RefMatcher()
    out_c = cpp_m.verify(qt, qkp, dt_, dkp, azimuth_shift=0,
                         config=vcfg, verbose=False)
    out_r = ref_m.verify(qt, qkp, dt_, dkp, azimuth_shift=0,
                         config=vcfg, verbose=False)
    assert out_c.n_correspondences == out_r.n_correspondences
    assert out_c.success == out_r.success
    assert isinstance(out_c.T_sensor, np.ndarray)
    if out_c.success:
        np.testing.assert_allclose(out_c.T_sensor[:3, 3],
                                   out_r.T_sensor[:3, 3], atol=2.0)
