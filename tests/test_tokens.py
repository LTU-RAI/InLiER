""" Token codec equivalence — C++ vs the numpy implementation."""

from __future__ import annotations

import numpy as np
import pytest

from inlier import _inlier_pybind as ip
from inlier.core.InLiER import InLiER

NH, NR, NS, NA = 10, 20, 7, 60


def test_pack_roundtrip_exhaustive():
    hb, rb, sb, ab = np.meshgrid(
        np.arange(NH), np.arange(NR), np.arange(NS), np.arange(NA), indexing="ij",
    )
    hb, rb, sb, ab = (a.ravel().astype(np.int64) for a in (hb, rb, sb, ab))

    tid = ip.pack_token_ids(hb, rb, sb, ab, NH, NR, NS, NA)
    assert tid.dtype == np.uint32
    assert np.unique(tid).size == NH * NR * NS * NA  # bijective
    
    ref = InLiER.pack_token_ids(hb, rb, sb, ab, NR, NS, NA, dtype=np.uint32)
    np.testing.assert_array_equal(tid, ref)
    
    hb2, rb2, sb2, ab2 = ip.unpack_token_ids(tid.astype(np.int64), NR, NS, NA)
    np.testing.assert_array_equal(hb2, hb)
    np.testing.assert_array_equal(rb2, rb)
    np.testing.assert_array_equal(sb2, sb)
    np.testing.assert_array_equal(ab2, ab)

    
def test_bin_radial_dense_sweep():
    r_max = 100.0
    r = np.concatenate([
        np.linspace(0.0, 2.0 * r_max, 200001),
        np.arange(NR + 1) * (r_max / NR),
        np.nextafter(np.arange(NR + 1) * (r_max / NR), np.inf),
        np.nextafter(np.arange(NR + 1) * (r_max / NR), -np.inf),
    ])  
    got = ip.bin_radial(r, r_max, NR)
    want = InLiER._bin_radial(r, r_max, NR)
    np.testing.assert_array_equal(got, want)
    
    
def test_bin_azimuth_dense_sweep():
    theta = np.linspace(-2.0 * np.pi, 2.0 * np.pi, 400001)
    got = ip.bin_azimuth(theta, NA)
    want = InLiER._bin_azimuth(theta, NA)
    np.testing.assert_array_equal(got, want)
    
    
def test_bin_height_matches_searchsorted():
    z_min, z_max = 0.0, 20.0
    z_edges = np.linspace(z_min, z_max, NH + 1)
    z = np.clip(np.linspace(-5.0, 25.0, 100001), z_min, z_max)
    want = np.clip(np.searchsorted(z_edges[1:], z, side="right").astype(np.int16), 0, NH - 1)
    got = ip.bin_height(z, z_min, z_max, NH)
    np.testing.assert_array_equal(got, want) 