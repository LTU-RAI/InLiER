"""InLiER: An Intermediate LiDAR Encoding for Robust Heterogeneous Place Recognition

Thin wrapper over the C++ core (``inlier._inlier_pybind``). The public
API, output dataclasses, and verbose prints are identical to the
original pure-numpy implementation, which is preserved verbatim in
``inlier.core.reference`` and used as:

* the base class — every private numpy helper (``_ransac_plane``,
  ``_align_ground``, ``pack_token_ids`` …) keeps working unchanged;
* the automatic fallback when the compiled extension is unavailable or
  ``INLIER_FORCE_PYTHON=1`` is set.

Ground-plane RANSAC intentionally stays in numpy (inherited): it is not
a hotspot and keeping numpy's seeded PCG64 stream makes the planes —
and therefore keypoints and tokens — bit-identical to the original
implementation. The C++ core accelerates everything downstream of the
plane: BEV slicing + NMS keypoints, shape PCA, and tokenization.

Public API
----------
extract_keypoints(points_xyz, ...)      →   InLiER_Keypoints
tokenize(points_xyz, keypoints, ...)    →   InLiER_Tokens
encode(points_xyz, ...)                 →  (InLiER_Keypoints, InLiER_Tokens)

Token definition
----------------
token_id = ((hb × N_r + rb) × N_s + sb) × N_a + ab   (uint32, or uint64
when the mixed-radix product exceeds 2³²) — fully invertible given
(N_h, N_r, N_s, N_a).
"""

from __future__ import annotations

import os as _os
import time as _time
import warnings as _warnings
from typing import Dict, Optional, Tuple

import numpy as np

from inlier.core.Dataclasses import InLiER_Config, InLiER_Keypoints, InLiER_Tokens
from inlier.core.reference.InLiER import InLiER as _ReferenceInLiER

_BACKEND = "python"
if _os.environ.get("INLIER_FORCE_PYTHON", "0") != "1":
    try:
        from inlier import _inlier_pybind as _ip
        from inlier.core import _cfg_bridge as _bridge

        _BACKEND = "cpp"
    except ImportError as _err:  # pragma: no cover - build-dependent
        _warnings.warn(
            f"inlier: C++ extension unavailable ({_err}); using the "
            f"pure-Python reference implementation.")


if _BACKEND == "python":

    InLiER = _ReferenceInLiER

else:

    class InLiER(_ReferenceInLiER):
        """InLiER encoder: keypoint extraction + token_id assignment.

        C++-accelerated; see the module docstring for the backend split.
        """

        def __init__(self, config: InLiER_Config = InLiER_Config()) -> None:
            super().__init__(config)
            self._cpp = _ip._Encoder(_bridge.to_cpp_inlier_config(config))
            self._token_dtype = _bridge.token_dtype(config)

        ## ---- public API (C++ backed) ----

        def extract_keypoints(
                self,
                points_xyz: np.ndarray,
                plane: Optional[Dict[str, np.ndarray]] = None,
                verbose: Optional[bool] = True,
            ) -> InLiER_Keypoints:
            t0 = _time.perf_counter()
            P = np.asarray(points_xyz, dtype=np.float64)
            if P.ndim != 2 or P.shape[1] != 3:
                raise ValueError("points_xyz must have shape (N, 3)")
            if P.shape[0] == 0:
                return self._empty_keypoints()

            if plane is None:
                plane = self._ransac_plane(P, verbose=verbose)

            p, T_ground = self._cpp.extract_keypoints(
                P, _bridge.plane_dict_to_cpp(plane))
            kp = InLiER_Keypoints(p=p, T_ground=T_ground)

            if verbose:
                dt = _time.perf_counter() - t0
                print(
                    f"[InLiER] extract_keypoints: {p.shape[0]} keypoints "
                    f"from {P.shape[0]} points in {dt * 1000:.0f}ms"
                )
            return kp

        def tokenize(
                self,
                points_xyz: np.ndarray,
                keypoints: InLiER_Keypoints,
                plane: Optional[Dict[str, np.ndarray]] = None,
                verbose: Optional[bool] = True,
            ) -> InLiER_Tokens:
            return self._tokenize_impl(points_xyz, keypoints, plane,
                                       verbose, mode="config")

        def tokenize_keypoints(
                self,
                points_xyz: np.ndarray,
                keypoints: InLiER_Keypoints,
                plane: Optional[Dict[str, np.ndarray]] = None,
                verbose: Optional[bool] = True,
            ) -> InLiER_Tokens:
            ## explicit mode argument — no config mutation as in the
            ## reference implementation
            return self._tokenize_impl(points_xyz, keypoints, plane,
                                       verbose, mode="keypoints")

        def encode(
                self,
                points_xyz: np.ndarray,
                plane: Optional[Dict[str, np.ndarray]] = None,
                verbose: Optional[bool] = True,
            ) -> Tuple[InLiER_Keypoints, InLiER_Tokens]:
            t0 = _time.perf_counter()

            P = np.asarray(points_xyz, dtype=np.float64)
            if plane is None:
                plane = self._ransac_plane(P, verbose=verbose)

            keypoints = self.extract_keypoints(P, plane=plane, verbose=verbose)
            tokens = self.tokenize(P, keypoints, plane=plane, verbose=verbose)

            if verbose:
                dt = _time.perf_counter() - t0
                print(f"[InLiER] encode: total pipeline in {dt * 1000:.0f}ms")

            return keypoints, tokens

        ## ---- internals ----

        def _tokenize_impl(self, points_xyz, keypoints, plane, verbose,
                           mode: str) -> InLiER_Tokens:
            t0 = _time.perf_counter()
            cfg = self._config
            pmode = cfg.point_mode.lower() if mode == "config" else mode

            P = np.asarray(points_xyz, dtype=np.float64)
            if pmode == "all_points" and plane is None:
                plane = self._ransac_plane(P, verbose=verbose)

            cpp_plane = (_bridge.plane_dict_to_cpp(plane)
                         if plane is not None else None)
            token_id = self._cpp.tokenize(
                P, np.asarray(keypoints.p, dtype=np.float64),
                np.asarray(keypoints.T_ground, dtype=np.float64),
                cpp_plane, mode)
            tokens = InLiER_Tokens(token_id=token_id)

            if verbose:
                K = token_id.shape[0]
                if K == 0:
                    print("[InLiER] tokenize: 0 tokens (empty input)")
                else:
                    dt = _time.perf_counter() - t0
                    n_unique = int(np.unique(token_id).shape[0])
                    print(
                        f"[InLiER] tokenize: {K} tokens ({n_unique} unique "
                        f"IDs), mode='{pmode}' in {dt * 1000:.0f}ms"
                    )
            return tokens
