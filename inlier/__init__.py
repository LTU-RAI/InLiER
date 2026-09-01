"""InLiER — Intermediate LiDAR Encoding for Robust Heterogeneous Place Recognition.

Public API
----------
``InLiER``            encoder: point cloud -> structural keypoints + mixed-radix tokens
``InLiER_Matcher``    MINT shortlist / BEAM rerank / token-guided verification
``InLiER_Config``     encoder configuration, plus the per-stage config dataclasses

Attribute access is lazy (PEP 562).  ``import inlier`` on its own touches
neither the compiled extension nor ``small_gicp``; the first attribute access
pulls in ``inlier.core``.  Two things depend on that: ``inlier --help`` stays
instant, and the ``INLIER_FORCE_PYTHON`` backend switch -- which is read at
*core* import time (``inlier/core/InLiER.py``) -- can still be set by the CLI
after the package itself has been imported.
"""

from typing import TYPE_CHECKING

from inlier.version import __version__

# public name -> module that defines it
_LAZY = {
    "InLiER": "inlier.core.InLiER",
    "InLiER_Matcher": "inlier.core.InLiER_Matcher",
    "InLiER_Config": "inlier.core.Dataclasses",
    "InLiER_Keypoints": "inlier.core.Dataclasses",
    "InLiER_Tokens": "inlier.core.Dataclasses",
    "InLiER_Output": "inlier.core.Dataclasses",
    "ShortlistConfig": "inlier.core.Dataclasses",
    "ShortlistOutput": "inlier.core.Dataclasses",
    "BEAMScoreConfig": "inlier.core.Dataclasses",
    "BEAMScoreOutput": "inlier.core.Dataclasses",
    "RerankConfig": "inlier.core.Dataclasses",
    "RerankOutput": "inlier.core.Dataclasses",
    "VerifyConfig": "inlier.core.Dataclasses",
    "VerifyOutput": "inlier.core.Dataclasses",
    "GICPRefineConfig": "inlier.core.Dataclasses",
    "GICPRefineOutput": "inlier.core.Dataclasses",
}


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if module_path is None:
        # Must be AttributeError: `from inlier import _inlier_pybind` relies on
        # the failure here to fall through to the submodule import machinery.
        raise AttributeError(f"module 'inlier' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache; __getattr__ is only consulted once per name
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:  # keep the lazy names visible to type checkers and IDEs
    from inlier.core.Dataclasses import (
        BEAMScoreConfig,
        BEAMScoreOutput,
        GICPRefineConfig,
        GICPRefineOutput,
        InLiER_Config,
        InLiER_Keypoints,
        InLiER_Output,
        InLiER_Tokens,
        RerankConfig,
        RerankOutput,
        ShortlistConfig,
        ShortlistOutput,
        VerifyConfig,
        VerifyOutput,
    )
    from inlier.core.InLiER import InLiER
    from inlier.core.InLiER_Matcher import InLiER_Matcher

__all__ = ["__version__", *sorted(_LAZY)]
