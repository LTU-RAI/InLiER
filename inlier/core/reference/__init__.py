"""Frozen pure-numpy reference implementation of the InLiER core.

Verbatim copy of the original Python implementation, kept as:
  1. ground truth for the C++ equivalence test-suite (tests/), and
  2. a zero-compiler fallback when the ``inlier._inlier_pybind``
     extension is unavailable (or ``INLIER_FORCE_PYTHON=1`` is set).

Do not modify — behaviour changes belong in the C++ core.
"""

from inlier.core.reference.InLiER import InLiER
from inlier.core.reference.InLiER_Matcher import InLiER_Matcher

__all__ = ["InLiER", "InLiER_Matcher"]
