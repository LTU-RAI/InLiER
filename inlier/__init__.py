"""InLiER — Intermediate LiDAR Encoding for Robust Heterogeneous Place Recognition."""

from importlib.metadata import PackageNotFoundError, version

from inlier.core.InLiER import InLiER
from inlier.core.InLiER_Matcher import InLiER_Matcher
from inlier.core.Dataclasses import InLiER_Config

try:
    __version__ = version("inlier")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["InLiER", "InLiER_Matcher", "InLiER_Config", "__version__"]
