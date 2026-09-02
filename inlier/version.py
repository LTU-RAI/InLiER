"""Single source of truth for the package version.

``inlier.__version__`` and everything that prints a version (the encoder banner,
the CLI's ``--version``, the provenance block written into results JSON) read
from here, so they cannot drift apart.  ``inlier/core/banner.py`` previously
hardcoded ``"0.2.0"`` in its signature default.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("inlier")
except PackageNotFoundError:  # source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
