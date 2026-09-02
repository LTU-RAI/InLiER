"""Compatibility shim.

The dataset handlers moved into the installed package
(``inlier.eval.datasets``) so the CLI can reach them -- ``pyproject.toml``
ships only ``inlier/``, so anything left here would work from a git checkout
and nowhere else.

These re-exports keep ``from utils.HeLiPR_Handler import HeLiPR_Handler``
working for the scripts under ``evaluation/`` until they are removed in 1.1.0.
New code should import from ``inlier.eval.datasets``.
"""

from inlier.eval.datasets.generic import Generic_Handler
from inlier.eval.datasets.helipr import HeLiPR_Handler

__all__ = ["Generic_Handler", "HeLiPR_Handler"]
