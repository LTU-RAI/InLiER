"""Dataset loaders.

``base.Sequence`` is the common shape; each module here produces one.  The
registry lets ``--dataset-type`` name a loader without the evaluation code
importing any of them directly.
"""

from typing import Dict, Type

from inlier.eval.datasets.base import Sequence, SequenceSource, load_transform
from inlier.eval.datasets.generic import Generic_Handler, GenericSource
from inlier.eval.datasets.helipr import HeLiPR_Handler, HeLiPRSource

REGISTRY: Dict[str, Type] = {
    HeLiPRSource.name: HeLiPRSource,
    GenericSource.name: GenericSource,
}


def get_source(name: str):
    """Look up a loader class by ``--dataset-type`` name."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown dataset type {name!r}; available: {', '.join(sorted(REGISTRY))}"
        ) from None


__all__ = [
    "REGISTRY",
    "Generic_Handler",
    "GenericSource",
    "HeLiPR_Handler",
    "HeLiPRSource",
    "Sequence",
    "SequenceSource",
    "get_source",
    "load_transform",
]
