"""Dataset loaders.

``base.Sequence`` is the common shape; each module here produces one.  The
registry lets ``--dataset-type`` name a loader without the evaluation code
importing any of them directly.
"""

from typing import Dict, Type

from inlier.eval.datasets.base import Sequence, SequenceSource, load_transform
from inlier.eval.datasets.generic import Generic_Handler, GenericSource
from inlier.eval.datasets.helipr import HeLiPR_Handler, HeLiPRSource
from inlier.eval.datasets.kitti import (KITTI_Handler, KITTISource,
                                        read_calib_tr)

REGISTRY: Dict[str, Type] = {
    HeLiPRSource.name: HeLiPRSource,
    GenericSource.name: GenericSource,
    KITTISource.name: KITTISource,
}


def source_from_describe(described, *, root=None, verbose=False):
    """Rebuild the loader a finished run used, from its results-JSON block.

    ``inlier play`` replays artifacts rather than re-running anything, so it
    needs the same scans the evaluation encoded -- including, for the generic
    loader, the submap accumulation.  Asking the user to retype ``--n-scans``
    and ``--stride`` would let a replay disagree with its own run.
    """
    return get_source(described.get("dataset_type", "helipr")).from_describe(
        described, root=root, verbose=verbose)


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
    "KITTI_Handler",
    "KITTISource",
    "read_calib_tr",
    "Sequence",
    "SequenceSource",
    "get_source",
    "source_from_describe",
    "load_transform",
]
