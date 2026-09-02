"""``inlier gt`` -- build and sanity-check overlap ground truth.

Wraps the two scripts that used to live under ``evaluation/scripts/``.  Their
argument surface is preserved exactly (both spellings of every flag), because
docs/helipr-benchmark.md documents those command lines; the one addition is that ``build``
now writes a provenance sidecar so the evaluation can verify a matrix was built
the way it is about to be read.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from typing import List


def register(subparsers, parent) -> None:
    p = subparsers.add_parser(
        "gt", parents=[parent],
        help="build or validate overlap ground truth",
        description="Precompute the pairwise scan-overlap matrices that label "
                    "true and false positives, or inspect one before trusting it.",
    )
    sub = p.add_subparsers(dest="gt_cmd", required=True)

    build = sub.add_parser(
        "build", parents=[parent], add_help=False,
        help="compute the pairwise overlap matrix",
        description="Build an overlap matrix (plus its provenance sidecar).",
    )
    build.set_defaults(func=_build, _forward=True)

    validate = sub.add_parser(
        "validate", parents=[parent], add_help=False,
        help="plot and sanity-check an existing overlap matrix",
        description="Re-load the poses and scans behind a matrix and plot "
                    "trajectories, overlap distribution, and example pairs.",
    )
    validate.set_defaults(func=_validate, _forward=True)


def _forwarded(args: argparse.Namespace) -> List[str]:
    """The raw arguments after the subcommand, passed straight through."""
    return list(getattr(args, "_rest", []))


@contextlib.contextmanager
def _prog(name: str):
    """Make the wrapped script's own --help say `inlier gt build`, not `main.py`.

    Those parsers derive prog from sys.argv[0], which is this CLI's entry point.
    """
    original = sys.argv[0]
    sys.argv[0] = name
    try:
        yield
    finally:
        sys.argv[0] = original


def _build(args: argparse.Namespace) -> int:
    from inlier.eval import overlap_build

    with _prog("inlier gt build"):
        return int(overlap_build.main(_forwarded(args)) or 0)


def _validate(args: argparse.Namespace) -> int:
    from inlier.eval import overlap_validate

    with _prog("inlier gt validate"):
        return int(overlap_validate.main(_forwarded(args)) or 0)
