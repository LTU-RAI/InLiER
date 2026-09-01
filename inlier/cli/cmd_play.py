"""``inlier play`` -- replay a finished evaluation as an animation.

Everything the playback needs -- sequences, sensors, thresholds -- is read back
out of the ``results_*.json`` in the run directory, so the command takes the
run and nothing else.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from typing import List


def register(subparsers, parent) -> None:
    p = subparsers.add_parser(
        "play", parents=[parent], add_help=False,
        help="replay a run: trajectory, loop closures, matched keypoints",
        description="Animate a completed evaluation from its artifacts. "
                    "SPACE plays/pauses, arrow keys step; --record writes mp4.",
    )
    p.set_defaults(func=run, _forward=True)


def run(args: argparse.Namespace) -> int:
    from inlier.eval import playback

    original = sys.argv[0]
    sys.argv[0] = "inlier play"
    try:
        return int(playback.main(list(getattr(args, "_rest", []))) or 0)
    finally:
        sys.argv[0] = original
