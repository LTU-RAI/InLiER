"""``inlier bench`` -- time the C++ core against the numpy reference.

Runs the same evaluation once per backend with the descriptor cache disabled,
so the encoder is actually exercised, and tabulates per-stage timings.  Each
backend needs its own process: ``INLIER_FORCE_PYTHON`` is read when
``inlier.core`` is first imported, so one interpreter cannot do both.
"""

from __future__ import annotations

import argparse
import sys


def register(subparsers, parent) -> None:
    p = subparsers.add_parser(
        "bench", parents=[parent],
        help="benchmark the backends against each other",
        description="Compare the C++ core and the numpy reference on the same run.",
    )
    sub = p.add_subparsers(dest="bench_cmd", required=True)
    cpp_vs_py = sub.add_parser(
        "cpp-vs-py", parents=[parent], aliases=["cpp_vs_py"], add_help=False,
        help="run the evaluation once per backend and tabulate the timings",
    )
    cpp_vs_py.set_defaults(func=run, _forward=True)


def run(args: argparse.Namespace) -> int:
    from inlier.eval import benchmark

    original = sys.argv[0]
    sys.argv[0] = "inlier bench cpp-vs-py"
    try:
        return int(benchmark.main(list(getattr(args, "_rest", []))) or 0)
    finally:
        sys.argv[0] = original
