"""``inlier bench`` -- time the C++ core against the numpy reference.

Runs the same evaluation once per backend with the descriptor cache disabled,
so the encoder is actually exercised, and tabulates per-stage timings.  Each
backend needs its own process: ``INLIER_FORCE_PYTHON`` is read when
``inlier.core`` is first imported, so one interpreter cannot do both.

The evaluation arguments are given here and forwarded verbatim, so a benchmark
run is the same command as the run it is timing.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from inlier.cli._common import forwarded_global_flags

USAGE = ("inlier bench cpp-vs-py --dataset /path/to/HeLiPR "
         "--db-sequence Roundabout01 --q-sequence Roundabout03 --pair O-Aeva")


def register(subparsers, parent) -> None:
    p = subparsers.add_parser(
        "bench", parents=[parent],
        help="benchmark the backends against each other",
        description="Compare the C++ core and the numpy reference on the same run.",
    )
    sub = p.add_subparsers(dest="bench_cmd", required=True, metavar="<subcommand>")
    cpp_vs_py = sub.add_parser(
        "cpp-vs-py", parents=[parent], add_help=False,
        help="run the evaluation once per backend and tabulate the timings",
    )
    cpp_vs_py.set_defaults(func=run, _forward=True)


def run(args: argparse.Namespace) -> int:
    from inlier.eval import benchmark

    # The global flags were consumed by the parser rather than left among the
    # leftovers, so put them back: without this, --config and --set would be
    # dropped on the way to the two eval subprocesses.
    rest: List[str] = forwarded_global_flags(args) + list(getattr(args, "_rest", []))
    if not rest:
        raise ValueError(
            "nothing to benchmark: pass the evaluation arguments, e.g.\n    "
            + USAGE)

    # --backend selects one backend; the whole point here is to run both, and
    # each subprocess gets its own INLIER_FORCE_PYTHON.  Saying so beats
    # honouring a flag that would make the comparison meaningless.
    backend = getattr(args, "backend", "auto")
    if backend != "auto":
        print(f"inlier bench: ignoring --backend {backend}; both backends are "
              "run. Use --backends to choose which.", file=sys.stderr)

    original = sys.argv[0]
    sys.argv[0] = "inlier bench cpp-vs-py"
    try:
        return int(benchmark.main(rest) or 0)
    finally:
        sys.argv[0] = original
