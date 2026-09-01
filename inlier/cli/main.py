"""``inlier`` -- the command-line entry point.

Registered as a console script in ``pyproject.toml``.  Subcommands live in
sibling ``cmd_*`` modules and are imported lazily by :func:`_register`, so
``inlier --help`` does not pay for open3d, matplotlib, or the compiled core.

The ``--backend`` flag is handled by a pre-pass before the real parse, because
``inlier.core`` reads ``INLIER_FORCE_PYTHON`` at import time and a command
module may import the core as soon as it is loaded.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from inlier.cli._common import (BACKENDS, add_global_flags, apply_backend,
                                 apply_verbosity, expand_user)
from inlier.version import __version__


class _VersionAction(argparse.Action):
    """``--version`` prints the banner; ``-q --version`` prints just the string.

    The banner is the friendlier answer for a human, but ``inlier --version``
    is also the obvious thing to parse in a script or a CI check, so quiet mode
    keeps the bare ``inlier <version>`` line.  Printing the banner costs
    nothing: ``inlier.core.banner`` pulls in neither the compiled extension nor
    small_gicp.
    """

    def __init__(self, option_strings, dest, **kwargs):
        kwargs.setdefault("nargs", 0)
        kwargs.setdefault("help", "show the version and exit")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        from inlier import verbosity
        from inlier.core.banner import print_banner

        if verbosity.is_quiet():
            print(f"inlier {__version__}")
        else:
            print_banner()
        parser.exit()

# subcommand name -> module providing register(subparsers, parent)
COMMANDS = {
    "doctor": "inlier.cli.cmd_doctor",
    "config": "inlier.cli.cmd_config",
    "encode": "inlier.cli.cmd_encode",
    "gt": "inlier.cli.cmd_gt",
    "eval": "inlier.cli.cmd_eval",
    "play": "inlier.cli.cmd_play",
    "bench": "inlier.cli.cmd_bench",
}

EPILOG = """\
examples:
  inlier doctor --dataset /data/HeLiPR      check install and data layout
  inlier config show                        what will actually run
  inlier config show --set stage1.topk=50   with one value overridden
  inlier encode scan.pcd -o tokens.npz      encode a single scan
  inlier encode scan.pcd --viz              plot it and its descriptors
  inlier eval cross-session --help          evaluation protocols

common flags (-c/--config, --set, --backend, -q) are accepted either
before or after the command name.

run 'inlier <command> --help' for per-command options.
"""


def _prescan_quiet(argv: Sequence[str]) -> None:
    """Apply ``-q`` before parsing, so actions that fire during it honour it.

    ``--version`` runs inside ``parse_args``, which is before the parsed
    namespace exists -- without this, ``inlier -q --version`` would still print
    the banner.
    """
    from inlier import verbosity

    if any(token in ("-q", "--quiet") for token in argv):
        verbosity.set_verbosity(verbosity.QUIET)


def _prescan_backend(argv: Sequence[str]) -> None:
    """Apply ``--backend`` before any command module imports ``inlier.core``."""
    backend = "auto"
    for i, token in enumerate(argv):
        if token == "--backend" and i + 1 < len(argv):
            backend = argv[i + 1]
        elif token.startswith("--backend="):
            backend = token.split("=", 1)[1]
    if backend in BACKENDS:
        apply_backend(backend)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inlier",
        description="InLiER -- learning-free heterogeneous LiDAR place recognition.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action=_VersionAction)

    # Global flags go on the top level *and*, through this parent, on every
    # subcommand -- so they are accepted on either side of the command name.
    # The subcommand copies suppress their defaults so that omitting a flag
    # there does not overwrite a value given before the command name.
    add_global_flags(parser)
    parent = argparse.ArgumentParser(add_help=False)
    add_global_flags(parent, suppress=True)

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    _register(subparsers, parent)
    return parser


def _register(subparsers, parent) -> None:
    import importlib

    for name, module_path in COMMANDS.items():
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:  # a missing optional dependency
            _register_unavailable(subparsers, parent, name, exc)
            continue
        module.register(subparsers, parent)


def _register_unavailable(subparsers, parent, name: str, exc: Exception) -> None:
    """Keep an unimportable command visible, and explain why it cannot run."""
    p = subparsers.add_parser(name, parents=[parent],
                              help=f"(unavailable: {exc})")

    def _fail(_args):
        print(f"inlier {name}: unavailable -- {exc}", file=sys.stderr)
        print('try: pip install -e ".[eval]"', file=sys.stderr)
        return 1

    p.set_defaults(func=_fail)


def main(argv: Optional[List[str]] = None) -> int:
    argv = expand_user(sys.argv[1:] if argv is None else argv)
    _prescan_backend(argv)
    _prescan_quiet(argv)

    parser = build_parser()
    # Some subcommands (inlier gt build/validate) wrap a script whose full flag
    # surface is documented in the README; they opt in to passthrough with
    # _forward and receive the leftovers verbatim.  Every other command still
    # rejects unknown flags, so a typo is an error rather than a silent no-op.
    args, rest = parser.parse_known_args(argv)
    if getattr(args, "_forward", False):
        args._rest = rest
    elif rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")

    if getattr(args, "command", None) is None:
        parser.print_help()
        return 0

    apply_backend(getattr(args, "backend", "auto"))
    apply_verbosity(args)

    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError, NotImplementedError) as exc:
        # Expected, user-facing failures: a bad config key, a missing dataset.
        # Anything else keeps its traceback, which is what a bug report needs.
        print(f"inlier: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
