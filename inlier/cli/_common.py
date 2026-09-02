"""Shared CLI plumbing: global flags, backend selection, verbosity.

Backend selection is the awkward one.  ``inlier/core/InLiER.py`` decides
between the compiled extension and the numpy reference at *import* time, by
reading ``INLIER_FORCE_PYTHON`` -- so ``--backend python`` has to reach the
environment before ``inlier.core`` is first imported.  That is why
``inlier/__init__.py`` is lazy and why nothing in this package imports
``inlier.core`` at module level: :func:`apply_backend` runs during argument
parsing, well before any command touches the encoder.

``scripts/benchmark_cpp_vs_py.py`` solved the same problem by re-executing a
subprocess with a modified environment; that still works and is what
``inlier bench`` does, since it needs a clean process per backend anyway.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional, Sequence

BACKENDS = ("auto", "cpp", "python")


def add_global_flags(parser: argparse.ArgumentParser, suppress: bool = False) -> None:
    """Flags every subcommand accepts.

    Added to both the top-level parser and (via ``parents=``) every
    subcommand, so ``inlier --backend python doctor`` and
    ``inlier doctor --backend python`` both work.  The subcommand copies use
    ``default=SUPPRESS`` (``suppress=True``) so that omitting a flag there does
    not overwrite a value given before the subcommand -- with ordinary
    defaults, argparse writes the subparser's default into the shared
    namespace and silently discards the top-level value.
    """
    def default(value):
        return argparse.SUPPRESS if suppress else value

    group = parser.add_argument_group("common options")
    group.add_argument(
        "-c", "--config", type=str, default=default(None), metavar="FILE",
        help="YAML config; merged onto the packaged defaults "
             "(inlier/config/default.yaml). Omit to use the defaults alone.",
    )
    group.add_argument(
        "--set", dest="overrides", action="append", default=default([]), metavar="KEY=VALUE",
        help="Override one config key, e.g. --set stage1.topk=50 "
             "--set verify.skip=true. Repeatable; applied after --config.",
    )
    group.add_argument(
        "--backend", choices=BACKENDS, default=default("auto"),
        help="Computation backend. 'auto' uses the C++ core when importable "
             "and falls back to the numpy reference. (default: auto)",
    )
    group.add_argument(
        "-q", "--quiet", action="store_true", default=default(False),
        help="Suppress the encoder banner and per-stage progress output.",
    )
    group.add_argument(
        "-v", "--verbose", action="store_true", default=default(False),
        help="Extra diagnostics.",
    )


def apply_backend(backend: str) -> None:
    """Set ``INLIER_FORCE_PYTHON`` before anything imports ``inlier.core``."""
    if backend == "python":
        os.environ["INLIER_FORCE_PYTHON"] = "1"
    elif backend == "cpp":
        # Clear any inherited force-python so 'cpp' really means cpp.  If the
        # extension is missing the core still falls back with a warning; that
        # is reported by `inlier doctor` rather than raised here.
        os.environ.pop("INLIER_FORCE_PYTHON", None)


def apply_verbosity(args: argparse.Namespace) -> None:
    from inlier import verbosity

    if getattr(args, "quiet", False):
        verbosity.set_verbosity(verbosity.QUIET)
    elif getattr(args, "verbose", False):
        verbosity.set_verbosity(verbosity.DEBUG)
    else:
        verbosity.set_verbosity(verbosity.NORMAL)


def load_config(args: argparse.Namespace):
    """Merged config dict from ``--config`` + ``--set``."""
    from inlier.config import load

    return load(args.config, args.overrides)


def resolved_config(args: argparse.Namespace, mode: str = "eval"):
    from inlier.config import resolve

    return resolve(load_config(args), mode=mode)


def current_backend() -> str:
    """Which backend actually loaded.  Imports the core, so call it late."""
    from inlier.core.InLiER import _BACKEND

    return _BACKEND


def existing_path(value: str) -> Path:
    p = Path(value).expanduser()
    if not p.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {p}")
    return p


def user_path(value: str) -> Path:
    """argparse ``type=`` for a path that may start with ``~`` or ``$VAR``."""
    return Path(os.path.expandvars(os.path.expanduser(value)))


# `--opt=~/x` only; a bare `~/x` is handled separately, and `--opt ~/x` was
# already expanded by the shell before argv was built.
_OPT_WITH_TILDE = re.compile(r"^(--[A-Za-z0-9][\w-]*)=(~.*)$")


def expand_user(argv: Sequence[str]) -> List[str]:
    """Expand a leading ``~`` that the shell left alone.

    bash only expands a tilde at the start of a word, so ``--dataset ~/data``
    arrives as an absolute path but ``--dataset=~/data`` arrives with a
    literal ``~`` -- which then fails as a missing directory that is plainly
    there.  zsh expands both, so the bug only bites bash users.  Doing this on
    argv rather than per-argument covers the commands that forward their
    leftovers to a wrapped parser too.

    Values that are not paths are left alone: ``--set stage1.topk=~x`` goes to
    the config parser, not the filesystem.
    """
    out: List[str] = []
    for token in argv:
        if token.startswith("~"):
            out.append(os.path.expanduser(token))
            continue
        match = _OPT_WITH_TILDE.match(token)
        if match:
            out.append(f"{match.group(1)}={os.path.expanduser(match.group(2))}")
            continue
        out.append(token)
    return out


# The option *name* only: `--set a.b_c=1` must keep its value untouched.
_SNAKE_OPT = re.compile(r"^(--[A-Za-z0-9][A-Za-z0-9_-]*)(=.*)?$", re.DOTALL)


def kebab_flags(argv: Sequence[str]) -> List[str]:
    """Accept a ``--snake_case`` spelling of any ``--kebab-case`` flag.

    Every flag in this CLI is kebab-case.  The scripts it absorbed
    (``overlap_build``, ``overlap_validate``, ``playback``) spelled theirs with
    underscores, and the 0.2.x README documented those spellings, so they have
    to keep working -- but listing both in ``add_argument`` doubles the width
    of every ``--help`` line.  Rewriting the name on argv instead keeps the old
    invocations working and the help output single-spelled.

    Applied at the entry point rather than per parser, so the commands that
    forward their leftovers to a wrapped parser are covered too.
    """
    out: List[str] = []
    for token in argv:
        match = _SNAKE_OPT.match(token)
        if match and "_" in match.group(1):
            out.append(match.group(1).replace("_", "-") + (match.group(2) or ""))
            continue
        out.append(token)
    return out
