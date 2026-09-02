#!/usr/bin/env python3
"""DEPRECATED -- use ``inlier eval cross-session``.

The evaluation moved into the installed package so that it is reachable after
``pip install inlier`` rather than only from a git checkout, and so the HeLiPR
and generic drivers -- which were ~90% the same file -- share one
implementation.  See ``inlier/eval/protocols/cross_session.py``.

This shim translates the old flags and forwards.  It is removed in 0.4.0.
"""

import sys

# old flag -> new flag.  Only genuine renames belong here: --snake_case is
# turned into --kebab-case for every flag by the CLI entry point.
_RENAMES = {
    "--pr-threshold": "--threshold",
}


def _translate(argv):
    from inlier.cli._common import kebab_flags

    out = ["eval", "cross-session", "--dataset-type", "helipr"]
    for token in kebab_flags(argv):
        head, sep, tail = token.partition("=")
        out.append(_RENAMES.get(head, head) + sep + tail if sep
                   else _RENAMES.get(token, token))
    return out


def main():
    new_argv = _translate(sys.argv[1:])
    print(
        "DEPRECATED: evaluation/evaluate_inlier_helipr.py will be removed in "
        "InLiER 0.4.0.\n"
        "Use the CLI instead:\n\n"
        "    inlier " + " ".join(new_argv) + "\n\n"
        "Forwarding...\n",
        file=sys.stderr,
    )
    from inlier.cli.main import main as cli_main

    return cli_main(new_argv)


if __name__ == "__main__":
    raise SystemExit(main())
