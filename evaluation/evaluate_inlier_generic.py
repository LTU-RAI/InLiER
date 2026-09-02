#!/usr/bin/env python3
"""DEPRECATED -- use ``inlier eval cross-session --dataset-type generic``.

Same protocol as the HeLiPR driver, different loader; both now share
``inlier/eval/protocols/cross_session.py``.  This shim translates the old flags
and forwards.  It is removed in 1.1.0.
"""

import sys

# Only genuine renames belong here: --snake_case is turned into --kebab-case
# for every flag by the CLI entry point.
_RENAMES = {
    "--pr-threshold": "--threshold",
}

# Flags the old generic driver accepted that the cross-session protocol does
# not yet expose.  Dropping one silently would change what the run measures, so
# they are reported rather than ignored.
_UNSUPPORTED = {
    "--refine-db-poses", "--refine-voxel-size", "--refine-distance-threshold",
    "--refine-icp-max-dist", "--refine-max-range", "--db-overlap-filter",
    "--local-radius",
}


def _translate(argv):
    from inlier.cli._common import kebab_flags

    out = ["eval", "cross-session", "--dataset-type", "generic"]
    dropped = []
    skip_next = False
    argv = kebab_flags(argv)
    for i, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        head = token.partition("=")[0]
        if head in _UNSUPPORTED:
            dropped.append(head)
            if "=" not in token and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                skip_next = True
            continue
        h, sep, tail = token.partition("=")
        out.append(_RENAMES.get(h, h) + sep + tail if sep else _RENAMES.get(token, token))
    return out, dropped


def main():
    new_argv, dropped = _translate(sys.argv[1:])
    print(
        "DEPRECATED: evaluation/evaluate_inlier_generic.py will be removed in "
        "InLiER 1.1.0.\n"
        "Use the CLI instead:\n\n"
        "    inlier " + " ".join(new_argv) + "\n",
        file=sys.stderr,
    )
    if dropped:
        print(
            "NOT FORWARDED -- these options have no equivalent yet, so this run "
            "is not the same as the one you asked for:\n    "
            + ", ".join(sorted(set(dropped)))
            + "\nRe-run without them, or stay on 0.2.x, if they matter.\n",
            file=sys.stderr,
        )
    print("Forwarding...\n", file=sys.stderr)
    from inlier.cli.main import main as cli_main

    return cli_main(new_argv)


if __name__ == "__main__":
    raise SystemExit(main())
