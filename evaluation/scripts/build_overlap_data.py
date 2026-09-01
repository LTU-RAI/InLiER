#!/usr/bin/env python3
"""DEPRECATED -- use ``inlier gt build``.  Removed in 0.4.0.

Moved to ``inlier.eval.overlap_build``.  Every flag is unchanged; the CLI also
writes a provenance sidecar next to the matrix so the evaluation can verify it
was built with the accumulation parameters it is about to assume.
"""

import sys


def main():
    print("DEPRECATED: use `inlier gt build " + " ".join(sys.argv[1:])
          + "`\nForwarding...\n", file=sys.stderr)
    from inlier.eval.overlap_build import main as build_main

    return build_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
