#!/usr/bin/env python3
"""DEPRECATED -- use ``inlier gt validate``.  Removed in 0.4.0.

Moved to ``inlier.eval.overlap_validate``.
"""

import sys


def main():
    print("DEPRECATED: use `inlier gt validate " + " ".join(sys.argv[1:])
          + "`\nForwarding...\n", file=sys.stderr)
    from inlier.eval.overlap_validate import main as validate_main

    return validate_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
