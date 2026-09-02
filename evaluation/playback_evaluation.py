#!/usr/bin/env python3
"""DEPRECATED -- use ``inlier play``.  Removed in 1.1.0.

Moved to ``inlier.eval.playback``.
"""

import sys


def main():
    print("DEPRECATED: use `inlier play " + " ".join(sys.argv[1:])
          + "`\nForwarding...\n", file=sys.stderr)
    from inlier.eval.playback import main as playback_main

    return playback_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
