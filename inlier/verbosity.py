"""Process-wide verbosity switch.

The core stage methods each take their own ``verbose`` argument and default to
``True``; that stays untouched -- the evaluation pipeline passes ``verbose=``
explicitly.  What this module adds is a global floor for the output that is
*not* behind a keyword: the figlet banner and encoder config table printed
unconditionally from ``InLiER.__init__``.  It lets ``inlier --quiet`` silence a
run without changing a single core signature.

Levels
------
``QUIET``   suppress the banner and config table
``NORMAL``  default: banner + config table, as before
``DEBUG``   reserved for extra diagnostics
"""

QUIET = 0
NORMAL = 1
DEBUG = 2

_level: int = NORMAL


def set_verbosity(level: int) -> None:
    """Set the global verbosity level (``QUIET`` / ``NORMAL`` / ``DEBUG``)."""
    global _level
    if level not in (QUIET, NORMAL, DEBUG):
        raise ValueError(f"verbosity must be one of 0/1/2, got {level!r}")
    _level = int(level)


def get_verbosity() -> int:
    """Return the current global verbosity level."""
    return _level


def is_quiet() -> bool:
    return _level <= QUIET


__all__ = ["QUIET", "NORMAL", "DEBUG", "set_verbosity", "get_verbosity", "is_quiet"]
