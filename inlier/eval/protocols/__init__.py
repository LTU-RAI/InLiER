"""Evaluation protocols.

Place recognition is not one task.  A protocol fixes two things -- which
database entries a query is allowed to match, and what counts as a correct
match -- and those choices change the numbers far more than any tuning does.
Results from different protocols are not comparable, so each one names itself
in the results JSON it writes.

Implemented:

``cross_session``
    Full database, full query sequence, everything visible, overlap-matrix
    ground truth.  Offline.  This is the protocol behind the published results.

Planned (see the plan in the repository):

``online_lcd``
    Single session; the database grows as the query streams and recent frames
    are excluded.  Scored the way SLAM loop closure is scored.

``online_global``
    Fixed prior map, streaming query, decision at a fixed threshold with no
    post-hoc selection.

``multi_session``
    N sessions aggregated into one benchmark table.
"""

from inlier.eval.protocols.base import RunResult

__all__ = ["RunResult"]
