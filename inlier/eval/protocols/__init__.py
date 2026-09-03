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

``online_lcd``
    Single session; the database grows as the query streams and recent frames
    are excluded.  Scored the way SLAM loop closure is scored -- F1max and
    max recall at 100% precision.

``inlier run`` is not here either, and that is the point: a protocol fixes
*both* halves above, and ``inlier run`` fixes only the first -- it produces
loop closures on data with no ground truth and declines to say whether any of
them is correct.  Filing it beside ``cross_session`` would make it look like a
peer whose output belongs in the same table, which is the one reading that must
never happen.  It lives in :mod:`inlier.eval.deploy`.

Two protocols are deliberately *absent*, and stay absent -- see
``docs/roadmap.md`` for the long version:

``online_global``
    Online localization against a fixed prior map.  A prior map is a finalized
    database that does not change while the queries run, so arrival order
    cannot affect what is retrievable and every query sees the whole database
    -- which is ``cross_session``.  The retrieval is identical; the only
    distinguishing requirement, a threshold not selected from the run being
    scored, is ``threshold_policy="fixed"``.

``multi_session``
    N sessions aggregated into one benchmark table.  Mechanically it is N(N-1)
    ``cross_session`` runs, which a caller can already loop over.  A table of
    pairwise retrieval scores is not evidence about a multi-session system --
    consistency as sessions accumulate depends on arrival order, on the
    back-end, and on drift that appears several sessions in, none of which
    decompose into pairs.  Left to field evaluation.
"""

from inlier.eval.protocols.base import RunResult

__all__ = ["RunResult"]
