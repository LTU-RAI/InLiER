"""Reading and writing evaluation artifacts.

The results JSON, the candidates CSV, and the per-pair verify CSV are a
contract: ``playback_evaluation.py`` reads the run's identity back out of the
JSON, and downstream tooling parses the CSVs.  So every key path the previous
schema used keeps its name and nesting, and ``schema_version`` is added
alongside rather than in place of anything.

What version 2 adds is provenance -- which protocol ran, which
operating-threshold policy chose the number quoted in ``confusion``, which
candidate filter was in force (and whether it consulted ground-truth poses),
and which backend produced the timings.  Without those, two results files with
different numbers are indistinguishable from two runs of the same thing.
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = 2

CANDIDATE_FIELDS = ["query_idx", "predicted_db_idx", "score", "match_type",
                    "overlap", "xy_distance_m", "has_gt_positive"]
#: The 4x4 transform, flattened.  Shared so the files that carry a pose cannot
#: drift apart about the column order -- and therefore about the frame
#: convention, which is ``p_db = T @ p_query`` in the sensor frame.
POSE_FIELDS = (["tx", "ty", "tz"]
               + [f"r{i}{j}" for i in range(3) for j in range(3)])
VERIFY_FIELDS = ["query_idx", "db_idx", "success"] + POSE_FIELDS
RANKED_FIELDS = ["query_idx", "rank", "db_idx", "score"]

#: ``inlier run``'s accepted loop closures.  No ground-truth column exists,
#: which is why ``CANDIDATE_FIELDS`` cannot be reused: three of its columns
#: (``match_type``, ``overlap``, ``has_gt_positive``) require labels.
CLOSURE_FIELDS = (
    ["query_idx", "db_idx", "rank", "score", "stage",
     "q_stamp", "db_stamp", "q_scan_idx", "db_scan_idx"]
    + POSE_FIELDS
    + ["yaw_deg",
       "n_correspondences", "n_ransac_inliers", "n_keypoint_inliers",
       "n_total_keypoints", "ransac_inlier_ratio", "inlier_rmse",
       "refined", "gicp_converged", "gicp_error", "gicp_inliers",
       "gicp_iterations",
       "odom_xy_distance_m", "odom_disagreement_m", "odom_disagreement_deg"]
)

#: Per-stage scores for the top candidates.  Deliberately carries no accept
#: decision: what a stage scored and what a run accepted are different facts,
#: and mixing them makes the file a record of neither.
SCORE_FIELDS = ["query_idx", "db_idx",
                "rank_mint", "mint",
                "rank_beam", "beam", "beam_shift",
                "rank_rerank", "rerank",
                "rank_verify", "verify"]


def git_sha() -> Optional[str]:
    """Short SHA of the working tree, when there is one."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def provenance(protocol: str, **extra: Any) -> Dict[str, Any]:
    """The block that says how this run was produced."""
    from inlier.version import __version__

    try:
        from inlier.core.InLiER import _BACKEND as backend
    except Exception:
        backend = "unknown"

    block = {
        "schema_version": SCHEMA_VERSION,
        "inlier_version": __version__,
        "protocol": protocol,
        "backend": backend,
        "git_sha": git_sha(),
    }
    block.update(extra)
    return block


def stage_block(
    recall_n: Optional[Dict[int, float]],
    recall_kpct: Optional[Dict[float, float]],
    pr_auc: Optional[float],
    **extra: Any,
) -> Optional[Dict[str, Any]]:
    """One ``stage1`` / ``stage2`` / ``verify`` entry, rounded as before."""
    if recall_n is None or pr_auc is None:
        return None
    block: Dict[str, Any] = {
        "recall_at_n": {str(n): round(v, 6) for n, v in recall_n.items()},
        "recall_at_kpct": {f"{k:.0f}pct": round(v, 6) for k, v in (recall_kpct or {}).items()},
        "pr_auc": round(pr_auc, 6),
    }
    block.update(extra)
    return block


def write_results(path: Path, results: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    return path


def read_results(path: Path) -> Dict[str, Any]:
    """Read a results file of either schema version.

    A version-1 file has no ``schema_version``; it is tagged as 1 on read so
    consumers can tell "produced before provenance existed" from "produced
    without a protocol".
    """
    data = json.loads(Path(path).read_text())
    data.setdefault("schema_version", 1)
    return data


def candidate_rows(
    conf_sims: Dict[int, Dict[int, float]],
    conf_rank: Optional[Dict[int, List[int]]],
    ground_truth: Dict[int, Any],
    n_queries: int,
    q_positions: Any,
    db_positions: Any,
    threshold: float,
    overlap: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """One top-1 decision per query at the operating threshold.

    Ordering follows :func:`inlier.eval.metrics.confusion` exactly so the CSV
    cannot disagree with the confusion matrix it accompanies.

    ``overlap`` is the scan-overlap matrix when the protocol has one; the
    single-session protocols do not, and report 0.0 rather than inventing a
    number.  For those, ``q_positions`` and ``db_positions`` are the same array.
    """
    rows: List[Dict[str, Any]] = []
    for j in range(n_queries):
        gt_set = set(ground_truth[j].tolist()) if ground_truth[j].size > 0 else set()
        q_sims = conf_sims.get(j, {})
        if conf_rank is not None and j in conf_rank:
            ordered = [d for d in conf_rank[j] if d in q_sims]
            if len(ordered) < len(q_sims):
                seen = set(ordered)
                ordered = ordered + sorted((d for d in q_sims if d not in seen),
                                           key=lambda d: q_sims[d], reverse=True)
            ranked_d = ordered
        else:
            ranked_d = sorted(q_sims, key=lambda d: q_sims[d], reverse=True)

        top1 = next((d for d in ranked_d if q_sims[d] >= threshold), None)
        if top1 is not None:
            match_type = ("TP" if top1 in gt_set else "FP") if gt_set else "FP"
            rows.append({
                "query_idx": j,
                "predicted_db_idx": top1,
                "score": round(float(q_sims[top1]), 6),
                "match_type": match_type,
                "overlap": (0.0 if overlap is None
                            else round(float(overlap[top1, j]), 6)),
                "xy_distance_m": round(float(np.linalg.norm(
                    q_positions[j, :2] - db_positions[top1, :2])), 3),
                "has_gt_positive": bool(gt_set),
            })
        else:
            rows.append({
                "query_idx": j, "predicted_db_idx": -1, "score": 0.0,
                "match_type": "FN" if gt_set else "TN",
                "overlap": 0.0, "xy_distance_m": 0.0,
                "has_gt_positive": bool(gt_set),
            })
    return rows


def write_candidates(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> Path:
    """The top-1 decision per query at the operating threshold."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_per_pair_verify(
    path: Path,
    verify_outputs: Dict[Tuple[int, int], Any],
) -> Path:
    """Every verified pair's estimated transform, flattened."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(VERIFY_FIELDS)
        for (q, d), out in sorted(verify_outputs.items()):
            T = np.asarray(out.T_sensor, dtype=np.float64)
            writer.writerow(
                [q, d, int(bool(out.success)), T[0, 3], T[1, 3], T[2, 3]]
                + [T[i, j] for i in range(3) for j in range(3)]
            )
    return path


def write_closures(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    """``inlier run``'s accepted closures, one row each.

    A header-only file is a legitimate result -- no place was revisited above
    the threshold -- so an empty ``rows`` still writes the header rather than
    nothing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CLOSURE_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_scores(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    """Per-stage scores for the top candidates, with no decision applied.

    Blank cells mean *that stage never scored this pair* -- the funnel narrows
    at every step -- and are not the same as a score of ``0.0``, which a stage
    can genuinely return.  ``write_scores`` therefore writes ``""`` for a
    missing value and never coerces it to zero.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCORE_FIELDS, restval="",
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def latency_block(latency_ms: np.ndarray, bounds: Sequence[int]) -> Dict[str, Any]:
    """Per-frame cost, over the frames that actually ran a query.

    Frames inside the opening exclusion window do nothing but an insertion, so
    averaging them in would report a latency no query ever experienced.
    """
    latency_ms = np.asarray(latency_ms, dtype=np.float64)
    queried = latency_ms[[t for t, b in enumerate(bounds) if b > 0]]
    if queried.size == 0:
        return {"n_frames": 0}
    return {
        "n_frames": int(queried.size),
        "mean_ms": round(float(queried.mean()), 3),
        "median_ms": round(float(np.median(queried)), 3),
        "p95_ms": round(float(np.percentile(queried, 95)), 3),
        "max_ms": round(float(queried.max()), 3),
    }


def write_ranked(
    path: Path,
    ranked: Dict[int, List[int]],
    scores: Dict[int, Dict[int, float]],
    top_k: int = 20,
) -> Path:
    """Top-K retrieval, unthresholded -- the interchange format.

    The candidates CSV is gated at the operating threshold and writes ``-1``
    below it, so it cannot be used to compare ranking against another method.
    This file can: it is the full ordering with raw scores, which also lets
    InLiER consume some other retrieval front-end's ranking and contribute only
    verification.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RANKED_FIELDS)
        for q in sorted(ranked):
            for rank, db in enumerate(ranked[q][:top_k]):
                writer.writerow([q, rank, db, scores.get(q, {}).get(db, float("nan"))])
    return path


def read_ranked(path: Path) -> Tuple[Dict[int, List[int]], Dict[int, Dict[int, float]]]:
    """Inverse of :func:`write_ranked`."""
    ranked: Dict[int, List[int]] = {}
    scores: Dict[int, Dict[int, float]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            q = int(row["query_idx"])
            db = int(row["db_idx"])
            ranked.setdefault(q, []).append(db)
            scores.setdefault(q, {})[db] = float(row["score"])
    return ranked, scores


# ---------------------------------------------------------------------------
#  Run directory naming
# ---------------------------------------------------------------------------

def _fmt_num(value: float) -> str:
    text = f"{value:g}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def short_sequence_tag(sequence: str) -> str:
    """``Roundabout01`` -> ``R01``: the run-directory abbreviation."""
    import re

    match = re.match(r"^([A-Za-z]+)(\d+)$", sequence)
    if match:
        return f"{match.group(1)[0].upper()}{match.group(2)}"
    return sequence[:4]


def experiment_dirname(
    db_sequence: str, db_sensor_tag: str,
    q_sequence: str, q_sensor_tag: str,
    voxel_size: float, cell_size: float,
    N_h: int, N_r: int, N_a: int, N_s: int,
) -> str:
    """Encoder settings in the directory name, so runs cannot overwrite."""
    return (
        f"db{short_sequence_tag(db_sequence)}-{db_sensor_tag}"
        f"-q{short_sequence_tag(q_sequence)}-{q_sensor_tag}"
        f"_vs{_fmt_num(voxel_size)}_cs{_fmt_num(cell_size)}"
        f"_nh{N_h}_nr{N_r}_na{N_a}_ns{N_s}"
    )
