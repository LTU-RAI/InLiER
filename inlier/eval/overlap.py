"""Overlap ground-truth matrices and their provenance.

The overlap matrix is indexed *by submap*, so the accumulation parameters used
to build it (``n_db``, ``n_q``, ``stride_db``, ``stride_q``) must be the same
ones the evaluation later uses.  If they differ, row *i* of the matrix
describes a different stretch of trajectory than query *i* of the run, and the
ground truth is silently misaligned -- every metric is wrong and nothing
errors.  README.md mitigates this with a bold warning that the parameters
"**must match what you later pass to** evaluate_inlier_generic.py".

A warning in prose is not a check.  :func:`save` writes a machine-readable
sidecar next to the matrix recording exactly how it was built, and
:func:`check` compares it against what the evaluation is about to do.  Matrices
built before the sidecar existed are not lost: the builder already wrote the
same facts into the ``#`` header as prose, so :func:`load_provenance` falls
back to parsing that.

Layout: rows are database submaps, columns are query submaps -- ``overlap[db,
q]`` -- which is the opposite of the ``[query, db]`` ordering used everywhere
else in the evaluation.  It is preserved because the on-disk files use it.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SIDECAR_VERSION = 1


@dataclass
class OverlapProvenance:
    """How an overlap matrix was built."""

    n_db: int = 1
    n_q: int = 1
    stride_db: int = 1
    stride_q: int = 1
    voxel_size: float = 0.5
    max_range: float = 100.0
    distance_threshold: float = 100.0
    shape: Optional[Tuple[int, int]] = None
    db_id: str = ""
    q_id: str = ""
    transform: Optional[str] = None
    icp: bool = False
    version: int = SIDECAR_VERSION
    #: set when the values were recovered from the legacy text header
    from_legacy_header: bool = False

    #: fields that must agree between build time and evaluation time
    CRITICAL = ("n_db", "n_q", "stride_db", "stride_q")

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("from_legacy_header", None)
        if data.get("shape") is not None:
            data["shape"] = list(data["shape"])
        return data

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "OverlapProvenance":
        data = dict(data)
        shape = data.get("shape")
        if shape is not None:
            data["shape"] = tuple(shape)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def sidecar_path(matrix_path: Path) -> Path:
    return Path(matrix_path).with_suffix(".json")


# ---------------------------------------------------------------------------
#  Naming
# ---------------------------------------------------------------------------

def helipr_name(db_seq: str, db_sensor: str, q_seq: str, q_sensor: str) -> str:
    """``overlap_Roundabout01_Ouster_Roundabout03_Aeva.txt``"""
    return f"overlap_{db_seq}_{db_sensor}_{q_seq}_{q_sensor}.txt"


def generic_name(db_name: str, q_name: str, n_db: int, n_q: int,
                 stride_db: int, stride_q: int) -> str:
    """Submap parameters go in the filename so two builds cannot collide."""
    suffix = ""
    if n_db > 1 or n_q > 1:
        suffix = f"_Ndb{n_db}_Nq{n_q}"
    if stride_db != n_db or stride_q != n_q:
        suffix += f"_Sdb{stride_db}_Sq{stride_q}"
    return f"overlap_{db_name}_{q_name}{suffix}.txt"


# ---------------------------------------------------------------------------
#  I/O
# ---------------------------------------------------------------------------

def load(path: Path) -> np.ndarray:
    """Read a matrix.  Rows are DB submaps, columns are query submaps."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"overlap matrix not found: {path}\n"
            f"build it first:  inlier gt build ..."
        )
    matrix = np.loadtxt(path)
    if matrix.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D matrix, got shape {matrix.shape}")
    return matrix


def save(matrix: np.ndarray, path: Path, provenance: OverlapProvenance,
         extra_header: Optional[List[str]] = None) -> Path:
    """Write the matrix plus its provenance sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance.shape = (int(matrix.shape[0]), int(matrix.shape[1]))

    header = [
        "InLiER overlap matrix",
        f"Database: {provenance.db_id}  ({matrix.shape[0]} submaps, "
        f"n_db={provenance.n_db}, stride_db={provenance.stride_db})",
        f"Query:    {provenance.q_id}  ({matrix.shape[1]} submaps, "
        f"n_q={provenance.n_q}, stride_q={provenance.stride_q})",
        f"Voxel size (delta): {provenance.voxel_size} m   "
        f"tau: {1.5 * provenance.voxel_size} m",
        f"Pose distance threshold: {provenance.distance_threshold} m",
        f"Max point range: {provenance.max_range} m",
        f"Inter-sequence transform: {provenance.transform or 'none (shared frame)'}",
        f"ICP refinement: {'yes' if provenance.icp else 'none'}",
        "Rows = DB submap index, Columns = Q submap index "
        "(keyframe = first scan of each submap)",
    ]
    header.extend(extra_header or [])
    np.savetxt(path, matrix, fmt="%.6f", header="\n".join(header))

    side = sidecar_path(path)
    side.write_text(json.dumps(provenance.to_json(), indent=2) + "\n")
    return path


_HEADER_PATTERNS = {
    "n_db": re.compile(r"n_db=(\d+)"),
    "n_q": re.compile(r"n_q=(\d+)"),
    "stride_db": re.compile(r"stride_db=(\d+)"),
    "stride_q": re.compile(r"stride_q=(\d+)"),
    "voxel_size": re.compile(r"Voxel size \(delta\):\s*([0-9.eE+-]+)"),
    "distance_threshold": re.compile(r"Pose distance threshold:\s*([0-9.eE+-]+)"),
    "max_range": re.compile(r"Max point range:\s*([0-9.eE+-]+)"),
}


def _parse_legacy_header(path: Path) -> Optional[OverlapProvenance]:
    """Recover provenance from the ``#`` header of a pre-sidecar matrix."""
    lines: List[str] = []
    with open(path, "r") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            lines.append(line)
    if not lines:
        return None
    text = "".join(lines)
    values: Dict[str, Any] = {}
    for field_name, pattern in _HEADER_PATTERNS.items():
        match = pattern.search(text)
        if match:
            raw = match.group(1)
            values[field_name] = int(raw) if field_name.startswith(("n_", "stride_")) else float(raw)
    if not values:
        return None
    return OverlapProvenance(from_legacy_header=True, **values)


def load_provenance(path: Path) -> Optional[OverlapProvenance]:
    """Sidecar if present, else the legacy header, else ``None``."""
    path = Path(path)
    side = sidecar_path(path)
    if side.exists():
        return OverlapProvenance.from_json(json.loads(side.read_text()))
    if path.exists():
        return _parse_legacy_header(path)
    return None


# ---------------------------------------------------------------------------
#  Consistency
# ---------------------------------------------------------------------------

class OverlapMismatch(ValueError):
    """The matrix was built with parameters the evaluation is not using."""


def check(
    path: Path,
    matrix: np.ndarray,
    expected: OverlapProvenance,
    strict: bool = True,
) -> Optional[OverlapProvenance]:
    """Verify a matrix matches how the evaluation intends to read it.

    Shape is always checked -- it is the one thing that needs no sidecar and it
    catches the coarsest mismatch.  The submap parameters are checked whenever
    provenance is recoverable; without it a warning is issued rather than an
    error, so the matrices shipped in ``overlap_matrices/`` keep working.
    """
    found = load_provenance(path)

    if expected.shape is not None and tuple(matrix.shape) != tuple(expected.shape):
        raise OverlapMismatch(
            f"{Path(path).name}: matrix is {matrix.shape[0]}x{matrix.shape[1]} but the "
            f"evaluation has {expected.shape[0]} database and {expected.shape[1]} query "
            f"submaps.\nThe matrix is indexed by submap, so this misaligns ground truth "
            f"against retrieval. Rebuild it with the same accumulation parameters:\n"
            f"  n_db={expected.n_db} n_q={expected.n_q} "
            f"stride_db={expected.stride_db} stride_q={expected.stride_q}"
        )

    if found is None:
        warnings.warn(
            f"{Path(path).name}: no provenance sidecar and no parsable header; "
            f"cannot verify it was built with n_db={expected.n_db} n_q={expected.n_q} "
            f"stride_db={expected.stride_db} stride_q={expected.stride_q}. "
            f"Rebuild with `inlier gt build` to record them.",
            stacklevel=2,
        )
        return None

    differences = [
        (name, getattr(found, name), getattr(expected, name))
        for name in OverlapProvenance.CRITICAL
        if getattr(found, name) != getattr(expected, name)
    ]
    if differences:
        detail = "\n".join(f"  {n}: matrix built with {a}, evaluation using {b}"
                           for n, a, b in differences)
        source = "text header" if found.from_legacy_header else "sidecar"
        message = (
            f"{Path(path).name}: submap parameters disagree ({source}):\n{detail}\n"
            f"The matrix is indexed by submap, so this misaligns ground truth against "
            f"retrieval and every metric will be wrong."
        )
        if strict:
            raise OverlapMismatch(message)
        warnings.warn(message, stacklevel=2)

    return found
