#!/usr/bin/env python3
"""Benchmark the FULL HeLiPR evaluation: C++ core vs pure-Python reference.

Runs ``evaluation/evaluate_inlier_helipr.py`` end-to-end twice — once with
the C++ backend (default) and once with ``INLIER_FORCE_PYTHON=1`` — both
encoding FROM SCRATCH (cache disabled), then diffs the two result JSONs.

The C++ core is a faithful port of the numpy reference, so the accuracy
metrics (PR-AUC, Recall@N) should match to within RANSAC noise — that
half of the table is an equivalence check.  The timing half is where the
C++ core pays off.

Pass the usual eval arguments straight through, e.g.:

    python scripts/benchmark_cpp_vs_py.py \
        --config config/default.yaml \
        --dataset ~/Documents/datasets/HeLiPR/ \
        --db_sequence Roundabout01 --q_sequence Roundabout03 --pair O-Aeva \
        --overlap_threshold 0.2 --max_pose_dist 10.0 --pr_threshold 0.3

Any ``--output_dir`` / ``--cache_dir`` you pass are overridden: results
land in ``<out-base>/{cpp,python}/`` and caching is forced off so both
backends really encode.  A comparison table is printed and saved to
``<out-base>/comparison_cpp_vs_py.{json,md}``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent
_EVAL = _REPO / "evaluation" / "evaluate_inlier_helipr.py"

# stage -> which timing key in results["timing"]
_TIMING_KEYS = [
    ("encoding", "encoding_s"),
    ("MINT", "stage1_retrieval_s"),
    ("BEAM", "stage2_reranking_s"),
    ("combined", "combined_reranking_s"),
    ("verify", "verify_s"),
    ("gicp", "gicp_s"),
]

# stage -> ms_per_query_* key (one query against the full DB)
_PERQUERY_KEYS = [
    ("MINT", "ms_per_query_s1"),
    ("BEAM", "ms_per_query_s2"),
    ("combined", "ms_per_query_comb"),
    ("verify", "ms_per_query_ver"),
]

# stage key in results dict -> label
_ACC_STAGES = [
    ("stage1", "MINT"),
    ("stage2", "BEAM"),
    ("combined", "Combined"),
    ("verify", "Verify"),
]


def run_backend(backend: str, out_dir: Path, passthrough: List[str]) -> float:
    """Run the eval with the given backend into out_dir. Returns wall time (s)."""
    env = dict(os.environ)
    env["INLIER_FORCE_PYTHON"] = "1" if backend == "python" else "0"

    argv = [sys.executable, str(_EVAL), *passthrough,
            "--output_dir", str(out_dir),
            "--cache_dir", ""]  # empty => no cache => encode from scratch

    print("=" * 74)
    print(f"  RUN [{backend}]  (backend={backend}, cache disabled)")
    print("  " + " ".join(argv))
    print("=" * 74, flush=True)

    t0 = time.time()
    proc = subprocess.run(argv, env=env, cwd=str(_REPO))
    wall = time.time() - t0
    if proc.returncode != 0:
        sys.exit(f"[{backend}] eval failed (exit {proc.returncode})")
    print(f"\n  [{backend}] wall time: {wall:.1f} s\n", flush=True)
    return wall


def load_results(out_dir: Path) -> Dict[str, Any]:
    """Load the newest results_*.json written under out_dir."""
    hits = sorted(glob.glob(str(out_dir / "**" / "results_*.json"), recursive=True),
                  key=os.path.getmtime)
    if not hits:
        sys.exit(f"no results_*.json found under {out_dir}")
    return json.load(open(hits[-1]))


def _fmt(v: Optional[float], width: int = 10, prec: int = 4) -> str:
    return " " * width if v is None else f"{v:>{width}.{prec}f}"


def _speedup(py: Optional[float], cpp: Optional[float]) -> str:
    if not py or not cpp:
        return "     -"
    return f"{py / cpp:>5.1f}x"


def compare(res: Dict[str, Dict[str, Any]], walls: Dict[str, float]) -> str:
    """Build a markdown comparison of the loaded per-backend result dicts."""
    have_py = "python" in res
    lines: List[str] = []
    lines.append("# C++ vs Python — full HeLiPR eval (from scratch)\n")

    cfg = next(iter(res.values())).get("config", {})
    di = next(iter(res.values())).get("dataset_info", {})
    lines.append(f"- DB/Q: `{cfg.get('db_sequence')}/{cfg.get('db_sensor')}` → "
                 f"`{cfg.get('q_sequence')}/{cfg.get('q_sensor')}`  "
                 f"(DB={di.get('n_db_scans')}, Q={di.get('n_q_scans')}, "
                 f"GT+={di.get('n_queries_with_gt')})")
    lines.append(f"- overlap_threshold={cfg.get('overlap_threshold')}  "
                 f"max_pose_dist={cfg.get('max_pose_dist')}  "
                 f"voxel_size={cfg.get('voxel_size')}\n")

    # ---- accuracy (equivalence check) ----
    lines.append("## Accuracy  (should match — C++ is a port of the reference)\n")
    lines.append("| stage | metric | c++ | python | Δ(py−cpp) |")
    lines.append("|---|---|---:|---:|---:|")
    for skey, slabel in _ACC_STAGES:
        cpp_s = res["cpp"].get(skey)
        py_s = res["python"].get(skey) if have_py else None
        if cpp_s is None and py_s is None:
            continue
        if skey == "stage1":  # MINT
            rows = [("Recall@100", "recall_at_n", "100")]
        elif skey == "stage2":  # BEAM
            rows = [("Recall@20", "recall_at_n", "20")]
        else:
            rows = [("Recall@1", "recall_at_n", "1"),
                    ("Recall@5", "recall_at_n", "5")]
        for label, key, sub in rows:
            def get(d):
                if d is None:
                    return None
                v = d.get(key)
                return (v.get(sub) if sub else v) if v is not None else None
            c, p = get(cpp_s), get(py_s)
            d = (p - c) if (have_py and c is not None and p is not None) else None
            lines.append(f"| {slabel} | {label} | {_fmt(c,8,4)} | "
                         f"{_fmt(p,8,4)} | {_fmt(d,8,4) if d is not None else '   -':>8} |")

    # ---- timing ----
    def _frames(d: Dict[str, Any]) -> int:
        di = d.get("dataset_info", {})
        return (di.get("n_db_scans") or 0) + (di.get("n_q_scans") or 0)

    def _enc_by_sensor(d: Dict[str, Any]):
        """{sensor_name: (encode_time_s, n_frames)} aggregated over DB and Q,
        or None if the JSON predates per-sensor encode timing."""
        t, di, cfg = d.get("timing", {}), d.get("dataset_info", {}), d.get("config", {})
        vals = (cfg.get("db_sensor"), cfg.get("q_sensor"),
                t.get("encoding_db_s"), t.get("encoding_q_s"),
                di.get("n_db_encoded"), di.get("n_q_encoded"))
        if any(v is None for v in vals):
            return None
        dbs, qs, tdb, tq, ndb, nq = vals
        agg: Dict[str, List[float]] = {}
        for s, tt, nn in [(dbs, tdb, ndb), (qs, tq, nq)]:
            a = agg.setdefault(s, [0.0, 0])
            a[0] += tt
            a[1] += nn
        return {s: (tt, nn) for s, (tt, nn) in agg.items()}

    n_cpp = _frames(res["cpp"])
    n_py = _frames(res["python"]) if have_py else 0

    lines.append("\n## Processing time (seconds)  — from scratch\n")
    lines.append("| stage | c++ (s) | python (s) | speedup |")
    lines.append("|---|---:|---:|---:|")
    tcpp = res["cpp"].get("timing", {})
    tpy = res["python"].get("timing", {}) if have_py else {}
    for label, key in _TIMING_KEYS:
        c = tcpp.get(key)
        p = tpy.get(key) if have_py else None
        if not c and not p:
            continue
        lines.append(f"| {label} | {_fmt(c,8,1)} | {_fmt(p,8,1)} | "
                     f"{_speedup(p, c) if have_py else '   -':>7} |")
        # per-frame encoding, broken down by sensor type (DB and Q sensors
        # aggregated by name, so O-O collapses to one row, O-Aeva → two).
        if key == "encoding_s":
            acpp = _enc_by_sensor(res["cpp"])
            apy = _enc_by_sensor(res["python"]) if have_py else None
            if acpp:
                sensors = list(acpp)
                for s in (apy or {}):
                    if s not in sensors:
                        sensors.append(s)
                for s in sensors:
                    tc, nc = acpp.get(s, (None, None))
                    ec = (tc / nc * 1000.0) if (tc and nc) else None
                    ep = None
                    if apy and s in apy:
                        tp, npf = apy[s]
                        ep = (tp / npf * 1000.0) if (tp and npf) else None
                    lines.append(f"| encoding/frame · {s} (ms) | {_fmt(ec,8,2)} | "
                                 f"{_fmt(ep,8,2)} | {_speedup(ep, ec) if have_py else '   -':>7} |")
            else:  # fallback for JSONs without per-sensor timing
                ec = (c / n_cpp * 1000.0) if (c and n_cpp) else None
                ep = (p / n_py * 1000.0) if (p and n_py) else None
                lines.append(f"| encoding / frame (ms) | {_fmt(ec,8,2)} | "
                             f"{_fmt(ep,8,2)} | {_speedup(ep, ec) if have_py else '   -':>7} |")

    # wall-clock totals (whole eval process, includes I/O + plotting)
    wc = walls.get("cpp")
    wp = walls.get("python") if have_py else None
    lines.append(f"| **wall total** | {_fmt(wc,8,1)} | {_fmt(wp,8,1)} | "
                 f"{_speedup(wp, wc) if have_py else '   -':>7} |")

    # ---- per-query timing: one query against the full DB ----
    lines.append("\n## Time per query (ms) — one query against the full DB\n")
    lines.append("| stage | c++ (ms) | python (ms) | speedup |")
    lines.append("|---|---:|---:|---:|")
    for label, key in _PERQUERY_KEYS:
        c = tcpp.get(key)
        p = tpy.get(key) if have_py else None
        if not c and not p:
            continue
        lines.append(f"| {label} | {_fmt(c,8,2)} | {_fmt(p,8,2)} | "
                     f"{_speedup(p, c) if have_py else '   -':>7} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full-eval C++ vs Python benchmark (from scratch).",
        epilog="All other args are forwarded verbatim to "
               "evaluation/evaluate_inlier_helipr.py.")
    ap.add_argument("--out-base", default="results/bench_cpp_vs_py",
                    help="Root for per-backend outputs and the comparison.")
    ap.add_argument("--backends", default="cpp,python",
                    help="Comma list of backends to run: cpp,python (order = run order).")
    args, passthrough = ap.parse_known_args()

    if not _EVAL.exists():
        sys.exit(f"eval script not found: {_EVAL}")

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for b in backends:
        if b not in ("cpp", "python"):
            sys.exit(f"unknown backend '{b}' (want cpp and/or python)")

    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    walls: Dict[str, float] = {}
    for b in backends:
        walls[b] = run_backend(b, out_base / b, passthrough)

    res = {b: load_results(out_base / b) for b in backends}

    md = compare(res, walls)
    print("\n" + md)

    (out_base / "comparison_cpp_vs_py.md").write_text(md)
    json.dump(
        {"walls_s": walls, "results": res},
        open(out_base / "comparison_cpp_vs_py.json", "w"), indent=2)
    print(f"Saved → {out_base/'comparison_cpp_vs_py.md'}")
    print(f"Saved → {out_base/'comparison_cpp_vs_py.json'}")


if __name__ == "__main__":
    main()
