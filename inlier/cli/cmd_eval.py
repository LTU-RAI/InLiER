"""``inlier eval`` -- run an evaluation protocol.

Replaces ``evaluation/evaluate_inlier_helipr.py`` and
``evaluate_inlier_generic.py``, which were the same protocol behind two
loaders.  ``--dataset-type`` picks the loader; the protocol is written once.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def register(subparsers, parent) -> None:
    p = subparsers.add_parser(
        "eval", parents=[parent],
        help="run an evaluation protocol",
        description="Evaluate InLiER under one of the place-recognition protocols.",
    )
    sub = p.add_subparsers(dest="protocol", required=True)
    _register_cross_session(sub, parent)
    _register_online_lcd(sub, parent)


def _register_cross_session(sub, parent) -> None:
    p = sub.add_parser(
        "cross-session", parents=[parent],
        help="full database vs full query sequence (offline)",
        description=(
            "Cross-session place recognition: the whole database is visible to "
            "every query and correctness comes from a precomputed overlap "
            "matrix. This is the protocol behind the published results."
        ),
    )
    p.add_argument("--dataset-type", dest="dataset_type",
                   choices=("helipr", "generic"), default="helipr",
                   help="which loader to use (default: helipr)")

    helipr = p.add_argument_group("helipr options")
    helipr.add_argument("--dataset", type=str, help="HeLiPR dataset root")
    helipr.add_argument("--db-sequence", dest="db_sequence", type=str)
    helipr.add_argument("--q-sequence", dest="q_sequence", type=str)
    helipr.add_argument("--pair", type=str,
                        help="'<DB sensor>-<Q sensor>', e.g. O-Aeva")
    helipr.add_argument("--overlap-dir", dest="overlap_dir",
                        type=str, default="overlap_matrices")

    generic = p.add_argument_group("generic options")
    generic.add_argument("--db-path", dest="db_path", type=str)
    generic.add_argument("--q-path", dest="q_path", type=str)
    generic.add_argument("--overlap-file", dest="overlap_file", type=str)
    generic.add_argument("--n-db", dest="n_db", type=int, default=1,
                         help="scans accumulated per database submap (default: 1)")
    generic.add_argument("--n-q", dest="n_q", type=int, default=1)
    generic.add_argument("--stride-db", dest="stride_db", type=int, default=None,
                         help="step between database submaps (default: --n-db)")
    generic.add_argument("--stride-q", dest="stride_q", type=int, default=None)
    generic.add_argument("--transform", type=str, default=None,
                         help="4x4 mapping the DB world frame into the query world "
                              "frame; defaults to <db-path>/transform.txt if present")
    generic.add_argument("--no-transform", dest="no_transform",
                         action="store_true",
                         help="both sequences already share a world frame")

    gt = p.add_argument_group("ground truth")
    gt.add_argument("--overlap-threshold", dest="overlap_threshold", type=float, default=0.3,
                    help="minimum scan overlap for a positive (default: 0.3)")
    gt.add_argument("--max-pose-dist", dest="max_pose_dist",
                    type=float, default=25.0,
                    help="maximum XY pose distance for a positive, 0 to disable "
                         "(default: 25.0)")
    gt.add_argument("--no-strict-gt-check", dest="strict_gt_check",
                    action="store_false",
                    help="downgrade an overlap-matrix parameter mismatch from an "
                         "error to a warning")

    out = p.add_argument_group("output")
    out.add_argument("-o", "--output-dir", dest="output_dir",
                     type=str, default="results")
    out.add_argument("--cache-dir", dest="cache_dir", type=str,
                     default="cache_inlier",
                     help="descriptor cache; '' disables (default: cache_inlier)")
    out.add_argument("--threshold-policy", dest="threshold_policy",
                     choices=("max_precision", "max_f1", "fixed"),
                     default="max_precision",
                     help="how to pick the operating threshold. max_precision is "
                          "the default so published numbers reproduce; max_f1 is "
                          "what most baselines use. (default: max_precision)")
    out.add_argument("--threshold", "--pr-threshold", dest="threshold_value", type=float, default=None,
                     help="operating threshold; implies --threshold-policy fixed")
    p.set_defaults(func=run_cross_session)


def _parse_exclusion(text: str):
    """``frames=N`` / ``seconds=S`` / ``metres=M`` -> an Exclusion.

    Spelled as a unit because the three are not interchangeable: 100 frames is
    a different window at 1 Hz than at 10 Hz, and neither equals 50 m. Making
    the unit part of the value stops a run from being silently mis-scoped.
    """
    from inlier.eval.gt import Exclusion

    unit, _, value = text.partition("=")
    unit = unit.strip().lower()
    if not value or unit not in ("frames", "seconds", "metres"):
        raise ValueError(
            f"--exclusion takes frames=N, seconds=S or metres=M; got {text!r}")
    try:
        number = int(value) if unit == "frames" else float(value)
    except ValueError:
        raise ValueError(f"--exclusion {unit}= needs a number, got {value!r}")
    return Exclusion(**{unit: number})


def _register_online_lcd(sub, parent) -> None:
    p = sub.add_parser(
        "online-lcd", parents=[parent],
        help="single-session online loop closure detection",
        description=(
            "Online loop closure detection: one session streams past, the "
            "database grows as it goes, and every frame may only match frames "
            "older than the exclusion window. Scored the way SLAM scores loop "
            "closure -- F1max and max recall at 100% precision."
        ),
    )
    p.add_argument("--dataset-type", dest="dataset_type",
                   choices=("helipr", "generic"), default="helipr",
                   help="which loader to use (default: helipr)")
    ## One sequence, so one path: the HeLiPR root, or the generic sequence
    ## directory.  Unlike cross-session there is no second session to name.
    p.add_argument("--dataset", type=str,
                   help="HeLiPR dataset root, or the sequence directory "
                        "with --dataset-type generic")

    helipr = p.add_argument_group("helipr options")
    helipr.add_argument("--sequence", type=str, help="sequence name")
    helipr.add_argument("--sensor", type=str, help="sensor, e.g. Ouster or Aeva")

    generic = p.add_argument_group("generic options")
    generic.add_argument("--n-scans", dest="n_scans", type=int, default=1,
                         help="scans accumulated per submap (default: 1)")
    generic.add_argument("--stride", type=int, default=None,
                         help="step between submaps (default: --n-scans)")

    gt = p.add_argument_group("ground truth")
    gt.add_argument("--exclusion", type=str, default="frames=100",
                    help="how much recent past a frame may not match: "
                         "frames=N, seconds=S or metres=M (default: frames=100)")
    gt.add_argument("--max-pose-dist", dest="max_pose_dist", type=float, default=10.0,
                    help="maximum XY pose distance for a true revisit "
                         "(default: 10.0)")
    gt.add_argument("--search-radius", dest="search_radius", type=float, default=0.0,
                    help="restrict the database to frames within this many "
                         "metres of the query, as a SLAM local map would. "
                         "0 searches the whole causal past (default: 0). "
                         "WARNING: this measures against the ground-truth "
                         "pose, so it is an oracle and inflates every metric; "
                         "runs that use it are flagged in the results JSON")

    out = p.add_argument_group("output")
    out.add_argument("-o", "--output-dir", dest="output_dir", type=str,
                     default="results")
    out.add_argument("--cache-dir", dest="cache_dir", type=str,
                     default="cache_inlier",
                     help="descriptor cache; '' disables (default: cache_inlier)")
    out.add_argument("--threshold-policy", dest="threshold_policy",
                     choices=("max_precision", "max_f1", "fixed"),
                     default="max_precision",
                     help="how to pick the operating threshold (default: "
                          "max_precision; f1_max and max-recall-at-100%%-precision "
                          "are reported whichever is chosen)")
    out.add_argument("--threshold", "--pr-threshold", dest="threshold_value",
                     type=float, default=None,
                     help="operating threshold; implies --threshold-policy fixed")
    p.set_defaults(func=run_online_lcd)


def run_online_lcd(args) -> int:
    from inlier.cli._common import resolved_config
    from inlier.eval import artifacts
    from inlier.eval.datasets import GenericSource, HeLiPRSource
    from inlier.eval.protocols.online_lcd import OnlineLCDSpec, run

    resolved = resolved_config(args, mode="eval")
    quiet = getattr(args, "quiet", False)
    policy = args.threshold_policy
    if args.threshold_value is not None:
        policy = "fixed"
    exclusion = _parse_exclusion(args.exclusion)

    if args.dataset_type == "helipr":
        _require(args, ["dataset", "sequence", "sensor"], "helipr")
        source = HeLiPRSource(args.dataset, args.sequence, args.sensor,
                              verbose=not quiet)
        name, sensor = args.sequence, args.sensor
    else:
        _require(args, ["dataset"], "generic")
        path = Path(args.dataset)
        stride = args.stride if args.stride is not None else args.n_scans
        source = GenericSource(path, args.n_scans, stride, verbose=not quiet)
        name, sensor = path.name, "q"

    exp_dir = artifacts.experiment_dirname(
        name, sensor, name, "online",
        resolved.voxel_size, resolved.inlier.cell_size,
        resolved.inlier.N_h, resolved.inlier.N_r,
        resolved.inlier.N_a, resolved.inlier.N_s)
    tag = (f"{name}_{sensor}_lcd_{exclusion.unit}"
           f"{getattr(exclusion, exclusion.unit)}_pd{args.max_pose_dist}m")
    if args.search_radius > 0:
        tag += f"_r{args.search_radius}m"

    spec = OnlineLCDSpec(
        resolved=resolved,
        source=source,
        exclusion=exclusion,
        output_dir=Path(args.output_dir) / exp_dir,
        max_pose_dist=args.max_pose_dist,
        search_radius=args.search_radius,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        threshold_policy=policy,
        threshold_value=args.threshold_value,
        config_path=Path(args.config) if args.config else None,
        verbose=not quiet,
        tag=tag,
    )
    result = run(spec)
    if not quiet:
        print("\n" + result.summary())
    return 0


def _require(args, names, why):
    missing = [n for n in names if not getattr(args, n.replace("-", "_"), None)]
    if missing:
        raise ValueError(f"--dataset-type {why} requires: "
                         + ", ".join("--" + n for n in missing))


def run_cross_session(args: argparse.Namespace) -> int:
    from inlier.cli._common import resolved_config
    from inlier.eval import artifacts, overlap as overlapmod
    from inlier.eval.datasets import GenericSource, HeLiPRSource, load_transform
    from inlier.eval.datasets.helipr import parse_pair
    from inlier.eval.protocols.cross_session import CrossSessionSpec, run

    resolved = resolved_config(args, mode="eval")
    quiet = getattr(args, "quiet", False)
    policy = args.threshold_policy
    if args.threshold_value is not None:
        policy = "fixed"

    transform = None
    if args.dataset_type == "helipr":
        _require(args, ["dataset", "db-sequence", "q-sequence", "pair"], "helipr")
        db_sensor, q_sensor = parse_pair(args.pair)
        db_source = HeLiPRSource(args.dataset, args.db_sequence, db_sensor, verbose=not quiet)
        q_source = HeLiPRSource(args.dataset, args.q_sequence, q_sensor, verbose=not quiet)
        overlap_path = Path(args.overlap_dir) / overlapmod.helipr_name(
            args.db_sequence, db_sensor, args.q_sequence, q_sensor)
        db_tag, q_tag = args.pair.split("-")
        exp_dir = artifacts.experiment_dirname(
            args.db_sequence, db_tag, args.q_sequence, q_tag,
            resolved.voxel_size, resolved.inlier.cell_size,
            resolved.inlier.N_h, resolved.inlier.N_r,
            resolved.inlier.N_a, resolved.inlier.N_s)
        tag = (f"{args.db_sequence}_{db_sensor}_{args.q_sequence}_{q_sensor}"
               f"_ov{args.overlap_threshold}_pd{args.max_pose_dist}m")
    else:
        _require(args, ["db-path", "q-path", "overlap-file"], "generic")
        db_path, q_path = Path(args.db_path), Path(args.q_path)
        if not args.no_transform:
            candidate = Path(args.transform) if args.transform else db_path / "transform.txt"
            if candidate.exists():
                transform = load_transform(candidate)
            elif args.transform:
                raise FileNotFoundError(f"transform not found: {candidate}")
        stride_db = args.stride_db if args.stride_db is not None else args.n_db
        stride_q = args.stride_q if args.stride_q is not None else args.n_q
        db_source = GenericSource(db_path, args.n_db, stride_db, verbose=not quiet)
        q_source = GenericSource(q_path, args.n_q, stride_q, verbose=not quiet)
        overlap_path = Path(args.overlap_file)
        exp_dir = artifacts.experiment_dirname(
            db_path.name, "db", q_path.name, "q",
            resolved.voxel_size, resolved.inlier.cell_size,
            resolved.inlier.N_h, resolved.inlier.N_r,
            resolved.inlier.N_a, resolved.inlier.N_s)
        tag = (f"{db_path.name}_n{args.n_db}s{stride_db}"
               f"_{q_path.name}_n{args.n_q}s{stride_q}"
               f"_ov{args.overlap_threshold}_pd{args.max_pose_dist}m")

    spec = CrossSessionSpec(
        resolved=resolved,
        db_source=db_source,
        q_source=q_source,
        overlap_path=overlap_path,
        output_dir=Path(args.output_dir) / exp_dir,
        overlap_threshold=args.overlap_threshold,
        max_pose_dist=args.max_pose_dist,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        threshold_policy=policy,
        threshold_value=args.threshold_value,
        config_path=Path(args.config) if args.config else None,
        db_transform=transform,
        strict_overlap_check=args.strict_gt_check,
        verbose=not quiet,
        tag=tag,
    )
    result = run(spec)
    if not quiet:
        print("\n" + result.summary())
    return 0
