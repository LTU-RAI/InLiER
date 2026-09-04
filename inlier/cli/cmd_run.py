"""``inlier run`` -- loop closures on data with no ground truth.

The deployment command.  Same pipeline as ``inlier eval``, no metrics: it
emits the closures and their 6-DoF constraints, and says nothing about whether
they are right, because without labels nothing can.

Two modes, inferred rather than declared.  Naming a prior map (any ``--db-*``
or ``--q-*`` flag) makes it a cross-session run against that map; otherwise it
streams one session causally against its own past.  Inference beats a ``--mode``
flag because a typo like ``--db-pth`` is an unrecognised argument and aborts,
where a mistyped mode value would just select the wrong one.  The resolved mode
is echoed and recorded in the run JSON either way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inlier.cli._common import add_generic_layout_flags

#: Flags that only mean something with a prior map.  Presence of any one of
#: these selects cross-session mode.
CROSS_DESTS = ("db_sequence", "q_sequence", "pair", "db_path", "q_path",
               "db_scans_dir", "db_pose_file", "q_scans_dir", "q_pose_file",
               "n_db", "n_q", "stride_db", "stride_q", "transform",
               "no_transform")

#: Flags that only mean something for a single streaming session.
SINGLE_DESTS = ("sequence", "sensor", "scans_dir", "pose_file", "n_scans",
                "stride", "exclusion")

DEFAULT_EXCLUSION = "frames=100"


def register(subparsers, parent) -> None:
    p = subparsers.add_parser(
        "run", parents=[parent],
        help="produce loop closures on data with no ground truth",
        description=(
            "Run the full pipeline and emit the loop closures it found, with "
            "a 6-DoF relative pose each -- what a SLAM back-end consumes. No "
            "metrics: there are no labels to score against, so the poses are "
            "treated as odometry and never used to accept or reject a "
            "closure. Streams one session against its own past, or queries a "
            "fixed prior map when a --db-* flag names one."
        ),
        epilog=(
            "examples:\n"
            "  inlier run --dataset-type kitti --dataset /data/kitti --sequence 00 \\\n"
            "      --n-scans 10 --exclusion seconds=30 --threshold 0.35 -o results/run\n"
            "\n"
            "  inlier run --dataset-type helipr --dataset /data/HeLiPR \\\n"
            "      --db-sequence Roundabout01 --q-sequence Roundabout03 \\\n"
            "      --pair O-Aeva --threshold 0.35 -o results/run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-type", dest="dataset_type",
                   choices=("helipr", "generic", "kitti"), default="helipr",
                   help="which loader to use (default: helipr)")
    p.add_argument("--dataset", type=str,
                   help="HeLiPR root, KITTI root (or one KITTI sequence "
                        "directory), or -- single-session generic only -- the "
                        "sequence directory")
    ## Shared between the helipr and kitti loaders, so it cannot live in
    ## either group: argparse rejects the same option string twice.
    p.add_argument("--sequence", type=str,
                   help="the session to stream (helipr name, or kitti id "
                        "e.g. 00)")

    single = p.add_argument_group("single session")
    single.add_argument("--sensor", type=str,
                        help="helipr sensor, e.g. Ouster or Aeva")
    single.add_argument("--n-scans", dest="n_scans", type=int, default=None,
                        help="scans accumulated per submap (default: 1)")
    single.add_argument("--stride", type=int, default=None,
                        help="step between submaps (default: --n-scans)")
    add_generic_layout_flags(single)
    single.add_argument("--exclusion", type=str, default=None,
                        help="how much recent past a frame may not match: "
                             f"frames=N, seconds=S or metres=M (default: "
                             f"{DEFAULT_EXCLUSION})")

    cross = p.add_argument_group("cross session (query against a prior map)")
    cross.add_argument("--db-sequence", dest="db_sequence", type=str,
                       help="the prior map's sequence")
    cross.add_argument("--q-sequence", dest="q_sequence", type=str,
                       help="the querying sequence")
    cross.add_argument("--pair", type=str,
                       help="helipr '<DB sensor>-<Q sensor>', e.g. O-Aeva")
    cross.add_argument("--db-path", dest="db_path", type=str)
    cross.add_argument("--q-path", dest="q_path", type=str)
    add_generic_layout_flags(cross, "db", "database")
    add_generic_layout_flags(cross, "q", "query")
    cross.add_argument("--n-db", dest="n_db", type=int, default=None)
    cross.add_argument("--n-q", dest="n_q", type=int, default=None)
    cross.add_argument("--stride-db", dest="stride_db", type=int, default=None)
    cross.add_argument("--stride-q", dest="stride_q", type=int, default=None)
    cross.add_argument("--transform", type=str, default=None,
                       help="4x4 mapping the prior map's world frame into the "
                            "query's; defaults to <db-path>/transform.txt")
    cross.add_argument("--no-transform", dest="no_transform",
                       action="store_true", default=None,
                       help="both sessions already share a world frame")

    scope = p.add_argument_group("retrieval scope")
    scope.add_argument("--search-radius", dest="search_radius", type=float,
                       default=0.0,
                       help="restrict candidates to frames within this many "
                            "metres, measured on the ODOMETRY poses -- the "
                            "same drifted estimate the running system has, so "
                            "unlike `inlier eval --search-radius` this is a "
                            "scope choice and not an oracle. 0 searches "
                            "everything (default: 0)")

    out = p.add_argument_group("output")
    out.add_argument("--threshold", dest="threshold", type=float, default=None,
                     help="REQUIRED. Accept a closure at or above this "
                          "verification score")
    out.add_argument("-o", "--output-dir", dest="output_dir", type=str,
                     default="results")
    out.add_argument("--cache-dir", dest="cache_dir", type=str,
                     default="cache_inlier",
                     help="descriptor cache; '' disables (default: cache_inlier)")
    out.add_argument("--top-k", dest="top_k", type=int, default=20,
                     help="candidates per query in scores_*.csv and "
                          "ranked_*.csv (default: 20)")
    out.add_argument("--live", action="store_true",
                     help="watch the run happen: process the session frame by "
                          "frame in a 3D viewer instead of stage by stage. "
                          "Same closures either way, but the query session is "
                          "encoded for real rather than read back out of the "
                          "descriptor cache, so a re-run costs full encoding. "
                          "Needs the [viz] extra")
    out.add_argument("--no-score-matrices", dest="score_matrices",
                     action="store_false",
                     help="skip scores_*.npz; it grows with the square of the "
                          "frame count")
    ## Registered only to answer for itself: it exists on both sibling
    ## commands, so people will reach for it, and argparse's "unrecognized
    ## arguments" would not explain why it is absent.
    out.add_argument("--threshold-policy", dest="threshold_policy",
                     default=None, help=argparse.SUPPRESS)
    p.set_defaults(func=run_deploy)


def _given(args, dests) -> list:
    """The flags in ``dests`` the user actually passed, as option strings."""
    return [f"--{d.replace('_', '-')}" for d in dests
            if getattr(args, d, None) not in (None, False)]


def resolve_mode(args) -> bool:
    """``True`` for cross-session.  Rejects a mix of the two vocabularies."""
    cross = _given(args, CROSS_DESTS)
    single = _given(args, SINGLE_DESTS)
    if cross and single:
        # No verb on either list: one flag or five, the sentence has to read.
        raise ValueError(
            f"cannot mix the two modes. Single-session flags given: "
            f"{', '.join(single)}. Cross-session flags given (these query a "
            f"prior map): {', '.join(cross)}. Drop one set.")
    return bool(cross)


def _sources(args, cross: bool, quiet: bool):
    """``(query_source, db_source_or_None)`` for the resolved mode."""
    from inlier.cli._sources import generic_source, kitti_source, require_flags
    from inlier.eval.datasets import HeLiPRSource, load_transform

    if not cross:
        n_scans = 1 if args.n_scans is None else args.n_scans
        stride = n_scans if args.stride is None else args.stride
        if args.dataset_type == "helipr":
            require_flags(args, ["dataset", "sequence", "sensor"],
                          "helipr (single-session)")
            return HeLiPRSource(args.dataset, args.sequence, args.sensor,
                                verbose=not quiet), None
        if args.dataset_type == "kitti":
            return kitti_source(args, n_scans=n_scans, stride=stride,
                                verbose=not quiet), None
        return generic_source(args, n_scans=n_scans, stride=stride,
                              verbose=not quiet), None

    n_db = 1 if args.n_db is None else args.n_db
    n_q = 1 if args.n_q is None else args.n_q
    stride_db = n_db if args.stride_db is None else args.stride_db
    stride_q = n_q if args.stride_q is None else args.stride_q

    if args.dataset_type == "helipr":
        from inlier.eval.datasets.helipr import parse_pair

        require_flags(args, ["dataset", "db-sequence", "q-sequence", "pair"],
                      "helipr (cross-session)")
        db_sensor, q_sensor = parse_pair(args.pair)
        return (HeLiPRSource(args.dataset, args.q_sequence, q_sensor,
                             verbose=not quiet),
                HeLiPRSource(args.dataset, args.db_sequence, db_sensor,
                             verbose=not quiet))
    if args.dataset_type == "kitti":
        require_flags(args, ["dataset", "db-sequence", "q-sequence"],
                      "kitti (cross-session)")
        return (kitti_source(args, prefix="q", n_scans=n_q, stride=stride_q,
                             verbose=not quiet),
                kitti_source(args, prefix="db", n_scans=n_db, stride=stride_db,
                             verbose=not quiet))

    transform = None
    q_source = generic_source(args, prefix="q", n_scans=n_q, stride=stride_q,
                              verbose=not quiet)
    db_source = generic_source(args, prefix="db", n_scans=n_db,
                               stride=stride_db, verbose=not quiet)
    if not args.no_transform:
        candidate = (Path(args.transform) if args.transform
                     else db_source.path / "transform.txt")
        if candidate.is_file():
            transform = load_transform(candidate)
        elif args.transform:
            raise FileNotFoundError(f"transform not found: {candidate}")
    if transform is not None:
        db_source.transform = transform
    return q_source, db_source


def run_deploy(args: argparse.Namespace) -> int:
    from inlier.cli._common import resolved_config
    from inlier.cli._sources import parse_exclusion
    from inlier.eval import artifacts
    from inlier.eval.deploy import DeploySpec, run

    quiet = getattr(args, "quiet", False)

    if args.threshold_policy is not None:
        raise ValueError(
            "inlier run has no --threshold-policy: selecting an operating "
            "point requires ground truth to sweep against, and there is "
            "none here. Pass --threshold VALUE, chosen on a sequence you "
            "evaluated with `inlier eval`.")
    if args.threshold is None:
        raise ValueError(
            "inlier run requires --threshold: there is no ground truth to "
            "select an operating point from, so it cannot be inferred. Pick "
            "one with `inlier eval` on a labelled sequence -- its "
            "max-recall-at-100%-precision threshold is the usual choice -- "
            "then fix it here.")
    # Also checked in deploy.validate, which guards the library entry point;
    # here so the CLI fails before printing a header for a run it will refuse.
    if args.threshold <= 0.0:
        raise ValueError(
            f"--threshold must be > 0, got {args.threshold}. A failed "
            f"verification scores exactly 0.0, so a threshold of 0 would emit "
            f"those failures as closures carrying an identity transform.")

    cross = resolve_mode(args)
    if cross and args.search_radius > 0.0 and not args.no_transform \
            and not args.transform and args.dataset_type != "generic":
        raise ValueError(
            "--search-radius compares the query's odometry position against "
            "the prior map's poses, which are in a different world frame "
            "until one is mapped into the other. Pass --transform FILE, or "
            "--no-transform if the two sessions already share a frame.")

    # Deploy mode, not eval: this is the deployment command, so the stage
    # score thresholds the config asks for are the ones that apply.  With the
    # shipped defaults that changes nothing -- stage2/rerank thresholds are
    # 0.0 and both stages score non-negatively, so eval's -2.0 is equally
    # always satisfied -- but a config that sets a positive threshold should
    # see it honoured here and relaxed there.
    resolved = resolved_config(args, mode="deploy")
    q_source, db_source = _sources(args, cross, quiet)
    exclusion = None if cross else parse_exclusion(
        args.exclusion or DEFAULT_EXCLUSION)

    described = q_source.describe()
    q_name = described.get("sequence") or Path(described.get("path", "")).name
    q_sensor = described.get("sensor") or "q"
    if cross:
        db_described = db_source.describe()
        db_name = (db_described.get("sequence")
                   or Path(db_described.get("path", "")).name)
        db_sensor = db_described.get("sensor") or "db"
    else:
        db_name, db_sensor = q_name, "self"

    exp_dir = artifacts.experiment_dirname(
        db_name, db_sensor, q_name, q_sensor,
        resolved.voxel_size, resolved.inlier.cell_size,
        resolved.inlier.N_h, resolved.inlier.N_r,
        resolved.inlier.N_a, resolved.inlier.N_s)
    tag = f"{q_name}_{q_sensor}_thr{args.threshold:g}"

    # `run_` prefixed: experiment_dirname is derived from the sequences and
    # encoder settings alone, so a run and an eval on the same pair would
    # otherwise share a directory -- and playback picks the newest
    # results_*.json it finds in one.
    output_dir = Path(args.output_dir) / f"run_{exp_dir}"

    if not quiet:
        print(f"\ninlier run  [{'cross-session' if cross else 'single-session'}]")
        print(f"  query      : {q_name} ({q_sensor})")
        print(f"  prior map  : {db_name} ({db_sensor})" if cross
              else f"  database   : the session's own past, {exclusion.describe()}")
        print(f"  threshold  : {args.threshold:g}   verify top-"
              f"{resolved.verify_topv}")
        print(f"  poses      : odometry -- used for submaps"
              + (", search radius" if args.search_radius > 0 else "")
              + " and diagnostics, never to accept a closure")
        print(f"  output     : {output_dir}")
        if args.live:
            print("  mode       : live -- streaming frame by frame, "
                  "encoding for real (the cache is not read)")

    spec = DeploySpec(
        resolved=resolved, source=q_source, db_source=db_source,
        threshold=float(args.threshold), exclusion=exclusion,
        search_radius=args.search_radius, output_dir=output_dir,
        cache_dir=(Path(args.cache_dir) if args.cache_dir else None),
        config_path=getattr(args, "config", None),
        top_k=args.top_k, score_matrices=args.score_matrices,
        verbose=not quiet, tag=tag, live=args.live)

    result = run(spec)
    if not quiet:
        print("\n" + result.summary())
        print("\nnote: `inlier play` cannot replay a run yet -- it needs the "
              "TP/FP labels a candidates_*.csv carries, which a run has no "
              "way to produce.")
    return 0
