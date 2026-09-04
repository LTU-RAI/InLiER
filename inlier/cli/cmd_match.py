"""``inlier match`` -- score two encoded scans against each other.

A deliberately small tool: two ``.npz`` files from ``inlier encode``, one
figure, one verdict.  It exists for the question you ask while tuning -- *why
did these two not match?* -- which an evaluation run answers only in aggregate,
several minutes later.

Pairs only, by design.  Directories, globs and all-vs-all belong to
``inlier eval``, which has the database, the ground truth and the metrics to
make those numbers mean something.  Here there is one pair and no ground
truth, so every number is a diagnostic rather than a result.

The point cloud is not stored in an ``.npz`` -- tokens and keypoints are -- so
the geometry panels reload it from the provenance the encoding carries, using
the loader that wrote it.  That last part matters for KITTI: its clouds only
reconstruct correctly through ``KITTI_Handler``, which applies the
camera->velodyne correction, so the encoding records which loader built it
rather than leaving it to be guessed.  When reloading is impossible (the
dataset moved, or the file predates the provenance), the keypoints are drawn
alone and the figure says so.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inlier.cli._common import user_path

VIZ_SUFFIXES = (".png", ".pdf", ".svg", ".jpg", ".jpeg")


def register(subparsers, parent) -> None:
    p = subparsers.add_parser(
        "match", parents=[parent],
        help="score two encoded scans against each other",
        description=(
            "Run the matching stages on exactly two `inlier encode` outputs "
            "and report what each one scored: MINT, BEAM, rerank, geometric "
            "verification, and the refined pose. A quick pairwise check, not "
            "an evaluation -- use `inlier eval` for anything with ground truth."
        ),
        epilog=(
            "examples:\n"
            "  inlier match a.npz b.npz                   print the scores\n"
            "  inlier match a.npz b.npz --viz             and plot the comparison\n"
            "  inlier match a.npz b.npz --viz-save c.png  write the figure\n"
            "  inlier match a.npz b.npz -o scores.json    machine-readable scores\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("query", type=user_path, metavar="QUERY.npz",
                   help="the scan being looked up")
    p.add_argument("database", type=user_path, metavar="DATABASE.npz",
                   help="the scan it is looked up against")

    # Same shape as `inlier encode`: -o is the data, --viz/--viz-save are the
    # picture.  Only the data format differs -- there is no .npz to write here,
    # so -o carries the scores.
    out = p.add_argument_group("output")
    out.add_argument("-o", "--output", type=user_path, default=None,
                     metavar="PATH", help="write the scores to PATH as JSON")
    out.add_argument("--no-clouds", dest="load_clouds", action="store_false",
                     help="skip reloading the point clouds; draw keypoints "
                          "only, and refine on keypoints")

    viz = p.add_argument_group("visualization")
    viz.add_argument("--viz", action="store_true",
                     help="plot the two scans, both descriptor stacks, and "
                          "the stage scores")
    viz.add_argument("--viz-save", dest="viz_save", type=user_path,
                     default=None, metavar="PATH",
                     help="write the figure to PATH instead of opening a "
                          "window (implies --viz)")
    viz.add_argument("--viz-dpi", dest="viz_dpi", type=int, default=150,
                     help="figure resolution when saving (150)")
    p.set_defaults(func=run)


def _handler_for(prov):
    """The loader that built this encoding, from its recorded provenance.

    KITTI has to be told apart: its clouds only reconstruct correctly through
    ``KITTI_Handler``, which applies the camera->velodyne correction from
    ``calib.txt``.  Rebuilding a KITTI submap with the generic loader would
    not merely fail to find the scans -- if it found them it would accumulate
    them with camera-frame poses, which is the bug ``--dataset-type kitti``
    exists to fix.
    """
    from inlier.eval.datasets.generic import Generic_Handler
    from inlier.eval.datasets.kitti import SCAN_SUBDIR, KITTI_Handler

    root = Path(prov.get("dataset", ""))
    kind = prov.get("dataset_type")
    if kind is None and str(root):
        # Encoded before the loader was recorded: a sequence directory holding
        # velodyne/ is KITTI and nothing else.
        kind = "kitti" if (root / SCAN_SUBDIR).is_dir() else "generic"
    if kind == "kitti":
        return KITTI_Handler(root, prov.get("sequence") or root.name,
                             verbose=False)
    return Generic_Handler(verbose=False)


def _reload_points(scan):
    """Recover the scan's points from the provenance the encoding carries.

    Returns ``(points, note)``; ``points`` is None when the cloud cannot be
    recovered, and ``note`` says why, for the figure and the log.
    """
    prov = scan.provenance
    try:
        if "submap_index" in prov and "dataset" in prov:
            data = _handler_for(prov).load_generic(
                Path(prov["dataset"]),
                n_scans=int(prov.get("n_scans", 1)),
                stride=int(prov.get("stride", prov.get("n_scans", 1))),
                select=[int(prov["submap_index"])])
            return data["point_clouds"][0], ""
        if "source" in prov:
            from inlier.eval.datasets.generic import Generic_Handler

            # A bare scan needs no poses, so the generic reader is right for
            # a .bin whatever wrote it.
            return Generic_Handler(verbose=False).load_scan_file(
                Path(prov["source"])), ""
    except (OSError, ValueError, RuntimeError, IndexError) as exc:
        return None, f"{scan.path.name}: {str(exc).splitlines()[0]}"
    return None, (f"{scan.path.name}: no source recorded (encoded before the "
                  f"provenance existed?)")


def _print_report(query, db, result, shortlist=None) -> None:
    from inlier.viz.match import mint_label

    v = result.verify
    print(f"\nquery    {query.path}  ({query.label}, {len(query.token_id)} tokens)")
    print(f"database {db.path}  ({db.label}, {len(db.token_id)} tokens)")
    print()
    print(f"  stage 1  MINT   {result.mint:.6f}   ({mint_label(shortlist)})")
    if result.mint_gate is not None:
        shared, required = result.mint_gate
        print(f"                  ^ not scored: the pair shares {shared} "
              f"occupied height slice(s), and stage1.min_shared_rows requires "
              f"{required}.")
        print(f"                    A 0 here means 'not compared', not 'not "
              f"alike'. Lower encoder.z_max (or raise N_h) so a flat sensor "
              f"fills more slices,")
        print(f"                    or lower stage1.min_shared_rows.")
    if result.beam is not None:
        print(f"  stage 2  BEAM   {result.beam:.6f}   "
              f"(azimuth shift {result.beam_shift})")
    if result.rerank is not None:
        print(f"  rerank          {result.rerank:.6f}   "
              f"(azimuth shift {result.rerank_shift})")
    if v is None:
        print("  verify          skipped (verify.skip)")
    else:
        print(f"  verify          {v.keypoint_inlier_ratio:.6f}   "
              f"({v.n_keypoint_inliers}/{v.n_total_keypoints} keypoint inliers, "
              f"{v.n_ransac_inliers}/{v.n_correspondences} RANSAC)")
        print(f"\n  {'VERIFIED' if v.success else 'NOT VERIFIED'}")
        if v.success:
            import numpy as np

            print(f"    yaw {np.degrees(v.yaw):+.3f} deg   "
                  f"t = [{v.tx:+.3f}, {v.ty:+.3f}, {v.tz:+.3f}] m   "
                  f"RMSE {v.inlier_rmse:.4f} m")
    if result.gicp is not None:
        g = result.gicp
        print(f"    GICP on {result.gicp_on}: "
              + ("converged" if g.converged else "did not converge")
              + f" in {g.n_iterations} iters, {g.n_inliers} inliers, "
                f"error {g.final_error:.4f}")


def run(args: argparse.Namespace) -> int:
    import json

    from inlier.cli._common import resolved_config
    from inlier.eval.pair import check_compatible, load_encoded, match_pair

    quiet = getattr(args, "quiet", False)
    viz = args.viz or args.viz_save is not None

    # -o writes JSON and --viz-save writes the picture.  Pointing -o at a .png
    # would otherwise write JSON into it under an image name and look like it
    # had worked, so say which flag was meant.
    if args.output is not None and args.output.suffix.lower() in VIZ_SUFFIXES:
        raise ValueError(
            f"-o writes the scores as JSON, but {args.output.name} names an "
            f"image. Use --viz-save {args.output} for the figure.")
    if args.viz_save is not None and args.viz_save.suffix.lower() not in VIZ_SUFFIXES:
        raise ValueError(
            f"--viz-save must name an image file "
            f"({', '.join(VIZ_SUFFIXES)}), got {args.viz_save}")

    query = load_encoded(args.query)
    db = load_encoded(args.database)

    # Stage thresholds relaxed: with one candidate, a threshold would replace
    # the score being asked for with an empty result.  See eval.pair.
    resolved = resolved_config(args, mode="eval")
    check_compatible(query, db, resolved.inlier)

    q_points = db_points = None
    notes = []
    if args.load_clouds:
        for scan, name in ((query, "q"), (db, "db")):
            points, note = _reload_points(scan)
            if note:
                notes.append(note)
            if name == "q":
                q_points = points
            else:
                db_points = points

    result = match_pair(resolved, query, db, q_points, db_points,
                        verbose=not quiet)

    if not quiet:
        _print_report(query, db, result, resolved.shortlist)
        for note in notes:
            print(f"  note: point cloud not reloaded -- {note}")

    if args.output is not None:
        payload = {
            "query": {"path": str(query.path), "label": query.label,
                      "n_tokens": int(len(query.token_id))},
            "database": {"path": str(db.path), "label": db.label,
                         "n_tokens": int(len(db.token_id))},
            "scores": result.as_dict(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        if not quiet:
            print(f"\nwrote {args.output}")

    if viz:
        import matplotlib

        if args.viz_save is not None and not args.viz:
            # Saving must work with no display, as everywhere else in the CLI.
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        from inlier.viz.match import match_figure

        figure = match_figure(query, db, result, resolved.inlier,
                              q_points=q_points, db_points=db_points,
                              shortlist=resolved.shortlist)
        if notes:
            figure.text(0.01, 0.005, "point clouds not shown: " + "; ".join(notes),
                        fontsize=7.5, color="#d62728")
        if args.viz_save is not None:
            args.viz_save.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.viz_save, dpi=args.viz_dpi, bbox_inches="tight")
            if not quiet:
                print(f"wrote {args.viz_save}")
        if args.viz:
            plt.show()
        plt.close(figure)

    return 0
