"""``inlier config`` -- inspect the merged, resolved configuration.

Answers "what is actually going to run?".  Before this, the effective values
were spread across a YAML file, four hand-written flattening blocks, and the
dataclass defaults, with values derived from ``voxel_size`` computed inside
``run_evaluation`` where nothing could show them.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass


def register(subparsers, parent) -> None:
    p = subparsers.add_parser(
        "config", parents=[parent],
        help="show or dump the effective configuration",
        description="Show the configuration after merging the packaged "
                    "defaults, --config, and any --set overrides.",
    )
    sub = p.add_subparsers(dest="config_cmd", required=True)

    show = sub.add_parser("show", parents=[parent],
                          help="print the resolved dataclasses (what runs)")
    show.add_argument("--mode", choices=("eval", "deploy"), default="eval",
                      help="'eval' forces the stage score thresholds to -2.0 so the "
                           "PR sweep sees every candidate; 'deploy' keeps the "
                           "configured thresholds. (default: eval)")
    show.set_defaults(func=_show)

    dump = sub.add_parser("dump", parents=[parent],
                          help="print the merged YAML (a valid --config file)")
    dump.set_defaults(func=_dump)

    p.set_defaults(func=_show, config_cmd="show", mode="eval")


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _print_dataclass(title: str, obj) -> None:
    print(f"\n{title}")
    print("-" * 58)
    if obj is None:
        print("  (disabled)")
        return
    for key, value in asdict(obj).items() if is_dataclass(obj) else []:
        print(f"  {key.ljust(30)}: {_fmt(value)}")


def _show(args: argparse.Namespace) -> int:
    from inlier.cli._common import load_config
    from inlier.config import resolve
    from inlier.core.banner import print_banner

    # `show` is for a human deciding whether the run is set up right, so it
    # gets the banner.  `dump` deliberately does not -- its output has to stay
    # loadable as a config file.
    print_banner()

    cfg = load_config(args)
    r = resolve(cfg, mode=args.mode)

    src = args.config or "(packaged defaults)"
    print(f"config source : {src}")
    if args.overrides:
        print(f"overrides     : {', '.join(args.overrides)}")
    print(f"mode          : {r.mode}"
          + ("   (stage thresholds forced to -2.0 for the PR sweep)"
             if r.mode == "eval" else ""))
    print(f"voxel_size    : {_fmt(r.voxel_size)}")

    _print_dataclass("encoder (InLiER_Config)", r.inlier)
    _print_dataclass("stage 1 - MINT shortlist (ShortlistConfig)", r.shortlist)
    _print_dataclass("stage 2 - BEAM rerank (BEAMScoreConfig)", r.beam)
    _print_dataclass("rerank (RerankConfig)", r.rerank)
    _print_dataclass("verify (VerifyConfig)", r.verify)
    _print_dataclass("GICP refine (GICPRefineConfig)", r.gicp)

    print("\norchestration")
    print("-" * 58)
    for key, value in (
        ("skip_stage2", r.skip_stage2),
        ("run_rerank", r.run_rerank),
        ("skip_verify", r.skip_verify),
        ("skip_gicp", r.skip_gicp),
        ("verify_topv", r.verify_topv),
        ("stage2 score_threshold (deploy)", r.stage2_score_threshold_deploy),
        ("rerank score_threshold (deploy)", r.rerank_score_threshold_deploy),
    ):
        print(f"  {key.ljust(30)}: {_fmt(value)}")

    if "cell_size" in cfg.get("encoder", {}):
        derived = 2.0 * r.voxel_size
        if abs(r.inlier.cell_size - derived) > 1e-9:
            print(f"\nnote: encoder.cell_size ({_fmt(r.inlier.cell_size)}) differs from the "
                  f"2 x voxel_size guideline ({_fmt(derived)}).")
    return 0


def _dump(args: argparse.Namespace) -> int:
    from inlier.cli._common import load_config
    from inlier.config import dump

    print(dump(load_config(args)), end="")
    return 0
