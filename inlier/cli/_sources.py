"""Turning CLI flags into dataset sources, shared by every command.

Lives apart from any one command module because ``inlier run`` needs the same
resolution as ``inlier eval`` and must not import it: the deployment command
depending on the evaluation command is backwards, and it is the kind of edge
that later grows an import of ``inlier.eval.protocols`` into a place that must
never have one.
"""

from __future__ import annotations

from pathlib import Path

def parse_exclusion(text: str):
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


def generic_source(args, *, prefix: str = "", n_scans: int, stride,
                    transform=None, verbose: bool = True):
    """Build a ``GenericSource`` from the dataset dir and/or explicit paths.

    ``--dataset`` stays the sequence's identity -- it names the cache entry,
    the run directory and the tag -- so it is only optional when both explicit
    paths are given and the scans directory's parent can stand in for it.
    """
    from inlier.eval.datasets import GenericSource

    dest = f"{prefix}_" if prefix else ""
    flag = f"--{prefix}-" if prefix else "--"
    dataset = getattr(args, f"{dest}path" if prefix else "dataset", None)
    scans_dir = getattr(args, f"{dest}scans_dir", None)
    pose_file = getattr(args, f"{dest}pose_file", None)

    if dataset is None and scans_dir is None:
        raise ValueError(
            f"--dataset-type generic needs a dataset directory or "
            f"{flag}scans; got neither")
    if dataset is None and pose_file is None:
        raise ValueError(
            f"{flag}scans without a dataset directory also needs {flag}poses: "
            f"there is nowhere else to look for the poses")

    if dataset is not None:
        return GenericSource(Path(dataset), n_scans, stride, transform,
                             verbose=verbose, scans_dir=scans_dir,
                             pose_file=pose_file)
    return GenericSource.from_paths(scans_dir, pose_file, n_scans, stride,
                                    transform, verbose=verbose)


def kitti_source(args, *, prefix: str = "", n_scans: int, stride,
                 verbose: bool = True):
    """Build a ``KITTISource``, resolving the sequence id here rather than there.

    The convenience of pointing ``--dataset`` straight at a sequence directory
    has to be settled before the source exists: ``KITTISource.tag`` names a
    descriptor-cache file and ``describe()`` is written into the results JSON,
    and neither may depend on what is currently on disk.
    """
    from inlier.eval.datasets import KITTISource
    from inlier.eval.datasets.kitti import SCAN_SUBDIR, normalise_sequence

    dest = f"{prefix}_" if prefix else ""
    flag = f"--{prefix}-" if prefix else "--"
    sequence_arg = getattr(args, f"{dest}sequence", None)

    if not args.dataset:
        raise ValueError("--dataset-type kitti requires --dataset")
    root = Path(args.dataset)
    if not root.is_dir():
        raise FileNotFoundError(f"--dataset {root} is not a directory")

    for name, value in ((f"{flag}scans", getattr(args, f"{dest}scans_dir", None)),
                        (f"{flag}poses", getattr(args, f"{dest}pose_file", None)),
                        ("--sensor", getattr(args, "sensor", None))):
        if value:
            raise ValueError(
                f"{name} does not apply to --dataset-type kitti: the loader "
                f"finds {SCAN_SUBDIR}/, the poses and calib.txt from the KITTI "
                f"layout itself, and KITTI odometry has one LiDAR.")

    is_sequence_dir = (root / SCAN_SUBDIR).is_dir()
    if is_sequence_dir:
        sequence = normalise_sequence(sequence_arg or root.name)
        if sequence_arg and normalise_sequence(sequence_arg) != normalise_sequence(root.name):
            raise ValueError(
                f"--dataset {root} is itself sequence {root.name!r}, but "
                f"--sequence says {sequence_arg!r}. Drop --sequence, or point "
                f"--dataset at the KITTI root instead.")
    elif sequence_arg:
        sequence = normalise_sequence(sequence_arg)
    else:
        raise ValueError(
            "--dataset-type kitti needs --sequence (e.g. --sequence 00), "
            f"unless --dataset points straight at a sequence directory -- one "
            f"containing {SCAN_SUBDIR}/.")

    return KITTISource(root, sequence, n_scans, stride, verbose=verbose)


def require_flags(args, names, why):
    missing = [n for n in names if not getattr(args, n.replace("-", "_"), None)]
    if missing:
        raise ValueError(f"--dataset-type {why} requires: "
                         + ", ".join("--" + n for n in missing))


