"""Reading a sequence one frame at a time.

``SequenceSource.load()`` returns the whole sequence at once, which is what
every batch protocol wants and what a live run cannot afford: a KITTI sequence
is thousands of scans and several gigabytes of points, and the point of
``inlier run --live`` is to process each one as it arrives rather than after
all of them have.

Nothing here is a new reader.  The loaders already had every primitive --
``list_scan_files``, ``load_scan_file``, ``load_poses``, and the submap rule in
``inlier.eval.submaps`` -- and ``load_generic(select=[i])`` already builds a
single submap out of a sequence.  What it does *not* do is remember anything
between calls, so driving it once per frame would re-parse the pose file and
re-list the scan directory for every frame.  This module lists once and then
walks the windows.

The frames it yields are the same frames, in the same order, with the same
indices, as ``source.load()`` would have produced.  That is the contract the
live driver's equivalence with the batch one rests on, including the detail
that a window whose scans were all empty is dropped rather than yielded, so
indices stay dense exactly as ``load_generic`` leaves them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import numpy as np

from inlier.eval.submaps import build_submap, submap_windows


@dataclass
class Frame:
    """One submap as it arrives."""

    index: int              #: position in the sequence, after empty windows are dropped
    points: np.ndarray      #: (N, 3) float32, in the keyframe's own frame
    pose: np.ndarray        #: (4, 4) float64, global
    stamp: float            #: pose timestamp; 0.0 when the loader has none


def _generic_parts(source):
    """``(handler, root)`` for the two loaders that accumulate submaps."""
    handler = source._handler
    root = handler.seq_dir if source.name == "kitti" else source.path
    return handler, root


def keyframe_stamps(source) -> List[float]:
    """The stamp of every submap's keyframe, without reading a single scan.

    The exclusion window needs a timestamp axis before the run starts; poses
    and their timestamps come from one text file, so this costs nothing and
    reads no points.
    """
    if source.name == "helipr":
        handler = source._handler
        seq_path = source.dataset_path / source.sequence
        files = handler.list_scan_files(seq_path, source.sensor, type=source.scan_type)
        _, p_stamps = handler.load_poses(seq_path, source.sensor)
        stamps = handler._bin_timestamps(files)
        return [float(p_stamps[handler.get_closest_pose_index(ts, p_stamps)])
                for ts in stamps]

    handler, root = _generic_parts(source)
    _, stamps = handler.load_poses(root)
    windows = submap_windows(len(handler.list_scan_files(root)),
                             source.n_scans, source.stride)
    return [float(stamps[w[0]]) if stamps else 0.0 for w in windows]


def frame_count(source) -> int:
    """How many frames :func:`iter_frames` will yield, at most.

    An upper bound rather than a promise: a window whose scans were all empty
    is dropped, the same way ``load_generic`` drops it.  Used to preallocate,
    never to index.
    """
    if source.name == "helipr":
        seq_path = source.dataset_path / source.sequence
        return len(source._handler.list_scan_files(
            seq_path, source.sensor, type=source.scan_type))
    handler, root = _generic_parts(source)
    return len(submap_windows(len(handler.list_scan_files(root)),
                              source.n_scans, source.stride))


def iter_frames(source) -> Iterator[Frame]:
    """Yield the sequence's submaps one at a time, in order."""
    if source.name == "helipr":
        yield from _iter_helipr(source)
    else:
        yield from _iter_generic(source)


def _iter_generic(source) -> Iterator[Frame]:
    handler, root = _generic_parts(source)
    poses, stamps = handler.load_poses(root)
    scan_files = handler.list_scan_files(root)
    if len(poses) != len(scan_files):
        # The same refusal load_generic makes, for the same reason: a silent
        # offset between poses and scans misaligns everything downstream.
        pose_path, _ = handler._pose_file(root)
        raise RuntimeError(
            f"Pose/scan count mismatch: {len(poses)} poses in {pose_path} "
            f"vs {len(scan_files)} scans in {handler.scan_dir(root)}.")

    transform = getattr(source, "transform", None)
    index = 0
    for window in submap_windows(len(poses), source.n_scans, source.stride):
        points = build_submap(handler.load_scan_file, scan_files, poses, window)
        if points is None:
            continue
        pose = poses[window[0]]
        if transform is not None:
            pose = np.asarray(transform, dtype=np.float64) @ pose
        yield Frame(index, points, pose,
                    float(stamps[window[0]]) if stamps else 0.0)
        index += 1


def _iter_helipr(source) -> Iterator[Frame]:
    ## HeLiPR is evaluated scan by scan -- there is no submap accumulation --
    ## so a frame is one .bin, with the pose whose timestamp is nearest.
    handler = source._handler
    seq_path = source.dataset_path / source.sequence
    poses, p_stamps = handler.load_poses(seq_path, source.sensor)
    files = handler.list_scan_files(seq_path, source.sensor, type=source.scan_type)

    for index, bin_file in enumerate(files):
        points, stamp = handler.load_scan_file(bin_file, source.sensor,
                                               type=source.scan_type)
        closest = handler.get_closest_pose_index(stamp, p_stamps)
        yield Frame(index, np.asarray(points, dtype=np.float32),
                    poses[closest], float(p_stamps[closest]))
