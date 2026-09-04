"""``inlier run --live`` -- the same pipeline, one frame at a time.

:mod:`inlier.eval.deploy` runs the pipeline in stages: encode everything, then
retrieve everything, then verify everything.  That is the right shape for a
batch run and the wrong shape for watching one happen, because nothing is
finished until everything is.

This module runs the identical arithmetic in the other loop order -- for each
frame: accumulate, encode, retrieve, verify, accept, refine -- so a viewer sees
the map grow and the closures land as they are found.

**It is not an approximation of the batch pipeline.**  Every stage in
:mod:`inlier.eval.pipeline` is already a per-query loop with no state carried
between queries; the one exception, ``online_shortlist_stage``, is causal by
design and streams already.  Reordering the loops therefore cannot change a
score.  The per-query bodies live in ``pipeline`` as ``shortlist_one``,
``beam_one``, ``rerank_one`` and ``verify_one``, and both drivers call them, so
the two cannot drift apart even if someone later changes one.

What does differ, deliberately:

* **Encoding is real.**  The descriptor cache is not consulted for the query
  session -- a cache hit would skip the very work the live view exists to show.
  A prior map is still encoded through the cache: it is built before the run
  starts either way, and re-encoding it shows nothing.
* **GICP runs per frame**, on the closures that frame produced, rather than in
  one batch at the end.  The clouds it needs are read back on demand
  (:class:`_LazyClouds`) instead of holding the sequence in memory.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from inlier.eval import gt as gtmod
from inlier.eval.datasets import stream as dstream
from inlier.eval.submaps import build_submap, submap_windows


class _LazyClouds:
    """``clouds[i]`` for a sequence, read from disk when asked and not before.

    ``refine_pairs`` only ever indexes what it needs, so it never notices this
    is not a list.  Holding the real one would put the whole sequence back in
    memory, which is the thing this module exists to avoid -- and a closure
    points at a frame from far enough in the past that no bounded cache would
    still have it.
    """

    def __init__(self, source) -> None:
        self._source = source
        self._helipr = source.name == "helipr"
        if self._helipr:
            self._seq_path = source.dataset_path / source.sequence
            self._files = source._handler.list_scan_files(
                self._seq_path, source.sensor, type=source.scan_type)
        else:
            self._handler, root = dstream._generic_parts(source)
            self._poses, _ = self._handler.load_poses(root)
            self._files = self._handler.list_scan_files(root)
            self._windows = submap_windows(len(self._poses), source.n_scans,
                                           source.stride)

    def __getitem__(self, index: int) -> np.ndarray:
        if self._helipr:
            points, _ = self._source._handler.load_scan_file(
                self._files[index], self._source.sensor,
                type=self._source.scan_type)
            return np.asarray(points, dtype=np.float32)
        points = build_submap(self._handler.load_scan_file, self._files,
                              self._poses, self._windows[index])
        return np.zeros((0, 3), dtype=np.float32) if points is None else points


def _encoded(tokens, kp_aligned, kp_sensor, T_grounds, poses):
    from inlier.eval.encode import EncodedSequence

    pose_arr = (np.asarray(poses, dtype=np.float64) if poses
                else np.zeros((0, 4, 4), dtype=np.float64))
    positions = (pose_arr[:, :3, 3] if len(pose_arr)
                 else np.zeros((0, 3), dtype=np.float64))
    return EncodedSequence(tokens, kp_aligned, kp_sensor, T_grounds,
                           positions, pose_arr)


def run_stream(spec, encoder, viewer=None):
    """Drive the pipeline frame by frame, reporting each frame to *viewer*.

    Returns the same :class:`~inlier.eval.deploy.StageResults` the batch driver
    returns, so everything downstream is shared.
    """
    from inlier.eval.deploy import (StageResults, accept, usable_stamps,
                                     validate_seconds_axis)
    from inlier.eval.encode import (encode_frame, encode_sequence,
                                    prepare_points)
    from inlier.eval.pipeline import (beam_one, build_matcher, minimal_keypoints,
                                      rerank_one, shortlist_one, verify_one)
    from inlier.eval.refine import refine_pairs
    from inlier.viz.live import StubViewer

    if viewer is None:
        viewer = StubViewer()
    r = spec.resolved
    cross = spec.cross_session

    # --- the prior map, if this mode has one -------------------------------
    if cross:
        db_enc = encode_sequence(
            encoder, load=spec.db_source.load,
            tag=f"{spec.db_source.tag}_Undistorted", voxel_size=r.voxel_size,
            cache_dir=spec.cache_dir, verbose=spec.verbose,
            desc="  Encoding prior map")
        db_stamps = _sequence_stamps(spec.db_source, len(db_enc.tokens))
        matcher = build_matcher(r, list(db_enc.tokens), verbose=spec.verbose)
        n_db_fixed: Optional[int] = len(db_enc.tokens)
        cand_filter: Any = gtmod.NoFilter()
    else:
        matcher = build_matcher(r, [], verbose=False)
        n_db_fixed = None
        db_stamps = []

    # --- axes the exclusion window needs, before a scan is read ------------
    n_frames = dstream.frame_count(spec.source)
    # Poses and their timestamps come out of one text file, so reading them up
    # front costs nothing and reads no points.  They are odometry either way:
    # a running system has this frame's pose when the frame arrives.
    positions = np.zeros((n_frames, 3), dtype=np.float64)
    arc = np.zeros(n_frames, dtype=np.float64)
    stamps_axis = np.zeros(n_frames, dtype=np.float64)
    if not cross and spec.exclusion is not None and spec.exclusion.unit == "seconds":
        keyframe_stamps = dstream.keyframe_stamps(spec.source)
        validate_seconds_axis(keyframe_stamps, n_frames)

    if not cross and spec.search_radius > 0.0:
        # `positions` is filled as frames arrive; the filter only ever reads
        # entries below the current frame's bound, all of which are written by
        # the time it is asked.
        cand_filter = gtmod.RadiusFilter(
            positions=positions, radius=spec.search_radius,
            exclusion=spec.exclusion, timestamps=stamps_axis,
            arc_length=arc, pose_source="odometry")
    elif not cross:
        cand_filter = gtmod.CausalFilter(
            exclusion=spec.exclusion, timestamps=stamps_axis, arc_length=arc)

    # `topk_pct` needs a sequence length before the sequence has been read.
    # frame_count is exact unless a window's scans were all empty, which drops
    # a frame here and would drop it in the batch path too.
    n_pool = n_db_fixed if cross else n_frames
    effective_topk = (max(1, round(n_pool * r.shortlist.topk_pct))
                      if r.shortlist.topk_pct is not None else r.shortlist.topk)

    tokens: List[Any] = []
    kp_aligned: List[np.ndarray] = []
    kp_sensor: List[np.ndarray] = []
    T_grounds: List[np.ndarray] = []
    poses: List[np.ndarray] = []
    q_stamps: List[float] = []
    bounds: List[int] = []

    ranked_s1: Dict[int, List[int]] = {}
    sims_s1: Dict[int, Dict[int, float]] = {}
    ranked_s2: Dict[int, List[int]] = {}
    sims_s2: Dict[int, Dict[int, float]] = {}
    shifts_s2: Dict[int, Dict[int, int]] = {}
    ranked_rr: Dict[int, List[int]] = {}
    sims_rr: Dict[int, Dict[int, float]] = {}
    shifts_rr: Dict[int, Dict[int, int]] = {}
    sims_ver: Dict[int, Dict[int, float]] = {}
    ver_rank: Dict[int, List[int]] = {}
    verify_outputs: Dict[Tuple[int, int], Any] = {}
    accepted: List[Tuple[int, int, float, int]] = []
    gicp_outputs: Dict[Tuple[int, int], Any] = {}
    latency = np.zeros(n_frames, dtype=np.float64)
    STAGES = ("voxel", "encode", "s1", "s2", "rr", "verify", "gicp")
    #: Cumulative over the run -- what the results JSON reports.
    times = {name: 0.0 for name in STAGES}
    #: This frame alone -- what a live view wants.  A running total only ever
    #: grows, so a panel showing one says nothing about the frame on screen.
    #: Cleared every frame, so a stage that did not run reads 0 rather than
    #: whatever it happened to cost last time.
    frame_times = {name: 0.0 for name in STAGES}

    def took(name: str, t0: float) -> float:
        dt = time.perf_counter() - t0
        times[name] += dt
        frame_times[name] = dt
        return dt
    n_converged = 0

    if not cross:
        matcher.reserve(n_frames)
    q_clouds = _LazyClouds(spec.source)
    db_clouds = _LazyClouds(spec.db_source) if cross else q_clouds

    viewer.start(spec, db_enc if cross else None, db_clouds, n_frames)

    for frame in dstream.iter_frames(spec.source):
        if viewer.closed():
            break
        t = frame.index
        frame_times.update((name, 0.0) for name in STAGES)
        poses.append(np.asarray(frame.pose, dtype=np.float64))
        positions[t] = poses[-1][:3, 3]
        stamps_axis[t] = frame.stamp
        q_stamps.append(float(frame.stamp))
        arc[t] = (0.0 if t == 0 else
                  arc[t - 1] + float(np.linalg.norm(positions[t, :2]
                                                    - positions[t - 1, :2])))

        # --- encode --------------------------------------------------------
        t0 = time.perf_counter()
        prepared = prepare_points(frame.points, r.voxel_size)
        took("voxel", t0)
        t0 = time.perf_counter()
        tok, kp_a, kp_s, T_g = encode_frame(encoder, prepared)
        took("encode", t0)
        tokens.append(tok)
        kp_aligned.append(kp_a)
        kp_sensor.append(kp_s)
        T_grounds.append(T_g)

        # --- retrieve ------------------------------------------------------
        t0 = time.perf_counter()
        if cross:
            bound = None
            ids, scores = shortlist_one(matcher, tok, topk=n_db_fixed)
        else:
            bound = int(spec.exclusion.cutoff(t, stamps_axis, arc))
            bounds.append(bound)
            if bound > 0:
                ids, scores = shortlist_one(matcher, tok, topk=bound,
                                            max_db_index=bound)
                if spec.search_radius > 0.0:
                    mask = cand_filter.allowed_mask(t, bound)
                    keep = [i for i, d in enumerate(ids) if mask[d]]
                    ids = [ids[i] for i in keep]
                    scores = [scores[i] for i in keep]
            else:
                ids, scores = [], []
        ranked_s1[t] = ids
        sims_s1[t] = {d: s for d, s in zip(ids, scores)}
        t_s1_frame = took("s1", t0)

        ranked_in = ids
        shifts_in: Optional[Dict[int, int]] = None
        if not r.skip_stage2:
            t0 = time.perf_counter()
            b_ids, b_scores, b_shifts = beam_one(matcher, tok,
                                                 ids[:effective_topk])
            took("s2", t0)
            ranked_s2[t] = b_ids
            sims_s2[t] = {d: s for d, s in zip(b_ids, b_scores)}
            shifts_s2[t] = {d: s for d, s in zip(b_ids, b_shifts)}
            ranked_in, shifts_in = b_ids, shifts_s2[t]

            if r.run_rerank:
                topk_rr = (max(1, round(effective_topk * r.beam.topk_pct))
                           if r.beam.topk_pct is not None else effective_topk)
                t0 = time.perf_counter()
                rr_ids, rr_scores, rr_shifts = rerank_one(
                    matcher, tok, b_ids[:topk_rr], shifts_s2[t])
                took("rr", t0)
                ranked_rr[t] = rr_ids
                sims_rr[t] = {d: s for d, s in zip(rr_ids, rr_scores)}
                shifts_rr[t] = {d: s for d, s in zip(rr_ids, rr_shifts)}
                ranked_in, shifts_in = rr_ids, shifts_rr[t]

        # --- verify --------------------------------------------------------
        q_kp = minimal_keypoints(kp_a, kp_s, T_g)
        db_kp_of = _db_keypoints(db_enc if cross else None, kp_aligned,
                                 kp_sensor, T_grounds)
        db_tokens = list(db_enc.tokens) if cross else tokens
        t0 = time.perf_counter()
        q_sims, verified, per_db = verify_one(
            matcher, tok, q_kp, db_tokens, db_kp_of, ranked_in, shifts_in,
            r.verify, top_v=r.verify_topv)
        took("verify", t0)
        sims_ver[t] = q_sims
        ver_rank[t] = verified
        for d, out in per_db.items():
            verify_outputs[(t, d)] = out

        # --- accept + refine ------------------------------------------------
        frame_accepted = accept({t: q_sims}, verify_outputs, spec.threshold)
        accepted.extend(frame_accepted)
        frame_gicp: Dict[Tuple[int, int], Any] = {}
        if frame_accepted and not r.skip_gicp:
            q_enc_now = _encoded(tokens, kp_aligned, kp_sensor, T_grounds, poses)
            frame_gicp, dt, converged = refine_pairs(
                [(q, d) for q, d, _, _ in frame_accepted], verify_outputs,
                q_enc_now, db_enc if cross else q_enc_now, r.gicp, r.voxel_size,
                load_q_clouds=lambda: q_clouds,
                load_db_clouds=lambda: db_clouds, verbose=False)
            times["gicp"] += dt
            frame_times["gicp"] = dt
            n_converged += converged
            gicp_outputs.update(frame_gicp)

        if not cross:
            # The reported per-frame latency is the shortlist plus the
            # insertion and nothing else -- the same quantity
            # ``online_shortlist_stage`` reports, so the two runs' numbers mean
            # the same thing.  The later stages are timed in ``times``.
            t0 = time.perf_counter()
            matcher.add(t, tok)
            matcher.finalize(verbose=False)
            latency[t] = (t_s1_frame + (time.perf_counter() - t0)) * 1e3

        viewer.on_frame(frame=frame, index=t, tokens=tok, kp_sensor=kp_s,
                        kp_aligned=kp_a, pose=poses[-1],
                        ranked=ranked_in, sims_s1=sims_s1[t],
                        sims_s2=sims_s2.get(t), sims_ver=q_sims,
                        verify=per_db, accepted=frame_accepted,
                        gicp=frame_gicp, db_poses=(db_enc.poses if cross else None),
                        q_poses=poses, times=dict(times),
                        frame_times=dict(frame_times), bound=bound)

    n_q = len(tokens)
    q_enc = _encoded(tokens, kp_aligned, kp_sensor, T_grounds, poses)
    q_stamps = usable_stamps(q_stamps, n_q)
    if not cross:
        db_enc = q_enc
        db_stamps = list(q_stamps)

    return StageResults(
        q_enc=q_enc, db_enc=db_enc, q_stamps=q_stamps, db_stamps=db_stamps,
        bounds=bounds, cand_filter=cand_filter,
        ranked_s1=ranked_s1, sims_s1=sims_s1,
        ranked_s2=ranked_s2 or None, sims_s2=sims_s2 or None,
        shifts_s2=shifts_s2 or None,
        ranked_rr=ranked_rr or None, sims_rr=sims_rr or None,
        shifts_rr=shifts_rr or None,
        sims_ver=sims_ver, ver_rank=ver_rank, verify_outputs=verify_outputs,
        accepted=accepted, gicp_outputs=gicp_outputs, n_converged=n_converged,
        latency_ms=latency[:n_q], effective_topk=effective_topk,
        encode_time=times["voxel"] + times["encode"], t_s1=times["s1"], t_s2=times["s2"],
        t_rr=times["rr"], t_verify=times["verify"], t_gicp=times["gicp"])


def _db_keypoints(db_enc, kp_aligned, kp_sensor, T_grounds):
    """``db_id -> InLiER_Keypoints``, from the prior map or the growing past."""
    from inlier.eval.pipeline import minimal_keypoints

    if db_enc is not None:
        return lambda d: minimal_keypoints(db_enc.kp_aligned[d],
                                           db_enc.kp_sensor[d],
                                           db_enc.T_grounds[d])
    return lambda d: minimal_keypoints(kp_aligned[d], kp_sensor[d], T_grounds[d])


def _sequence_stamps(source, n: int) -> List[float]:
    """Keyframe timestamps for a prior map, or ``[]`` when it carries none."""
    from inlier.eval.deploy import usable_stamps

    return usable_stamps(dstream.keyframe_stamps(source), n)
