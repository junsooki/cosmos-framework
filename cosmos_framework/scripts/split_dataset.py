# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Split a LeRobot v3.0 dataset into standalone train/val datasets by DIRECT parquet edits.

This manipulates the data parquet + episodes metadata + info.json directly. Episodes are
partitioned into two disjoint sets; each becomes its own standalone v3.0 dataset under
``<dst>/train`` and ``<dst>/val`` with densely re-indexed episodes.
The packed video is truly split per set: each episode's segment is extracted by lossless
ffmpeg **stream-copy** and re-concatenated into the split's own video file — so ``val``'s video
contains only val footage. This relies on episode boundaries being keyframes (true for videos
produced by our converters, which concatenate per-episode keyframe-started clips); the script
verifies this and errors rather than silently misaligning. The concatenated result is then
**re-encoded to exact constant-frame-rate** (``_retime_cfr``): the ``-ss``-seek + concat-demuxer
re-timing leaves sub-frame PTS drift that would desync the video from the parquet ``timestamp``
grid for torchcodec's frame queries, and a lossless container re-timestamp can't fix it (B-frame
edit-list offsets that torchcodec ignores). The re-encode is high quality (crf 23).

Val selection:
  * default: a random ``--val-fraction`` (5%) of episodes, chosen with ``--seed`` (reproducible).
  * or ``--val-episodes I J K ...``: use exactly those original episode indices as val.

Per split, for its episodes (in ascending original order):
  * data rows: sliced by each episode's ``[dataset_from_index, dataset_to_index)``, then
    ``episode_index`` re-indexed dense 0..k-1 and global ``index`` renumbered 0..N-1
    (``frame_index`` / ``timestamp`` per-episode, unchanged); collapsed to one ``file-000``.
  * episodes meta: filtered + re-indexed; ``dataset_from/to_index`` recomputed from lengths;
    ``data/*`` and ``meta/episodes/*`` pointers set to the single file-000; video pointers
    repointed to the split's new video (chunk/file 0, cumulative from/to_timestamp).
  * videos: each episode segment is stream-copied out and re-concatenated into the split's own
    ``chunk-000/file-000.mp4``, then re-encoded to exact CFR (frame i at pts i/fps) so it stays
    in sync with the row timestamps; the split's video holds only its own episodes.
  * meta: ``tasks.parquet`` / ``stats.json`` / ``modality.json`` / etc. copied verbatim;
    ``info.json`` gets the split's ``total_episodes`` / ``total_frames``.

Note: per-episode string path features (e.g. ``sub_goal_image_path``) still reference the
*original* episode numbers; ``images/`` is copied wholesale so those paths resolve, but they
are not renumbered. ``stats.json`` is the full-dataset stats (fine for normalization; not
recomputed per split).

Usage::

    python -m cosmos_framework.scripts.split_dataset --src <v3.0 root> --dst <out dir>
    python -m cosmos_framework.scripts.split_dataset --src <root> --dst <out> --val-fraction 0.1
    python -m cosmos_framework.scripts.split_dataset --src <root> --dst <out> --val-episodes 3 7 12
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_KF_TOL = 1e-3  # episode boundary must be within this (s) of a keyframe for stream-copy extraction


def _set(table: pa.Table, name: str, array) -> pa.Table:
    return table.set_column(table.schema.get_field_index(name), name, array)


def _zeros_like(table: pa.Table, name: str) -> pa.Array:
    return pa.array([0] * table.num_rows, type=table.column(name).type)


def _keyframe_times(video: Path) -> list[float]:
    # The per-frame presentation-time field was renamed ``pkt_pts_time`` -> ``pts_time`` in
    # ffmpeg 5.0, and ffprobe silently omits fields it doesn't recognise. Requesting only
    # ``pts_time`` therefore yields an *empty* list on ffmpeg 4.x (e.g. Ubuntu 22.04's 4.4.2),
    # which makes even a true keyframe at t=0 fail the boundary check. Ask for every timestamp
    # variant and take the first populated one per keyframe so this works on ffmpeg 4.x and >=5.
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
         "-show_entries", "frame=pts_time,pkt_pts_time,best_effort_timestamp_time",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout
    times = []
    for line in out.splitlines():
        for tok in line.split(","):
            tok = tok.strip()
            if tok and tok != "N/A":
                times.append(float(tok))
                break
    return sorted(times)


def _extract_segment(src_video: Path, from_ts: float, n_frames: int, out: Path) -> None:
    """Stream-copy exactly ``n_frames`` starting at keyframe ``from_ts`` (lossless, no re-encode).

    ``-nostdin`` stops ffmpeg from reading the parent's terminal stdin (which otherwise makes it
    appear to hang under a shell)."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{from_ts:.6f}", "-i", str(src_video),
         "-frames:v", str(n_frames), "-c", "copy", str(out)], check=True)


def _concat_segments(segments: list[Path], out: Path) -> None:
    """Concatenate mp4 segments losslessly (ffmpeg concat demuxer, stream copy)."""
    if len(segments) == 1:
        shutil.copy2(segments[0], out)
        return
    listfile = out.parent / "_concat_list.txt"
    listfile.write_text("".join(f"file '{s.resolve()}'\n" for s in segments))
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listfile),
         "-c", "copy", str(out)], check=True)
    listfile.unlink()


def _count_video_frames(video: Path) -> int:
    """Exact decoded frame count of ``video`` (ffprobe -count_frames)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


def _retime_cfr(video: Path, fps: float, n_frames: int, crf: int = 23) -> None:
    """Re-encode ``video`` to exact constant-frame-rate so frame ``i`` sits at PTS ``i/fps``.

    WHY: the ``-ss``-seek extraction + concat-demuxer stitching above is lossless but does NOT
    preserve exact CFR — the concatenated stream ends up mildly variable-frame-rate, its frame
    PTS drifting up to ~half a frame off the ideal ``i/fps`` grid (the source is perfect CFR;
    the re-timing introduces the drift). The parquet ``timestamp`` column and the per-episode
    cumulative ``from_timestamp`` stay on the ideal grid, so the drift desyncs video queries
    from the data rows: torchcodec (lerobot's decoder) reads raw PTS and rejects a query whose
    nearest frame is outside ``tolerance_s``.

    A lossless container re-timestamp can't fix this robustly: the streams carry B-frames, so a
    packet-level PTS rewrite either scrambles display order or leaves an edit-list start offset
    that torchcodec (which ignores edit lists) mis-reads. So we re-encode to true CFR with NO
    B-frames — every frame ``pts == dts``, the stream starts at PTS 0, and frame ``i`` lands
    exactly at ``i/fps`` — which is what torchcodec's timestamp queries need. High quality
    (crf 23, libx264 default); the only cost is a re-encode pass and a modestly larger file.
    """
    tmp = video.with_name(video.stem + ".cfr.mp4")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(video),
         "-vf", f"fps={fps:g}", "-c:v", "libx264", "-crf", str(crf), "-bf", "0",
         "-preset", "medium", "-pix_fmt", "yuv420p", "-an", str(tmp)], check=True)
    # The sub-frame input drift keeps the ``fps`` filter a 1:1 remap (no drop/dup); assert the
    # frame count is preserved so a video/data-row desync fails loudly instead of silently.
    got = _count_video_frames(tmp)
    if got != n_frames:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"CFR re-encode of {video.name}: frame count {got} != expected {n_frames} "
            "(drift exceeded half a frame?); aborting to avoid a video/data-row desync."
        )
    tmp.replace(video)


def _write_split(src: Path, dst: Path, ep_indices: list[int], info: dict) -> None:
    """Write the episodes in ``ep_indices`` (original indices) as a standalone v3.0 dataset."""
    if dst.exists():
        raise SystemExit(f"refusing to overwrite existing {dst}")
    ep_indices = sorted(ep_indices)
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    print(f"  writing {dst.name} ({len(ep_indices)} episodes)...", flush=True)

    base = pa.concat_tables([pq.read_table(p) for p in sorted((src / "data").glob("chunk-*/file-*.parquet"))])
    ebase = pa.concat_tables([pq.read_table(p) for p in sorted((src / "meta" / "episodes").glob("chunk-*/file-*.parquet"))])
    erows = ebase.to_pylist()
    by_ep = {int(r["episode_index"]): r for r in erows}

    # --- data rows: slice each episode's [from, to), concatenate in order -----------------
    data_slices, ep_col, idx_col = [], [], []
    running = 0
    new_ep_meta = []  # rebuilt episode-metadata rows
    for new_ei, old_ei in enumerate(ep_indices):
        r = by_ep[old_ei]
        f, t = int(r["dataset_from_index"]), int(r["dataset_to_index"])
        length = t - f
        sl = base.slice(f, length)
        data_slices.append(sl)
        ep_col.extend([new_ei] * length)
        idx_col.extend(range(running, running + length))
        nr = dict(r)
        nr["episode_index"] = new_ei
        nr["dataset_from_index"] = running
        nr["dataset_to_index"] = running + length
        new_ep_meta.append(nr)
        running += length

    data_out = pa.concat_tables(data_slices)
    data_out = _set(data_out, "episode_index",
                    pa.array(ep_col, type=data_out.column("episode_index").type))
    data_out = _set(data_out, "index", pa.array(idx_col, type=data_out.column("index").type))
    n_frames = data_out.num_rows

    # --- videos: extract runs of consecutive episodes and re-concat into the split's file ---
    # Stream-copy (lossless, no re-encode): episode boundaries are keyframes, so each maximal run
    # of consecutive original episodes (same source video file) is extracted in ONE ffmpeg call
    # (seek to the run's keyframe, take exactly its total frames). This is far fewer subprocesses
    # than one-per-episode (e.g. a 5% val split -> a handful of runs), so it stays fast even under
    # heavy CPU load. Video pointers are repointed to the new file with cumulative from/to_timestamp.
    vpath = info["video_path"]
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        for vk in video_keys:
            kf_cache: dict[tuple[int, int], list[float]] = {}
            # Assign each episode its new (cumulative) output timestamps and pointers first.
            running_ts = 0.0
            for new_ei, old_ei in enumerate(ep_indices):
                r = by_ep[old_ei]
                dur = float(r[f"videos/{vk}/to_timestamp"]) - float(r[f"videos/{vk}/from_timestamp"])
                new_ep_meta[new_ei][f"videos/{vk}/chunk_index"] = 0
                new_ep_meta[new_ei][f"videos/{vk}/file_index"] = 0
                new_ep_meta[new_ei][f"videos/{vk}/from_timestamp"] = running_ts
                new_ep_meta[new_ei][f"videos/{vk}/to_timestamp"] = running_ts + dur
                running_ts += dur
            # Group consecutive original episodes sharing a source (chunk,file) into runs.
            runs: list[list[int]] = []
            for new_ei, old_ei in enumerate(ep_indices):
                r = by_ep[old_ei]
                cf = (int(r[f"videos/{vk}/chunk_index"]), int(r[f"videos/{vk}/file_index"]))
                prev = ep_indices[new_ei - 1] if new_ei > 0 else None
                prev_cf = (runs[-1] and (int(by_ep[prev][f"videos/{vk}/chunk_index"]),
                                         int(by_ep[prev][f"videos/{vk}/file_index"]))) if prev is not None else None
                if runs and prev is not None and old_ei == prev + 1 and cf == prev_cf:
                    runs[-1].append(old_ei)
                else:
                    runs.append([old_ei])
            # Extract one segment per run (all frames of its episodes, contiguous in the source).
            segments = []
            for ri, run in enumerate(runs):
                r0, rL = by_ep[run[0]], by_ep[run[-1]]
                ci, fi = int(r0[f"videos/{vk}/chunk_index"]), int(r0[f"videos/{vk}/file_index"])
                src_video = src / vpath.format(video_key=vk, chunk_index=ci, file_index=fi)
                from_ts = float(r0[f"videos/{vk}/from_timestamp"])
                run_frames = int(rL["dataset_to_index"]) - int(r0["dataset_from_index"])
                kf = kf_cache.setdefault((ci, fi), _keyframe_times(src_video))
                if not any(abs(from_ts - k) <= _KF_TOL for k in kf):
                    raise SystemExit(
                        f"{vk} ep {run[0]}: from_timestamp {from_ts:.3f}s is not a keyframe in "
                        f"{src_video.name}; lossless stream-copy split not possible for this video."
                    )
                seg = tmpd / f"{vk.replace('/', '_')}_{ri:04d}.mp4"
                _extract_segment(src_video, from_ts, run_frames, seg)
                segments.append(seg)
            out_video = dst / vpath.format(video_key=vk, chunk_index=0, file_index=0)
            out_video.parent.mkdir(parents=True, exist_ok=True)
            _concat_segments(segments, out_video)
            # Re-encode to exact CFR so frame i sits at pts i/fps (matches the parquet grid);
            # the lossless extract+concat above leaves sub-frame PTS drift that desyncs the
            # video from the data rows for torchcodec's timestamp queries. See _retime_cfr.
            _retime_cfr(out_video, float(info.get("fps", 30.0)), n_frames)
            print(f"    {dst.name}/{vk}: {len(runs)} run(s) -> repacked video (CFR {info.get('fps', 30)}fps)", flush=True)

    # --- episodes metadata: rebuild table from new_ep_meta, fix self/data pointers --------
    ep_out = pa.Table.from_pylist(new_ep_meta, schema=ebase.schema)
    for col in ("data/chunk_index", "data/file_index",
                "meta/episodes/chunk_index", "meta/episodes/file_index"):
        if col in ep_out.schema.names:
            ep_out = _set(ep_out, col, _zeros_like(ep_out, col))

    # --- write dst tree -------------------------------------------------------------------
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(data_out, dst / "data" / "chunk-000" / "file-000.parquet")
    (dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(ep_out, dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    # meta: copy everything except episodes/ (rewritten) and info.json (rewritten below).
    # This carries the source's derived artifacts too — notably ``cosmos3_stats_flat.json``
    # (the action-policy server/eval stats), which the converters emit alongside stats.json.
    for item in (src / "meta").iterdir():
        if item.name in ("episodes", "info.json"):
            continue
        (shutil.copytree if item.is_dir() else shutil.copy2)(item, dst / "meta" / item.name)
    # Warn if the source lacks the flat policy stats: the split can't derive it (the recipe is
    # embodiment-specific and lives in the converters), so a downstream --stats-path will 404.
    if not (dst / "meta" / "cosmos3_stats_flat.json").exists():
        print(f"  WARNING: {dst.name}: no meta/cosmos3_stats_flat.json (source lacks it). "
              "Re-run the dataset converter (write_policy_stats_flat) or regenerate it from "
              "meta/stats.json, else the action-policy server/eval --stats-path will be missing.",
              file=sys.stderr)

    # images/: copy wholesale if present (per-episode string paths reference original numbers).
    if (src / "images").exists():
        shutil.copytree(src / "images", dst / "images")

    out_info = dict(info)
    out_info["total_episodes"] = len(ep_indices)
    out_info["total_frames"] = n_frames
    if isinstance(out_info.get("splits"), dict):
        out_info["splits"] = {"train": f"0:{len(ep_indices)}"}
    # Order info.json ``features`` to match the data parquet's column order: lerobot's
    # LeRobotDataset does a strict ``table.cast(features.arrow_schema)`` that fails if they
    # differ, and some source datasets ship them out of order. Parquet columns first (in
    # their on-disk order); non-parquet features (video streams, etc.) keep their original
    # relative order, appended after.
    feats = out_info.get("features")
    if isinstance(feats, dict):
        pq_cols = list(data_out.schema.names)
        ordered = {k: feats[k] for k in pq_cols if k in feats}
        ordered.update({k: v for k, v in feats.items() if k not in ordered})
        out_info["features"] = ordered
    (dst / "meta" / "info.json").write_text(json.dumps(out_info, indent=4))

    print(f"  {dst.name}: {len(ep_indices)} episodes / {n_frames} frames "
          f"(video re-packed, {len(video_keys)} key(s))")


def split_dataset(src: str | Path, dst: str | Path, *, val_fraction: float = 0.05,
                  val_episodes: list[int] | None = None, seed: int = 42,
                  train_name: str = "train", val_name: str = "val") -> None:
    src, dst = Path(src), Path(dst)
    info = json.loads((src / "meta" / "info.json").read_text())
    n_eps = int(info["total_episodes"])
    all_eps = list(range(n_eps))

    if val_episodes is not None:
        bad = [e for e in val_episodes if e < 0 or e >= n_eps]
        if bad:
            raise SystemExit(f"--val-episodes out of range [0,{n_eps}): {bad}")
        val_set = sorted(set(val_episodes))
    else:
        k = max(1, round(n_eps * val_fraction))
        val_set = sorted(random.Random(seed).sample(all_eps, k))
    train_set = [e for e in all_eps if e not in set(val_set)]
    if not train_set or not val_set:
        raise SystemExit(f"empty split (train={len(train_set)}, val={len(val_set)}) from {n_eps} episodes")

    print(f"Splitting {src} ({n_eps} episodes) -> train {len(train_set)} / val {len(val_set)}")
    print(f"  val episodes: {val_set}")
    _write_split(src, dst / train_name, train_set, info)
    _write_split(src, dst / val_name, val_set, info)
    print(f"Done: {dst}/{train_name} + {dst}/{val_name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Split a v3.0 dataset into train/val via direct parquet edits.")
    ap.add_argument("--src", required=True, help="source v3.0 dataset root")
    ap.add_argument("--dst", required=True, help="output dir; creates <dst>/train and <dst>/val")
    ap.add_argument("--val-fraction", type=float, default=0.05,
                    help="fraction of episodes for val (default 0.05); ignored if --val-episodes given")
    ap.add_argument("--val-episodes", type=int, nargs="+", default=None,
                    help="explicit original episode indices for val (overrides --val-fraction)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for random val selection")
    args = ap.parse_args()
    split_dataset(args.src, args.dst, val_fraction=args.val_fraction,
                  val_episodes=args.val_episodes, seed=args.seed)
