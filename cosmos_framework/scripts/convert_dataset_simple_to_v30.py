# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert a "simple" LeRobot v2.1 dataset to v3.0 (minimal, feature-agnostic).

For datasets like ``G1WholebodyLocomotionPickBetweenTablesTeleop-v0``: a v2.1 layout
(per-episode ``data/chunk-*/episode_*.parquet`` + ``videos/chunk-*/<cam>/episode_*.mp4``)
with GR00T-style episode metadata (integer ``tasks`` + inclusive ``dataset_from/to_index``).

Unlike the G1 "sonic" converter, this does NOT apply that post-processing
(``task_description``->task remap, ``action.hand`` slicing, neck zero-padding,
``cosmos3_stats_flat.json``) — those assume the 80-D ``[body_token, hand, neck]`` action +
``observation.state`` layout, which this dataset doesn't have. It performs only the core
repack:

  1. ``convert_info`` / ``convert_tasks`` — v3.0 info + tasks.parquet.
  2. ``convert_data`` — pack per-episode parquet into ``data/chunk-*/file-*.parquet``.
  3. ``convert_videos`` — concatenate per-episode mp4 into shared video files.
  4. ``convert_episodes_metadata`` (inlined below) — cleans the GR00T legacy
     fields (drops inclusive ranges so v3.0 recomputes exclusive ``[from, to)``; maps integer
     ``tasks`` -> task strings) and writes ``meta/episodes/`` + aggregated ``stats.json``.
  5. ``write_meta_stats_with_quantiles`` (inlined below) — recompute ``meta/stats.json`` with
     per-feature ``min/max/mean/std/count/q01/q99`` for every numeric column.
  6. copy ``meta/modality.json`` if present.

The source ``index`` column is already globally monotonic here, so no
global-frame-index rewrite is needed.

Usage::

    python cosmos_framework/scripts/convert_dataset_simple_to_v30.py \\
        --repo-id G1WholebodyLocomotionPickBetweenTablesTeleop-v0 --root .data/simple
    # -> .data/simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0_v30
"""

import argparse
import logging
import shutil
from pathlib import Path

from datasets import Dataset

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import (
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    load_info,
    write_episodes,
    write_stats,
)
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
    convert_data,
    convert_info,
    convert_tasks,
    convert_videos,
    generate_episode_metadata_dict,
    legacy_load_episodes,
    legacy_load_episodes_stats,
    legacy_load_tasks,
)
from lerobot.utils.utils import init_logging


def clean_psix_legacy_episodes(episodes_legacy_metadata: dict, idx_to_label: dict) -> dict:
    """Drop GR00T inclusive frame ranges and map integer `tasks` -> task label strings.

    ``idx_to_label`` maps a task index to the canonical task string written to the
    episodes ``tasks`` field. We pass the human-readable description here (not the
    slug) so the episodes metadata agrees with the remapped ``meta/tasks.parquet``
    (whose ``task`` is also the description); the slug stays available per-frame as
    ``task_uid``.
    """
    for ep in episodes_legacy_metadata.values():
        # let convert_data recompute exclusive [from, to); GR00T stored inclusive last
        ep.pop("dataset_from_index", None)
        ep.pop("dataset_to_index", None)
        if "tasks" in ep:
            ep["tasks"] = [
                idx_to_label.get(t, t) if isinstance(t, int) else t for t in ep["tasks"]
            ]
    return episodes_legacy_metadata


def convert_episodes_metadata(root, new_root, episodes_metadata, episodes_video_metadata=None):
    """Like the v21 converter, but cleans GR00T legacy episode fields first."""
    logging.info(f"Converting episodes metadata from {root} to {new_root}")

    episodes_legacy_metadata = legacy_load_episodes(root)
    episodes_stats = legacy_load_episodes_stats(root)
    idx_to_task, _ = legacy_load_tasks(root)  # index -> slug
    # Map index -> human-readable description (from each source episode's
    # task_description) so the episodes 'tasks' field matches the remapped
    # tasks.parquet; fall back to the slug for any index without a description.
    idx_to_desc: dict[int, str] = {}
    for ep in episodes_legacy_metadata.values():
        desc = ep.get("task_description")
        if desc is None:
            continue
        for t in ep.get("tasks", []):
            if isinstance(t, int):
                idx_to_desc.setdefault(int(t), desc)
    idx_to_label = {i: idx_to_desc.get(i, slug) for i, slug in idx_to_task.items()}
    episodes_legacy_metadata = clean_psix_legacy_episodes(episodes_legacy_metadata, idx_to_label)

    num_eps_set = {len(episodes_legacy_metadata), len(episodes_metadata)}
    if episodes_video_metadata is not None:
        num_eps_set.add(len(episodes_video_metadata))
    if len(num_eps_set) != 1:
        raise ValueError(f"Number of episodes is not the same ({num_eps_set}).")

    ds_episodes = Dataset.from_generator(
        lambda: generate_episode_metadata_dict(
            episodes_legacy_metadata, episodes_metadata, episodes_stats, episodes_video_metadata
        )
    )
    write_episodes(ds_episodes, new_root)

    stats = aggregate_stats(list(episodes_stats.values()))
    write_stats(stats, new_root)


def write_meta_stats_with_quantiles(new_root: Path) -> None:
    """(Re)compute ``meta/stats.json`` with per-feature ``min/max/mean/std/count/q01/q99``
    for every numeric data column.

    LeRobot's own ``aggregate_stats`` only emitted ``min/max/mean/std`` for the 36-D
    ``action`` column here, and never the quantiles. The Cosmos action policy
    quantile-normalizes ``[q01, q99] -> [-1, 1]``, so we compute stats directly from
    the data — covering every numeric column (per-feature dict, à la a canonical
    LeRobot ``stats.json``) so they always match the data and travel with the dataset.

    Video columns (not in the parquet) and string columns (e.g. ``task_uid``,
    ``subtask_prompt``) are skipped. Scalar columns are stored as length-1 lists,
    matching the LeRobot stats shape convention.
    """
    import json

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    stats_path = new_root / "meta" / "stats.json"
    data_paths = sorted((new_root / "data").glob("chunk-*/file-*.parquet"))
    schema = pq.read_schema(data_paths[0])
    numeric_cols = [
        f.name
        for f in schema
        if not pa.types.is_string(f.type) and not pa.types.is_large_string(f.type)
    ]

    accum: dict[str, list] = {c: [] for c in numeric_cols}
    for path in data_paths:
        table = pq.read_table(path, columns=numeric_cols)
        for c in numeric_cols:
            a = np.asarray(table[c].to_pylist(), dtype=np.float32)  # [N] or [N, D]
            accum[c].append(a if a.ndim == 2 else a.reshape(-1, 1))  # scalars -> [N, 1]

    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for c in numeric_cols:
        a = np.concatenate(accum[c], axis=0)  # [N, D]
        stats[c] = {
            "min": a.min(0).tolist(),
            "max": a.max(0).tolist(),
            "mean": a.mean(0).tolist(),
            "std": a.std(0).tolist(),
            "count": [int(a.shape[0])],
            "q01": np.quantile(a, 0.01, axis=0).astype(np.float32).tolist(),
            "q99": np.quantile(a, 0.99, axis=0).astype(np.float32).tolist(),
        }
    stats_path.write_text(json.dumps(stats, indent=4))
    logging.info(
        f"wrote meta/stats.json with min/max/mean/std/count/q01/q99 for {len(numeric_cols)} numeric columns"
    )


def write_policy_stats_flat(new_root: Path) -> None:
    """Write ``meta/cosmos3_stats_flat.json`` consumed by the action policy server/eval.

    A dict with two parallel namespaces — ``"action"`` and ``"state"`` — each a sub-dict of
    the stat names (``mean/std/min/max/q01/q99``) over the modality's full vector. The simple
    embodiment has a FLAT action, so this mirrors the sonic converter's version but with the
    simple layout:

      * ``action`` — the flat 36-D ``action`` feature (no body/hand/neck sub-blocks).
      * ``state``  — the 32-D proprioceptive ``states`` feature, renamed to ``state`` (the
        server reads state stats under ``"state"``; simple's stats.json stores it as
        ``"states"``).

    ``use_state`` prepends the state as the action chunk's conditioning row and must be
    normalized with its own stats; keeping both here lets the server normalize each from one
    file. Skips with a warning if ``action`` is absent (non-simple inputs).
    """
    import json

    meta = new_root / "meta"
    stats = json.loads((meta / "stats.json").read_text())
    keys = ["mean", "std", "min", "max", "q01", "q99"]
    if "action" not in stats:
        logging.warning("skipping cosmos3_stats_flat.json: stats.json missing 'action'")
        return
    out = {"action": {k: list(stats["action"][k]) for k in keys}}
    if "states" in stats:  # simple stores proprioceptive state under 'states'; server wants 'state'
        out["state"] = {k: list(stats["states"][k]) for k in keys}
        state_dim = len(out["state"]["q01"])
    else:
        state_dim = 0
        logging.warning("'states' absent from stats.json; cosmos3_stats_flat.json will omit state stats")
    out_path = meta / "cosmos3_stats_flat.json"
    out_path.write_text(json.dumps(out, indent=2))
    logging.info(f"wrote {out_path.name} (action {len(out['action']['mean'])}-D, state {state_dim}-D)")


def stage_video_symlinks(root: Path) -> list[Path]:
    """Symlink each video-feature key to its actual (short-named) camera dir, if they differ.

    The base ``convert_videos`` globs ``videos/*/<feature_key>/*.mp4`` (feature_key e.g.
    ``observation.images.egocentric``), but this source names the dir with the short cam name
    (e.g. ``egocentric``) per its ``video_path`` template. Add relative symlinks so the glob
    matches; return them for cleanup. Non-destructive (symlinks only, removed in ``finally``).
    """
    info = load_info(root)
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    created: list[Path] = []
    for chunk_dir in sorted((root / "videos").glob("chunk-*")):
        subdirs = [d for d in chunk_dir.iterdir() if d.is_dir()]
        for key in video_keys:
            if (chunk_dir / key).exists():
                continue
            # short cam dirs = subdirs that aren't already named by a feature key.
            candidates = [d for d in subdirs if d.name not in video_keys]
            if len(video_keys) == 1 and len(candidates) == 1:
                src_name = candidates[0].name
            elif key.rsplit(".", 1)[-1] in {d.name for d in subdirs}:
                src_name = key.rsplit(".", 1)[-1]  # e.g. 'egocentric' from 'observation.images.egocentric'
            else:
                raise FileNotFoundError(
                    f"cannot map video key {key!r} to a camera dir under {chunk_dir} "
                    f"(found {[d.name for d in subdirs]})"
                )
            link = chunk_dir / key
            link.symlink_to(src_name, target_is_directory=True)  # relative -> sibling short dir
            created.append(link)
            logging.info(f"staged video symlink {link.name} -> {src_name} in {chunk_dir}")
    return created


def fix_info_feature_shapes(new_root: Path) -> None:
    """Patch info.json feature shapes that are ``[-1]`` (unknown) to the real vector length.

    This source declares ``states`` / ``action`` with shape ``[-1]``; LeRobot then maps them
    to a scalar ``float`` and fails to cast the ``list<double>`` parquet data
    (``Couldn't cast list<double> to float``). Read each such column's per-row list length
    from the packed parquet and write the concrete ``[N]`` shape so the dataset loads.
    """
    import json

    import pyarrow.parquet as pq

    info_path = new_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    data_file = next((new_root / "data").glob("chunk-*/file-*.parquet"))
    schema = pq.read_schema(data_file)
    list_cols = [f.name for f in schema if "list" in str(f.type)]

    to_fix = [
        k for k, v in info["features"].items()
        if v.get("dtype") != "video" and k in list_cols and (-1 in (v.get("shape") or []) or not v.get("shape"))
    ]
    if not to_fix:
        return
    row = pq.read_table(data_file, columns=to_fix).slice(0, 1).to_pydict()
    for k in to_fix:
        n = len(row[k][0])
        info["features"][k]["shape"] = [n]
        logging.info(f"fixed info.json shape: {k} [-1] -> [{n}]")
    info_path.write_text(json.dumps(info, indent=4))


def name_tasks_index(new_root: Path) -> None:
    """Name the ``meta/tasks.parquet`` index column ``task``.

    The base LeRobot ``convert_tasks`` writes the task string as the (unnamed) pandas index,
    which serializes to ``__index_level_0__``. ``ActionBaseDataset`` resolves a frame's task via
    a ``task`` column, so name the index ``task`` (matching a canonical converter output).
    """
    import pandas as pd

    p = new_root / "meta" / "tasks.parquet"
    df = pd.read_parquet(p)
    if df.index.name != "task":
        df.index.name = "task"
        df.to_parquet(p)
        logging.info("named tasks.parquet index 'task'")


def downcast_float64_to_float32(new_root: Path) -> None:
    """Cast float64 data columns to float32 to match the ``float32`` info.json dtype.

    This source stores floats as ``double`` / ``list<double>`` (float64), but declares them
    ``float32``; HF's loader then fails to cast ``list<double>`` into the float32 schema
    (``Couldn't cast list<double> to float``). Downcast the affected columns (scalar and
    list) to float32 so the parquet matches the declared dtype (like a canonical v3.0 dataset).
    """
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    info = json.loads((new_root / "meta" / "info.json").read_text())
    f32_feats = {k for k, v in info["features"].items()
                 if v.get("dtype") == "float32" and v.get("dtype") != "video"}
    for path in sorted((new_root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        changed = False
        for name in table.column_names:
            if name not in f32_feats:
                continue
            col = table.column(name)
            t = col.type
            if t == pa.float64():
                target = pa.float32()
            elif pa.types.is_list(t) and t.value_type == pa.float64():
                target = pa.list_(pa.float32())
            elif pa.types.is_large_list(t) and t.value_type == pa.float64():
                target = pa.large_list(pa.float32())
            else:
                continue
            table = table.set_column(table.schema.get_field_index(name), name, col.cast(target))
            changed = True
        if changed:
            pq.write_table(table, path)
    logging.info(f"downcast float64 -> float32 for declared-float32 columns in {new_root.name}")


def flatten_unit_list_columns(new_root: Path) -> None:
    """Flatten shape-``[1]`` features stored as 1-element lists to scalars.

    LeRobot maps a ``(1,)`` shape to a scalar HF ``Value`` (not a length-1 Sequence), so a
    shape-``[1]`` feature stored as ``list<...>`` (e.g. ``observation.prev_height``) fails to
    load (``Couldn't cast list<float> to float``). Replace such columns with their single
    element so the parquet matches the scalar schema.
    """
    import json

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    info = json.loads((new_root / "meta" / "info.json").read_text())
    unit_feats = {k for k, v in info["features"].items()
                  if v.get("dtype") != "video" and list(v.get("shape") or []) == [1]}
    for path in sorted((new_root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        changed = False
        for name in table.column_names:
            if name not in unit_feats:
                continue
            col = table.column(name)
            if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
                table = table.set_column(table.schema.get_field_index(name), name,
                                         pc.list_flatten(col))  # 1 elem/list -> scalar column
                changed = True
        if changed:
            pq.write_table(table, path)
    logging.info(f"flattened shape-[1] list columns to scalars in {new_root.name}")


def convert_dataset(
    repo_id: str,
    root: str | Path,
    data_file_size_in_mb: int | None = None,
    video_file_size_in_mb: int | None = None,
) -> None:
    if data_file_size_in_mb is None:
        data_file_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_file_size_in_mb is None:
        video_file_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB

    root = Path(root) / repo_id
    if not root.exists():
        raise FileNotFoundError(f"Dataset not found at {root}")

    version = load_info(root).get("codebase_version", "unknown")
    if version not in ("v2.0", "v2.1", "v3.0"):
        logging.warning(f"Unexpected codebase_version {version!r}; proceeding as v2.1 layout.")

    # Non-destructive: write the converted dataset alongside the source as ``<name>_v30``.
    new_root = root.parent / f"{root.name}_v30"
    if new_root.is_dir():
        shutil.rmtree(new_root)

    convert_info(root, new_root, data_file_size_in_mb, video_file_size_in_mb)
    convert_tasks(root, new_root)
    name_tasks_index(new_root)  # base convert_tasks leaves the task string as an unnamed index
    episodes_metadata = convert_data(root, new_root, data_file_size_in_mb)
    # Source declares states/action with placeholder shape [-1]; set concrete lengths so the
    # v3.0 dataset loads (LeRobot else maps [-1] to scalar float and mis-casts the list data).
    fix_info_feature_shapes(new_root)
    # Source stores floats as float64 but declares float32; downcast to match (else HF's
    # loader can't cast list<double> into the float32 schema).
    downcast_float64_to_float32(new_root)
    # Shape-[1] features stored as 1-element lists must be scalars (LeRobot maps (1,)->scalar).
    flatten_unit_list_columns(new_root)
    # Source names camera dirs with the short cam name (e.g. 'egocentric') rather than the
    # feature key; stage symlinks so convert_videos' glob finds them, then remove them.
    staged_links = stage_video_symlinks(root)
    try:
        episodes_videos_metadata = convert_videos(root, new_root, video_file_size_in_mb)
    finally:
        for link in staged_links:
            link.unlink(missing_ok=True)
    convert_episodes_metadata(root, new_root, episodes_metadata, episodes_videos_metadata)
    # Recompute meta/stats.json with per-feature min/max/mean/std/count/q01/q99.
    write_meta_stats_with_quantiles(new_root)
    # Derive meta/cosmos3_stats_flat.json ({action, state} stats) from stats.json for the
    # action policy server/eval (splits copy it verbatim; see split_dataset.py).
    write_policy_stats_flat(new_root)

    modality = root / "meta" / "modality.json"
    if modality.exists():
        shutil.copy2(modality, new_root / "meta" / "modality.json")
        logging.info("copied meta/modality.json")

    logging.info(f"Done. v3.0 dataset at {new_root} (original kept at {root})")


if __name__ == "__main__":
    init_logging()
    parser = argparse.ArgumentParser(description="Convert a simple LeRobot v2.1 dataset to v3.0.")
    parser.add_argument("--repo-id", type=str, required=True, help="dataset dir name under --root")
    parser.add_argument("--root", type=str, required=True, help="parent dir containing <repo-id>/")
    parser.add_argument("--data-file-size-in-mb", type=int, default=None)
    parser.add_argument("--video-file-size-in-mb", type=int, default=None)
    args = parser.parse_args()
    convert_dataset(**vars(args))
