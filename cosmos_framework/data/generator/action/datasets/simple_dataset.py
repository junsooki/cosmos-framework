# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

""" "SIMPLE" whole-body loco-manipulation LeRobot v3.0 dataset (custom embodiment).

Single egocentric camera. This embodiment stores one flat 36-D ``action`` column per frame:

    [left_hand(7), right_hand(7), left_arm(7), right_arm(7), rpy(3), height(1),
     base_vel(3), target_yaw(1)]  =  36   (per meta/modality.json)

The proprioceptive ``states`` (32-D — note the key is ``states``, not ``observation.state``)
is, when ``use_state=True``, zero-padded to the 36-D action width and prepended as the first
row of the action sequence (the model treats row 0 as the given conditioning frame and predicts
rows ``1:``). With ``action_normalization="minmax"`` actions are mapped ``[min, max] -> [-1, 1]``
using the per-column ``action`` stats in ``meta/stats.json``; the prepended state row is instead
normalized with the ``states`` stats (different modality/range). Pass ``action_normalization=None``
to train on raw values.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from lerobot.datasets.video_utils import decode_video_frames

from cosmos_framework.data.generator.action.action_normalization import normalize_action
from cosmos_framework.data.generator.action.action_spec import ActionSpec, Joint, build_action_spec
from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset

PoseConvention = Literal["backward_framewise"]
Viewpoint = Literal["ego_view"]

_EGO_KEY = "observation.images.egocentric"
_STATE_FEATURE = "states"      # [32] proprioception (note: not "observation.state")
_ACTION_FEATURE = "action"     # [36] single flat action column

_ACTION_DIM = 36
_STATE_DIM = 32


class SimpleActionDataset(ActionBaseDataset):
    """G1 "simple" action dataset — flat 36-D action + 32-D states, single ego view."""

    def __init__(
        self,
        root: str,
        fps: float = 50.0,
        chunk_length: int = 16,
        mode: str = "policy",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 3e-4,
        viewpoint: Viewpoint = "ego_view",
        use_state: bool = True,
        action_normalization: str | None = None,
        sample_stride: int = 1,
        domain_name: str = "g1_simple",
    ) -> None:
        if viewpoint != "ego_view":
            raise NotImplementedError("SimpleActionDataset only supports the single-camera ego_view.")
        super().__init__(
            root=root,
            domain_name=domain_name,
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )
        self._use_state = bool(use_state)
        self._state_norm_stats: dict[str, torch.Tensor] | None = None

        # Canonical LeRobot v3.0: ``index`` is a dataset-global unique monotonic id (0..N-1),
        # episode-major; the base sorts ``self._rows`` by it so episodes are contiguous blocks.
        n = len(self._rows)
        row_index = np.fromiter((int(r["index"]) for r in self._rows), dtype=np.int64, count=n)
        if not np.array_equal(row_index, np.arange(n, dtype=np.int64)):
            raise ValueError(
                f"{type(self).__name__}: 'index' is not a dataset-global 0..N-1 id "
                f"({len(np.unique(row_index))} unique over {n} rows)."
            )

        # Fail fast on a layout mismatch (clear message instead of a mid-training broadcast error).
        if self._rows:
            r0 = self._rows[0]
            if _ACTION_FEATURE not in r0:
                raise KeyError(f"{type(self).__name__}: missing action column {_ACTION_FEATURE!r}.")
            adim = int(np.asarray(r0[_ACTION_FEATURE], dtype=np.float32).reshape(-1).shape[0])
            if adim != _ACTION_DIM:
                raise ValueError(f"{type(self).__name__}: {_ACTION_FEATURE} has dim {adim}, expected {_ACTION_DIM}.")
            if self._use_state:
                if _STATE_FEATURE not in r0:
                    raise KeyError(f"{type(self).__name__}: missing state column {_STATE_FEATURE!r} (use_state=True).")
                sdim = int(np.asarray(r0[_STATE_FEATURE], dtype=np.float32).reshape(-1).shape[0])
                if sdim != _STATE_DIM:
                    raise ValueError(f"{type(self).__name__}: {_STATE_FEATURE} has dim {sdim}, expected {_STATE_DIM}.")

        # Per-dim absolute mask: True -> repeat last on trailing pad, False -> zero-pad.
        modality_path = self._root / "meta" / "modality.json"
        if modality_path.exists():
            action_mod = json.loads(modality_path.read_text()).get("action", {})
            mask = np.zeros(_ACTION_DIM, dtype=bool)
            for seg in action_mod.values():
                if seg.get("absolute", True):
                    mask[seg["start"] : seg["end"]] = True
            self._action_absolute_mask: np.ndarray = mask
        else:
            self._action_absolute_mask = np.ones(_ACTION_DIM, dtype=bool)

        # Group global-index-ordered rows into contiguous per-episode blocks; every frame
        # is a valid sample start — trailing windows are padded in _build_action/_load_ego_video.
        self._episode_rows: list[list[dict[str, Any]]] = []
        cur_ep: int | None = None
        for row in self._rows:
            ep = int(row["episode_index"])
            if ep != cur_ep:
                self._episode_rows.append([])
                cur_ep = ep
            self._episode_rows[-1].append(row)
        valid_counts = [len(rows) for rows in self._episode_rows]
        self._valid_cum = np.cumsum(valid_counts).astype(np.int64)

    @property
    def action_dim(self) -> int:
        return _ACTION_DIM

    def _action_spec(self) -> ActionSpec:
        # Labeled joint blocks per meta/modality.json (informational; sums to 36).
        return build_action_spec(
            Joint(n=7, label="left_hand"),
            Joint(n=7, label="right_hand"),
            Joint(n=7, label="left_arm"),
            Joint(n=7, label="right_arm"),
            Joint(n=3, label="rpy"),
            Joint(n=1, label="height"),
            Joint(n=3, label="base_vel"),
            Joint(n=1, label="target_yaw"),
        )

    @classmethod
    def _stats_path(cls) -> Path:
        raise NotImplementedError(
            "SimpleActionDataset reads action stats from meta/stats.json via _load_norm_stats."
        )

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        """36-D ``q01``/``q99``/``min``/``max`` for the flat ``action`` column from meta/stats.json."""
        if self._norm_stats is None:
            stats = json.loads((self._root / "meta" / "stats.json").read_text())
            if _ACTION_FEATURE not in stats or "min" not in stats.get(_ACTION_FEATURE, {}):
                raise KeyError(
                    f"{type(self).__name__}: meta/stats.json missing stats for {_ACTION_FEATURE!r}. "
                    "Re-convert with cosmos_framework/scripts/convert_dataset_simple_to_v30.py, "
                    "or set action_normalization=None."
                )
            self._norm_stats = {
                key: torch.from_numpy(np.asarray(stats[_ACTION_FEATURE][key], dtype=np.float32)).float()
                for key in ("q01", "q99", "min", "max")
            }
        return self._norm_stats

    def _load_state_norm_stats(self) -> dict[str, torch.Tensor]:
        """32-D ``min``/``max``/``q01``/``q99`` for ``states`` (normalizes the prepended row)."""
        if self._state_norm_stats is None:
            stats = json.loads((self._root / "meta" / "stats.json").read_text())
            if _STATE_FEATURE not in stats or "min" not in stats.get(_STATE_FEATURE, {}):
                raise KeyError(
                    f"{type(self).__name__}: meta/stats.json missing stats for {_STATE_FEATURE!r}."
                )
            self._state_norm_stats = {
                key: torch.from_numpy(np.asarray(stats[_STATE_FEATURE][key], dtype=np.float32)).float()
                for key in ("q01", "q99", "min", "max")
            }
        return self._state_norm_stats

    def _build_result(self, *, action: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        """Build the sample, then re-normalize the prepended ``use_state`` row with STATE stats.

        The base normalizes the whole ``[chunk+1, 36]`` action with ACTION stats, but row 0 holds
        the raw 32-D proprioceptive state (zero-padded to 36) — a different modality. Re-normalize
        its state dims ``[0:32]`` with the STATE stats and zero the padding ``[32:36]``.
        """
        result = super()._build_result(action=action, **kwargs)
        if self._use_state and self.action_normalization is not None:
            raw_state_row = action[0, :_STATE_DIM]
            result["action"][0, :_STATE_DIM] = normalize_action(
                raw_state_row, self.action_normalization, self._load_state_norm_stats()
            )
            result["action"][0, _STATE_DIM:] = 0.0
        return result

    def __len__(self) -> int:
        return int(self._valid_cum[-1]) if self._valid_cum.size else 0

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        ep = int(np.searchsorted(self._valid_cum, idx, side="right"))
        prev = int(self._valid_cum[ep - 1]) if ep > 0 else 0
        start = idx - prev
        rows = self._episode_rows[ep]
        observation_rows = rows[start : start + self._chunk_length + 1]

        episode = self._episodes[int(observation_rows[0]["episode_index"])]
        task = self._tasks[int(observation_rows[0]["task_index"])]
        ai_caption = random.choice(task.split(" | "))

        video = self._load_ego_video(episode, observation_rows)
        action = self._build_action(observation_rows)

        return self._build_result(
            mode=mode,
            video=video,
            action=action,
            ai_caption=ai_caption,
            action_spec_names=self.action_names,
        )

    def _load_ego_video(self, episode: dict[str, Any], observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        timestamps = [float(row["timestamp"]) for row in observation_rows]
        # Repeat last timestamp for trailing pad frames so decode_video_frames returns
        # the full chunk_length+1 frames even when we're near the end of an episode.
        expected = self._chunk_length + 1
        if len(timestamps) < expected:
            timestamps += [timestamps[-1]] * (expected - len(timestamps))
        return decode_video_frames(
            self._video_path(episode, _EGO_KEY),
            [float(episode.get(f"videos/{_EGO_KEY}/from_timestamp", 0.0)) + ts for ts in timestamps],
            self._tolerance_s,
        )

    def _build_action(self, observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        """36-D action over the chunk (single flat ``action`` column).

        Window is ``chunk + 1`` rows; the ``chunk`` actions that advance it are ``rows[:-1]``.
        Near episode boundaries ``action_rows`` may be shorter than ``chunk_length``; trailing
        frames are padded: absolute dims repeat the last available action, non-absolute dims
        (velocities) are zero-padded.
        ``rows[0]`` additionally supplies the initial ``states`` (zero-padded to 36) prepended as
        the conditioning row when ``use_state``.
        """
        action_rows = observation_rows[:-1]
        k = len(action_rows)

        if k > 0:
            action = np.asarray([r[_ACTION_FEATURE] for r in action_rows], dtype=np.float32)  # [k, 36]
        else:
            action = np.empty((0, _ACTION_DIM), dtype=np.float32)

        pad = self._chunk_length - k
        if pad > 0:
            # Seed for absolute dims: last available action row, or the current frame's action.
            if k > 0:
                seed = action[-1].copy()
            else:
                seed = np.asarray(observation_rows[0][_ACTION_FEATURE], dtype=np.float32).reshape(-1)
            pad_row = seed.copy()
            pad_row[~self._action_absolute_mask] = 0.0  # zero non-absolute (velocity) dims
            action = np.concatenate([action, np.tile(pad_row, (pad, 1))], axis=0)  # [chunk, 36]

        if self._use_state:
            init = observation_rows[0]
            state = np.asarray(init[_STATE_FEATURE], dtype=np.float32).reshape(-1)  # [32]
            init_row = np.zeros((1, _ACTION_DIM), dtype=np.float32)                 # [1, 36]
            init_row[0, :_STATE_DIM] = state[:_STATE_DIM]
            action = np.concatenate([init_row, action], axis=0)  # [chunk + 1, 36]

        return torch.from_numpy(action).float()

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Per-episode flat-index blocks ``(start, length)`` over the packed valid-sample index."""
        blocks: list[tuple[int, int]] = []
        prev = 0
        for c in self._valid_cum.tolist():
            c = int(c)
            if c > prev:
                blocks.append((prev, c - prev))
            prev = c
        return blocks
