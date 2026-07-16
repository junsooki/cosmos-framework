# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action normalization helpers."""

import json
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.utils import log


def load_action_stats(stats_path: str) -> dict[str, np.ndarray]:
    """Load pre-computed action normalization stats from a JSON file."""
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Action normalization stats not found at {stats_path}.")
    log.info(f"Loading action normalization stats from {stats_path}")
    with path.open("r") as f:
        raw = json.load(f)
    stat_keys = {"mean", "std", "min", "max", "q01", "q99"}
    return {key: np.array(value, dtype=np.float32) for key, value in raw.items() if key in stat_keys}


def normalize_action(
    action: torch.Tensor,
    method: str,
    stats: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Normalize action tensor.

    Dimensions with a zero range (``min==max`` / ``q01==q99``, e.g. zero-padded dummy
    channels for a reduced embodiment) map to 0 (the neutral center) instead of an
    arbitrary offset, so a constant channel contributes no signal.
    """
    if method == "quantile":
        q01, q99 = stats["q01"], stats["q99"]
        rng = q99 - q01
        out = 2.0 * (action - q01) / rng.clamp(min=1e-8) - 1.0
        return torch.where(rng > 1e-8, out, torch.zeros_like(out))
    if method == "meanstd":
        return (action - stats["mean"]) / stats["std"].clamp(min=1e-8)
    if method == "minmax":
        lo, hi = stats["min"], stats["max"]
        rng = hi - lo
        out = 2.0 * (action - lo) / rng.clamp(min=1e-8) - 1.0
        return torch.where(rng > 1e-8, out, torch.zeros_like(out))
    raise ValueError(f"Unknown normalization method: {method!r}")


def denormalize_action(
    action: torch.Tensor,
    method: str,
    stats: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Denormalize action tensor."""
    if method == "quantile":
        q01, q99 = stats["q01"], stats["q99"]
        return 0.5 * (action + 1.0) * (q99 - q01) + q01
    if method == "meanstd":
        return action * stats["std"] + stats["mean"]
    if method == "minmax":
        lo, hi = stats["min"], stats["max"]
        return 0.5 * (action + 1.0) * (hi - lo) + lo
    raise ValueError(f"Unknown normalization method: {method!r}")
