# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Domain ID helpers for cross-embodiment action datasets."""

EMBODIMENT_TO_DOMAIN_ID: dict[str, int] = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,  # Both Droid and RoboMIND-Franka are using robotiq and franka
    "embodiment_b": 9,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "embodiment_c_gripper": 15,
    "embodiment_c_gripper_ext": 15,
    "xdof_yam": 16,
    "g1_simple": 18,  # G1 whole-body locomotion+pick; single 36-D action, 32-D states
    "fractal": 20,
}


EMBODIMENT_TO_RAW_ACTION_DIM: dict[str, int] = {
    "av": 9,
    "camera_pose": 9,
    "pusht": 2,
    "umi": 10,
    "bridge_orig_lerobot": 10,
    "droid_lerobot": 10,
    "robomind-franka": 10,
    "robomind-franka-dual": 20,
    "robomind-ur": 10,
    "embodiment_b": 30,
    "agibotworld": 29,
    "embodiment_c_gripper": 29,
    "embodiment_c_gripper_ext": 29,
    "xdof_yam": 20,
    "g1_simple": 36,  # single flat action column (hands/arms/rpy/height/base-vel/target_yaw)
    "fractal": 10,
    # NOTE: ``libero`` (7/10/13 depending on ``rotation_space``) and ``hand_pose``
    # (variable with ``keypoint_option`` and ``rotation_format``) are absent
    # because their raw width is set per-dataset at construction time. Inference
    # in inverse_dynamics/policy modes is not supported for these domains until
    # canonical widths are added here.
}


def get_domain_id(embodiment_type: str) -> int:
    """Get the domain ID for a given embodiment type."""
    key = embodiment_type.lower().strip()
    if key not in EMBODIMENT_TO_DOMAIN_ID:
        raise KeyError(
            f"Unknown embodiment type: {embodiment_type!r}. "
            f"Available embodiments: {sorted(EMBODIMENT_TO_DOMAIN_ID.keys())}"
        )
    return EMBODIMENT_TO_DOMAIN_ID[key]


def get_action_dim(embodiment_type: str) -> int:
    """Get the raw action dimension for a given embodiment type."""
    key = embodiment_type.lower().strip()
    if key not in EMBODIMENT_TO_RAW_ACTION_DIM:
        raise KeyError(
            f"Unknown embodiment type: {embodiment_type!r}. "
            f"Available embodiments: {sorted(EMBODIMENT_TO_RAW_ACTION_DIM.keys())}"
        )
    return EMBODIMENT_TO_RAW_ACTION_DIM[key]


def is_valid_domain_name(embodiment_type: str) -> bool:
    """Check if the given embodiment type is recognized."""
    key = embodiment_type.lower().strip()
    return key in EMBODIMENT_TO_RAW_ACTION_DIM


# Per-embodiment action-modality layout, in the model's action space (the width the model
# predicts / validates in, i.e. ``max_action_dim``). Maps a modality name -> a contiguous
# ``(start, end)`` slice of the action vector, used to report per-modality validation L1.
# Embodiments absent here report only the total L1.
EMBODIMENT_ACTION_MODALITIES: dict[str, list[tuple[str, int, int]]] = {
    # g1_simple 36-D: 14 hand_joints + 14 arm_joints + 3 torso_rpy + base_height +
    # base_vx + base_vy + base_vyaw + target_yaw (1-D each).
    "g1_simple": [
        ("hand_joints", 0, 14),
        ("arm_joints", 14, 28),
        ("torso_rpy", 28, 31),
        ("base_height", 31, 32),
        ("base_vx", 32, 33),
        ("base_vy", 33, 34),
        ("base_vyaw", 34, 35),
        ("target_yaw", 35, 36),
    ],
}

# Inverse of EMBODIMENT_TO_DOMAIN_ID (first embodiment wins when ids collide, e.g. id 8/15).
DOMAIN_ID_TO_EMBODIMENT: dict[int, str] = {}
for _emb, _id in EMBODIMENT_TO_DOMAIN_ID.items():
    DOMAIN_ID_TO_EMBODIMENT.setdefault(_id, _emb)


def embodiment_from_domain_id(domain_id: int) -> str | None:
    """Map a domain id back to its embodiment name (``None`` if unknown)."""
    return DOMAIN_ID_TO_EMBODIMENT.get(int(domain_id))


def get_action_modalities(embodiment_type: str | None) -> list[tuple[str, int, int]]:
    """Per-modality ``(name, start, end)`` slices of an embodiment's action vector.

    Returns ``[]`` for a flat or unrecognized embodiment (caller reports total L1 only).
    """
    if embodiment_type is None:
        return []
    return EMBODIMENT_ACTION_MODALITIES.get(embodiment_type.lower().strip(), [])


# Fallback lookup of the modality layout by action WIDTH, auto-derived from
# EMBODIMENT_ACTION_MODALITIES (each embodiment's width is the max end of its slices).
# Used when the embodiment can't be identified from the batch's domain_id (e.g. it is
# absent in the validation batch). First registered embodiment wins if two share a width.
_MODALITIES_BY_WIDTH: dict[int, list[tuple[str, int, int]]] = {}
for _mods in EMBODIMENT_ACTION_MODALITIES.values():
    if _mods:
        _MODALITIES_BY_WIDTH.setdefault(max(end for _, _, end in _mods), _mods)


def action_modalities_for_width(width: int) -> list[tuple[str, int, int]]:
    """Modality slices for a raw action ``width``, derived from EMBODIMENT_ACTION_MODALITIES.
    The fallback when the embodiment can't be read from the batch's domain_id. Returns ``[]``
    for an unregistered width (caller reports total L1 only)."""
    return _MODALITIES_BY_WIDTH.get(width, [])


def resolve_action_modalities(embodiment: str | None, width: int) -> list[tuple[str, int, int]]:
    """Per-modality slices for reporting validation L1: prefer the embodiment's registered
    layout; fall back to deriving from the action ``width`` (robust when the embodiment is
    unknown / domain_id is unavailable in the batch). Slices are clamped to ``width``."""
    mods = get_action_modalities(embodiment) or action_modalities_for_width(width)
    return [(name, start, end) for (name, start, end) in mods if end <= width]
