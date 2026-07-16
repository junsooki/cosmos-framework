#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Per-experiment launcher: G1 "simple" (whole-body locomotion+pick) action-policy SFT.
# Selects examples/toml/sft_config/action_policy_simple.toml (experiment
# `action_policy_simple_nano`) and delegates ALL logic — .env loading, --resume,
# run-name/log timestamping — to launch_sft_action_policy_psix.sh, which derives
# the run-name prefix from the selected TOML. Pass --resume [latest|<ts>] and any
# EXTRA_TAIL_OVERRIDES / EXTRA_TRAIN_ARGS exactly as with the other launchers.
# ============================================================================
export TOML_FILE="examples/toml/sft_config/action_policy_simple.toml"
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_sft_action_policy_psix.sh" "$@"
