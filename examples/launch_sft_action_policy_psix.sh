#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Shared launcher BODY for the G1 action-policy SFT wrappers ("psix"). Drives
# cosmos_framework.scripts.train through _sft_launcher_common.sh. The
# per-experiment wrapper — launch_sft_action_policy_simple.sh — does `export
# TOML_FILE=<recipe>.toml` and `exec bash` this script; running it directly
# falls back to the simple recipe.
#
# Everything common to the wrapper lives here and runs once per launch:
#   * load the repo-root .env so its values win over the defaults below;
#   * derive the run-name prefix from the selected TOML's [job].name default;
#   * resolve `--resume [latest|<ts>]` to a concrete run to (auto-)resume;
#   * mint ONE timestamp shared by the run name and its log filename;
# then hand off to _sft_launcher_common.sh (input checks + the torchrun call).
#
# Env vars (override for your filesystem):
#   TOML_FILE             recipe TOML; set by the wrapper (default: simple recipe).
#   PSI_HOME              Data/cache root; the experiment SKU reads dataset roots as
#                         "${oc.env:PSI_HOME}/data/..." (dataset paths live in the SKU, not here).
#   BASE_CHECKPOINT_PATH  DCP of nvidia/Cosmos3-Nano (convert_model_to_dcp; see docs)
#   WAN_VAE_PATH          Wan2.2 VAE .pth (Wan-AI/Wan2.2-TI2V-5B)
#   WANDB_API_KEY         for online logging (set TOML wandb_mode="online" first)
#   NPROC_PER_NODE        torchrun --nproc_per_node (default 8)
#   EXTRA_TAIL_OVERRIDES  space-separated Hydra overrides
#   EXTRA_TRAIN_ARGS      space-separated train.py flags before the `--` opts, e.g.
#                         --attach_vscode_debugger (debugpy on :3002, rank 0) or --dryrun
#
# The wrapper and this script both accept `--resume [latest|<ts>]` to reuse a run.
#
# Single-node smoke (config/data sanity, a few iters):
#   export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10 \
#                                dataloader_train.max_samples_per_batch=1"
#   bash examples/launch_sft_action_policy_simple.sh
#
# Multi-node: launch on every worker; the trainer reads torchrun's
# --nnodes/--node_rank. For HSDP set
# model.parallelism.data_parallel_replicate_degree = <num_nodes> (shard stays 8).
# ============================================================================

# Load the repo-root .env (PSI_HOME, HF_TOKEN, DATASET_PATH, BASE_CHECKPOINT_PATH,
# WAN_VAE_PATH, IMAGINAIRE_OUTPUT_ROOT, WANDB_API_KEY, …) before the defaults below so
# its values win. `set -a` auto-exports every assignment so they reach the train.py child.
_ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env"
if [[ -f "$_ENV_FILE" ]]; then
    set -a; source "$_ENV_FILE"; set +a
    echo ">>> loaded env from $_ENV_FILE:"
    while IFS= read -r _line || [[ -n "$_line" ]]; do
        _line="${_line#"${_line%%[![:space:]]*}"}"   # ltrim leading spaces
        [[ -z "$_line" || "$_line" == \#* ]] && continue
        _line="${_line#export }"                       # tolerate `export KEY=...`
        [[ "$_line" != *=* ]] && continue
        _k="${_line%%=*}"; _k="${_k// /}"
        case "$_k" in
            *TOKEN*|*SECRET*|*PASSWORD*|*API_KEY*|*APIKEY*) _v="<redacted>" ;;
            *) _v="${!_k}" ;;
        esac
        echo ">>>   $_k=$_v"
    done < "$_ENV_FILE"
    unset _line _k _v
fi

# TOML_FILE is overridable so the per-experiment wrapper launcher can point this shared
# body at its recipe. Defaults to the simple recipe when run directly.
: "${TOML_FILE:=examples/toml/sft_config/action_policy_simple.toml}"
# Run-name prefix for --resume lookup, run naming, and the log filename — derived from the
# selected TOML's [job].name default (action_policy_..._${now}), so a wrapper only needs to
# set TOML_FILE. Falls back to the simple prefix if extraction fails.
_RUN_PREFIX="$(sed -n 's/.*RESUME_RUN_NAME,\(.*\)_\${now.*/\1/p' "$TOML_FILE" 2>/dev/null)"
_RUN_PREFIX="${_RUN_PREFIX:-action_policy_simple}"
# NOTE: no DATASET_PATH default here on purpose. The action-policy experiments read their
# dataset roots directly from the SKU (root="${oc.env:PSI_HOME}/data/..."), so DATASET_PATH
# is unused; leaving it unset skips the common launcher's DATASET_PATH existence check.
: "${BASE_CHECKPOINT_PATH:=$PSI_HOME/cache/Cosmos3-Nano}"


# `--resume [latest|<ts>]`: reuse an existing run so training auto-resumes from its
# checkpoints. `--resume` or `--resume latest` picks the most recent run by timestamp
# suffix; `--resume <ts>` picks ${_RUN_PREFIX}_<ts> (full run name also
# accepted). Exported as RESUME_RUN_NAME, which the TOML's [job].name reads; without it
# the name falls back to a fresh ${now} timestamp. (The launcher-common script forwards
# only EXTRA_TRAIN_ARGS / TAIL_OVERRIDES, never "$@", so this flag never reaches train.py.)
_resume=0
_resume_arg=""
_expect_val=0
for _arg in "$@"; do
    if [[ "$_expect_val" == "1" ]]; then
        _expect_val=0
        # Consume the token after --resume as the target only if it isn't another flag.
        if [[ "$_arg" != -* ]]; then _resume_arg="$_arg"; continue; fi
    fi
    [[ "$_arg" == "--resume" ]] && { _resume=1; _expect_val=1; }
done

if [[ "$_resume" == "1" ]]; then
    _out_root="${IMAGINAIRE_OUTPUT_ROOT:-outputs/train}"
    _project=$(sed -n 's/^[[:space:]]*project[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$TOML_FILE")
    _group=$(sed -n 's/^[[:space:]]*group[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$TOML_FILE")
    _runs_dir="$_out_root/$_project/$_group"
    if [[ -z "$_resume_arg" || "$_resume_arg" == "latest" ]]; then
        # Timestamp suffix is fixed-width (%y%m%d%H%M), so lexical sort == chronological.
        _latest=$(ls -d "$_runs_dir/${_RUN_PREFIX}_"* 2>/dev/null | sort | tail -n 1)
        [[ -n "$_latest" ]] || { echo "ERROR: --resume: no run found under $_runs_dir/${_RUN_PREFIX}_*" >&2; exit 1; }
        RESUME_RUN_NAME="$(basename "$_latest")"
    else
        # Explicit target: a bare timestamp suffix or a full run directory name. Unlike
        # `--resume [latest]` (which requires an existing run), an explicit name is
        # create-if-missing: the SAME command starts a fresh run with that fixed name the
        # first time and auto-resumes it (from its checkpoints) on every later invocation.
        case "$_resume_arg" in
            ${_RUN_PREFIX}_*) RESUME_RUN_NAME="$_resume_arg" ;;
            *) RESUME_RUN_NAME="${_RUN_PREFIX}_$_resume_arg" ;;
        esac
        _latest="$_runs_dir/$RESUME_RUN_NAME"
    fi
    export RESUME_RUN_NAME
    if [[ -f "$_latest/checkpoints/latest_checkpoint.txt" ]]; then
        echo "[launch] --resume -> $RESUME_RUN_NAME (resuming from ckpt: $(cat "$_latest/checkpoints/latest_checkpoint.txt"))"
    elif [[ -d "$_latest" ]]; then
        echo "[launch] --resume -> $RESUME_RUN_NAME (run exists but no checkpoint yet; starting from BASE_CHECKPOINT_PATH)" >&2
    else
        echo "[launch] --resume -> $RESUME_RUN_NAME (new run with fixed name; starting from BASE_CHECKPOINT_PATH, resumable on re-run)"
    fi
fi

# One timestamp shared by the run name and its log file, so the log
# (outputs/train/logs/<toml-stem>_sft_<ts>.log) correlates with the run dir and runs no
# longer clobber a single flat log. On --resume, reuse the resumed run's suffix; otherwise
# mint one now and pin the run name to it (RESUME_RUN_NAME is the run-name env the TOML reads,
# i.e. [job].name = ${oc.env:RESUME_RUN_NAME, action_policy_simple_${now}}).
if [[ -n "${RESUME_RUN_NAME:-}" ]]; then
    _RUN_TS="${RESUME_RUN_NAME##*_}"          # trailing %y%m%d%H%M suffix of the run name
else
    _RUN_TS="$(date +%y%m%d%H%M)"
    export RESUME_RUN_NAME="${_RUN_PREFIX}_${_RUN_TS}"
fi
export LOG_FILENAME="${_RUN_PREFIX}_sft_${_RUN_TS}.log"

# Extra Hydra overrides from the environment: a space-separated string word-split into
# the TAIL_OVERRIDES array. An exported string survives `bash <wrapper>` (a child process),
# unlike a TAIL_OVERRIDES array set in your shell.
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
