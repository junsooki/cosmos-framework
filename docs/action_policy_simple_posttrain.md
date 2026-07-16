# Cosmos3-Nano G1 "simple" Whole-Body Action Policy Post-Training

The G1 **`g1_simple`** action policy is post-trained from [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) on the G1 "simple" whole-body locomotion+pick dataset. The model predicts a **flat 36-D whole-body action** conditioned on a **32-D proprioceptive state** and a single ego-camera video observation at `256`px. This example reproduces the post-training procedure — both the single-task recipe and the 14-task multitask blend.

Two external inputs are required: (1) a prepared G1 "simple" dataset (downloaded, converted to LeRobotDataset v3.0, and split into train/val), and (2) a DCP base checkpoint converted from Cosmos3-Nano.

The recipe runs single node / 8 GPUs, and multi-node via HSDP.

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Prerequisites](#prerequisites)
- [Inputs You Provide](#inputs-you-provide)
- [Recipe](#recipe)
- [Full Reproduction](#full-reproduction)
  - [0. Dataset — download, convert, split](#0-dataset--download-convert-split)
  - [1. Fine-tuning](#1-fine-tuning)
- [Checkpoints](#checkpoints)

______________________________________________________________________

<!--TOC-->

## Prerequisites

- [Setup](../README.md#setup) — clone the repo, install the training extras, and activate the environment.
- [Environment Variables](./environment_variables.md) — set environment variables.
- [FAQ](./faq.md) — troubleshooting (OOM during SFT, defaults) and common pitfalls.

The runnable artifacts (TOML recipes, paired launch shells) live in [`examples/`](../examples); all commands below run from the repo root with the environment activated.

## Inputs You Provide

This package ships the training stack — the registered `action_policy_simple_nano` (single-task)
and `action_policy_simple_mixed_nano` / `action_policy_simple_mixed50_nano` (multitask) experiments,
the G1 "simple" action dataset wiring with the recipe knobs (flat 36-D `action`, 32-D `states`
prepend, `use_state`, `minmax` normalization, `fps=50`), and the dataset conversion/split tooling.
Three inputs are external and must be provided per environment:

1. **G1 "simple" dataset (LeRobotDataset v3.0 format)** — download the raw demos, convert them
   to v3.0, and split into train/val (see [Full Reproduction](#0-dataset--download-convert-split)).
   Point `PSI_HOME` at your data root so the experiment resolves each task's split under
   `${PSI_HOME}/data/simple/…`.
2. **DCP base checkpoint** — convert [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) to
   DCP and point `BASE_CHECKPOINT_PATH` at it. The action heads are **not** loaded from it (they
   init fresh — the base has no G1 "simple" action head).
3. **Wan2.2 VAE** — download the tokenizer VAE (`Wan2.2_VAE.pth`) and point `WAN_VAE_PATH` at it.

## Recipe

The dataset/action knobs are fixed in the registered experiment; the TOML sets only run-level
scalars (iters, checkpoint cadence, parallelism, batch, VAE).

| knob                | value                                                                                                        |
| ------------------- | ----------------------------------------------------------------------------------------------------------- |
| init                | `Cosmos3-Nano` (public Hugging Face repo)                                                                    |
| embodiment / domain | `g1_simple` (its own action head)                                                                            |
| action space        | flat **36-D** whole-body action                                                                              |
| state               | `use_state=true` (32-D proprioception, prepended)                                                            |
| normalization       | `minmax` to `[-1,1]`, **per task** (from `meta/cosmos3_stats_flat.json`)                                     |
| resolution          | `256`                                                                                                        |
| camera / video      | single ego camera                                                                                            |
| chunk length        | `32` (tokenizer `encode_exact_durations=[33]`)                                                               |
| fps                 | `50`                                                                                                         |
| trained params      | generation + action heads (action-head LR ×5, action `loss_scale=10`)                                       |
| lr                  | `2e-4` (for the `8192` global batch)                                                                         |
| global batch        | `8192` = `max_samples_per_batch` × world size × `grad_accum_iter` (reduce the first to fit GPU memory)      |
| scheduler           | `LambdaLinear`, `cycle_lengths=[max_iter]` — LR anneals to `~0` at the cycle end (loss flattens by design)  |
| eval                | in-loop validation on the held-out `val` split                                                              |

## Full Reproduction

The OSS flow mirrors the other action recipes (see [docs/training.md](./training.md)). Steps 0.a–0.d
prepare the dataset once per task; step 1 fine-tunes.

### 0. Dataset — download, convert, split

Each task's raw demos ship as `simple/<task>.zip`. The 14 tasks used by the multitask blend are
enumerated in `action_policy_simple_mixed_nano.py` (`SIMPLE_MIXED_TASKS`); pick one for the
single-task recipe or prepare all 14 for the blend.

```bash
export TASK=G1WholebodyBendPickTeleop-v0

# (a) download the task's train demos from the gated Hugging Face dataset repo.
#     Request access first, then `hf auth login` (or export HF_TOKEN).
hf download <HF_DATASET_REPO> simple/$TASK.zip --repo-type dataset --local-dir /path/to/dl
unzip -oq /path/to/dl/simple/$TASK.zip -d ${PSI_HOME}/data/simple/

# (b) convert LeRobot v2.1 -> v3.0. Also writes meta/cosmos3_stats_flat.json (the merged
#     action+state min/max the split carries forward and the server/eval consume).
python -m cosmos_framework.scripts.convert_dataset_simple_to_v30 \
  --repo-id=$TASK --root=${PSI_HOME}/data/simple

# (c) reproducible train/val split of the pristine _v30 (seed 42). split_dataset.py carries
#     cosmos3_stats_flat.json into each split. Single-task -> <TASK>_v30_split/{train,val}:
python -m cosmos_framework.scripts.split_dataset \
  --src ${PSI_HOME}/data/simple/${TASK}_v30 \
  --dst ${PSI_HOME}/data/simple/${TASK}_v30_split \
  --val-fraction 0.1

# (d) convert the base checkpoint -> $BASE_CHECKPOINT_PATH (once, shared by all tasks).
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path Cosmos3-Nano -o $BASE_CHECKPOINT_PATH
```

> **Split the pristine `_v30`, never a split-of-a-split.** `split_dataset.py` re-encodes video to
> exact constant-frame-rate, so a second pass would misalign GOP keyframes with episode boundaries.
> Use `--val-episodes I J K …` to pin exact held-out episodes instead of a random fraction.

**Multitask:** prepare all 14 tasks. The blend experiments read each task's split directly from
`${PSI_HOME}/data/simple/<TASK>_v30_20ep/{train,val}` (18 train / 2 val, for
`action_policy_simple_mixed_nano`) or `…_v30_50tr/{train,val}` (50 train / 2 val, for
`action_policy_simple_mixed50_nano`). Split each task's `_v30` into the matching directory name.

### 1. Fine-tuning

The launcher's `--resume <name>` selects the run name: a **new** name starts a fresh fine-tune from
`BASE_CHECKPOINT_PATH`; re-running the same name auto-resumes from that run's latest checkpoint.
Runs use all 8 GPUs.

**Single task** — the launcher selects `action_policy_simple.toml` (experiment
`action_policy_simple_nano`):

```bash
export PSI_HOME=/path/to/data
export BASE_CHECKPOINT_PATH=/path/to/base_checkpoint
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=/path/to/output_root
export NPROC_PER_NODE=8
# The single-task TOML defaults job.wandb_mode="online"; export WANDB_API_KEY to log,
# or disable it via EXTRA_TAIL_OVERRIDES as below. max_iter/save_iter are overridable too.
export EXTRA_TAIL_OVERRIDES="job.wandb_mode=disabled trainer.max_iter=10000 checkpoint.save_iter=10000"

bash examples/launch_sft_action_policy_simple.sh --resume bendpick20
```

**14-task multitask blend** — one policy over all 14 tasks. `IterativeJointDataLoader` picks one
task-stream per step at equal ratio; each stream is a single-dataset `RankPartitionedDataLoader`
that shards its train episodes over all 8 ranks (wrapping single-dataset loaders dodges the
`world_size >= num_datasets` assert that kills the stock mixed template at 14 tasks > 8 GPUs). All
tasks share `domain_name="g1_simple"` (one 36-D head) and self-normalize on their own stats. Select
the blend TOML explicitly and run the shared launcher body:

```bash
export TOML_FILE="examples/toml/sft_config/action_policy_simple_mixed50.toml"  # experiment action_policy_simple_mixed50_nano
export EXTRA_TAIL_OVERRIDES="job.wandb_mode=disabled trainer.max_iter=40000 checkpoint.save_iter=10000"

bash examples/launch_sft_action_policy_psix.sh --resume 50ep
```

**Overrides / knobs to know:**

| Key                            | Value             | Note                                                                                       |
| ------------------------------ | ----------------- | ------------------------------------------------------------------------------------------ |
| `job.wandb_mode`               | `disabled`        | disable to run without a W&B key; the single-task TOML defaults to `"online"`              |
| `[scheduler] cycle_lengths`    | `[40000]`         | LR (`LambdaLinear`) decays to `~0` over the cycle — **loss flattens because LR→0 at the end** |
| `[trainer] max_iter`           | `10000` / `40000` | set `≤ cycle_length` to stop early *without* bending the LR curve                           |
| `[checkpoint] save_iter`       | `1000`–`10000`    | each full checkpoint ≈ 137 GB (model + optimizer + EMA) — space accordingly                |
| `num_workers`                  | 1–4               | keep `world_size(8) × num_workers ≤ train_eps` or the iterable loader hangs                 |

Registered (fixed in the experiment): `action_normalization="minmax"`, `use_state=True`,
`resolution="256"`, `chunk_length=32`, `max_action_dim=36`, `fps=50`, `domain_name="g1_simple"`,
`lr=2e-4`. For multi-node HSDP, set
`model.parallelism.data_parallel_replicate_degree = <num_nodes>` (the intra-node shard stays 8).

## Checkpoints

- Saved every `save_iter` iters to `IMAGINAIRE_OUTPUT_ROOT` at
  `<project>/<group>/<job.name>/checkpoints/iter_<N>/` (project `psi`, group `cosmos3_action_sft`).
- Each full DCP checkpoint carries model + optimizer + EMA (≈137 GB); size for your milestone
  cadence (e.g. `save_iter=10000` → four checkpoints over a 40k run).
- The run is **resumable** from the latest checkpoint (re-launch with the same `--resume <name>`).
- Judge "done" by the loss actually flattening (per-window drop → 0) and read the **val** curve for
  generalization, not just the training loss — the `LambdaLinear` schedule drives LR→0 at
  `cycle_lengths`, so the loss flattens at the schedule end regardless.
- Export to Hugging Face safetensors via `cosmos_framework.scripts.export_model` (see
  [docs/training.md](./training.md)).
