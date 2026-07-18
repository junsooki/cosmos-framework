# Cosmos3-Nano G1 "simple" Whole-Body Action Policy Post-Training

The G1 **`g1_simple`** action policy is post-trained from [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) on the G1 "simple" whole-body locomotion+pick dataset. The model predicts a **flat 36-D whole-body action** conditioned on a **32-D proprioceptive state** and a single ego-camera video observation at `256`px. This example reproduces the single-task post-training procedure.

Two external inputs are required: (1) a pre-downloaded G1 "simple" dataset in LeRobotDataset v3.0 format, and (2) a DCP base checkpoint converted from Cosmos3-Nano.

The recipe runs single node / 8 GPUs, and multi-node via HSDP.

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Prerequisites](#prerequisites)
- [Inputs You Provide](#inputs-you-provide)
- [Recipe](#recipe)
- [Full Reproduction](#full-reproduction)
  - [0. Dataset + base checkpoint](#0-dataset--base-checkpoint)
  - [1. Fine-tuning](#1-fine-tuning)
- [Checkpoints](#checkpoints)

______________________________________________________________________

<!--TOC-->

## Prerequisites

- [Setup](../README.md#setup) — clone the repo, install the training extras, and activate the environment.
- [Environment Variables](./environment_variables.md) — set environment variables.
- [FAQ](./faq.md) — troubleshooting (OOM during SFT, defaults) and common pitfalls.

The runnable artifacts (TOML recipe, paired launch shell) live in [`examples/`](../examples); all commands below run from the repo root with the environment activated.

## Inputs You Provide

This package ships the training stack — the registered `action_policy_simple_nano` experiment and
the G1 "simple" action dataset wiring with the recipe knobs (flat 36-D `action`, 32-D `states`
prepend, `use_state`, `minmax` normalization, `fps=50`). Three inputs are external and must be
provided per environment:

1. **G1 "simple" dataset (LeRobotDataset v3.0 format)** — pre-download the dataset and point
   `SIMPLE_ROOT` at the resulting directory (must contain `meta/info.json`, `meta/stats.json`, and
   `meta/modality.json`). The `simple` dataset schema is a single flat 36-D `action` column and a
   32-D `state`; `minmax` normalization is read from the standard `meta/stats.json`. The launcher
   bridges `DATASET_PATH → SIMPLE_ROOT`, so either variable works.
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
| normalization       | `minmax` to `[-1,1]`, from the dataset's `meta/stats.json`                                                   |
| resolution          | `256`                                                                                                        |
| camera / video      | single ego camera                                                                                            |
| chunk length        | `32` (tokenizer `encode_exact_durations=[33]`)                                                               |
| fps                 | `50`                                                                                                         |
| trained params      | generation + action heads (action-head LR ×5, action `loss_scale=10`)                                       |
| lr                  | `2e-4`                                                                                                       |
| scheduler           | `LambdaLinear`, `cycle_lengths=[max_iter]` — LR anneals to `~0` at the cycle end (loss flattens by design)  |
| eval                | in-loop validation on the dataset                                                                            |

## Full Reproduction

The OSS flow mirrors the other action recipes (see [docs/training.md](./training.md)). Step 0 stages
the two external inputs; step 1 fine-tunes.

### 0. Dataset + base checkpoint

```bash
# (a) pre-download the G1 "simple" dataset (LeRobotDataset v3.0) and point SIMPLE_ROOT at it.
#     The directory must contain meta/{info,stats,modality}.json.
export SIMPLE_ROOT=/path/to/Cosmos3-SIMPLE

# (b) convert the Cosmos3-Nano base checkpoint -> $BASE_CHECKPOINT_PATH (once).
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path Cosmos3-Nano -o $BASE_CHECKPOINT_PATH
```

> The dataset must already be in LeRobotDataset v3.0 format with a flat 36-D `action` column and a
> 32-D `state`, and a `meta/stats.json` carrying `min`/`max`/`q01`/`q99` for the `action` feature
> (used for `minmax` normalization). This recipe does not ship raw-demo conversion tooling — provide
> the prepared v3.0 dataset, as with the [DROID recipe](./action_policy_droid_posttrain.md).

### 1. Fine-tuning

The launcher's `--resume <name>` selects the run name: a **new** name starts a fresh fine-tune from
`BASE_CHECKPOINT_PATH`; re-running the same name auto-resumes from that run's latest checkpoint.
Runs use all 8 GPUs. The launcher selects `action_policy_simple.toml` (experiment
`action_policy_simple_nano`):

```bash
export SIMPLE_ROOT=/path/to/Cosmos3-SIMPLE        # or DATASET_PATH; the launcher bridges it
export BASE_CHECKPOINT_PATH=/path/to/base_checkpoint
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=/path/to/output_root
export NPROC_PER_NODE=8
# The TOML defaults job.wandb_mode="online"; export WANDB_API_KEY to log, or disable it via
# EXTRA_TAIL_OVERRIDES as below. max_iter/save_iter are overridable too.
export EXTRA_TAIL_OVERRIDES="job.wandb_mode=disabled trainer.max_iter=10000 checkpoint.save_iter=10000"

bash examples/launch_sft_action_policy_simple.sh --resume simple_run
```

**Overrides / knobs to know:**

| Key                            | Value             | Note                                                                                       |
| ------------------------------ | ----------------- | ------------------------------------------------------------------------------------------ |
| `job.wandb_mode`               | `disabled`        | disable to run without a W&B key; the TOML defaults to `"online"`                           |
| `[scheduler] cycle_lengths`    | `[max_iter]`      | LR (`LambdaLinear`) decays to `~0` over the cycle — **loss flattens because LR→0 at the end** |
| `[trainer] max_iter`           | e.g. `10000`      | set `≤ cycle_length` to stop early *without* bending the LR curve                           |
| `[checkpoint] save_iter`       | `1000`–`10000`    | each full checkpoint ≈ 137 GB (model + optimizer + EMA) — space accordingly                |
| `[dataloader_train] max_samples_per_batch` | e.g. `16` | per-rank micro-batch; global batch = this × world size × `grad_accum_iter`                  |

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
- Judge "done" by the loss actually flattening (per-window drop → 0); the `LambdaLinear` schedule
  drives LR→0 at `cycle_lengths`, so the loss flattens at the schedule end regardless.
- Serve the trained checkpoint with `action_policy_server_simple.py` — see
  [action_policy_simple_server.md](./action_policy_simple_server.md).
- Export to Hugging Face safetensors via `cosmos_framework.scripts.export_model` (see
  [docs/training.md](./training.md)).
