# G1 `simple` Whole-Body Action Policy Server

The G1 **`g1_simple`** whole-body action policy (Cosmos3-Nano) is served by an HTTP **policy server**
that streams predicted action chunks to a client. The server (`action_policy_server_simple.py`) is the
single-ego-camera, `use_state` variant of the action server: it takes one observation image plus a
proprioceptive state vector and returns a chunk of raw (denormalized) actions and a short rollout video.

This guide covers serving a trained checkpoint, the `/predict` HTTP protocol, **open-loop** evaluation
(predicted-vs-recorded actions) with `examples/eval_g1_openloop.py`, and **closed-loop** evaluation in
the SIMPLE simulator. The policy is a flat **36-D** action + **32-D** proprioceptive `use_state`,
`chunk_length=32`, `fps=50`, `minmax` action normalization, `domain_name="g1_simple"`.

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Policy Server](#policy-server)
- [`/predict` Protocol](#predict-protocol)
- [Open-Loop Evaluation](#open-loop-evaluation)
- [Closed-Loop Evaluation (SIMPLE Simulator)](#closed-loop-evaluation-simple-simulator)
- [Notes](#notes)

______________________________________________________________________

<!--TOC-->

## Policy Server

The server runs **natively** in the cosmos virtual environment (see [`setup.md`](setup.md)); call
`.venv/bin/python` directly. Serve with the **task's own** `cosmos3_stats_flat.json` — normalization is
**per task**, so a multitask checkpoint needs one server restart per task, each with that task's stats file.

```bash
export TASK=G1WholebodyBendPickTeleop-v0
export CKPT=.runs/psi/cosmos3_action_sft/action_policy_simple_bendpick20/checkpoints/iter_000010000
export DS=/path/to/data/simple/${TASK}_v30_20ep

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -m cosmos_framework.scripts.action_policy_server_simple \
  --checkpoint-path $CKPT --config-file cosmos_framework/configs/base/config.py \
  --experiment action_policy_simple_nano \
  --experiment-overrides model.config.tokenizer.vae_path=$WAN_VAE_PATH model.config.compile.enabled=False \
  --action-chunk-size 32 --no-guardrails --fps 50 \
  --stats-path $DS/train/meta/cosmos3_stats_flat.json --port 22085
```

| Flag | Meaning |
| --- | --- |
| `--experiment` | `action_policy_simple_nano` (the 36-D arch; same for single-task and multitask checkpoints) |
| `--stats-path` | that task's merged `cosmos3_stats_flat.json`; the server returns **raw** actions and normalizes the state row |
| `--action-chunk-size` / `--fps` | `32` / `50` — match training |
| `model.config.compile.enabled` | default `True` → first `/predict` compiles (~min) then runs fast; set `False` for one-shot/debug |
| `--no-guardrails` | skip the gated video guardrail |

`cosmos3_stats_flat.json` holds the merged action+state min/max the server consumes to denormalize actions
and normalize the prepended `use_state` conditioning row. Because the checkpoint was trained with
`use_state=True`, requests **must** include `state`; omitting it silently runs `use_state=False`, a
train/inference mismatch.

## `/predict` Protocol

The server exposes three HTTP endpoints:

| Method / Path | Purpose |
| --- | --- |
| `GET /` | Health check — returns an OK response when the server is up. |
| `GET /info` | Model / runtime info: `run_name`, checkpoint, sampling params, etc. |
| `POST /predict` | Run policy inference on one observation and return an action chunk + rollout video. |

**`POST /predict` request** (JSON):

```json
{
  "image": "<base64_png>",
  "prompt": "<task instruction>",
  "domain_name": "g1_simple",
  "image_size": 256,
  "view_point": "ego_view",
  "state": [/* 32-D proprioceptive state */]
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `image` | base64 PNG | The current observation frame from the single ego camera. A `data:image/png;base64,…` prefix is accepted. |
| `prompt` | string | Natural-language task instruction. |
| `domain_name` | string | Embodiment / domain id — `"g1_simple"`. |
| `image_size` | int > 0 | Square edge length of the observation (e.g. `256`). |
| `view_point` | string | Viewpoint tag; defaults to `"ego_view"`. Selects the viewpoint sentence appended to the prompt to match training. |
| `state` | list[float] | Proprioceptive state vector (32-D for `g1_simple`). Required because the checkpoint was trained with `use_state=True`. |

**`POST /predict` response** (JSON):

```json
{
  "action": [[a0, a1, /* … */], /* … */],
  "video":  ["<base64_png>", /* … */]
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `action` | list[list[float]] | The predicted action chunk — `action_chunk_size` rows (32), each a 36-D action vector. Returned **raw** (denormalized) when the server is launched with `--stats-path`. |
| `video` | list[base64 PNG] | `action_chunk_size + 1` rollout frames: frame 0 is the conditioning frame, frames `1..H` are the predicted future. Frames are content-cropped (reflection padding removed). |

## Open-Loop Evaluation

Open-loop eval feeds **recorded** observation frames to `/predict` and compares predicted vs ground-truth
actions. The train-vs-val gap is the generalization signal (train L1 ≪ val L1 ⇒ data-limited).

```bash
srv=http://localhost:22085; stats=$DS/train/meta/cosmos3_stats_flat.json
.venv/bin/python examples/eval_g1_openloop.py --domain-name g1_simple --root $DS/val \
  --episode 0 --stride 32 --image-size 256 --server $srv --stats-path $stats --out /tmp/val0.mp4
.venv/bin/python examples/eval_g1_openloop.py --domain-name g1_simple --root $DS/train \
  --episode 0 --stride 32 --image-size 256 --server $srv --stats-path $stats --out /tmp/tr0.mp4
```

The client walks the episode in strided steps, POSTing the real observation frame at each step (open-loop:
observations come from the dataset, not the model's own rollout), then tiles the predicted rollout segments
into one continuous video. Each `/predict` returns `action_chunk_size` actions and `action_chunk_size + 1`
frames; to tile without overlap or gaps it takes the `stride` future frames `[1 : 1+stride]` per request and
advances by `stride`, so `--stride` defaults to `action_chunk_size` (32).

Pass the **server's** flat `cosmos3_stats_flat.json` to `--stats-path`, not the default per-episode
`stats.json` — the latter's constant-on-some-dims ranges blow up the normalized error. The eval reports
per-modality (the 8 whole-body components for `g1_simple`) `raw` and `minmax[-1,1]` × `MSE`/`L1`, and writes a
side-by-side `predicted | ground-truth` mp4 spanning the episode. `--mock` runs the whole pipeline without a
server (ground truth used as the prediction, MSE ≈ 0).

## Closed-Loop Evaluation (SIMPLE Simulator)

The SIMPLE simulator runs in **Docker** and talks to the native cosmos server over `127.0.0.1:<port>` (the
container uses host networking). **Never pass `--max-episode-steps`** — the task metadata step budget is the
source of truth; capping it makes episodes fail on the clock.

**Teleop tasks — WBC path** (agent `cosmos3_decoupled_wbc`, `eval-decoupled-wbc` service). With the server up
on port `22085` (see [Policy Server](#policy-server)):

```bash
cd ~/SIMPLE
GPUs=1 ./run_closedloop.sh $TASK 10 level-0 22085
# results: data/evals_decoupled_wbc/eval_stats.txt  (episode_N: True/False)
# videos:  data/evals/cosmos3_decoupled_wbc/<task>/level-0/episode_*/
```

**Motion-planning tasks — MP path** (agent `cosmos3`, `eval` service; the non-WBC twin):

```bash
cd ~/SIMPLE
GPUs=1 docker compose -p simplemp run --rm eval "simple/$TASK" cosmos3 train \
  --data-format lerobot --data-dir "data/evals/simple-eval/$TASK/level-0" \
  --host 127.0.0.1 --port 22085 --sim-mode mujoco_isaac --headless --num-episodes 10
# verdict encoded in the video filename suffix: <episode>/*_success.mp4 vs *_failed.mp4
```

Success = object-in-target per the task's criterion. For a full multitask sweep, `parallel_closedloop.sh`
(per-task server + stats, `compile.enabled=True` for speed) runs the teleop tasks and `mp_closedloop_eval.sh`
the motion-planning tasks.

## Notes

- **Native venv:** the server runs in the cosmos virtual environment; call `.venv/bin/python` directly and do
  **not** use a bare `uv run` (it re-syncs the environment and can break CUDA).
- **Per-task normalization:** the server denormalizes with `--stats-path`, so a multitask checkpoint must be
  re-served per task with that task's `cosmos3_stats_flat.json`, or actions come out wrong-scale.
- **Compile trade-off:** `model.config.compile.enabled=True` pays a one-time (~minute) compile on the first
  `/predict` then runs fast — leave it on for long sweeps, turn it off for one-shot debugging.
