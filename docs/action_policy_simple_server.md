# G1 `simple` Whole-Body Action Policy Server

The G1 **`g1_simple`** whole-body action policy (Cosmos3-Nano) is served by an HTTP **policy server**
that streams predicted action chunks to a client. The server (`action_policy_server_simple.py`) is the
single-ego-camera, `use_state` variant of the action server: it takes one observation image plus a
proprioceptive state vector and returns a chunk of raw (denormalized) actions and a short rollout video.

This guide covers serving a trained checkpoint, the `/predict` HTTP protocol, and closed-loop
evaluation in the SIMPLE simulator. The policy is a flat **36-D** action + **32-D** proprioceptive
`use_state`, `chunk_length=32`, `fps=50`, `minmax` action normalization, `domain_name="g1_simple"`.

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Policy Server](#policy-server)
- [`/predict` Protocol](#predict-protocol)
- [Closed-Loop Evaluation (SIMPLE Simulator)](#closed-loop-evaluation-simple-simulator)
- [Notes](#notes)

______________________________________________________________________

<!--TOC-->

## Policy Server

The server runs **natively** in the cosmos virtual environment (see [`setup.md`](setup.md)); call
`.venv/bin/python` directly.

```bash
export CKPT=/path/to/output_root/psi/cosmos3_action_sft/<run_name>/checkpoints/iter_000010000

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -m cosmos_framework.scripts.action_policy_server_simple \
  --checkpoint-path $CKPT --config-file cosmos_framework/configs/base/config.py \
  --experiment action_policy_simple_nano \
  --experiment-overrides model.config.tokenizer.vae_path=$WAN_VAE_PATH model.config.compile.enabled=False \
  --action-chunk-size 32 --no-guardrails --fps 50 \
  --stats-path $SIMPLE_ROOT/meta/stats.json --port 22085
```

| Flag | Meaning |
| --- | --- |
| `--experiment` | `action_policy_simple_nano` (the 36-D `g1_simple` architecture) |
| `--stats-path` | the dataset's `meta/stats.json` — the server reads its `action` + `states` min/max to return **raw** (denormalized) actions and normalize the state row; optional (omit → normalized actions returned) |
| `--action-chunk-size` / `--fps` | `32` / `50` — match training |
| `model.config.compile.enabled` | default `True` → first `/predict` compiles (~min) then runs fast; set `False` for one-shot/debug |
| `--no-guardrails` | skip the gated video guardrail |

Because the checkpoint was trained with `use_state=True`, requests **must** include `state`; omitting
it silently runs `use_state=False`, a train/inference mismatch.

> **Stats source.** The server reads action + state min/max from the dataset's standard
> `meta/stats.json` (the `action` and `states` feature keys — the same source training normalizes
> with), so point `--stats-path` at `$SIMPLE_ROOT/meta/stats.json`. No separate merged stats file or
> conversion tooling is needed. (A merged file with top-level `action`/`state` keys is also accepted.)

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

## Closed-Loop Evaluation (SIMPLE Simulator)

Closed-loop evaluation is driven by the **[SIMPLE](https://github.com/songlin/SIMPLE) simulator**, which
runs in Docker and talks to this native cosmos server over `127.0.0.1:<port>` (the container uses host
networking). Start the server as above, then run the SIMPLE eval client from the SIMPLE repo pointing
`--host 127.0.0.1 --port 22085` at it; the SIMPLE-side agents wrap the `/predict` protocol. See the
SIMPLE repo for its eval commands. (**Never** cap `--max-episode-steps` — the task metadata step budget
is the source of truth; capping it makes episodes fail on the clock.)

## Notes

- **Native venv:** the server runs in the cosmos virtual environment; call `.venv/bin/python` directly and do
  **not** use a bare `uv run` (it re-syncs the environment and can break CUDA).
- **Compile trade-off:** `model.config.compile.enabled=True` pays a one-time (~minute) compile on the first
  `/predict` then runs fast — leave it on for long sweeps, turn it off for one-shot debugging.
