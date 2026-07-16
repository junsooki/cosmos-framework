#!/usr/bin/env python
"""Whole-episode open-loop eval of a G1 action policy vs ground truth.

Walks an episode in strided steps; at each step it POSTs the *real* observation
frame to ``/predict`` (open-loop: observations come from the dataset, not the
model's own rollout), then tiles the predicted rollout segments into one
continuous video. It reports aggregate **video** and **action** error against the
ground truth and writes a side-by-side ``predicted | ground-truth`` mp4 spanning
the whole episode, plus a per-step per-modality L1 plot.

Each ``/predict`` returns ``action_chunk_size`` predicted actions and
``action_chunk_size + 1`` video frames (frame 0 is the conditioning frame, frames
``1..H`` are the predicted future). To tile without overlap or gaps, take the
``stride`` predicted future frames ``[1 : 1+stride]`` from each request and advance
by ``stride``. ``--stride`` therefore defaults to ``action_chunk_size`` (use the
full 32-frame rollout per request) and is clamped to ``<= action_chunk_size``.

The g1 server (``action_policy_server_psix``) returns RAW (denormalized) actions
when launched with ``--stats-path``, and content-cropped video frames.

The embodiment is selected by ``--domain-name`` (see ``_EMBODIMENTS`` below), which
sets the action layout, state key, fps, and per-modality split:
  * g1_simple         — flat 36-D action, states(32), 50fps, 8 whole-body modalities

    python examples/eval_g1_openloop.py \
        --root /path/to/data/g1_simple/g1_v30 --episode 0 \
        --server http://localhost:8000 --image-size 256 --stride 32 --domain-name g1_simple

``--mock`` runs the whole pipeline without a server (GT as prediction, MSE ~ 0).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

_EGO_KEY = "observation.images.egocentric"


@dataclass(frozen=True)
class EmbodimentSpec:
    """Per-embodiment layout the eval works in.

    ``action_dim``   client working action width (== the dataset's raw action width).
    ``effective_dim``the model-active dims; trailing ``[effective_dim:action_dim]`` are
                     zero-padded on BOTH prediction and GT (matches mixed-training padding),
                     so inactive dims contribute no error and don't blow up normalization.
    ``state_key``    dataset column for the use_state conditioning row.
    ``action_cols``  dataset columns concatenated to form the raw action, or ``None`` for a
                     single flat ``action`` column.
    ``fps``          dataset/video fps (drives the output mp4 + prompt duration on the server).
    ``modalities``   ``(name, start, end)`` slices used for the per-modality L1 report + plot.
    """

    action_dim: int
    effective_dim: int
    state_key: str
    action_cols: tuple[str, ...] | None
    fps: int
    modalities: tuple[tuple[str, int, int], ...]


# G1 "simple" flat 36-D whole-body action.
_SIMPLE_MODALITIES = (
    ("hand_joints", 0, 14), ("arm_joints", 14, 28), ("torso_rpy", 28, 31),
    ("base_height", 31, 32), ("base_vx", 32, 33), ("base_vy", 33, 34),
    ("base_vyaw", 34, 35), ("target_yaw", 35, 36),
)

_EMBODIMENTS: dict[str, EmbodimentSpec] = {
    "g1_simple": EmbodimentSpec(
        action_dim=36, effective_dim=36, state_key="states",
        action_cols=None, fps=50, modalities=_SIMPLE_MODALITIES),
}


def get_spec(domain_name: str) -> EmbodimentSpec:
    """EmbodimentSpec for a domain; unknown domains fall back to g1_simple (prior default)."""
    return _EMBODIMENTS.get(domain_name, _EMBODIMENTS["g1_simple"])


def split_by_modality(err_per_dim: np.ndarray, spec: EmbodimentSpec) -> dict[str, float]:
    """Per-modality mean of a per-dim error vector, keyed by modality name."""
    return {name: float(err_per_dim[a:b].mean()) for name, a, b in spec.modalities}


def load_action_minmax(path: Path, spec: EmbodimentSpec) -> tuple[np.ndarray, np.ndarray]:
    """Load the action min/max from a stats JSON, supporting two layouts:

    * per-column dataset stats (``<root>/meta/stats.json``): the embodiment's ``action_cols``
      (the per-modality action columns), concatenated.
    * flat stats (``cosmos3_stats_flat.json``, what the SERVER normalizes with): an ``"action"``
      block with ``"min"`` / ``"max"`` of width ``spec.action_dim``.

    Prefer the flat/global stats so the minmax metric is meaningful (per-episode dataset stats
    can have constant, zero-range dims that blow the normalized error up via the 1e-6 clip).
    """
    stats = json.loads(Path(path).read_text())
    cols = spec.action_cols
    if cols and all(c in stats for c in cols):  # per-column dataset layout
        mn = np.concatenate([np.asarray(stats[c]["min"], np.float32) for c in cols])
        mx = np.concatenate([np.asarray(stats[c]["max"], np.float32) for c in cols])
    elif isinstance(stats.get("action"), dict) and "min" in stats["action"]:  # flat layout
        mn = np.asarray(stats["action"]["min"], np.float32)
        mx = np.asarray(stats["action"]["max"], np.float32)
    else:
        need = f"{cols} columns" if cols else "a flat 'action' block with 'min'/'max'"
        raise ValueError(f"{path}: unrecognized stats layout (need {need})")
    if mn.shape[0] != spec.action_dim or mx.shape[0] != spec.action_dim:
        raise ValueError(f"{path}: expected {spec.action_dim}-D action min/max, got {mn.shape[0]}")
    return mn, mx


def minmax_normalize(a: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    """Minmax-normalize to [-1, 1], mirroring training's ``normalize_action`` (minmax):
    ``2*(a-min)/(max-min) - 1``, with **constant (zero-range) dims mapped to 0** rather than
    amplified by the epsilon clip.

    This matters for the minmax[-1,1] metric to be comparable to the in-training-loop
    validation L1: training sends a zero-range channel (e.g. a constant/inactive dim in the
    per-episode val stats) to 0 (no signal, no error), whereas dividing by a 1e-6 clip
    would amplify any prediction there into a huge spurious error. See
    ``cosmos_framework/data/generator/action/action_normalization.py:normalize_action``.
    """
    rng = mx - mn
    out = 2.0 * (a - mn) / np.clip(rng, 1e-8, None) - 1.0
    return np.where(rng > 1e-8, out, 0.0)


def frame_to_png_b64(img_chw: torch.Tensor) -> str:
    """[C,H,W] float in [0,1] -> base64 PNG (RGB uint8)."""
    hwc = (img_chw.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(hwc).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_png_to_np(s: str) -> np.ndarray:
    """base64 PNG -> [H,W,3] uint8."""
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB"))


def gt_frame_to_np(img_chw: torch.Tensor, size_hw: tuple[int, int]) -> np.ndarray:
    """Dataset ego frame [C,H,W] float [0,1] -> [H,W,3] uint8 resized to size_hw=(H,W)."""
    hwc = (img_chw.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    h, w = size_hw
    return np.asarray(Image.fromarray(hwc).resize((w, h), Image.Resampling.BILINEAR))


def save_mp4(frames_thwc_uint8: np.ndarray, path: Path, fps: int) -> None:
    """Write a [T,H,W,3] uint8 array to an mp4."""
    import torchvision  # local import to avoid a hard top-level dep

    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.io.write_video(str(path), torch.from_numpy(frames_thwc_uint8), fps=float(fps))
    print(f"wrote video -> {path}")


def save_l1_plot(
    step_x: list[int],
    step_l1: dict[str, list[float]],
    means: dict[str, float],
    path: Path,
    episode: int,
    modality_names: list[str],
) -> None:
    """Plot per-step raw L1 for each modality, with a dashed horizontal line at each mean.

    ``step_l1`` / ``means`` are keyed by modality name; ``means`` are the whole-episode L1
    means (matching the printed report), drawn as the horizontal lines. Colors are assigned
    by modality index, so this works for any embodiment's modality set.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(modality_names):
        color = cmap(i % 10)
        m = means[name]
        ax.plot(step_x, step_l1[name], marker="o", ms=3, lw=1.3, color=color,
                label=f"{name} L1 (mean={m:.4f})")
        ax.axhline(m, color=color, ls="--", lw=1.0, alpha=0.7)  # mean line
    ax.set_xlabel("observation frame t (segment start)")
    ax.set_ylabel("raw L1  (mean |pred - gt|)")
    ax.set_title(f"Per-step per-modality action L1 — episode {episode}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote L1 plot -> {path}")


def query_server(
    server: str, image_b64: str, prompt: str, image_size: int, state: list[float] | None = None,
    domain_name: str = "g1_simple",
) -> dict:
    """POST /predict and return the raw response dict ({'action': [...], 'video': [...]}).

    ``state`` is the raw proprioceptive state for the observation frame (embodiment-specific
    width). The g1 policies are trained with ``use_state``, so it MUST be sent — the server
    treats it as the prepended conditioning row 0. Omitting it silently makes the server run
    use_state=False, a train/inference mismatch.
    """
    payload: dict = {
        "image": image_b64,
        "prompt": prompt,
        "domain_name": domain_name,
        "image_size": image_size,
        "view_point": "ego_view",
    }
    if state is not None:
        payload["state"] = state
    resp = requests.post(server.rstrip("/") + "/predict", json=payload, timeout=1200)
    resp.raise_for_status()
    return resp.json()


def query_info(server: str) -> dict:
    """GET /info -> the server's model metadata (run_name, checkpoint, sampling params)."""
    r = requests.get(server.rstrip("/") + "/info", timeout=30)
    r.raise_for_status()
    return r.json()


def _embodiment_tag(domain_name: str) -> str:
    """g1_simple -> simple (strip the g1_ prefix)."""
    return re.sub(r"^g1_", "", domain_name) or domain_name


def _dataset_tag(root: Path) -> str:
    """Dataset name for the output filename: the dir above the version/split subdir when the
    root ends in one (…/<dataset>/g1_v30 or …/<dataset>/val -> <dataset>), else the root name."""
    if root.name in ("val", "train", "test") or re.match(r"^(g1_)?v\d", root.name):
        return root.parent.name
    return root.name


def _checkpoint_tags(checkpoint: str) -> tuple[str, str]:
    """Parse a checkpoint path into (run_timestamp, ckpt_tag), e.g.
    …/action_policy_g1_simple_2607040407/checkpoints/iter_000040000/model
    -> ("2607040407", "ckpt40000"). Missing pieces come back as "".
    """
    parts = Path(checkpoint).parts
    ts = ckpt = ""
    for p in parts:
        m = re.fullmatch(r"iter_0*(\d+)", p)
        if m:
            ckpt = f"ckpt{int(m.group(1))}"
    if "checkpoints" in parts:
        run_dir = parts[parts.index("checkpoints") - 1]
        m = re.search(r"_(\d{6,})$", run_dir)  # trailing %y%m%d%H%M-style run timestamp
        if m:
            ts = m.group(1)
    return ts, ckpt


def auto_output_path(server: str, domain_name: str, root: Path, episode: int, mock: bool) -> Path:
    """Build ``outputs_<embodiment>_<run_ts>_ckpt<iter>/<dataset>_ep<episode>.mp4`` from the
    server's /info metadata. Run-ts/ckpt tags are dropped if /info is unavailable (e.g. --mock)."""
    ts = ckpt = ""
    if not mock:
        try:
            ts, ckpt = _checkpoint_tags(query_info(server).get("checkpoint", ""))
        except Exception as e:  # server down / no /info — still produce a usable folder
            print(f"WARNING: could not GET {server.rstrip('/')}/info ({e}); "
                  "output folder will omit the run/ckpt tags")
    folder = "_".join(p for p in ("outputs", _embodiment_tag(domain_name), ts, ckpt) if p)
    return Path(folder) / f"{_dataset_tag(root)}_ep{episode}.mp4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LeRobot v3.0 dataset root (…/g1_v30 or …/val)")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="first observation frame index")
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--action-chunk-size", type=int, default=32, help="must match training chunk_length")
    ap.add_argument("--image-size", type=int, default=256, help="server resize target; 256 for g1 (matches training)")
    ap.add_argument("--domain-name", default="g1_simple", choices=sorted(_EMBODIMENTS),
                    help="embodiment tag sent to the server (selects the DomainAwareLinear action "
                         "head) AND the eval's action/state/fps/modality layout.")
    ap.add_argument(
        "--stride",
        type=int,
        default=0,
        help="predicted future frames used per request (0 => action_chunk_size); clamped to <= chunk. "
        "Segments tile with no overlap/gap.",
    )
    ap.add_argument("--out", default=None,
                    help="side-by-side (pred | gt) video path. If omitted, auto-derived from the "
                         "server /info + dataset as outputs_<embodiment>_<run_ts>_ckpt<iter>/"
                         "<dataset>_ep<episode>.mp4 (the folder is created automatically). The L1 "
                         "plot is written next to it as <same-stem>_l1.png.")
    ap.add_argument("--stats-path", default=None,
                    help="stats JSON for minmax-normalizing the action-error metric. Pass the GLOBAL "
                         "training stats the server normalizes with (cosmos3_stats_flat.json) so the "
                         "minmax metric is meaningful and comparable to the server's 'Val action MSE'. "
                         "Default: <root>/meta/stats.json, whose per-episode ranges can be constant "
                         "(zero) on some dims and blow the normalized error up.")
    ap.add_argument("--mock", action="store_true", help="no server; use GT as prediction (MSE ~ 0)")
    ap.add_argument("--no-state", action="store_true",
                    help="do NOT send the state (only for models trained WITHOUT use_state)")
    args = ap.parse_args()

    spec = get_spec(args.domain_name)
    modality_names = [name for name, _, _ in spec.modalities]

    root = Path(args.root)
    stats_path = Path(args.stats_path) if args.stats_path else root / "meta" / "stats.json"
    mn, mx = load_action_minmax(stats_path, spec)
    _zero_rng = np.where((mx - mn) <= 1e-6)[0]
    if len(_zero_rng):
        print(f"NOTE: {len(_zero_rng)} action dim(s) have zero min-max range in {stats_path} "
              f"(dims {_zero_rng.tolist()}); these constant channels are normalized to 0 (no signal), "
              f"matching training's normalize_action, so they contribute 0 to the minmax[-1,1] metric.")
    H = args.action_chunk_size
    stride = args.stride if args.stride > 0 else H
    stride = min(stride, H)  # can't consume more future frames than the chunk provides
    A = spec.action_dim

    ds = LeRobotDataset(root.name, root=root, episodes=[args.episode])
    n = ds.num_frames
    prompt = ds[0]["task"]
    _pad_note = "" if spec.effective_dim >= A else (
        f" | effective_dim={spec.effective_dim} (dims {spec.effective_dim}..{A} padded to zero)")
    print(f"episode {args.episode}: {n} frames | domain={args.domain_name} action_dim={A} fps={spec.fps} "
          f"| prompt={prompt!r} | chunk={H} stride={stride} image_size={args.image_size}{_pad_note}")

    # Resolve the output path up front (creates the folder at save time). If --out is omitted,
    # auto-derive outputs_<embodiment>_<run_ts>_ckpt<iter>/<dataset>_ep<episode>.mp4 from /info.
    out_path = Path(args.out) if args.out else auto_output_path(
        args.server, args.domain_name, root, args.episode, args.mock)
    if out_path.suffix.lower() != ".mp4":
        out_path = out_path.with_suffix(".mp4")
    print(f"output -> {out_path}")

    def raw_action(i: int) -> np.ndarray:
        """Raw [action_dim] GT action for frame i, per the embodiment's action columns."""
        s = ds[i]
        if spec.action_cols is None:
            return s["action"].numpy().astype(np.float32)
        return np.concatenate([s[c].numpy() for c in spec.action_cols]).astype(np.float32)

    def predict(t: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (pred_action [Ta,action_dim] raw, pred_video [Tv,H,W,3] uint8) for frame t."""
        if args.mock:
            n_v = min(H + 1, n - t)
            vid = np.stack(
                [(ds[t + k][_EGO_KEY].clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy() for k in range(n_v)]
            )
            act = np.stack([raw_action(t + k) for k in range(min(H, n - t))])
            return act, vid
        state = None if args.no_state else ds[t][spec.state_key].numpy().astype(np.float32).tolist()
        resp = query_server(args.server, frame_to_png_b64(ds[t][_EGO_KEY]), prompt, args.image_size, state=state,
                            domain_name=args.domain_name)
        act = np.asarray(resp["action"], dtype=np.float32)  # [Ta, D_raw]
        # A reduced-embodiment server (raw_action_dim < action_dim) returns fewer than
        # action_dim dims; right-pad the missing trailing dims with zeros so the rest
        # of the pipeline (GT concat, stats, per-modality split) lines up.
        if act.shape[-1] < A:
            act = np.pad(act, ((0, 0), (0, A - act.shape[-1])))
        return act, np.stack([b64_png_to_np(s) for s in resp["video"]])

    # Accumulators.
    pred_frames: list[np.ndarray] = []
    gt_frames: list[np.ndarray] = []
    raw_sse = np.zeros(A, np.float32)    # sum of squared action error (raw), per dim
    norm_sse = np.zeros(A, np.float32)   # ... minmax-normalized
    raw_sae = np.zeros(A, np.float32)    # sum of absolute action error (raw), per dim
    norm_sae = np.zeros(A, np.float32)   # ... minmax-normalized
    n_actions = 0
    hp = wp = None

    # Per-step raw L1 per modality (mean |pred - gt| over the step's frames + that modality's
    # dims), for the per-step trace + plot below.
    step_x: list[int] = []                                   # segment-start frame t
    step_l1: dict[str, list[float]] = {name: [] for name in modality_names}

    t = args.start
    first = True
    while t < n - 1:
        pred_action, pred_video = predict(t)
        if hp is None:
            hp, wp = int(pred_video.shape[1]), int(pred_video.shape[2])
        # frames [1:1+k] are the predicted future; k limited by chunk, video length, and episode tail.
        k = min(stride, pred_video.shape[0] - 1, len(pred_action), n - 1 - t)
        if k <= 0:
            break

        if first:  # seed the tracks with the observation frame at `start`
            pred_frames.append(pred_video[0:1])
            gt_frames.append(gt_frame_to_np(ds[t][_EGO_KEY], (hp, wp))[None])
            first = False

        pred_frames.append(pred_video[1 : 1 + k])
        gt_frames.append(np.stack([gt_frame_to_np(ds[t + 1 + j][_EGO_KEY], (hp, wp)) for j in range(k)]))

        # actions a_t .. a_{t+k-1} drive s_t -> s_{t+k}.
        pa = pred_action[:k].copy()
        ga = np.stack([raw_action(t + j) for j in range(k)])
        if spec.effective_dim < A:
            # Pad inactive trailing dims (dims [effective_dim:action_dim]) to zero on BOTH pred
            # and GT, matching mixed-training padding, so they contribute no error.
            pa[:, spec.effective_dim:] = 0.0
            ga[:, spec.effective_dim:] = 0.0
        pan, gan = minmax_normalize(pa, mn, mx), minmax_normalize(ga, mn, mx)
        raw_sse += ((pa - ga) ** 2).sum(axis=0)
        norm_sse += ((pan - gan) ** 2).sum(axis=0)
        raw_sae += np.abs(pa - ga).sum(axis=0)
        norm_sae += np.abs(pan - gan).sum(axis=0)
        n_actions += k

        # Per-step raw L1 per modality: mean |pred - gt| over the step's k frames, per dim.
        step_parts = split_by_modality(np.abs(pa - ga).mean(axis=0), spec)  # [action_dim] -> per modality
        step_x.append(t)
        for name in modality_names:
            step_l1[name].append(step_parts[name])

        _pm = " ".join(f"{name}={step_parts[name]:.4f}" for name in modality_names)
        print(f"  t={t:>4}..{t + k:<4} ({k} frames)  L1 {_pm}")
        t += k

    # ---- assemble videos + MSE --------------------------------------------------
    pred_video_full = np.concatenate(pred_frames, axis=0)  # [T,hp,wp,3] uint8
    gt_video_full = np.concatenate(gt_frames, axis=0)
    T = min(len(pred_video_full), len(gt_video_full))
    pv = pred_video_full[:T].astype(np.float32) / 255.0 * 2.0 - 1.0  # -> [-1,1]
    gv = gt_video_full[:T].astype(np.float32) / 255.0 * 2.0 - 1.0
    video_mse = float(((pv - gv) ** 2).mean())
    video_l1 = float(np.abs(pv - gv).mean())

    na = max(n_actions, 1)
    raw_mse, norm_mse = raw_sse / na, norm_sse / na
    raw_l1, norm_l1 = raw_sae / na, norm_sae / na

    side_by_side = np.concatenate([pred_video_full[:T], gt_video_full[:T]], axis=2)  # [T,hp,2wp,3]
    save_mp4(side_by_side, out_path, fps=spec.fps)  # out_path resolved above; save_mp4 mkdirs the folder

    # Per-step per-modality L1 plot (horizontal line = whole-episode mean, matching report).
    means = split_by_modality(raw_l1, spec)
    plot_path = out_path.with_name(out_path.stem + "_l1").with_suffix(".png")
    save_l1_plot(step_x, step_l1, means, plot_path, args.episode, modality_names)

    # ---- report -----------------------------------------------------------------
    def _fmt(err: np.ndarray) -> str:
        parts = split_by_modality(err, spec)
        return f"total={err.mean():.5f}  " + "  ".join(f"{k}={v:.5f}" for k, v in parts.items())

    print(f"\ncovered {n_actions} actions / {T} video frames over episode {args.episode}")
    print(f"=== action error (open-loop, all {A} dims) ===")
    # print(f"  raw          MSE : {_fmt(raw_mse)}")
    print(f"  raw          L1  : {_fmt(raw_l1)}")
    # print(f"  minmax[-1,1] MSE : {_fmt(norm_mse)}   (compare to server 'Val action MSE')")
    print(f"  minmax[-1,1] L1  : {_fmt(norm_l1)}   (normalized with {stats_path})")
    print("=== video error (concatenated rollout vs GT) ===")
    print(f"  MSE : [-1,1]={video_mse:.5f}  [0,1]={video_mse / 4.0:.5f}   (compare to server 'Val video MSE')")
    print(f"  L1  : [-1,1]={video_l1:.5f}  [0,1]={video_l1 / 2.0:.5f}")


if __name__ == "__main__":
    main()
