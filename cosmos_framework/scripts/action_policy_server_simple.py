# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""HTTP inference server for the PsiX / G1 action policy (OmniMoTModel).

A g1-tailored fork of ``action_policy_server_libero`` that fixes train/inference
discrepancies for ``use_state`` / single-ego-camera action policies:
  * adds the required ``action_processing_record`` so generated actions can be
    externalized,
  * builds the prompt to match training exactly — viewpoint sentence (from the
    request's ``view_point``, default ``ego_view``), the *padded* resolution, and
    integer-truncated duration,
  * exposes ``--no-guardrails`` to skip the gated video guardrail,
  * writes the rollout **inference video** (rollout.mp4) under ``--dump-dir``.

The server exposes two endpoints:

- POST /predict: run policy inference.
  - Input: {"image": "<base64_png>", "prompt": "<task>", "domain_name": "<name>",
            "image_size": <int>, "view_point": "<ego_view|...>"}
  - Output: {"action": [[a0, a1, ...], ...], "video": ["<base64_png>", ...]}
- GET /info: model / runtime info (run_name, checkpoint, sampling params, ...).

Direct DCP loading works when given the matching training config:

  PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_psix \
    --checkpoint-path /path/to/job/checkpoints/iter_000010000 \
    --config-file /path/to/train-output/config.yaml \
    --action-chunk-size 32 --no-guardrails --fps 30 \
    --stats-path /path/to/g1_v30/meta/cosmos3_stats_flat.json \
    --dump-dir /path/to/rollouts \
    --port 8000
"""

from cosmos_framework.inference.common.init import (  # noqa: F401  (is_rank0 may be useful for logging)
    init_script,
    is_rank0,
)

init_script()

import base64
import binascii
import datetime
import io
import json
import threading
import time
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic
import torch
import tyro
from omegaconf import DictConfig
from PIL import Image

# Action-specific helpers live in the in-tree data/utils generator packages.
from cosmos_framework.data.generator.action.action_processing import (
    ActionProcessingRecord,
    make_batched_action_processing_fields,
)
from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.viewpoint_utils import DEFAULT_VIEWPOINT_TEMPLATES
from cosmos_framework.data.generator.action.transforms import (
    build_sequence_plan_from_mode,
    find_closest_target_size,
    reflection_pad_to_target,
    remove_reflection_padding,
)
from cosmos_framework.inference.args import OmniSetupArgs, OmniSetupOverrides
from cosmos_framework.inference.common.args import CheckpointOverrides, ConfigFileType, tyro_cli
from cosmos_framework.inference.common.config import deserialize_config_dict
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.action_policy_server_utils import (
    DEFAULT_FALLBACK_OUTPUT_DIR,
    disable_runtime_ema_for_frozen_config,
    get_local_ip,
    maybe_init_distributed,
)
from cosmos_framework.utils import log
from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution

_DEFAULT_ACTION_CHUNK_SIZE = 16
ActionNormalization = Literal["auto", "meanstd", "minmax", "quantile", "quantile_rot"]
ResolvedActionNormalization = Literal["meanstd", "minmax", "quantile", "quantile_rot"]

_DURATION_FPS_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."


# ---------------------------------------------------------------------------
# Pre/post processing helpers (copied verbatim from the previous server, with
# minor adjustments so config-introspection helpers also accept plain dicts).
# ---------------------------------------------------------------------------


def _augment_prompt_with_metadata(
    prompt: str,
    *,
    t_frames: int,
    fps: int,
    height: int,
    width: int,
    view_point: str | None = None,
    append_viewpoint_info: bool = True,
    append_duration_fps: bool = True,
    append_resolution_info: bool = True,
) -> str:
    """Append viewpoint, duration/FPS, and resolution metadata to match training.

    Mirrors ``ViewpointTextInfo``, ``DurationFPSTextTimeStamps`` and
    ``ResolutionTextInfo`` augmentors (in that order) from the Action training
    transform pipeline. Each piece is appended only when its flag is ``True``.
    ``height``/``width`` must be the *padded* canvas size (training appends the
    padded resolution), and ``duration`` is ``int(t_frames / fps)`` to match
    ``DurationFPSTextTimeStamps`` (integer truncation), not float seconds.
    """
    if append_viewpoint_info and view_point:
        template = DEFAULT_VIEWPOINT_TEMPLATES.get(view_point)
        if template:
            sep = " " if prompt.rstrip().endswith(".") else ". "
            prompt = prompt + sep + template
    if append_duration_fps:
        duration = int(t_frames / fps)
        sep = " " if prompt.rstrip().endswith(".") else ". "
        prompt = prompt + sep + _DURATION_FPS_TEMPLATE.format(duration=duration, fps=fps)
    if append_resolution_info:
        sep = " " if prompt.rstrip().endswith(".") else ". "
        prompt = prompt + sep + _RESOLUTION_TEMPLATE.format(height=height, width=width)
    return prompt


def _extract_bool_from_config(config: Any, key: str, default: bool) -> bool:
    """Recursively search dataloader_train config for a boolean flag."""

    def _search(obj: Any) -> bool | None:
        if isinstance(obj, (DictConfig, dict)):
            if key in obj:
                val = obj[key]
                if isinstance(val, bool):
                    return val
            iterable = obj.values()
            for v in iterable:
                result = _search(v)
                if result is not None:
                    return result
        return None

    try:
        if isinstance(config, dict):
            dl_train = config.get("dataloader_train")
        else:
            dl_train = getattr(config, "dataloader_train", None)
        if dl_train is not None:
            result = _search(dl_train)
            if result is not None:
                return result
    except Exception:
        pass
    return default


def _extract_str_from_config(config: Any, key: str) -> str | None:
    """Recursively search dataloader_train config for a string field."""

    def _search(obj: Any) -> str | None:
        if isinstance(obj, (DictConfig, dict)):
            if key in obj:
                val = obj[key]
                if isinstance(val, str):
                    return val
            iterable = obj.values()
            for v in iterable:
                result = _search(v)
                if result is not None:
                    return result
        return None

    try:
        if isinstance(config, dict):
            dl_train = config.get("dataloader_train")
        else:
            dl_train = getattr(config, "dataloader_train", None)
        if dl_train is not None:
            return _search(dl_train)
    except Exception:
        pass
    return None


def _extract_chunk_length_from_config(config: Any) -> int | None:
    """Try to extract chunk_length from the experiment's dataloader config.

    Recursively searches ``config.dataloader_train`` for ``chunk_length`` or
    ``num_action_per_chunk`` to determine the action chunk size the model was
    trained with.  Returns ``None`` when neither key is found.
    """

    def _search(obj: Any, keys: tuple[str, ...] = ("chunk_length", "num_action_per_chunk")) -> int | None:
        if isinstance(obj, (DictConfig, dict)):
            for key in keys:
                if key in obj:
                    val = obj[key]
                    if isinstance(val, int):
                        return val
            for v in obj.values():
                result = _search(v, keys)
                if result is not None:
                    return result
        return None

    try:
        if isinstance(config, dict):
            dl_train = config.get("dataloader_train")
        else:
            dl_train = getattr(config, "dataloader_train", None)
        if dl_train is not None:
            return _search(dl_train)
    except Exception:
        pass
    return None


def _strip_data_url_prefix(b64: str) -> str:
    # Accept "data:image/png;base64,...." as well as raw base64.
    if "," in b64 and b64[:64].lower().startswith("data:"):
        return b64.split(",", 1)[1].strip()
    return b64.strip()


def _b64decode_loose(b64: str) -> bytes:
    """
    Decode base64 permissively.

    The simulator/client may include whitespace/newlines, omit padding, or use urlsafe base64.
    """
    s = _strip_data_url_prefix(b64)
    s = "".join(s.split())  # remove whitespace/newlines
    pad = (-len(s)) % 4
    if pad:
        s = s + ("=" * pad)
    try:
        return base64.b64decode(s, validate=False)
    except binascii.Error:
        return base64.urlsafe_b64decode(s)


def _decode_base64_png_to_rgb_uint8(image_b64: str) -> torch.Tensor:
    """
    Returns a tensor with shape (3, H, W), dtype uint8, RGB.
    """
    try:
        raw = _b64decode_loose(image_b64)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Invalid base64 image: {e}") from e

    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")
        # Pillow can expose a read-only view; make it writable to avoid PyTorch warnings/UB.
        arr = np.asarray(img, dtype=np.uint8).copy()
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape {arr.shape}")
    chw = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,H,W]
    return chw


def _video_tensor_to_pil_images(video_c_t_h_w: torch.Tensor) -> list[Image.Image]:
    """
    Convert (C, T, H, W) float tensor in [-1,1] or [0,1] to a list of PIL RGB frames.
    """
    if video_c_t_h_w.dim() != 4:
        raise ValueError(f"Expected (C,T,H,W), got {tuple(video_c_t_h_w.shape)}")
    if int(video_c_t_h_w.shape[0]) != 3:
        raise ValueError(f"Expected C=3 RGB, got C={int(video_c_t_h_w.shape[0])}")

    images: list[Image.Image] = []
    t = int(video_c_t_h_w.shape[1])
    for ti in range(t):
        frame = video_c_t_h_w[:, ti].detach().cpu().float()
        if frame.min().item() < 0.0:
            frame = (frame + 1.0) / 2.0
        frame = frame.clamp(0.0, 1.0)
        frame_uint8 = (frame * 255.0).round().to(torch.uint8)  # [3,H,W]
        hwc = frame_uint8.permute(1, 2, 0).numpy()  # [H,W,3]
        images.append(Image.fromarray(hwc))
    return images


def _save_gif(frames: list[Image.Image], path: Path, fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / float(fps))))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

    print(f"Saved gif to {path}")


def _save_mp4(frames: list[Image.Image], path: Path, fps: int) -> None:
    """Write the rollout frames out as an mp4 (the inference video)."""
    if not frames:
        return
    import torchvision  # local import: avoid a hard top-level dep for the server

    path.parent.mkdir(parents=True, exist_ok=True)
    thwc = torch.from_numpy(np.stack([np.asarray(f.convert("RGB")) for f in frames]))  # [T,H,W,3] uint8
    torchvision.io.write_video(str(path), thwc, fps=float(fps))
    print(f"Saved mp4 to {path}")


def _save_policy_request_dump(
    *,
    dump_root: Path,
    request_id: int,
    request_json: dict[str, Any],
    obs_chw_uint8: torch.Tensor,
    pred_action: list[list[float]],
    pred_video_c_t_h_w: torch.Tensor | None,
    fps: int,
) -> None:
    """
    Dump input observation, predicted actions, and rollout video for offline debugging.
    Creates:
      - request.json
      - observation.png
      - action_output.json
      - rollout.mp4 (the inference video, if pred_video provided)
      - rollout.gif (if pred_video provided)
      - rollout_frames/frame_XXX.png (if pred_video provided)
    """
    ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    out_dir = dump_root / f"{ts}_req{request_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=False)

    # Save request JSON (without the base64 image to save space, image is saved separately)
    request_json_copy = {k: v for k, v in request_json.items() if k != "image"}
    request_json_copy["image"] = "<saved as observation.png>"
    (out_dir / "request.json").write_text(json.dumps(request_json_copy, indent=2), encoding="utf-8")

    # Save observation image
    obs_hwc = obs_chw_uint8.permute(1, 2, 0).cpu().numpy()  # [H,W,3]
    obs_img = Image.fromarray(obs_hwc)
    obs_img.save(out_dir / "observation.png")

    # Save predicted actions
    action_output = {"action": pred_action}
    (out_dir / "action_output.json").write_text(json.dumps(action_output, indent=2), encoding="utf-8")

    if pred_video_c_t_h_w is None:
        return

    # Save rollout video (mp4 = the inference video, plus a gif for quick preview)
    frames = _video_tensor_to_pil_images(pred_video_c_t_h_w)
    _save_mp4(frames, out_dir / "rollout.mp4", fps=fps)
    _save_gif(frames, out_dir / "rollout.gif", fps=fps)

    # Save individual frames
    frames_dir = out_dir / "rollout_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(frames_dir / f"frame_{i:03d}.png")


def _save_failed_request_dump(
    *,
    dump_root: Path,
    request_id: int,
    request_json: dict[str, Any],
    error: str,
) -> None:
    """
    Dump request + error even if inference fails.
    Best-effort: will try to decode and save observation image if present.
    """
    ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    out_dir = dump_root / f"{ts}_req{request_id:06d}_ERROR"
    out_dir.mkdir(parents=True, exist_ok=False)

    # Save request JSON (without base64 image)
    request_json_copy = {k: v for k, v in request_json.items() if k != "image"}
    if "image" in request_json:
        request_json_copy["image"] = "<attempted to save as observation.png>"
    (out_dir / "request.json").write_text(json.dumps(request_json_copy, indent=2), encoding="utf-8")
    (out_dir / "error.txt").write_text(error, encoding="utf-8")

    try:
        image_b64 = request_json.get("image")
        if isinstance(image_b64, str):
            img_chw_uint8 = _decode_base64_png_to_rgb_uint8(image_b64)
            obs_hwc = img_chw_uint8.permute(1, 2, 0).cpu().numpy()  # [H,W,3]
            Image.fromarray(obs_hwc).save(out_dir / "observation.png")
    except Exception:
        # Ignore any dump failures.
        return


def _ts() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _load_introspection_config_dict(setup_args: OmniSetupArgs) -> dict:
    """Load the full experiment config as a plain dict for prompt-augmentation
    introspection.

    For ``MODULE`` (``.py``) configs, ``OmniInference.create`` saves the
    structured config to ``setup_args.output_dir / 'config.yaml'`` via
    ``cosmos_framework.inference.common.config.save_config`` immediately after model load. For
    ``YAML`` / ``JSON`` configs we just deserialize the source file directly.
    """
    if setup_args.config_file_type == ConfigFileType.MODULE:
        saved = Path(setup_args.output_dir) / "config.yaml"
        if saved.exists():
            return deserialize_config_dict(saved)
        # Fallback: re-parse the .py module without instantiating anything.
        import importlib

        from cosmos_framework.inference.common.config import unstructure_config
        from cosmos_framework.utils import config_helper

        config_module = importlib.import_module(config_helper.get_config_module(setup_args.config_file))
        config = config_module.make_config()
        config = config_helper.override(
            config, ["--", f"experiment={setup_args.experiment}", *setup_args.experiment_overrides]
        )
        return unstructure_config(config, invalid="ignore")

    return deserialize_config_dict(Path(setup_args.config_file))


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


class ActionServerArgs(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    # We deliberately do NOT expose the full ``OmniSetupOverrides`` (i.e. no
    # ``setup: SetupOverrides`` field). That would surface every
    # ``setup.sample_overrides.*`` knob (``--guidance``, ``--seed``,
    # ``--num-steps``, ...) which collides with our own server-level defaults.
    # Instead we reuse the shared checkpoint/config args and build
    # ``OmniSetupOverrides`` programmatically in ``build_setup_overrides``.

    checkpoint: tyro.conf.OmitArgPrefixes[CheckpointOverrides] = CheckpointOverrides.model_construct()
    """Checkpoint and config loading configuration."""

    output_dir: Path | None = None
    """Output directory for ``OmniInference`` (saved config.yaml, benchmarks).
    Defaults to ``--dump-dir`` if set, else ``/tmp/cosmos3_action_server``."""

    # ----- single-rank parallelism / sampler ----------------------------------
    sampler: Literal["unipc", "edm"] = "unipc"
    """Diffusion sampler used by ``OmniInference``."""

    # ----- sampling defaults (per-request, used when client doesn't override) --
    seed: int = 0
    """Random seed for ``model.generate_samples_from_batch``."""
    guidance: float = 1.0
    """Guidance scale for denoising."""
    num_steps: int = 4
    """Number of denoising steps."""
    fps: int = 30
    """Frames per second used for both prompt augmentation and rollout encoding."""

    # ----- action policy parameters -------------------------------------------
    action_chunk_size: int | None = None
    """Number of action steps to predict. Defaults to ``chunk_length`` /
    ``num_action_per_chunk`` from the experiment config (or 16)."""
    max_action_dim: int | None = None
    """Maximum action dimension. Defaults to ``model.config.max_action_dim``
    from the experiment config (or 64)."""
    raw_action_dim: int | None = None
    """Unpadded action dimension used for action-channel masking. Inferred
    from action stats when omitted."""

    # ----- action/state normalization -----------------------------------------
    stats_path: Path | None = None
    """Path to the merged stats JSON (``cosmos3_stats_flat.json``): ``"action"`` stats
    for denormalizing predicted actions and ``"state"`` stats for normalizing the
    prepended use_state conditioning row."""
    action_normalization: ActionNormalization = "auto"
    """Action normalization to invert. ``auto`` reads ``action_normalization``
    from the experiment config (default ``minmax`` if unspecified)."""

    # ----- guardrails ---------------------------------------------------------
    guardrails: bool = True
    """Run the video/text content-safety guardrail (requires the gated
    ``nvidia/Cosmos-Guardrail1`` model). Pass ``--no-guardrails`` to skip it —
    irrelevant for robot action-policy serving and avoids the gated download."""

    # ----- debug dumps --------------------------------------------------------
    dump_dir: Path | None = None
    """If set, dump observations, predicted actions, and rollout videos under
    this directory for offline debugging."""
    dump_every: int = 1
    """Dump every N-th request (only used when ``--dump-dir`` is set)."""

    # ----- HTTP server --------------------------------------------------------
    host: str = "0.0.0.0"
    """HTTP host to bind."""
    port: int = 8000
    """HTTP port to bind."""
    http_400_on_error: bool = False
    """If set, return HTTP 400 on inference errors. Default is HTTP 200 with an
    empty action list, matching the legacy simulator client expectations."""

    # ----- developer utilities ------------------------------------------------
    run_validation: bool = False
    """If set, run a one-shot validation/training batch through the model on
    startup (developer debugging only)."""
    attach_vscode_debugger: bool = False
    """Debug mode: start a debugpy server on port 3002 and block until VS Code
    attaches (rank 0 only) before loading the model. Attach with the
    ".vscode/launch.json" "Attach to Cosmos (debugpy :3002)" configuration."""

    def build_setup_overrides(self) -> OmniSetupOverrides:
        """Build an ``OmniSetupOverrides`` from checkpoint and server fields.

        Required fields (``checkpoint_path``) must be present; optional fields
        keep their ``OmniSetupOverrides`` defaults when not specified by the
        user.
        """
        if not getattr(self.checkpoint, "checkpoint_path", ""):
            raise ValueError("--checkpoint-path is required")

        output_dir = self.output_dir or self.dump_dir or DEFAULT_FALLBACK_OUTPUT_DIR

        base = OmniSetupOverrides.model_validate(self.checkpoint.model_dump())
        base.output_dir = output_dir
        base.sampler = self.sampler
        base.guardrails = self.guardrails
        return base


# ---------------------------------------------------------------------------
# Service implementation (predict path is a verbatim port from the previous
# evaluation/action/http_inference_server.py).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionServerConfig:
    """Internal snapshot of the resolved CLI args, threaded through the service.

    Kept as a plain dataclass (vs. carrying ``ActionServerArgs`` directly) so
    request-time code can ``replace(...)`` individual fields after model load.
    """

    seed: int
    guidance: float
    num_steps: int
    fps: int
    action_chunk_size: int
    max_action_dim: int
    raw_action_dim: int | None
    dump_dir: Path | None
    dump_every: int
    http_400_on_error: bool
    stats_path: Path | None
    action_normalization: ActionNormalization
    experiment_name: str
    checkpoint_dir: str


class ActionModelService:
    def __init__(self, args: ActionServerArgs) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for OmniMoTModel inference in this repo.")

        # OmniInference internally calls into FSDP / DTensor parallelize utilities
        # that expect a process group; create a single-rank PG when not under
        # torchrun (init_script() only inits the PG when WORLD_SIZE>1).
        maybe_init_distributed()

        setup_overrides = args.build_setup_overrides()
        setup_args = setup_overrides.build_setup()
        init_output_dir(setup_args.output_dir)
        setup_args = disable_runtime_ema_for_frozen_config(setup_args)

        # Surface the resolved max_action_dim into the experiment config when
        # the user explicitly overrode it on the CLI; matches the previous
        # ``experiment_opts=[f"model.config.max_action_dim={...}"]`` plumbing.
        if args.max_action_dim is not None:
            setup_args.experiment_overrides = [
                *setup_args.experiment_overrides,
                f"model.config.max_action_dim={int(args.max_action_dim)}",
            ]

        log.info(
            f"[action-server] loading model: config_file='{setup_args.config_file}' "
            f"({setup_args.config_file_type}) experiment='{setup_args.experiment}' "
            f"checkpoint_path='{setup_args.checkpoint_path}'"
        )

        # OmniInference dispatches between MODULE (.py) and YAML/JSON loaders.
        pipe = OmniInference.create(setup_args)
        self.pipe: OmniInference = pipe
        self.model = pipe.model
        self.model.eval()
        # OmniInference always uses OmniSetupArgs at runtime, but the base class
        # types the attribute as the more general SetupArgs.
        assert isinstance(pipe.setup_args, OmniSetupArgs)
        self.setup_args: OmniSetupArgs = pipe.setup_args
        self.experiment_config: dict = _load_introspection_config_dict(self.setup_args)

        # Resolve action_chunk_size: CLI arg > experiment config > default.
        if args.action_chunk_size is not None:
            resolved_chunk_size = int(args.action_chunk_size)
        else:
            config_chunk = _extract_chunk_length_from_config(self.experiment_config)
            if config_chunk is not None:
                resolved_chunk_size = config_chunk
                log.info(
                    f"[action-server] --action-chunk-size not specified, "
                    f"using chunk_length={resolved_chunk_size} from experiment config"
                )
            else:
                resolved_chunk_size = _DEFAULT_ACTION_CHUNK_SIZE
                log.info(
                    f"[action-server] --action-chunk-size not specified and not found in experiment config, "
                    f"using default={resolved_chunk_size}"
                )

        # Resolve max_action_dim: CLI > model config > default (64).
        if args.max_action_dim is not None:
            resolved_max_action_dim = int(args.max_action_dim)
        else:
            model_max_action_dim = getattr(self.model.config, "max_action_dim", None)
            if isinstance(model_max_action_dim, int):
                resolved_max_action_dim = model_max_action_dim
                log.info(
                    f"[action-server] --max-action-dim not specified, "
                    f"using max_action_dim={resolved_max_action_dim} from model config"
                )
            else:
                resolved_max_action_dim = 64
                log.info(
                    f"[action-server] --max-action-dim not specified and not found in model config, "
                    f"using default={resolved_max_action_dim}"
                )

        self.cfg = ActionServerConfig(
            seed=int(args.seed),
            guidance=float(args.guidance),
            num_steps=int(args.num_steps),
            fps=int(args.fps),
            action_chunk_size=resolved_chunk_size,
            max_action_dim=resolved_max_action_dim,
            raw_action_dim=int(args.raw_action_dim) if args.raw_action_dim is not None else None,
            dump_dir=args.dump_dir,
            dump_every=int(args.dump_every),
            http_400_on_error=bool(args.http_400_on_error),
            stats_path=args.stats_path,
            action_normalization=args.action_normalization,
            experiment_name=setup_args.experiment or "",
            checkpoint_dir=setup_args.checkpoint_path,
        )

        self._lock = threading.Lock()
        self._req_id_lock = threading.Lock()
        self._req_id = 0

        self.append_duration_fps = _extract_bool_from_config(
            self.experiment_config, "append_duration_fps", default=True
        )
        self.append_resolution_info = _extract_bool_from_config(
            self.experiment_config, "append_resolution_info", default=True
        )
        log.info(
            f"[action-server] prompt augmentation: "
            f"append_duration_fps={self.append_duration_fps}, append_resolution_info={self.append_resolution_info}"
        )

        # Whether the model was trained with use_state (state prepended as the action
        # conditioning row). Used to reject requests that omit 'state', which would
        # silently run use_state=False — a train/inference mismatch.
        self.expects_state = _extract_bool_from_config(self.experiment_config, "use_state", default=False)
        log.info(f"[action-server] use_state={self.expects_state} (from training config)")

        # Action denormalization stats.
        self.action_min: torch.Tensor | None = None
        self.action_range: torch.Tensor | None = None
        self.action_mean: torch.Tensor | None = None
        self.action_std: torch.Tensor | None = None
        # State normalization stats (for the prepended use_state conditioning row;
        # state is a different modality from the action, so it gets its own stats).
        self.state_min: torch.Tensor | None = None
        self.state_range: torch.Tensor | None = None
        self.state_mean: torch.Tensor | None = None
        self.state_std: torch.Tensor | None = None
        self.state_const: torch.Tensor | None = None  # constant (min==max) state channels -> normalize to 0
        self.action_normalization: ResolvedActionNormalization = "minmax"
        self.raw_action_dim: int | None = self.cfg.raw_action_dim
        self._load_action_normalization_stats()
        if self.raw_action_dim is None:
            self.raw_action_dim = 7

        if args.run_validation:
            self._run_developer_validation()

    # ------------------------------------------------------------------
    # Action denormalization
    # ------------------------------------------------------------------

    def _load_action_normalization_stats(self) -> None:
        """Load action denormalization tensors from ``cfg.stats_path``.

        Populates ``self.action_normalization``, the relevant tensor attributes
        (``action_mean`` / ``action_std`` for meanstd; ``action_min`` /
        ``action_range`` for minmax / quantile / quantile_rot), and infers
        ``self.raw_action_dim`` when not explicitly set.
        """
        if self.cfg.stats_path is None:
            return

        self.action_normalization = self._resolve_action_normalization(self.cfg.action_normalization)
        stats_path = Path(self.cfg.stats_path)
        if not stats_path.is_absolute():
            stats_path = Path.cwd() / stats_path
        with open(stats_path) as f:
            raw_stats = json.load(f)
        if not isinstance(raw_stats, dict):
            raise ValueError(f"Action stats file must contain a dict: {stats_path}")
        # cosmos3_stats_flat.json nests action under "action" (symmetric with "state");
        # fall back to the legacy "global"/"global_raw" namespace or a flat top-level dict.
        stats_key = "global_raw" if self.action_normalization == "quantile_rot" else "global"
        stats = raw_stats.get("action", raw_stats.get(stats_key, raw_stats))
        if not isinstance(stats, dict):
            raise ValueError(f"Action stats file must contain a dict or {stats_key} stats dict: {stats_path}")
        if self.action_normalization == "meanstd":
            if "mean" not in stats or "std" not in stats:
                raise ValueError(f"Mean/std action normalization requires 'mean' and 'std' in {stats_path}")
            self.action_mean = torch.tensor(stats["mean"], dtype=torch.float32)  # [D]
            action_std = torch.tensor(stats["std"], dtype=torch.float32)  # [D]
            self.action_std = torch.clamp(action_std, min=1e-8)  # [D]
            stats_dim = int(self.action_mean.shape[0])
            stats_summary = f"mean={self.action_mean.tolist()}, std={self.action_std.tolist()}"
        elif self.action_normalization in ("quantile", "quantile_rot"):
            if "q01" not in stats or "q99" not in stats:
                raise ValueError(f"Quantile action normalization requires 'q01' and 'q99' in {stats_path}")
            self.action_min = torch.tensor(stats["q01"], dtype=torch.float32)  # [D]
            action_max = torch.tensor(stats["q99"], dtype=torch.float32)  # [D]
            action_range = action_max - self.action_min  # [D]
            self.action_range = torch.clamp(action_range, min=1e-6)  # [D]
            stats_dim = int(self.action_min.shape[0])
            stats_summary = f"q01={self.action_min.tolist()}, q99={action_max.tolist()}"
        else:
            if "min" not in stats or "max" not in stats:
                raise ValueError(f"Min/max action normalization requires 'min' and 'max' in {stats_path}")
            self.action_min = torch.tensor(stats["min"], dtype=torch.float32)  # [D]
            action_max = torch.tensor(stats["max"], dtype=torch.float32)  # [D]
            action_range = action_max - self.action_min  # [D]
            self.action_range = torch.clamp(action_range, min=1e-6)  # [D]
            stats_dim = int(self.action_min.shape[0])
            stats_summary = f"min={self.action_min.tolist()}, max={action_max.tolist()}"
        if self.raw_action_dim is None:
            self.raw_action_dim = stats_dim
        if stats_dim != self.raw_action_dim:
            raise ValueError(f"Action stats dimension {stats_dim} does not match raw_action_dim={self.raw_action_dim}")
        log.info(
            f"[action-server] Loaded action stats for denormalization from {stats_path}: "
            f"normalization={self.action_normalization}, {stats_summary}"
        )

        # State stats for the prepended use_state conditioning row (merged into the
        # same file under "state"). Normalized with the same method as the action, but
        # the state's OWN stats — state and action are different modalities.
        state_stats = raw_stats.get("state")
        if isinstance(state_stats, dict):
            if self.action_normalization == "meanstd":
                self.state_mean = torch.tensor(state_stats["mean"], dtype=torch.float32)
                self.state_std = torch.clamp(torch.tensor(state_stats["std"], dtype=torch.float32), min=1e-8)
                state_dim = int(self.state_mean.shape[0])
            else:
                lo_key = "q01" if self.action_normalization in ("quantile", "quantile_rot") else "min"
                hi_key = "q99" if self.action_normalization in ("quantile", "quantile_rot") else "max"
                self.state_min = torch.tensor(state_stats[lo_key], dtype=torch.float32)
                state_hi = torch.tensor(state_stats[hi_key], dtype=torch.float32)
                self.state_range = torch.clamp(state_hi - self.state_min, min=1e-6)
                # Constant channels (min==max, e.g. zero-padded dummy neck) normalize to 0,
                # matching normalize_action; otherwise the clamp would map them to -1.
                self.state_const = (state_hi - self.state_min).abs() < 1e-6
                state_dim = int(self.state_min.shape[0])
            log.info(f"[action-server] Loaded {state_dim}-D state normalization stats from {stats_path}")
        else:
            log.warning(
                f"[action-server] No 'state' stats in {stats_path}; the use_state conditioning row "
                "will fall back to RAW (un-normalized) — re-convert with convert_dataset_psix_to_v30.py "
                "to embed merged state stats."
            )

    def _resolve_action_normalization(
        self, requested_normalization: ActionNormalization
    ) -> ResolvedActionNormalization:
        """Resolve auto action normalization from the loaded experiment config."""
        if requested_normalization != "auto":
            return requested_normalization

        configured_normalization = _extract_str_from_config(self.experiment_config, "action_normalization")
        if configured_normalization is None:
            return "minmax"
        if configured_normalization in ("meanstd", "minmax", "quantile", "quantile_rot"):
            return configured_normalization  # type: ignore[return-value]
        raise ValueError(
            "action_policy_server_libero.py can denormalize action_normalization='minmax', 'meanstd', "
            "'quantile', or 'quantile_rot'; "
            f"loaded experiment config requested {configured_normalization!r}. "
            "Pass --action-normalization explicitly if this checkpoint should use a supported method."
        )

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Invert the configured action normalization."""
        if self.action_normalization == "meanstd":
            if self.action_mean is None or self.action_std is None:
                return action
            action_dim = self.action_mean.shape[0]
            normalized = action[..., :action_dim]  # [...,D]
            action_mean = self.action_mean.to(action.device)  # [D]
            action_std = self.action_std.to(action.device)  # [D]
            return normalized * action_std + action_mean  # [...,D]

        if self.action_min is None or self.action_range is None:
            return action
        action_dim = self.action_min.shape[0]
        normalized = action[..., :action_dim]  # [...,D]
        action_min = self.action_min.to(action.device)  # [D]
        action_range = self.action_range.to(action.device)  # [D]
        return (normalized + 1.0) / 2.0 * action_range + action_min  # [...,D]

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _should_dump(self, request_id: int) -> bool:
        """Decide whether to dump this request.

        We always dump the first request (request_id == 1) as a quick sanity
        check that dumping is wired up, then dump every N-th request controlled
        by ``dump_every``.
        """
        if self.cfg.dump_dir is None:
            return False
        n = int(self.cfg.dump_every)
        if n <= 0:
            return False
        return request_id == 1 or (request_id % n == 0)

    def get_info(self) -> dict[str, Any]:
        """Return model / server info for the /info endpoint.

        Includes all runtime-relevant config so clients can record reproducible
        params.json without needing to know CLI flags.
        """
        return {
            "run_name": self.cfg.experiment_name,
            "checkpoint": self.cfg.checkpoint_dir,
            "config_file": str(self.setup_args.config_file),
            "config_file_type": str(self.setup_args.config_file_type),
            "guidance": self.cfg.guidance,
            "num_steps": self.cfg.num_steps,
            "fps": self.cfg.fps,
            "seed": self.cfg.seed,
            "action_chunk_size": self.cfg.action_chunk_size,
            "max_action_dim": self.cfg.max_action_dim,
            "raw_action_dim": self.cfg.raw_action_dim,
            "stats_path": str(self.cfg.stats_path) if self.cfg.stats_path else None,
        }

    def _normalized_state_row(self, state: Any) -> torch.Tensor:
        """Build the normalized ``[max_action_dim]`` conditioning row from a raw state.

        The state occupies the leading dims (zero-padded to ``max_action_dim``) and is
        normalized with the STATE stats (its own ``min``/``max`` or ``q01``/``q99``),
        matching training — state and action are different modalities, so the row is
        NOT normalized with the action stats. Padding dims stay zero. Without merged
        state stats, the raw (un-normalized) padded state is used.
        """
        s = torch.as_tensor(np.asarray(state, dtype=np.float32)).reshape(-1)  # [S]
        row = torch.zeros(int(self.cfg.max_action_dim), dtype=torch.float32)  # [D_model]
        if self.state_mean is not None and self.state_std is not None:
            n = min(s.shape[0], self.state_mean.shape[0])
            row[:n] = (s[:n] - self.state_mean[:n]) / self.state_std[:n]
        elif self.state_min is not None and self.state_range is not None:
            n = min(s.shape[0], self.state_min.shape[0])
            row[:n] = 2.0 * (s[:n] - self.state_min[:n]) / self.state_range[:n] - 1.0
            if self.state_const is not None:  # constant channels -> 0 (match normalize_action)
                row[:n] = torch.where(self.state_const[:n], torch.zeros_like(row[:n]), row[:n])
        else:
            # Fallback: no state stats -> raw padded state.
            n = min(s.shape[0], row.shape[0])
            row[:n] = s[:n]
        return row  # [D_model]

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_policy(self, req: dict[str, Any]) -> dict[str, Any]:
        """
        Run policy inference: given an observation image and prompt, predict actions.

        Input request format:
        {
            "image": "<base64_encoded_png>",
            "prompt": "<task_description>",
            "domain_name": "<domain_name>",
            "image_size": <int>
        }

        Output format:
        {
            "action": [[a0_0, a0_1, ...], ..., [aN_0, aN_1, ...]],
            "video": ["<base64_png>", ...]  # List of T base64-encoded PNG frames
        }

        All action dimensions are returned. Video is the decoded predicted rollout as base64 PNGs.
        """
        t0 = time.monotonic()

        # Get or assign request ID
        injected_id = req.get("request_id", None)
        if isinstance(injected_id, int) and injected_id > 0:
            request_id = int(injected_id)
        else:
            with self._req_id_lock:
                self._req_id += 1
                request_id = int(self._req_id)

        # Validate request
        image_b64 = req.get("image")
        if not isinstance(image_b64, str):
            raise ValueError("'image' must be a base64 string")

        prompt = req.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("'prompt' must be a string")

        domain_name = req.get("domain_name")
        if not isinstance(domain_name, str):
            raise ValueError("'domain_name' must be a string")

        image_size = req.get("image_size")
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("'image_size' must be a positive integer")

        # Camera viewpoint label for the prompt's viewpoint sentence (matches training).
        # Single-camera action policies are ego_view; clients may override per request.
        view_point = req.get("view_point", "ego_view")

        # Decode image
        t_decode0 = time.monotonic()
        img_chw_uint8 = _decode_base64_png_to_rgb_uint8(image_b64)
        img_h, img_w = img_chw_uint8.shape[-2:]
        t_decode1 = time.monotonic()

        t_frames = self.cfg.action_chunk_size + 1
        video_c_t_h_w_uint8 = img_chw_uint8.unsqueeze(1).repeat(1, t_frames, 1, 1)  # [3,T,H,W] (native)
        resolution = get_vision_data_resolution((image_size, image_size))
        target_w, target_h = find_closest_target_size(img_h, img_w, resolution)
        pad_dict: dict[str, Any] = {"video": video_c_t_h_w_uint8}
        reflection_pad_to_target(pad_dict, ["video"], True, target_w, target_h)
        video_padded = pad_dict["video"]  # (C, T, target_h, target_w)
        padded_image_size = pad_dict["image_size"]  # (4,)

        # Action: noise starting point for policy mode. When the request carries a
        # proprioceptive ``state`` (use_state), prepend it as the *normalized*
        # conditioning row 0 (action_length becomes chunk+1, so the sequence plan
        # marks slot 0 as given) — matching training's ``_build_action`` + quantile
        # normalization. Without ``state`` this is the legacy chunk-length noise.
        state_in = req.get("state")
        use_state = state_in is not None
        if self.expects_state and not use_state:
            raise ValueError(
                "Model was trained with use_state=True but the request has no 'state' field. "
                "Send the raw observation.state (matching training) in the request; omitting it "
                "would run use_state=False and mismatch training."
            )
        if use_state:
            action_t_d = torch.zeros((self.cfg.action_chunk_size + 1, self.cfg.max_action_dim), dtype=torch.float32)
            action_t_d[0] = self._normalized_state_row(state_in)  # [max_action_dim]
        else:
            action_t_d = torch.zeros((self.cfg.action_chunk_size, self.cfg.max_action_dim), dtype=torch.float32)

        input_video_key = getattr(self.model, "input_video_key", None)
        if input_video_key is None:
            input_video_key = getattr(self.model, "config", None).input_video_key  # type: ignore[union-attr]

        sequence_plan = build_sequence_plan_from_mode(
            mode="policy",
            video_length=self.cfg.action_chunk_size + 1,
            action_length=self.cfg.action_chunk_size + (1 if use_state else 0),
            has_text=True,
        )

        augmented_prompt = _augment_prompt_with_metadata(
            prompt,
            t_frames=t_frames,
            fps=self.cfg.fps,
            # padded canvas size (training appends the padded resolution, not the pre-pad resize).
            height=target_h,
            width=target_w,
            view_point=view_point,
            append_duration_fps=self.append_duration_fps,
            append_resolution_info=self.append_resolution_info,
        )

        batch: dict[str, Any] = {
            input_video_key: [[video_padded]],
            # raw_action_dim + action_processing_record (needed by the model to externalize
            # the generated action). action_normalizer=None: the model only un-pads, then
            # this server denormalizes via _denormalize_action (avoids double-denorm).
            **make_batched_action_processing_fields(
                ActionProcessingRecord(raw_action_dim=self.raw_action_dim, action_normalizer=None),
                1,
            ),
            "action": [[action_t_d]],
            "mode": ["policy"],
            "ai_caption": [augmented_prompt],
            "prompt": [augmented_prompt],
            "conditioning_fps": [torch.tensor(self.cfg.fps, dtype=torch.long)],
            "image_size": padded_image_size.unsqueeze(0).to(device="cuda"),
            "domain_id": [torch.tensor(get_domain_id(domain_name), dtype=torch.long)],
            "sequence_plan": [sequence_plan],
        }

        if getattr(self.model, "training", False):
            log.warning(f"[action-server] request_id={request_id} WARNING: model.training=True")

        log.info(
            f"[action-server] request_id={request_id} mode=policy "
            f"prompt={augmented_prompt!r} domain_name={domain_name!r} image_size={image_size} "
            f"receiving img={tuple(img_chw_uint8.shape)} model_res={target_h}x{target_w} "
            f"steps={self.cfg.num_steps} guidance={self.cfg.guidance}"
        )

        # Run inference
        t_inf0 = time.monotonic()
        with self._lock:
            with torch.inference_mode():
                samples = self.model.generate_samples_from_batch(
                    batch,
                    guidance=self.cfg.guidance,
                    seed=[self.cfg.seed],
                    num_steps=self.cfg.num_steps,
                    has_negative_prompt=False,
                )
                pred_action = samples["action"][0]  # [T,D] or [1,T,D]

                # Decode vision for rollout video (samples["vision"] is a list; take first sample)
                pred_video_c_t_h_w = self.model.decode(samples["vision"][0]).squeeze(0)  # [C,T,H,W]

                # Remove reflection padding so the reported video matches the original resolution
                pred_video_c_t_h_w = remove_reflection_padding(pred_video_c_t_h_w, padded_image_size)
        t_inf1 = time.monotonic()

        # Extract actions: return all dimensions — (T, D) or (1, T, D)
        pred_action = pred_action.float().squeeze(0)  # [T,D]
        if use_state:
            pred_action = pred_action[1:]  # drop the conditioning state row, keep the chunk predictions
        pred_action = self._denormalize_action(pred_action)
        pred_action_np = pred_action.detach().cpu().numpy()  # [T,D]
        pred_action_list = pred_action_np.tolist()  # List of [a0, a1, ..., aD]

        # Convert video to base64-encoded PNG frames
        pred_video_frames = _video_tensor_to_pil_images(pred_video_c_t_h_w)
        pred_video_b64: list[str] = []
        for frame in pred_video_frames:
            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            pred_video_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))

        # Optional offline debug dump
        if self._should_dump(request_id):
            dump_dir = self.cfg.dump_dir
            assert dump_dir is not None
            dump_root = Path(dump_dir)
            dump_root.mkdir(parents=True, exist_ok=True)
            try:
                log.info(f"[action-server] request_id={request_id} dumping to {str(dump_root)}")
                _save_policy_request_dump(
                    dump_root=dump_root,
                    request_id=request_id,
                    request_json=req,
                    obs_chw_uint8=img_chw_uint8,
                    pred_action=pred_action_list,
                    pred_video_c_t_h_w=pred_video_c_t_h_w,
                    fps=int(self.cfg.fps),
                )
            except Exception as e:
                # Never fail serving a request due to dump failures
                log.error(f"[action-server] dump failed for request_id={request_id}: {e}")

        dt_total_ms = (time.monotonic() - t0) * 1000.0
        dt_decode_ms = (t_decode1 - t_decode0) * 1000.0
        dt_inf_ms = (t_inf1 - t_inf0) * 1000.0
        log.info(
            f"[action-server] request_id={request_id} done action_steps={len(pred_action_list)} "
            f"video_frames={len(pred_video_b64)} "
            f"ms_total={dt_total_ms:.1f} ms_decode={dt_decode_ms:.1f} ms_infer={dt_inf_ms:.1f}"
        )
        return {"action": pred_action_list, "video": pred_video_b64}

    # ------------------------------------------------------------------
    # Developer validation (optional, --run-validation)
    # ------------------------------------------------------------------

    def _run_developer_validation(self) -> None:
        """Run a single validation/training batch through the model on startup."""
        # Re-instantiate config so we can spin up dataloaders without the
        # inference-time freezes applied to the model config.
        if self.setup_args.config_file_type != ConfigFileType.MODULE:
            log.warning(
                "[action-server] --run-validation requires a .py config-file (got "
                f"{self.setup_args.config_file_type}); skipping."
            )
            return

        try:
            config = self.setup_args.load_config()
        except Exception as e:
            log.warning(f"[action-server] --run-validation could not load config: {e}; skipping.")
            return

        # Some experiments define no val dataloader (e.g. the G1 recipe sets
        # dataloader_val=None); skip cleanly instead of crashing serve startup.
        if config.dataloader_val is None or config.dataloader_train is None:
            log.warning(
                "[action-server] --run-validation requires both dataloader_train and dataloader_val "
                f"(train={config.dataloader_train is not None}, val={config.dataloader_val is not None}); skipping."
            )
            return

        try:
            val_dataset = instantiate(config.dataloader_val)
            train_dataset = instantiate(config.dataloader_train)
            val_batch = next(iter(val_dataset))  # pyrefly: ignore[no-matching-overload]
            train_batch = next(iter(train_dataset))  # pyrefly: ignore[no-matching-overload]
        except Exception as e:
            log.warning(f"[action-server] --run-validation could not build validation batches: {e}; skipping.")
            return

        with torch.inference_mode():
            sample_num = 1
            val_result = self.model.generate_samples_from_batch(
                val_batch,
                guidance=1.0,
                seed=[0 for _ in range(sample_num)],
                num_steps=8,
                n_sample=sample_num,
            )
            video_mse_list = []
            action_mse_list = []

            # Write predicted | ground-truth videos side-by-side (concatenated along width) for
            # visual inspection, under --dump-dir (or the fallback output dir).
            sbs_out_dir = Path(self.cfg.dump_dir) if self.cfg.dump_dir else Path(DEFAULT_FALLBACK_OUTPUT_DIR)

            def _save_pred_vs_gt(pred_bcthw: torch.Tensor, gt_bcthw: torch.Tensor, name: str) -> None:
                # pred/gt: [1, 3, T, H, W] in [-1, 1]; concat along width -> [3, T, H, 2W] (pred | gt).
                sbs = torch.cat(
                    [pred_bcthw[0].detach().cpu().float(), gt_bcthw[0].detach().cpu().float()], dim=-1
                )
                _save_mp4(_video_tensor_to_pil_images(sbs), sbs_out_dir / f"{name}.mp4", fps=int(self.cfg.fps))
                log.info(f"[action-server] wrote {name}.mp4 (pred | gt) to {sbs_out_dir / f'{name}.mp4'}")

            for i in range(sample_num):
                val_video = self.model.decode(val_result["vision"][i]).detach().cpu()
                # The generated vision latent is cropped back to the original (pre-padding)
                # content size (_remove_padding_from_latent), so the decoded video is smaller
                # than the reflection-padded GT still held in val_batch["video"]. Crop the GT
                # to the decoded content size (top-left, matching the latent crop) so the MSE
                # compares like-for-like. Both are already [-1, 1] (video normalized in-place
                # by generate_samples_from_batch).
                gh, gw = val_video.shape[-2:]
                gt_video = val_batch["video"][i].cpu()[..., :gh, :gw].to(val_video.dtype)
                val_video_mse = torch.nn.functional.mse_loss(val_video, gt_video)
                val_action = val_result["action"][i].detach().cpu()
                # Row 0 is the given state conditioning (use_state), not a prediction; score rows 1:.
                val_action_mse = torch.nn.functional.mse_loss(val_action[1:], val_batch["action"][i][0][1:].cpu())
                video_mse_list.append(val_video_mse.item())
                action_mse_list.append(val_action_mse.item())
                _save_pred_vs_gt(val_video, gt_video, "val_pred_vs_gt")
            log.info(f"Val video MSE: {np.mean(video_mse_list)}")
            log.info(f"Val action MSE: {np.mean(action_mse_list)}")

            train_result = self.model.generate_samples_from_batch(
                train_batch,
                guidance=1.0,
                seed=[0],
                num_steps=8,
                n_sample=1,  # only sample [0] is used below; no need to draw 20
            )
            train_video = self.model.decode(train_result["vision"][0])
            # Same content-vs-padded crop as the val branch (see comment above).
            gh, gw = train_video.shape[-2:]
            train_gt_video = train_batch["video"][0][..., :gh, :gw].to(train_video.device, train_video.dtype)
            train_action = train_result["action"][0].detach().cpu()
            # Mirror the val branch: batch["action"][sample][0] is the [T, D] tensor; row 0 is the
            # state conditioning (use_state), so score only the predicted actions (rows 1:).
            train_action_mse = torch.nn.functional.mse_loss(
                train_action[1:], train_batch["action"][0][0][1:].cpu()
            )
            train_video_mse = torch.nn.functional.mse_loss(train_video, train_gt_video)
            log.info(f"Train action MSE: {train_action_mse}; Train video MSE: {train_video_mse}")
            _save_pred_vs_gt(train_video, train_gt_video, "train_pred_vs_gt")


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class _ActionHandler(BaseHTTPRequestHandler):
    """
    ThreadingHTTPServer handler.

    The service instance is injected via ``server.service``.
    """

    server: ThreadingHTTPServer  # type: ignore[assignment]

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        # Avoid caches/proxies returning stale results.
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Client closed the connection (often due to request timeout).
            # Avoid noisy tracebacks; nothing to do server-side.
            return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/info":
            service: ActionModelService = getattr(self.server, "service")  # type: ignore[attr-defined]
            self._send_json(200, service.get_info())
        elif self.path == "/":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/", "/predict"):
            self._send_json(404, {"error": "Not found"})
            return

        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._send_json(415, {"error": "Content-Type must be application/json"})
            return

        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return

        body = self.rfile.read(max(0, length))
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON: {e}"})
            return

        if not isinstance(req, dict):
            self._send_json(400, {"error": "JSON body must be an object"})
            return

        service: ActionModelService = getattr(self.server, "service")  # type: ignore[attr-defined]

        # Generate a per-request id at the HTTP layer to correlate logs and dumps.
        req_id_lock: threading.Lock | None = getattr(self.server, "_req_id_lock", None)  # type: ignore[attr-defined]
        if req_id_lock is None:
            req_id_lock = threading.Lock()
            setattr(self.server, "_req_id_lock", req_id_lock)  # type: ignore[attr-defined]
            setattr(self.server, "_req_id", 0)  # type: ignore[attr-defined]
        with req_id_lock:
            next_id = int(getattr(self.server, "_req_id")) + 1  # type: ignore[attr-defined]
            setattr(self.server, "_req_id", next_id)  # type: ignore[attr-defined]
        req["request_id"] = next_id

        log.info(
            f"[action-server] HTTP request_id={next_id} from={self.client_address[0]}:{self.client_address[1]} "
            f"path={self.path} bytes={length}"
        )

        try:
            out = service.predict_policy(req)
        except Exception as e:
            err = str(e)
            traceback.print_exc()

            payload = {"action": [], "error": err, "request_id": req.get("request_id")}
            log.error(f"[action-server] request_id={req.get('request_id')} ERROR: {err}")

            # Dump failed request for offline debugging if enabled.
            if service.cfg.dump_dir is not None:
                try:
                    dump_root = Path(service.cfg.dump_dir)
                    dump_root.mkdir(parents=True, exist_ok=True)
                    _save_failed_request_dump(
                        dump_root=dump_root,
                        request_id=int(req.get("request_id") or 0),
                        request_json=req,
                        error=err,
                    )
                except Exception:
                    pass

            status = 400 if service.cfg.http_400_on_error else 200
            self._send_json(status, payload)
            return

        self._send_json(200, out)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Silence default request logging (the simulator can be chatty).
        return


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def serve(args: ActionServerArgs) -> None:
    if args.attach_vscode_debugger and is_rank0():
        import debugpy  # noqa: T100

        debugpy.listen(3002)  # noqa: T100
        log.info("[action-server] Waiting for VS Code debugger to attach on port 3002...")
        debugpy.wait_for_client()  # noqa: T100

    if args.dump_dir is not None:
        # Create dump dir up front so it's obvious where outputs will go.
        dump_root = Path(args.dump_dir).resolve()
        dump_root.mkdir(parents=True, exist_ok=True)
        log.info(f"[action-server] dump_root={str(dump_root)} dump_every={args.dump_every}")

    service = ActionModelService(args)

    local_ip = get_local_ip()
    log.info(
        f"[action-server] starting host={args.host} port={int(args.port)} "
        f"experiment_name={service.cfg.experiment_name!r} "
        f"steps={service.cfg.num_steps} guidance={service.cfg.guidance} fps={service.cfg.fps} "
        f"action_chunk_size={service.cfg.action_chunk_size} max_action_dim={service.cfg.max_action_dim} "
        f"raw_action_dim={service.cfg.raw_action_dim} "
        f"dump_dir={service.cfg.dump_dir} dump_every={service.cfg.dump_every} "
        f"http_400_on_error={service.cfg.http_400_on_error}"
    )
    log.info(f"[action-server] Server accessible at: http://{local_ip}:{int(args.port)}/")
    log.info("[action-server] Endpoints:")
    log.info("  - GET  /       : Health check")
    log.info("  - GET  /info   : Model info (run_name, checkpoint, sampling params)")
    log.info("  - POST /predict: Policy inference (image + prompt + domain_name + image_size -> action)")

    httpd: ThreadingHTTPServer = ThreadingHTTPServer((args.host, int(args.port)), _ActionHandler)
    setattr(httpd, "service", service)
    httpd.serve_forever()


def main() -> None:
    args = tyro_cli(
        ActionServerArgs,
        description=__doc__,
        config=(
            tyro.conf.OmitArgPrefixes,
            tyro.conf.CascadeSubcommandArgs,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
    serve(args)


if __name__ == "__main__":
    main()
