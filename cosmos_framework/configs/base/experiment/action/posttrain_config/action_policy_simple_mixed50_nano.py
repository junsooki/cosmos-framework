# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_simple_mixed50_nano`` — Cosmos3-Nano G1 "simple" MULTITASK (14-task) SFT, 50 train eps/task.

Same 14-task equal-weight blend as ``action_policy_simple_mixed_nano``, but each stream is fed the
**50-train / 2-val** split (``_v30_50tr``) instead of the 20-ep (18/2) split. The intent is to ramp up
to 50 training episodes per task while keeping the val the same (50*N total for training). val is the
SAME 2 held-out episodes as the 20-ep run, so open-loop val L1 is directly comparable 20-ep vs 50-ep
(controlled data-efficiency comparison; only the training-set size changes).

Why this shape (not a physical merge, not the stock mixed template):
- The stock ``RankPartitionedDataLoader`` with N datasets partitions ranks
  (ranks_per_dataset = world_size / N) and hard-asserts world_size >= N. 14 tasks on
  8 GPUs => 8/14 < 1 => dies. So instead we wrap 14 *single-dataset* RankPartitioned
  loaders (each shards over all 8 ranks, satisfies world_size >= 1) inside an
  ``IterativeJointDataLoader`` that picks ONE task per step at equal ratio.
- Equal ratio (1 each) == size-proportional here anyway, since every task is
  subsampled to 50 train eps. No task dominates.
- Each task self-normalizes (minmax on its own 50-ep stats) and is served with its own
  stats at eval, so no fragile cross-task combined-stats file is needed.

num_workers=1 per stream: 14 streams x 8 ranks x 1 worker = 112 loader procs (kept low
on purpose); and per-stream 8 ranks x 1 worker = 8 shards <= 50 train eps (no shard hang).

Usage (1 node, 8 GPU): select experiment=action_policy_simple_mixed50_nano; TOML sets max_iter.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.joint_dataloader import (
    IterativeJointDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_simple_sft_dataset

cs = ConfigStore.instance()

# 14 tasks that have BOTH simple/ (train) and simple-eval/ (closed-loop) data.
SIMPLE_MIXED_TASKS = [
    "G1WholebodyBendHandoverTeleop-v0",
    "G1WholebodyBendPickMP-v0",
    "G1WholebodyBendPickTeleop-v0",
    "G1WholebodyCloseDoorTeleop-v0",
    "G1WholebodyHandoverTeleop-v0",
    "G1WholebodyLocomotionPickBetweenTablesTeleop-v0",
    "G1WholebodyOpenFaucetTeleop-v0",
    "G1WholebodyOpenOvenTeleop-v0",
    "G1WholebodyOpenTrashCanTeleop-v0",
    "G1WholebodyPickAndPlaceAndHugContainerTeleop-v0",
    "G1WholebodyPushOfficeChairTeleop-v0",
    "G1WholebodyTabletopGraspMP-v0",
    "G1WholebodyXMoveBendPickTeleop-v0",
    "G1WholebodyXMovePickTeleop-v0",
]


def _stream(task: str, split: str, shuffle: bool, cfg_drop: float, in_order: bool):
    """One equal-ratio stream = a single-dataset RankPartitionedDataLoader (all 8 ranks)."""
    return dict(
        ratio=1,
        dataloader=L(RankPartitionedDataLoader)(
            batch_size=4,
            in_order=in_order,
            num_workers=1,  # 8 ranks x 1 = 8 shards <= 50 train eps; 14 streams => keep procs low
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            sampler=None,
            datasets={
                task: dict(
                    ratio=1,
                    dataset=L(get_action_simple_sft_dataset)(
                        root=f"${{oc.env:PSI_HOME}}/data/simple/{task}_v30_50tr/{split}",
                        fps=50.0,
                        chunk_length=32,
                        mode="policy",
                        use_state=True,
                        iterable_shuffle=shuffle,
                        episode_shuffle_seed=42,
                        action_normalization="minmax",
                        viewpoint="ego_view",
                        resolution="256",
                        max_action_dim="${model.config.max_action_dim}",
                        cfg_dropout_rate=cfg_drop,
                        tokenizer_config="${model.config.vlm_config.tokenizer}",
                        domain_name="g1_simple",
                    ),
                )
            },
        ),
    )


_TRAIN_STREAMS = {t: _stream(t, "train", True, 0.1, False) for t in SIMPLE_MIXED_TASKS}
_VAL_STREAMS = {t: _stream(t, "val", False, 0.0, True) for t in SIMPLE_MIXED_TASKS}


action_policy_simple_mixed50_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /checkpoint": "s3"},
            {"override /callbacks": ["basic", "optimization", "job_monitor"]},
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="action_sft",
            name="action_policy_simple_mixed50_nano",
            wandb_mode="disabled",
        ),
        model=dict(config=copy.deepcopy(NANO_MODEL_CONFIG)),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=2.0e-04,
            lr_multipliers={"action2llm": 5.0, "llm2action": 5.0, "action_modality_embed": 5.0},
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],  # smoke default; real run sets via TOML
            f_max=[0.4],
            f_min=[0.0],
            f_start=[0.0],
            verbosity_interval=0,
            warm_up_steps=[0],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=100,  # smoke default; TOML overrides
            max_val_iter=10,
            run_validation=True,
            run_validation_on_start=True,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=500,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                norm_monitor=dict(every_n=1000),
                param_count=dict(save_s3=False),
                sigma_loss_analysis=dict(every_n=1000, every_n_viz=1000),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            keep_only_model_for_older_checkpoints=True,
            strict_resume=False,
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        # MULTITASK: IterativeJointDataLoader picks one of 14 task-streams per step (equal ratio).
        # It does its own packing (replaces PackingDataLoader). Each stream is a single-dataset
        # RankPartitionedDataLoader that shards its 50 train eps over all 8 ranks.
        dataloader_train=L(IterativeJointDataLoader)(
            audio_sample_rate=48000,
            sound_latent_fps=0,
            patch_spatial=2,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            max_samples_per_batch=1,
            max_sequence_length=None,
            dataloaders=_TRAIN_STREAMS,
        ),
        dataloader_val=L(IterativeJointDataLoader)(
            audio_sample_rate=48000,
            sound_latent_fps=0,
            patch_spatial=2,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            max_samples_per_batch="${dataloader_train.max_samples_per_batch}",
            max_sequence_length=None,
            dataloaders=_VAL_STREAMS,
        ),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


# Flat 36-D action head (below NANO default 64). Heads init fresh (keys_to_skip_loading).
action_policy_simple_mixed50_nano["model"]["config"]["max_action_dim"] = 36
action_policy_simple_mixed50_nano["model"]["config"]["tokenizer"]["encode_exact_durations"] = [33]
action_policy_simple_mixed50_nano["model"]["config"]["max_num_tokens_after_packing"] = -1
action_policy_simple_mixed50_nano["model"]["config"]["rectified_flow_training_config"]["loss_scale"] = 10.0


for _item in [action_policy_simple_mixed50_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
