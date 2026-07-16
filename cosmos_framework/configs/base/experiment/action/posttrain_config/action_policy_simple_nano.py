# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_simple_nano`` — Cosmos3-Nano G1 "simple" action policy SFT recipe.

Mirrors the DROID/g1_sonic_neck action recipes, but feeds the G1 "simple" whole-body
locomotion+pick dataset (flat 36-D ``action`` + 32-D ``states`` prepend, single ego
camera, fps 50) through ``ActionTransformPipeline``, and trains the generation + action
heads from the public ``nvidia/Cosmos3-Nano`` base. ``max_action_dim`` is set to 36 to
match the action width; the ``g1_simple`` embodiment tag gives it its own action head.
Train/val roots point at the train/val split produced by split_dataset.py.

Usage (1 node, 8 GPU)::

    PSI_HOME=/path/to/data BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir> WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_simple.toml
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_simple_sft_dataset

cs = ConfigStore.instance()


action_policy_simple_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            # FusedAdam with fp32 master_weights + eps 1e-8 (bf16 params + eps 1e-6
            # diverged on the action loss).
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},  # linear LR decay
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
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
            name="action_policy_simple_nano",
            wandb_mode="disabled",
        ),
        model=dict(
            config=copy.deepcopy(NANO_MODEL_CONFIG),  # action_gen=True, max_action_dim=64
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,  # popped by build_optimizer for FusedAdam (fused by construction)
            # Train the generation + action heads.
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=2.0e-04,  # for the 8192 global batch
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],  # smoke: 100 iters (real run sets via TOML)
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
            max_iter=100,  # smoke
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
            # Skip net_ema. (EMA warm-starts from net, see dcp.py) and the action
            # heads, so they init fresh from the base (the base has no DROID-trained
            # action heads).
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Nano DCP dir; supply via TOML/env
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            keep_only_model_for_older_checkpoints=True,
            strict_resume=False,  # base init: tolerate key set differences
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
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_simple",
            max_samples_per_batch=1,  # per rank; lower it on small topologies to fit memory
            max_sequence_length=None,  # None disables token packing (TOML can't express null)
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=4,
                in_order=False,
                num_workers=4,  # train (iterable_shuffle): world_size(8) x num_workers(4) = 32 shards <= 94 train episodes OK. Val is map-style (5 eps), no shard hang.
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                datasets=dict(
                    simple=dict(
                        ratio=1,
                        dataset=L(get_action_simple_sft_dataset)(
                            root="${oc.env:PSI_HOME}/data/simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0_v30_split/train",
                            fps=50.0,
                            chunk_length=32,
                            mode="policy",
                            use_state=True,
                            iterable_shuffle=True,
                            episode_shuffle_seed=42,
                            action_normalization="minmax",
                            viewpoint="ego_view",
                            resolution="256",
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                            domain_name="g1_simple"
                        ),
                    )
                ),
            ),
        ),
        # Validation reuses the training dataset (no held-out split for this small
        # recipe). Enables the server's --run-validation sanity check; eval-appropriate
        # tweaks vs. train: deterministic order (iterable_shuffle=False) and no CFG
        # dropout (cfg_dropout_rate=0.0).
        dataloader_val=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_simple",
            # Track the train loader so a TOML override of [dataloader_train] applies here too
            # (val has no TOML section; single source of truth for the per-rank batch cap).
            max_samples_per_batch="${dataloader_train.max_samples_per_batch}",
            max_sequence_length=None,
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=4,
                in_order=True,
                num_workers=4,  # train (iterable_shuffle): world_size(8) x num_workers(4) = 32 shards <= 94 train episodes OK. Val is map-style (5 eps), no shard hang.
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                datasets=dict(
                    simple=dict(
                        ratio=1,
                        dataset=L(get_action_simple_sft_dataset)(
                            root="${oc.env:PSI_HOME}/data/simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0_v30_split/val",
                            fps=50.0,
                            chunk_length=32,
                            mode="policy",
                            use_state=True,
                            iterable_shuffle=False,  # deterministic for validation
                            episode_shuffle_seed=42,
                            action_normalization="minmax",
                            viewpoint="ego_view",
                            resolution="256",
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.0,  # no CFG dropout at eval
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                            domain_name="g1_simple"
                        ),
                    ),
                ),
            ),
        ),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


# The "simple" embodiment has a flat 36-D action; set max_action_dim=36 (below the NANO
# default 64) so the action head is exactly this width. Action heads init fresh (they are in
# keys_to_skip_loading), so resizing them is safe. The dataset reads this via
# ``max_action_dim="${model.config.max_action_dim}"``.
action_policy_simple_nano["model"]["config"]["max_action_dim"] = 36


# chunk_length=32 -> 33 observation frames; pin the VAE encode duration to match.
# Set post-construction so it lands on the deep-copied NANO_MODEL_CONFIG.tokenizer.
action_policy_simple_nano["model"]["config"]["tokenizer"]["encode_exact_durations"] = [33]


# Uncap the packed-sequence length. The NANO default (45056) caps the packed sequence;
# -1 (uncapped) processes the full vision sequence per step. Does not change the
# per-token loss; widens the effective vision context per step.
action_policy_simple_nano["model"]["config"]["max_num_tokens_after_packing"] = -1


# Weight the vision flow-matching loss 10x in the total loss (the NANO default is 1.0).
# loss_scale multiplies only the vision term, balancing it against the action loss
# (action_loss_weight=10) so both heads train at comparable gradient magnitude.
action_policy_simple_nano["model"]["config"]["rectified_flow_training_config"]["loss_scale"] = 10.0


for _item in [action_policy_simple_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
