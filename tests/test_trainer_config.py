"""
test_trainer_config.py: Tests for Trainer/model/feature config integration in Keisei.
"""

from typing import Any, Dict, cast  # Add Dict, Any

import pytest

from keisei.config_schema import (
    AppConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    ParallelConfig,
    TrainingConfig,
    WandBConfig,
    DisplayConfig,
)
from keisei.training.models.resnet_tower import ActorCriticResTower, ResidualBlock
from keisei.training.trainer import Trainer


class DummyArgs:
    def __init__(self, **kwargs):
        self.run_name = "test_run"  # Default run_name
        self.resume = None  # Add resume attribute with default None
        self.__dict__.update(kwargs)


def make_config_and_args(**overrides):
    # TrainingConfig: Start with all defaults, then apply overrides
    training_data: Dict[str, Any] = {  # Explicitly type training_data
        "total_timesteps": 500_000,
        "steps_per_epoch": 2048,
        "ppo_epochs": 10,
        "minibatch_size": 64,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "clip_epsilon": 0.2,
        "value_loss_coeff": 0.5,
        "entropy_coef": 0.01,
        "render_every_steps": 1,
        "refresh_per_second": 4,
        "enable_spinner": True,
        "input_features": "core46",
        "tower_depth": 9,
        "tower_width": 256,
        "se_ratio": 0.25,
        "model_type": "resnet",
        "mixed_precision": False,
        "ddp": False,
        "gradient_clip_max_norm": 0.5,
        "lambda_gae": 0.95,
        "checkpoint_interval_timesteps": 10000,
        "evaluation_interval_timesteps": 50000,
    }
    training_data.update({k: v for k, v in overrides.items() if k in training_data})
    training = TrainingConfig(
        **training_data,
        normalize_advantages=True,
        lr_schedule_type=None,
        lr_schedule_kwargs=None,
        lr_schedule_step_on="epoch",
    )

    # EnvConfig: Start with all defaults, then apply overrides
    env_data: Dict[str, Any] = {  # Explicitly type env_data
        "device": "cpu",
        "input_channels": 46,
        "num_actions_total": 13527,
        "seed": 42,
    }
    env_data.update({k: v for k, v in overrides.items() if k in env_data})
    env = EnvConfig(**env_data)

    # Other configs: Instantiate with their explicit defaults from schema
    evaluation = EvaluationConfig(
        num_games=20,
        opponent_type="random",
        evaluation_interval_timesteps=50000,
        enable_periodic_evaluation=True,
        max_moves_per_game=500,
        log_file_path_eval="eval_log.txt",
        wandb_log_eval=False,
    )
    logging = LoggingConfig(
        log_file="logs/training_log.txt", model_dir="models/", run_name=None
    )
    wandb_enabled = overrides.get("wandb_enabled", False)  # Default to False for tests
    wandb = WandBConfig(
        enabled=wandb_enabled,
        project="keisei-shogi",
        entity=None,
        run_name_prefix="keisei",
        watch_model=True,
        watch_log_freq=1000,
        watch_log_type="all",
        log_model_artifact=False,
    )
    display = DisplayConfig(display_moves=False, turn_tick=0.5)

    config = AppConfig(
        parallel=ParallelConfig(
            enabled=False,
            num_workers=1,
            batch_size=64,
            sync_interval=100,
            compression_enabled=False,
            timeout_seconds=30,
            max_queue_size=1000,
            worker_seed_offset=1000,
        ),
        training=training,
        env=env,
        evaluation=evaluation,
        logging=logging,
        wandb=wandb,
        display=display,
    )
    args = DummyArgs(**overrides)
    return config, args


def test_trainer_instantiates_resnet_and_features():
    config, args = make_config_and_args(
        input_features="core46+all",
        model_type="resnet",
        tower_depth=3,
        tower_width=32,
        se_ratio=0.5,
    )
    trainer = Trainer(config, args)
    # Check model and feature spec
    assert trainer.model is not None
    assert trainer.feature_spec is not None
    assert trainer.feature_spec.name == "core46+all"
    assert trainer.obs_shape == (51, 9, 9)  # For core46+all (46 + 1 + 1 + 1 + 2 = 51)

    # Check model config with explicit casting for type checker
    model = cast(ActorCriticResTower, trainer.model)
    assert (
        model.res_blocks[0].se is not None
    )  # SE block should be enabled with se_ratio > 0

    first_res_block = cast(ResidualBlock, model.res_blocks[0])
    assert first_res_block.conv1.out_channels == trainer.tower_width
    assert first_res_block.conv1.in_channels == trainer.tower_width


def test_trainer_invalid_feature_raises():
    config, args = make_config_and_args(
        input_features="not_a_feature", model_type="resnet"
    )
    with pytest.raises(KeyError):
        Trainer(config, args)


def test_trainer_invalid_model_raises():
    config, args = make_config_and_args(
        input_features="core46", model_type="not_a_model"
    )
    with pytest.raises(ValueError):
        Trainer(config, args)


def test_cli_overrides_config():
    # CLI args should override config
    config, _ = make_config_and_args(
        input_features="core46", model_type="resnet", tower_depth=3
    )
    args = DummyArgs(
        run_name="cli_override_test",
        input_features="core46+all",
        model="resnet",
        tower_depth=5,
    )
    trainer = Trainer(config, args)
    assert trainer.feature_spec is not None
    assert trainer.feature_spec.name == "core46+all"
    assert trainer.tower_depth == 5
