"""
Unit tests for ActorCritic in neural_network.py
"""

import pytest
import torch

from keisei.config_schema import (
    AppConfig,
    DemoConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    ParallelConfig,
    TrainingConfig,
    WandBConfig,
)
from keisei.core.neural_network import ActorCritic


def test_actor_critic_init_and_forward():
    """Test ActorCritic initializes and forward pass works with dummy input."""
    config = AppConfig(
        parallel=ParallelConfig(
            enabled=False, start_method="fork", num_envs=1, base_port=50000
        ),
        env=EnvConfig(
            device="cpu", input_channels=46, num_actions_total=13527, seed=42
        ),
        training=TrainingConfig(
            total_timesteps=500_000,
            steps_per_epoch=2048,
            ppo_epochs=10,
            minibatch_size=64,
            learning_rate=3e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            render_every_steps=1,
            refresh_per_second=4,
            enable_spinner=True,
            input_features="core46",
            tower_depth=9,
            tower_width=256,
            se_ratio=0.25,
            model_type="resnet",
            mixed_precision=False,
            ddp=False,
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=10000,
            evaluation_interval_timesteps=50000,
            weight_decay=0.0,
        ),
        evaluation=EvaluationConfig(
            num_games=20, opponent_type="random", evaluation_interval_timesteps=50000
        ),
        logging=LoggingConfig(
            log_file="logs/training_log.txt", model_dir="models/", run_name=None
        ),
        wandb=WandBConfig(
            enabled=True,
            project="keisei-shogi",
            entity=None,
            run_name_prefix="keisei",
            watch_model=True,
            watch_log_freq=1000,
            watch_log_type="all",
        ),
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.5),
    )
    input_channels = config.env.input_channels
    model = ActorCritic(input_channels=input_channels, num_actions_total=13527)
    x = torch.zeros((2, input_channels, 9, 9))  # batch of 2
    policy_logits, value = model(x)
    assert policy_logits.shape == (2, 13527)
    assert value.shape == (2, 1)
