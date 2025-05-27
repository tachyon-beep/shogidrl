"""
Unit tests for ActorCritic in neural_network.py
"""

import torch
from keisei.config_schema import AppConfig, EnvConfig, TrainingConfig, EvaluationConfig, LoggingConfig, WandBConfig, DemoConfig

from keisei.neural_network import ActorCritic


def test_actor_critic_init_and_forward():
    """Test ActorCritic initializes and forward pass works with dummy input."""
    config = AppConfig(
        env=EnvConfig(device="cpu", input_channels=46, num_actions_total=13527, seed=42),
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
        ),
        evaluation=EvaluationConfig(num_games=20, opponent_type="random"),
        logging=LoggingConfig(log_file="logs/training_log.txt", model_dir="models/"),
        wandb=WandBConfig(enabled=True, project="keisei-shogi", entity=None),
        demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.5),
    )
    input_channels = config.env.input_channels
    model = ActorCritic(input_channels=input_channels, num_actions_total=3159)
    x = torch.zeros((2, input_channels, 9, 9))  # batch of 2
    policy_logits, value = model(x)
    assert policy_logits.shape == (2, 3159)
    assert value.shape == (2, 1)
