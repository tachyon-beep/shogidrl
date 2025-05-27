"""
Unit tests for PPOAgent model saving and loading.
"""

import os

import torch

from keisei.config_schema import AppConfig, EnvConfig, TrainingConfig, EvaluationConfig, LoggingConfig, WandBConfig, DemoConfig
from keisei.ppo_agent import PPOAgent
from keisei.shogi.shogi_game import ShogiGame
from keisei.utils import PolicyOutputMapper


def test_model_save_and_load(tmp_path):
    """Test saving and loading of the PPO agent's model."""
    # Setup dimensions and policy mapper
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
    max_moves_per_game = 512  # Use a sensible default or pull from config if available
    game_for_dims = ShogiGame(max_moves_per_game=max_moves_per_game)
    obs_sample = game_for_dims.get_observation()
    input_channels = config.env.input_channels  # Use config schema

    policy_output_mapper = PolicyOutputMapper()
    device = config.env.device
    agent = PPOAgent(input_channels, policy_output_mapper, device=device)
    # Corrected to use agent.model instead of agent.policy
    original_model_state_dict = {
        k: v.cpu() for k, v in agent.model.state_dict().items()
    }

    model_path = tmp_path / "test_model.pth"
    # Provide default values for the new arguments
    agent.save_model(model_path, global_timestep=0, total_episodes_completed=0)

    assert os.path.exists(model_path)

    # Create a new agent and load the model
    new_agent = PPOAgent(input_channels, policy_output_mapper, device=device)
    new_agent.load_model(model_path)
    # Corrected to use new_agent.model
    loaded_model_state_dict = {
        k: v.cpu() for k, v in new_agent.model.state_dict().items()
    }

    # Compare model parameters
    for key in original_model_state_dict:
        assert torch.equal(
            original_model_state_dict[key], loaded_model_state_dict[key]
        ), f"Model parameter mismatch for key: {key}"

    # Test loading into an agent with a different network instance but same architecture
    third_agent = PPOAgent(input_channels, policy_output_mapper, device=device)
    # Access a specific layer's weights, e.g., the first conv layer's weights
    # This depends on the structure of your ActorCritic model in neural_network.py
    # Assuming self.model.conv is the first convolutional layer
    if hasattr(third_agent.model, "conv") and hasattr(third_agent.model.conv, "weight"):
        third_agent.model.conv.weight.data.fill_(0.12345)
    elif hasattr(third_agent.model, "policy_head") and hasattr(
        third_agent.model.policy_head, "weight"
    ):  # Fallback to policy_head if conv not found
        third_agent.model.policy_head.weight.data.fill_(0.12345)
    else:
        # If neither conv nor policy_head with weights are found, skip this specific modification part of the test
        # or raise an error if this modification is critical for the test's intent.
        print(
            "Warning: Could not find a suitable layer to modify for testing model loading into a modified agent."
        )

    third_agent.load_model(model_path)
    # Corrected to use third_agent.model
    third_loaded_model_state_dict = {
        k: v.cpu() for k, v in third_agent.model.state_dict().items()
    }
    for key in original_model_state_dict:
        assert torch.equal(
            original_model_state_dict[key], third_loaded_model_state_dict[key]
        ), f"Model parameter mismatch for key: {key} after loading into a third agent"

    # Clean up
    os.remove(model_path)
