"""
Unit tests for PPOAgent model saving and loading.
"""

import os
import torch

from keisei.ppo_agent import PPOAgent
from keisei.shogi.shogi_game import ShogiGame
from keisei.utils import PolicyOutputMapper
import config


def test_model_save_and_load(tmp_path):
    """Test saving and loading of the PPO agent's model."""
    # Setup dimensions and policy mapper
    game_for_dims = ShogiGame(max_moves_per_game=config.MAX_MOVES_PER_GAME)
    obs_sample = game_for_dims.get_observation()
    input_channels = obs_sample.shape[0]  # Assuming shape is (channels, height, width)

    policy_output_mapper = (
        PolicyOutputMapper()
    )  # Corrected: No arguments for constructor

    device = config.DEVICE
    agent = PPOAgent(input_channels, policy_output_mapper, device=device)
    # Corrected to use agent.model instead of agent.policy
    original_model_state_dict = {
        k: v.cpu() for k, v in agent.model.state_dict().items()
    }

    model_path = tmp_path / "test_model.pth"
    agent.save_model(model_path)

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
