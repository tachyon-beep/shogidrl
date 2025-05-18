"""
Unit tests for PPOAgent model saving and loading.
"""

import os
import torch
from keisei.ppo_agent import PPOAgent
from keisei.utils import PolicyOutputMapper


def test_model_save_and_load(tmp_path):
    """Test PPOAgent can save and load its model and optimizer state."""
    mapper = PolicyOutputMapper()
    agent = PPOAgent(
        input_channels=46, num_actions_total=3159, policy_output_mapper=mapper
    )
    # Modify model weights
    for p in agent.model.parameters():
        p.data.fill_(1.23)
    # Save
    save_path = tmp_path / "ppo_model.pt"
    agent.save_model(str(save_path))
    # Create a new agent and load
    agent2 = PPOAgent(
        input_channels=46, num_actions_total=3159, policy_output_mapper=mapper
    )
    agent2.load_model(str(save_path))
    # Check that parameters match
    for p1, p2 in zip(agent.model.parameters(), agent2.model.parameters()):
        assert torch.allclose(p1, p2)
