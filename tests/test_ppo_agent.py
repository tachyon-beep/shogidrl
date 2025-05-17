"""
Unit tests for PPOAgent in ppo_agent.py
"""

import pytest
import torch
import numpy as np
from keisei.ppo_agent import PPOAgent
from keisei.utils import PolicyOutputMapper


def test_ppo_agent_init_and_select_action():
    """Test PPOAgent initializes and select_action returns a valid index."""
    mapper = PolicyOutputMapper()
    agent = PPOAgent(
        input_channels=44, num_actions_total=3159, policy_output_mapper=mapper
    )
    obs = torch.zeros((1, 44, 9, 9))
    idx = agent.select_action(obs)
    assert isinstance(idx, (int, np.integer))
    assert 0 <= idx < 3159
    # Test with legal indices
    legal = np.array([1, 5, 9])
    idx2 = agent.select_action(obs, legal_indices=legal)
    assert idx2 in legal


def test_ppo_agent_ppo_update_and_gae():
    """Test PPOAgent's ppo_update and compute_gae methods with dummy data."""
    mapper = PolicyOutputMapper()
    agent = PPOAgent(
        input_channels=44, num_actions_total=3159, policy_output_mapper=mapper
    )
    # Create dummy batch
    obs = [torch.zeros((44, 9, 9)) for _ in range(8)]
    actions = [0, 1, 2, 3, 4, 5, 6, 7]
    old_log_probs = [0.0] * 8
    rewards = [1.0] * 8
    values = [0.5] * 8
    dones = [0] * 8
    next_value = 0.0
    returns = agent.compute_gae(rewards, values, dones, next_value)
    advantages = [r - v for r, v in zip(returns, values)]
    # Should not raise error
    agent.ppo_update(
        obs, actions, old_log_probs, returns, advantages, epochs=1, batch_size=4
    )
    assert len(returns) == 8
    assert isinstance(returns[0], float)
