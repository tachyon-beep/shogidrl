"""
Unit tests for PPOAgent in ppo_agent.py
"""

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
