"""
Unit tests for ActorCritic in neural_network.py
"""

import torch
from keisei.neural_network import ActorCritic


def test_actor_critic_init_and_forward():
    """Test ActorCritic initializes and forward pass works with dummy input."""
    model = ActorCritic(input_channels=46, num_actions_total=3159)
    x = torch.zeros((2, 46, 9, 9))  # batch of 2
    policy_logits, value = model(x)
    assert policy_logits.shape == (2, 3159)
    assert value.shape == (2, 1)
