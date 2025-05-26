"""
Unit tests for ActorCritic in neural_network.py
"""

import torch
import config

from keisei.neural_network import ActorCritic


def test_actor_critic_init_and_forward():
    """Test ActorCritic initializes and forward pass works with dummy input."""
    model = ActorCritic(input_channels=config.INPUT_CHANNELS, num_actions_total=3159)
    x = torch.zeros((2, config.INPUT_CHANNELS, 9, 9))  # batch of 2
    policy_logits, value = model(x)
    assert policy_logits.shape == (2, 3159)
    assert value.shape == (2, 1)
