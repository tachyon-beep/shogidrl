"""
Unit tests for ActorCritic in neural_network.py
"""

import pytest
import torch

from keisei.core.neural_network import ActorCritic


def test_actor_critic_init_and_forward(minimal_app_config):
    """Test ActorCritic initializes and forward pass works with dummy input."""
    config = minimal_app_config
    input_channels = config.env.input_channels
    num_actions = config.env.num_actions_total

    model = ActorCritic(input_channels=input_channels, num_actions_total=num_actions)
    x = torch.zeros((2, input_channels, 9, 9))  # batch of 2
    policy_logits, value = model(x)
    assert policy_logits.shape == (2, num_actions)
    assert value.shape == (2, 1)
