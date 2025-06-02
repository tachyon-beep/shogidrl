"""
Test to verify the base actor-critic refactoring works correctly.
"""

import pytest
import torch

from keisei.core.base_actor_critic import BaseActorCriticModel
from keisei.core.neural_network import ActorCritic
from keisei.training.models.resnet_tower import ActorCriticResTower


def test_both_models_inherit_from_base():
    """Test that both ActorCritic and ActorCriticResTower inherit from BaseActorCriticModel."""
    model1 = ActorCritic(input_channels=46, num_actions_total=100)
    model2 = ActorCriticResTower(
        input_channels=46, num_actions_total=100, tower_depth=2, tower_width=32
    )

    assert isinstance(model1, BaseActorCriticModel)
    assert isinstance(model2, BaseActorCriticModel)


def test_shared_methods_work_identically():
    """Test that both models provide the same interface through shared methods."""
    model1 = ActorCritic(input_channels=46, num_actions_total=100)
    model2 = ActorCriticResTower(
        input_channels=46, num_actions_total=100, tower_depth=2, tower_width=32
    )

    obs = torch.randn(1, 46, 9, 9)
    legal_mask = torch.ones(1, 100, dtype=torch.bool)
    actions = torch.randint(0, 100, (1,))

    # Test get_action_and_value
    action1, log_prob1, value1 = model1.get_action_and_value(obs, legal_mask)
    action2, log_prob2, value2 = model2.get_action_and_value(obs, legal_mask)

    # Test shapes are consistent
    assert action1.shape == action2.shape
    assert log_prob1.shape == log_prob2.shape
    assert value1.shape == value2.shape

    # Test evaluate_actions
    log_probs1, entropy1, value1_eval = model1.evaluate_actions(
        obs, actions, legal_mask
    )
    log_probs2, entropy2, value2_eval = model2.evaluate_actions(
        obs, actions, legal_mask
    )

    # Test shapes are consistent
    assert log_probs1.shape == log_probs2.shape
    assert entropy1.shape == entropy2.shape
    assert value1_eval.shape == value2_eval.shape


def test_deterministic_mode():
    """Test that deterministic mode works for both models."""
    model1 = ActorCritic(input_channels=46, num_actions_total=100)
    model2 = ActorCriticResTower(
        input_channels=46, num_actions_total=100, tower_depth=2, tower_width=32
    )

    obs = torch.randn(1, 46, 9, 9)

    # Test deterministic mode produces same action for same input
    action1_det, _, _ = model1.get_action_and_value(obs, deterministic=True)
    action1_det2, _, _ = model1.get_action_and_value(obs, deterministic=True)

    action2_det, _, _ = model2.get_action_and_value(obs, deterministic=True)
    action2_det2, _, _ = model2.get_action_and_value(obs, deterministic=True)

    assert torch.equal(action1_det, action1_det2)
    assert torch.equal(action2_det, action2_det2)


if __name__ == "__main__":
    test_both_models_inherit_from_base()
    test_shared_methods_work_identically()
    test_deterministic_mode()
    print("All refactoring tests passed!")
