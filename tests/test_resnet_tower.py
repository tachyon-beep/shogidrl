"""
test_resnet_tower.py: Unit tests for keisei/training/models/resnet_tower.py
"""

import pytest
import torch

from keisei.training.models.resnet_tower import ActorCriticResTower


def test_resnet_tower_forward_shapes():
    # Test with C=46, 51, and a large config
    for c in [46, 51]:
        model = ActorCriticResTower(
            input_channels=c,
            num_actions_total=13527,
            tower_depth=9,
            tower_width=256,
            se_ratio=0.25,
        )
        x = torch.randn(2, c, 9, 9)
        policy, value = model(x)
        assert policy.shape == (2, 13527)
        assert value.shape == (2,)


def test_resnet_tower_fp16_memory():
    # This is a smoke test for memory, not a strict limit
    model = ActorCriticResTower(
        input_channels=51,
        num_actions_total=13527,
        tower_depth=9,
        tower_width=256,
        se_ratio=0.25,
    )
    x = torch.randn(8, 51, 9, 9).half()
    model = model.half()
    with torch.no_grad():
        policy, value = model(x)
        assert policy.shape == (8, 13527)
        assert value.shape == (8,)


def test_resnet_tower_se_toggle():
    # Test with and without SE block
    model_se = ActorCriticResTower(
        46, 13527, tower_depth=3, tower_width=64, se_ratio=0.5
    )
    model_no_se = ActorCriticResTower(
        46, 13527, tower_depth=3, tower_width=64, se_ratio=None
    )
    x = torch.randn(1, 46, 9, 9)
    p1, v1 = model_se(x)
    p2, v2 = model_no_se(x)
    assert p1.shape == (1, 13527)
    assert v1.shape == (1,)
    assert p2.shape == (1, 13527)
    assert v2.shape == (1,)
