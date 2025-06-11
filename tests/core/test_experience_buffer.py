"""
Unit tests for ExperienceBuffer in experience_buffer.py
"""

import numpy as np
import pytest
import torch

from keisei.constants import (
    CORE_OBSERVATION_CHANNELS,
    SHOGI_BOARD_SIZE,
)
from keisei.core.experience_buffer import ExperienceBuffer
from keisei.utils import PolicyOutputMapper


def test_experience_buffer_add_and_len():
    """Test ExperienceBuffer add and __len__ methods."""
    buf = ExperienceBuffer(
        buffer_size=3, gamma=0.99, lambda_gae=0.95
    )  # Added gamma and lambda_gae
    assert len(buf) == 0
    # Create a properly sized dummy legal_mask using PolicyOutputMapper
    mapper = PolicyOutputMapper()
    dummy_legal_mask = torch.zeros(
        mapper.get_total_actions(), dtype=torch.bool, device=buf.device
    )  # Proper size (13527) with explicit device

    buf.add(
        torch.zeros(1, device=buf.device), 1, 0.5, 0.0, 0.0, False, dummy_legal_mask
    )  # Added log_prob=0.0, value=0.0, done=False, dummy_legal_mask with device
    assert len(buf) == 1
    buf.add(
        torch.ones(1, device=buf.device), 2, 1.0, 0.0, 0.0, False, dummy_legal_mask
    )  # Added log_prob=0.0, value=0.0, done=False, dummy_legal_mask with device
    assert len(buf) == 2
    buf.add(
        torch.ones(1, device=buf.device), 3, -1.0, 0.0, 0.0, True, dummy_legal_mask
    )  # Added log_prob=0.0, value=0.0, done=True, dummy_legal_mask with device
    assert len(buf) == 3
    # Should not add beyond buffer_size
    buf.add(
        torch.ones(1, device=buf.device), 4, 2.0, 0.0, 0.0, False, dummy_legal_mask
    )  # Added log_prob=0.0, value=0.0, done=False, dummy_legal_mask with device
    assert len(buf) == 3
    # Test the actual data through the public API by getting a batch
    buf.compute_advantages_and_returns(0.0)
    batch = buf.get_batch()
    assert torch.equal(batch["actions"], torch.tensor([1, 2, 3]))
    assert torch.allclose(batch["rewards"], torch.tensor([0.5, 1.0, -1.0]))


def test_experience_buffer_compute_advantages_and_returns():
    """Test ExperienceBuffer compute_advantages_and_returns method with GAE calculation."""
    gamma = 0.99
    lambda_gae = 0.95
    buf = ExperienceBuffer(buffer_size=3, gamma=gamma, lambda_gae=lambda_gae)
    mapper = PolicyOutputMapper()
    dummy_legal_mask = torch.zeros(mapper.get_total_actions(), dtype=torch.bool, device=buf.device)

    # Add test data: simple sequence with known rewards and values
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 1.0, 1.5]
    dones = [False, False, True]  # Last step is terminal

    for i in range(3):
        buf.add(
            obs=torch.randn(
                CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE, device=buf.device
            ),  # Dummy observation
            action=i,
            reward=rewards[i],
            log_prob=0.1,
            value=values[i],
            done=dones[i],
            legal_mask=dummy_legal_mask,
        )

    # Compute advantages with last_value=0.0 (terminal state)
    last_value = 0.0
    buf.compute_advantages_and_returns(last_value)

    # Verify advantages and returns are computed
    assert len(buf.advantages) == 3
    assert len(buf.returns) == 3

    # Check that all advantages and returns are tensors
    for i in range(3):
        assert isinstance(buf.advantages[i], torch.Tensor)
        assert isinstance(buf.returns[i], torch.Tensor)
        assert buf.advantages[i].device == buf.device
        assert buf.returns[i].device == buf.device

    # Manually verify GAE calculation for the last step (terminal)
    # For t=2 (terminal): delta = reward + gamma * 0 * (1-done) - value = 3.0 + 0 - 1.5 = 1.5
    # GAE = delta = 1.5
    # Return = GAE + value = 1.5 + 1.5 = 3.0
    expected_advantage_2 = 1.5
    expected_return_2 = 3.0
    assert torch.isclose(buf.advantages[2], torch.tensor(expected_advantage_2))
    assert torch.isclose(buf.returns[2], torch.tensor(expected_return_2))


def test_experience_buffer_compute_advantages_empty_buffer():
    """Test compute_advantages_and_returns on empty buffer."""
    buf = ExperienceBuffer(buffer_size=5, gamma=0.99, lambda_gae=0.95)

    # Should handle empty buffer gracefully
    buf.compute_advantages_and_returns(0.0)

    # With tensor pre-allocation, advantages tensor exists but all values are zero
    assert len(buf) == 0  # No experiences added
    assert buf.ptr == 0  # Pointer should be at start
    # Test that get_batch returns empty when no experiences added
    empty_batch = buf.get_batch()
    assert empty_batch == {}  # Should return empty dict when no data


def test_experience_buffer_get_batch():
    """Test ExperienceBuffer get_batch method."""
    buf = ExperienceBuffer(buffer_size=2, gamma=0.99, lambda_gae=0.95)
    mapper = PolicyOutputMapper()

    # Create test data
    obs1 = torch.randn(CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE)
    obs2 = torch.randn(CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE)
    legal_mask1 = torch.zeros(mapper.get_total_actions(), dtype=torch.bool)
    legal_mask2 = torch.ones(mapper.get_total_actions(), dtype=torch.bool)
    legal_mask1[100] = True  # Make some actions legal
    legal_mask2[200] = False  # Make some actions illegal

    # Add experiences
    buf.add(obs1, 1, 1.0, 0.1, 0.5, False, legal_mask1)
    buf.add(obs2, 2, 2.0, 0.2, 1.0, True, legal_mask2)

    # Compute advantages
    buf.compute_advantages_and_returns(0.0)

    # Get batch
    batch = buf.get_batch()

    # Verify batch structure
    expected_keys = [
        "obs",
        "actions",
        "log_probs",
        "values",
        "advantages",
        "returns",
        "dones",
        "legal_masks",
    ]
    for key in expected_keys:
        assert key in batch, f"Missing key: {key}"

    # Verify tensor shapes and types
    assert batch["obs"].shape == (2, 46, 9, 9)
    assert batch["actions"].shape == (2,)
    assert batch["log_probs"].shape == (2,)
    assert batch["values"].shape == (2,)
    assert batch["advantages"].shape == (2,)
    assert batch["returns"].shape == (2,)
    assert batch["dones"].shape == (2,)
    assert batch["legal_masks"].shape == (2, mapper.get_total_actions())

    # Verify data types
    assert batch["actions"].dtype == torch.int64
    assert batch["log_probs"].dtype == torch.float32
    assert batch["values"].dtype == torch.float32
    assert batch["advantages"].dtype == torch.float32
    assert batch["returns"].dtype == torch.float32
    assert batch["dones"].dtype == torch.bool
    assert batch["legal_masks"].dtype == torch.bool

    # Verify actual values
    assert torch.equal(batch["actions"], torch.tensor([1, 2]))
    assert torch.allclose(batch["log_probs"], torch.tensor([0.1, 0.2]))
    assert torch.allclose(batch["values"], torch.tensor([0.5, 1.0]))
    assert torch.equal(batch["dones"], torch.tensor([False, True]))


def test_experience_buffer_get_batch_empty():
    """Test get_batch on empty buffer returns empty dict."""
    buf = ExperienceBuffer(buffer_size=5, gamma=0.99, lambda_gae=0.95)

    batch = buf.get_batch()
    assert batch == {}


def test_experience_buffer_get_batch_stack_error():
    """Test that add() raises RuntimeError on tensor shape mismatch (improved with pre-allocation)."""
    buf = ExperienceBuffer(buffer_size=2, gamma=0.99, lambda_gae=0.95)
    mapper = PolicyOutputMapper()

    # Create obs tensors with different shapes to cause a shape error
    obs1 = torch.randn(46, 9, 9)
    obs2 = torch.randn(46, 8, 8)  # Different shape
    legal_mask = torch.zeros(mapper.get_total_actions(), dtype=torch.bool)

    buf.add(obs1, 1, 1.0, 0.1, 0.5, False, legal_mask)

    # With tensor pre-allocation, shape errors are caught during add() rather than get_batch()
    # This is actually better behavior - fail fast at insertion time
    with pytest.raises(RuntimeError, match="The expanded size of the tensor"):
        buf.add(obs2, 2, 2.0, 0.2, 1.0, True, legal_mask)

    # Test for legal_mask shape error - also caught at add() time now
    buf.clear()
    obs = torch.randn(46, 9, 9)
    legal_mask1 = torch.zeros(mapper.get_total_actions(), dtype=torch.bool)
    legal_mask2 = torch.zeros(
        mapper.get_total_actions() + 1, dtype=torch.bool
    )  # Different shape

    buf.add(obs, 1, 1.0, 0.1, 0.5, False, legal_mask1)

    # Legal mask shape error should also be caught at add() time
    with pytest.raises(RuntimeError, match="The expanded size of the tensor"):
        buf.add(obs, 2, 2.0, 0.2, 1.0, True, legal_mask2)


def test_experience_buffer_clear():
    """Test ExperienceBuffer clear method."""
    buf = ExperienceBuffer(buffer_size=3, gamma=0.99, lambda_gae=0.95)
    mapper = PolicyOutputMapper()
    dummy_legal_mask = torch.zeros(mapper.get_total_actions(), dtype=torch.bool)

    # Add some data
    buf.add(torch.randn(46, 9, 9), 1, 1.0, 0.1, 0.5, False, dummy_legal_mask)
    buf.add(torch.randn(46, 9, 9), 2, 2.0, 0.2, 1.0, False, dummy_legal_mask)

    # Compute advantages
    buf.compute_advantages_and_returns(0.0)

    # Verify buffer has data
    assert len(buf) == 2
    assert buf.ptr == 2  # Pointer should be at 2

    # Verify that data exists by checking the batch
    batch = buf.get_batch()
    assert batch["obs"].shape[0] == 2  # 2 observations
    assert batch["actions"].shape[0] == 2  # 2 actions

    # Clear buffer
    buf.clear()

    # Verify all data is cleared
    assert len(buf) == 0
    assert buf.ptr == 0
    # With tensor pre-allocation, tensors still exist but pointer is reset
    # Test that get_batch returns empty when ptr == 0
    empty_batch = buf.get_batch()
    assert empty_batch == {}  # Should return empty dict when no data


def test_experience_buffer_full_buffer_warning(capsys):
    """Test that adding to a full buffer prints warning and doesn't add."""
    buf = ExperienceBuffer(buffer_size=2, gamma=0.99, lambda_gae=0.95)
    mapper = PolicyOutputMapper()
    dummy_legal_mask = torch.zeros(mapper.get_total_actions(), dtype=torch.bool)

    # Fill buffer to capacity
    buf.add(torch.randn(46, 9, 9), 1, 1.0, 0.1, 0.5, False, dummy_legal_mask)
    buf.add(torch.randn(46, 9, 9), 2, 2.0, 0.2, 1.0, False, dummy_legal_mask)

    assert len(buf) == 2

    # Try to add one more - should print warning and not add
    buf.add(torch.randn(46, 9, 9), 3, 3.0, 0.3, 1.5, True, dummy_legal_mask)

    # Verify warning was logged to stderr
    captured = capsys.readouterr()
    assert "Buffer is full. Cannot add new experience." in captured.err
    assert "[ExperienceBuffer] WARNING:" in captured.err

    # Buffer should still be size 2, not 3
    assert len(buf) == 2
    # Test the actual data through the public API
    buf.compute_advantages_and_returns(0.0)
    batch = buf.get_batch()
    assert torch.equal(
        batch["actions"], torch.tensor([1, 2])
    )  # Should not contain the third action


def test_experience_buffer_device_consistency():
    """Test that ExperienceBuffer maintains device consistency."""
    device = "cpu"  # Use CPU for consistent testing
    buf = ExperienceBuffer(buffer_size=2, gamma=0.99, lambda_gae=0.95, device=device)
    mapper = PolicyOutputMapper()

    # Create tensors on the specified device
    obs = torch.randn(46, 9, 9, device=device)
    legal_mask = torch.zeros(
        mapper.get_total_actions(), dtype=torch.bool, device=device
    )

    buf.add(obs, 1, 1.0, 0.1, 0.5, False, legal_mask)
    buf.compute_advantages_and_returns(0.0)
    batch = buf.get_batch()

    # Verify all tensors in batch are on correct device
    for key, tensor in batch.items():
        if isinstance(tensor, torch.Tensor):
            assert tensor.device == torch.device(
                device
            ), f"Tensor {key} on wrong device"
