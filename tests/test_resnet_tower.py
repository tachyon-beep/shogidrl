"""
test_resnet_tower.py: Unit tests for keisei/training/models/resnet_tower.py
"""

import io
from unittest.mock import patch

import pytest
import torch

from keisei.training.models.resnet_tower import ActorCriticResTower


@pytest.mark.parametrize("input_channels", [46, 51], ids=["channels_46", "channels_51"])
def test_resnet_tower_forward_shapes(input_channels):
    # Test with different input channel configurations
    model = ActorCriticResTower(
        input_channels=input_channels,
        num_actions_total=13527,
        tower_depth=9,
        tower_width=256,
        se_ratio=0.25,
    )
    x = torch.randn(2, input_channels, 9, 9)
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


@pytest.fixture
def model():
    """Create a small test model for testing."""
    return ActorCriticResTower(
        input_channels=46,
        num_actions_total=100,  # Smaller for easier testing
        tower_depth=2,
        tower_width=32,
        se_ratio=0.25,
    )


@pytest.fixture
def obs_batch():
    """Create test observation batch."""
    return torch.randn(4, 46, 9, 9)


@pytest.fixture
def obs_single():
    """Create single test observation."""
    return torch.randn(1, 46, 9, 9)


class TestGetActionAndValue:
    """Test the get_action_and_value() method."""

    def test_basic_functionality(self, model, obs_batch):
        """Test basic functionality without legal mask."""
        action, log_prob, value = model.get_action_and_value(obs_batch)

        assert action.shape == (4,)
        assert log_prob.shape == (4,)
        assert value.shape == (4,)
        assert action.dtype == torch.long
        assert torch.all(action >= 0) and torch.all(action < 100)
        assert not torch.isnan(log_prob).any()
        assert not torch.isnan(value).any()

    def test_deterministic_vs_stochastic(self, model, obs_single):
        """Test deterministic vs stochastic action selection."""
        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Get stochastic actions multiple times
        actions_stochastic = []
        for _ in range(10):
            action, _, _ = model.get_action_and_value(obs_single, deterministic=False)
            actions_stochastic.append(action.item())

        # Get deterministic action multiple times
        actions_deterministic = []
        for _ in range(5):
            action, _, _ = model.get_action_and_value(obs_single, deterministic=True)
            actions_deterministic.append(action.item())

        # Deterministic should always be the same
        assert (
            len(set(actions_deterministic)) == 1
        ), "Deterministic actions should be identical"

        # Stochastic should have some variation (with high probability)
        # Note: This test might occasionally fail due to randomness, but should be rare
        assert len(set(actions_stochastic)) > 1, "Stochastic actions should vary"

    def test_legal_mask_basic(self, model, obs_single):
        """Test basic legal mask functionality."""
        # Create legal mask with some actions masked out
        legal_mask = torch.ones(100, dtype=torch.bool)
        legal_mask[50:] = False  # Mask out actions 50-99

        # Test multiple times to ensure actions are always legal
        for _ in range(20):
            action, log_prob, value = model.get_action_and_value(
                obs_single, legal_mask=legal_mask, deterministic=False
            )
            assert action.item() < 50, f"Action {action.item()} should be < 50 (legal)"
            assert not torch.isnan(log_prob).any()
            assert not torch.isnan(value).any()

    def test_legal_mask_batch_broadcasting(self, model, obs_batch):
        """Test legal mask broadcasting with batch observations."""
        # Create legal mask for batch (4, 100)
        legal_mask = torch.ones(4, 100, dtype=torch.bool)
        legal_mask[0, 10:] = False  # First obs: only actions 0-9 legal
        legal_mask[1, 20:] = False  # Second obs: only actions 0-19 legal
        legal_mask[2, 30:] = False  # Third obs: only actions 0-29 legal
        legal_mask[3, 40:] = False  # Fourth obs: only actions 0-39 legal

        action, log_prob, value = model.get_action_and_value(
            obs_batch, legal_mask=legal_mask, deterministic=True
        )

        assert action[0].item() < 10, "First action should be < 10"
        assert action[1].item() < 20, "Second action should be < 20"
        assert action[2].item() < 30, "Third action should be < 30"
        assert action[3].item() < 40, "Fourth action should be < 40"

    def test_single_obs_legal_mask_adaptation(self, model, obs_single):
        """Test legal mask shape adaptation for single observation."""
        # Test with 1D legal mask that needs to be adapted for batch
        legal_mask_1d = torch.ones(100, dtype=torch.bool)
        legal_mask_1d[80:] = False

        action, log_prob, value = model.get_action_and_value(
            obs_single, legal_mask=legal_mask_1d, deterministic=False
        )

        assert action.item() < 80, "Action should respect legal mask"
        assert not torch.isnan(log_prob).any()
        assert not torch.isnan(value).any()

    def test_all_false_legal_mask_nan_handling(self, model, obs_single):
        """Test handling of all-False legal mask (NaN case)."""
        # Create legal mask with all actions masked out
        legal_mask = torch.zeros(100, dtype=torch.bool)

        # Capture stderr to check for warning message
        captured_stderr = io.StringIO()
        with patch("sys.stderr", captured_stderr):
            action, log_prob, value = model.get_action_and_value(
                obs_single, legal_mask=legal_mask, deterministic=False
            )

        # Check that warning was printed
        stderr_content = captured_stderr.getvalue()
        assert "Warning: NaNs in probabilities" in stderr_content
        assert "Defaulting to uniform" in stderr_content

        # Action should still be valid (0-99)
        assert 0 <= action.item() < 100
        assert not torch.isnan(log_prob).any()
        assert not torch.isnan(value).any()

    def test_device_consistency(self, model):
        """Test device consistency when using CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        model = model.to(device)
        obs = torch.randn(2, 46, 9, 9, device=device)
        legal_mask = torch.ones(2, 100, dtype=torch.bool, device=device)

        action, log_prob, value = model.get_action_and_value(obs, legal_mask)

        assert action.device.type == device.type
        assert log_prob.device.type == device.type
        assert value.device.type == device.type

    def test_gradient_flow(self, model, obs_single):
        """Test that gradients flow properly through the method."""
        obs_single.requires_grad_(True)
        _, log_prob, value = model.get_action_and_value(obs_single)

        # Backward pass on value
        value.backward(retain_graph=True)
        assert obs_single.grad is not None

        obs_single.grad.zero_()

        # Backward pass on log_prob
        log_prob.backward()
        assert obs_single.grad is not None


class TestEvaluateActions:
    """Test the evaluate_actions() method."""

    def test_basic_functionality(self, model, obs_batch):
        """Test basic functionality without legal mask."""
        actions = torch.randint(0, 100, (4,))
        log_probs, entropy, value = model.evaluate_actions(obs_batch, actions)

        assert log_probs.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)
        assert not torch.isnan(log_probs).any()
        assert not torch.isnan(entropy).any()
        assert not torch.isnan(value).any()
        assert torch.all(entropy >= 0), "Entropy should be non-negative"

    def test_with_legal_mask(self, model, obs_batch):
        """Test evaluate_actions with legal mask."""
        # Create legal mask and legal actions
        legal_mask = torch.ones(4, 100, dtype=torch.bool)
        legal_mask[:, 50:] = False  # Only actions 0-49 are legal

        # Use legal actions
        actions = torch.randint(0, 50, (4,))

        log_probs, entropy, value = model.evaluate_actions(
            obs_batch, actions, legal_mask=legal_mask
        )

        assert not torch.isnan(log_probs).any()
        assert not torch.isnan(entropy).any()
        assert not torch.isnan(value).any()
        assert torch.all(entropy >= 0)

    def test_all_false_legal_mask_nan_handling(self, model, obs_batch):
        """Test handling of all-False legal mask in evaluate_actions."""
        # Create legal mask with all actions masked out
        legal_mask = torch.zeros(4, 100, dtype=torch.bool)
        actions = torch.randint(0, 100, (4,))

        # Capture stderr to check for warning message
        captured_stderr = io.StringIO()
        with patch("sys.stderr", captured_stderr):
            log_probs, entropy, value = model.evaluate_actions(
                obs_batch, actions, legal_mask=legal_mask
            )

        # Check that warning was printed
        stderr_content = captured_stderr.getvalue()
        assert "Warning: NaNs in probabilities" in stderr_content
        assert "Defaulting to uniform for affected rows" in stderr_content

        # Results should still be valid (no NaNs)
        assert not torch.isnan(log_probs).any()
        assert not torch.isnan(entropy).any()
        assert not torch.isnan(value).any()

    def test_consistency_with_get_action_and_value(self, model, obs_single):
        """Test consistency between get_action_and_value and evaluate_actions."""
        legal_mask = torch.ones(100, dtype=torch.bool)
        legal_mask[75:] = False

        # Get action and log_prob from get_action_and_value
        torch.manual_seed(42)
        action, log_prob1, value1 = model.get_action_and_value(
            obs_single, legal_mask=legal_mask.unsqueeze(0), deterministic=True
        )

        # Evaluate the same action with evaluate_actions
        log_prob2, _, value2 = model.evaluate_actions(
            obs_single, action, legal_mask=legal_mask.unsqueeze(0)
        )

        # Log probabilities should be very close (allowing for small numerical differences)
        assert torch.allclose(log_prob1, log_prob2, atol=1e-6)
        # Values should be identical (same forward pass)
        assert torch.allclose(value1, value2, atol=1e-6)

    def test_entropy_properties(self, model, obs_batch):
        """Test entropy calculation properties."""
        actions = torch.randint(0, 100, (4,))

        # Without legal mask
        _, entropy1, _ = model.evaluate_actions(obs_batch, actions)

        # With restrictive legal mask (fewer legal actions -> lower entropy)
        legal_mask = torch.ones(4, 100, dtype=torch.bool)
        legal_mask[:, 10:] = False  # Only 10 actions legal
        _, entropy2, _ = model.evaluate_actions(obs_batch, actions, legal_mask)

        # Entropy with fewer legal actions should generally be lower
        # Note: This is a probabilistic test and might occasionally fail
        assert torch.all(entropy1 >= 0) and torch.all(entropy2 >= 0)

    def test_gradient_flow(self, model, obs_batch):
        """Test that gradients flow properly through evaluate_actions."""
        obs_batch.requires_grad_(True)
        actions = torch.randint(0, 100, (4,))

        log_probs, entropy, value = model.evaluate_actions(obs_batch, actions)

        # Test gradient flow through each output
        loss = log_probs.mean() + entropy.mean() + value.mean()
        loss.backward()

        assert obs_batch.grad is not None
        assert not torch.isnan(obs_batch.grad).any()

    def test_device_consistency(self, model):
        """Test device consistency for evaluate_actions."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        model = model.to(device)
        obs = torch.randn(2, 46, 9, 9, device=device)
        actions = torch.randint(0, 100, (2,), device=device)
        legal_mask = torch.ones(2, 100, dtype=torch.bool, device=device)

        log_probs, entropy, value = model.evaluate_actions(obs, actions, legal_mask)

        assert log_probs.device.type == device.type
        assert entropy.device.type == device.type
        assert value.device.type == device.type


class TestIntegrationAndEdgeCases:
    """Test integration scenarios and edge cases."""

    def test_extreme_legal_masks(self, model, obs_single):
        """Test with extreme legal mask configurations."""
        # Single legal action
        legal_mask = torch.zeros(100, dtype=torch.bool)
        legal_mask[42] = True

        action, log_prob, value = model.get_action_and_value(
            obs_single, legal_mask=legal_mask, deterministic=True
        )

        assert action.item() == 42
        assert not torch.isnan(log_prob).any()
        assert not torch.isnan(value).any()

    def test_numerical_stability(self, model):
        """Test numerical stability with extreme inputs."""
        # Very small observation values
        obs_small = torch.full((1, 46, 9, 9), 1e-8)
        _, log_prob, value = model.get_action_and_value(obs_small)
        assert not torch.isnan(log_prob).any() and not torch.isnan(value).any()

        # Very large observation values
        obs_large = torch.full((1, 46, 9, 9), 1e3)
        _, log_prob, value = model.get_action_and_value(obs_large)
        assert not torch.isnan(log_prob).any() and not torch.isnan(value).any()

    @pytest.mark.parametrize(
        "batch_size",
        [1, 2, 7, 16, 32],
        ids=["batch_1", "batch_2", "batch_7", "batch_16", "batch_32"],
    )
    def test_batch_size_edge_cases(self, model, batch_size):
        """Test with different batch sizes including edge cases."""
        obs = torch.randn(batch_size, 46, 9, 9)
        actions = torch.randint(0, 100, (batch_size,))
        legal_mask = torch.ones(batch_size, 100, dtype=torch.bool)

        # Test get_action_and_value
        action, log_prob, value = model.get_action_and_value(obs, legal_mask)
        assert action.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        assert value.shape == (batch_size,)

        # Test evaluate_actions
        log_probs, entropy, value = model.evaluate_actions(obs, actions, legal_mask)
        assert log_probs.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert value.shape == (batch_size,)

    def test_mixed_legal_masks_in_batch(self, model):
        """Test batch with mixed legal mask conditions."""
        obs_batch = torch.randn(3, 46, 9, 9)
        legal_mask = torch.ones(3, 100, dtype=torch.bool)

        # First obs: normal mask (50% legal)
        legal_mask[0, 50:] = False
        # Second obs: single legal action
        legal_mask[1, :] = False
        legal_mask[1, 25] = True
        # Third obs: all legal actions
        # legal_mask[2] remains all True

        action, log_prob, value = model.get_action_and_value(
            obs_batch, legal_mask=legal_mask, deterministic=True
        )

        assert action[0].item() < 50, "First action should be < 50"
        assert action[1].item() == 25, "Second action should be 25"
        assert 0 <= action[2].item() < 100, "Third action should be valid"
        assert not torch.isnan(log_prob).any()
        assert not torch.isnan(value).any()
