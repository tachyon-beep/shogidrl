"""
Unit tests for ActorCritic in neural_network.py
"""

import pytest
import torch

from keisei.constants import (
    ALTERNATIVE_ACTION_SPACE,
    CORE_OBSERVATION_CHANNELS,
    EXTENDED_OBSERVATION_CHANNELS,
    FULL_ACTION_SPACE,
    SHOGI_BOARD_SIZE,
    SHOGI_BOARD_SQUARES,
)
from keisei.core.neural_network import ActorCritic


@pytest.fixture
def sample_observation(minimal_app_config):
    """Create a sample observation tensor for testing."""
    config = minimal_app_config
    return torch.randn(
        (2, config.env.input_channels, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE)
    )


class TestActorCriticBasicFunctionality:
    """Test basic ActorCritic functionality."""

    def test_actor_critic_init_and_forward(self, minimal_app_config):
        """Test ActorCritic initializes and forward pass works with dummy input."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )
        x = torch.zeros((2, input_channels, 9, 9))  # batch of 2
        policy_logits, value = model(x)
        assert policy_logits.shape == (2, num_actions)
        assert value.shape == (2, 1)

    @pytest.mark.parametrize(
        "batch_size",
        [1, 2, 4, 8, 16],
        ids=["single", "pair", "small", "medium", "large"],
    )
    def test_actor_critic_different_batch_sizes(self, minimal_app_config, batch_size):
        """Test ActorCritic works with different batch sizes."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )

        x = torch.randn((batch_size, input_channels, 9, 9))
        policy_logits, value = model(x)
        assert policy_logits.shape == (batch_size, num_actions)
        assert value.shape == (batch_size, 1)

    def test_actor_critic_output_types(self, minimal_app_config):
        """Test ActorCritic outputs have correct data types."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )
        x = torch.randn((2, input_channels, 9, 9))
        policy_logits, value = model(x)

        assert policy_logits.dtype == torch.float32
        assert value.dtype == torch.float32
        assert isinstance(policy_logits, torch.Tensor)
        assert isinstance(value, torch.Tensor)


class TestActorCriticParameterized:
    """Test ActorCritic with various parameter configurations."""

    @pytest.mark.parametrize(
        "input_channels", [1, 3, 46, 64], ids=["minimal", "rgb", "default", "large"]
    )
    def test_input_channels(self, input_channels):
        """Test ActorCritic with different input channel configurations."""
        num_actions = 100
        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )
        x = torch.randn((2, input_channels, 9, 9))

        policy_logits, value = model(x)
        assert policy_logits.shape == (2, num_actions)
        assert value.shape == (2, 1)
        assert not torch.isnan(policy_logits).any()
        assert not torch.isnan(value).any()

    @pytest.mark.parametrize(
        "num_actions", [1, 10, 100, 1000], ids=["single", "small", "default", "large"]
    )
    def test_action_spaces(self, num_actions):
        """Test ActorCritic with different action space sizes."""
        input_channels = 46
        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )
        x = torch.randn((2, input_channels, 9, 9))

        policy_logits, value = model(x)
        assert policy_logits.shape == (2, num_actions)
        assert value.shape == (2, 1)
        assert not torch.isnan(policy_logits).any()
        assert not torch.isnan(value).any()

    @pytest.mark.parametrize("spatial_dims", [(9, 9)], ids=["default"])
    def test_spatial_dimensions(self, spatial_dims):
        """Test ActorCritic with different spatial dimensions (currently only supports 9x9)."""
        input_channels = 46
        num_actions = 100
        height, width = spatial_dims

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )
        x = torch.randn((2, input_channels, height, width))

        policy_logits, value = model(x)
        assert policy_logits.shape == (2, num_actions)
        assert value.shape == (2, 1)
        assert not torch.isnan(policy_logits).any()
        assert not torch.isnan(value).any()


class TestActorCriticAdvanced:
    """Test advanced ActorCritic functionality."""

    def test_actor_critic_gradient_flow(self, minimal_app_config):
        """Test that gradients flow correctly through ActorCritic."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )
        x = torch.randn((2, input_channels, 9, 9), requires_grad=True)

        policy_logits, value = model(x)

        # Create a simple loss and compute gradients
        loss = policy_logits.sum() + value.sum()
        loss.backward()

        # Check that gradients were computed for model parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert not torch.all(
                param.grad == 0
            ), f"Zero gradient for parameter: {name}"

    def test_gradient_accumulation(self, minimal_app_config):
        """Test gradient accumulation across multiple forward/backward passes."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )
        x1 = torch.randn((1, input_channels, 9, 9))
        x2 = torch.randn((1, input_channels, 9, 9))

        # First accumulation
        policy1, value1 = model(x1)
        loss1 = policy1.sum() + value1.sum()
        loss1.backward()

        # Store first gradients (ensure gradients exist)
        first_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                first_grads[name] = param.grad.clone()

        # Second accumulation (without zero_grad)
        policy2, value2 = model(x2)
        loss2 = policy2.sum() + value2.sum()
        loss2.backward()

        # Check that gradients accumulated (are larger than first gradients)
        for name, param in model.named_parameters():
            if param.grad is not None and name in first_grads:
                assert not torch.allclose(
                    param.grad, first_grads[name], rtol=1e-5
                ), f"Gradient for {name} did not accumulate"

    def test_actor_critic_device_compatibility(self, minimal_app_config):
        """Test ActorCritic device compatibility (CPU and potentially CUDA)."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        # Test CPU
        model_cpu = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )
        x_cpu = torch.randn((2, input_channels, 9, 9))

        policy_logits_cpu, value_cpu = model_cpu(x_cpu)
        assert policy_logits_cpu.device.type == "cpu"
        assert value_cpu.device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = ActorCritic(
                input_channels=input_channels, num_actions_total=num_actions
            ).cuda()
            x_cuda = torch.randn((2, input_channels, 9, 9)).cuda()

            policy_logits_cuda, value_cuda = model_cuda(x_cuda)
            assert policy_logits_cuda.device.type == "cuda"
            assert value_cuda.device.type == "cuda"

            # Test moving model between devices
            model_back_to_cpu = model_cuda.cpu()
            x_cpu_again = torch.randn((2, input_channels, 9, 9))
            policy_logits_back, value_back = model_back_to_cpu(x_cpu_again)
            assert policy_logits_back.device.type == "cpu"
            assert value_back.device.type == "cpu"

    def test_actor_critic_serialization(self, minimal_app_config, tmp_path):
        """Test ActorCritic serialization and deserialization."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        # Create and save model
        model_original = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )
        save_path = tmp_path / "test_model.pt"
        torch.save(model_original.state_dict(), save_path)

        # Load model and verify
        model_loaded = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )
        model_loaded.load_state_dict(torch.load(save_path))

        # Test that loaded model produces same outputs
        x = torch.randn((2, input_channels, 9, 9))
        model_original.eval()
        model_loaded.eval()

        with torch.no_grad():
            policy_orig, value_orig = model_original(x)
            policy_loaded, value_loaded = model_loaded(x)

            assert torch.allclose(policy_orig, policy_loaded, atol=1e-6)
            assert torch.allclose(value_orig, value_loaded, atol=1e-6)

    def test_actor_critic_training_vs_eval_mode(self, minimal_app_config):
        """Test ActorCritic behavior in training vs evaluation mode."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )
        x = torch.randn((2, input_channels, 9, 9))

        # Test training mode
        model.train()
        assert model.training is True
        policy_train, value_train = model(x)

        # Test evaluation mode
        model.eval()
        assert model.training is False
        policy_eval, value_eval = model(x)

        # For this simple model without dropout/batchnorm, outputs should be identical
        # but we test that both modes work without error
        assert policy_train.shape == policy_eval.shape
        assert value_train.shape == value_eval.shape


class TestActorCriticEdgeCases:
    """Test ActorCritic edge cases and robustness."""

    def test_extreme_inputs(self, minimal_app_config):
        """Test ActorCritic with extreme input values."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )

        # Test very large values
        x_large = torch.full((2, input_channels, 9, 9), 100.0)
        policy_large, value_large = model(x_large)
        assert not torch.isnan(policy_large).any()
        assert not torch.isnan(value_large).any()
        assert not torch.isinf(policy_large).any()
        assert not torch.isinf(value_large).any()

        # Test very small values
        x_small = torch.full((2, input_channels, 9, 9), -100.0)
        policy_small, value_small = model(x_small)
        assert not torch.isnan(policy_small).any()
        assert not torch.isnan(value_small).any()
        assert not torch.isinf(policy_small).any()
        assert not torch.isinf(value_small).any()

        # Test zero inputs
        x_zero = torch.zeros((2, input_channels, 9, 9))
        policy_zero, value_zero = model(x_zero)
        assert not torch.isnan(policy_zero).any()
        assert not torch.isnan(value_zero).any()

    def test_minimal_configurations(self):
        """Test ActorCritic with minimal valid configurations (but maintains 9x9 spatial constraint)."""
        # Test minimal input channels and actions (but keep 9x9 spatial dimension)
        model_minimal = ActorCritic(input_channels=1, num_actions_total=1)
        x_minimal = torch.randn((1, 1, 9, 9))  # Changed from (1,1,1,1) to (1,1,9,9)

        policy, value = model_minimal(x_minimal)
        assert policy.shape == (1, 1)
        assert value.shape == (1, 1)
        assert not torch.isnan(policy).any()
        assert not torch.isnan(value).any()

    def test_parameter_counts(self, minimal_app_config):
        """Test that parameter counts are reasonable for different configurations."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Basic sanity checks
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

        # Verify all parameters are trainable by default
        assert total_params == trainable_params

    def test_weight_initialization(self, minimal_app_config):
        """Test that weights are properly initialized."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )

        # Check that weights are not all zeros or all ones
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() > 1:  # Skip bias and 1D parameters
                assert not torch.all(
                    param == 0
                ), f"Weight parameter {name} is all zeros"
                assert not torch.all(param == 1), f"Weight parameter {name} is all ones"
                assert (
                    torch.var(param, unbiased=False) > 1e-6
                ), f"Weight parameter {name} has very low variance"

    def test_output_ranges(self, minimal_app_config):
        """Test that outputs are in reasonable ranges."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )
        x = torch.randn((10, input_channels, 9, 9))  # Larger batch for statistics

        policy_logits, value = model(x)

        # Policy logits should not be extreme
        assert torch.all(
            torch.abs(policy_logits) < 100
        ), "Policy logits are too extreme"

        # Values should be reasonable (not extremely large)
        assert torch.all(torch.abs(value) < 1000), "Values are too extreme"

        # Check for reasonable variance in outputs
        assert (
            torch.var(policy_logits, unbiased=False) > 1e-6
        ), "Policy logits have very low variance"
        assert torch.var(value, unbiased=False) > 1e-6, "Values have very low variance"


class TestActorCriticSerialization:
    """Test ActorCritic model serialization and deserialization."""

    def test_state_dict_save_load(self, minimal_app_config):
        """Test saving and loading model state dictionary."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        # Create and initialize model
        model1 = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )

        # Save state dict
        state_dict = model1.state_dict()

        # Create new model and load state dict
        model2 = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )
        model2.load_state_dict(state_dict)

        # Test that models produce identical outputs
        x = torch.randn((3, input_channels, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE))

        model1.eval()
        model2.eval()

        with torch.no_grad():
            policy1, value1 = model1(x)
            policy2, value2 = model2(x)

        torch.testing.assert_close(policy1, policy2, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(value1, value2, atol=1e-6, rtol=1e-6)

    def test_torch_save_load(self, minimal_app_config):
        """Test saving and loading entire model with torch.save/load."""
        import os
        import tempfile

        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        # Create and initialize model
        model1 = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )

        # Save model to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
            torch.save(model1.state_dict(), tmp_file.name)

            # Load model
            model2 = ActorCritic(
                input_channels=input_channels, num_actions_total=num_actions
            )
            model2.load_state_dict(torch.load(tmp_file.name))

            # Test equivalence
            x = torch.randn((2, input_channels, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE))

            model1.eval()
            model2.eval()

            with torch.no_grad():
                policy1, value1 = model1(x)
                policy2, value2 = model2(x)

            torch.testing.assert_close(policy1, policy2, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(value1, value2, atol=1e-6, rtol=1e-6)

            # Clean up
            os.unlink(tmp_file.name)

    def test_partial_state_dict_loading(self, minimal_app_config):
        """Test loading partial state dictionaries (missing or extra keys)."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )

        # Get original state dict
        original_state_dict = model.state_dict()

        # Create partial state dict (missing some keys)
        partial_state_dict = {
            k: v for k, v in original_state_dict.items() if "conv" in k
        }

        # Should be able to load with strict=False
        model.load_state_dict(partial_state_dict, strict=False)

        # Model should still be functional
        x = torch.randn((1, input_channels, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE))
        policy, value = model(x)
        assert policy.shape == (1, num_actions)
        assert value.shape == (1, 1)


class TestActorCriticAdvancedEdgeCases:
    """Test ActorCritic behavior in advanced edge cases."""

    def test_zero_input(self, minimal_app_config):
        """Test model behavior with zero input."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )
        x = torch.zeros((1, input_channels, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE))

        policy_logits, value = model(x)

        # Outputs should be finite
        assert torch.isfinite(
            policy_logits
        ).all(), "Policy logits contain non-finite values"
        assert torch.isfinite(value).all(), "Values contain non-finite values"

    def test_extreme_input_values(self, minimal_app_config):
        """Test model behavior with extreme input values."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )

        # Very large inputs
        x_large = torch.full(
            (1, input_channels, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE), 1e6
        )
        policy_large, value_large = model(x_large)
        assert torch.isfinite(
            policy_large
        ).all(), "Policy logits not finite for large inputs"
        assert torch.isfinite(value_large).all(), "Values not finite for large inputs"

        # Very small inputs
        x_small = torch.full(
            (1, input_channels, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE), 1e-6
        )
        policy_small, value_small = model(x_small)
        assert torch.isfinite(
            policy_small
        ).all(), "Policy logits not finite for small inputs"
        assert torch.isfinite(value_small).all(), "Values not finite for small inputs"

    def test_single_vs_batch_consistency(self, minimal_app_config):
        """Test that single inputs and batch inputs produce consistent results."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )
        model.eval()

        # Single input
        x_single = torch.randn((1, input_channels, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE))

        with torch.no_grad():
            policy_single, value_single = model(x_single)

        # Batch with same input repeated
        x_batch = x_single.repeat(3, 1, 1, 1)

        with torch.no_grad():
            policy_batch, value_batch = model(x_batch)

        # Each batch item should match the single result
        for i in range(3):
            torch.testing.assert_close(
                policy_batch[i : i + 1], policy_single, atol=1e-6, rtol=1e-6
            )
            torch.testing.assert_close(
                value_batch[i : i + 1], value_single, atol=1e-6, rtol=1e-6
            )

    def test_model_mode_consistency(self, minimal_app_config):
        """Test that model behaves consistently in train/eval modes."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )
        x = torch.randn((2, input_channels, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE))

        # Get outputs in training mode
        model.train()
        policy_train, value_train = model(x)

        # Get outputs in eval mode
        model.eval()
        with torch.no_grad():
            policy_eval, value_eval = model(x)

        # Shapes should be consistent
        assert policy_train.shape == policy_eval.shape
        assert value_train.shape == value_eval.shape

        # Outputs should be finite in both modes
        assert torch.isfinite(policy_train).all() and torch.isfinite(policy_eval).all()
        assert torch.isfinite(value_train).all() and torch.isfinite(value_eval).all()


class TestActorCriticMemoryEfficiency:
    """Test ActorCritic memory usage and efficiency."""

    def test_memory_cleanup_after_forward(self, minimal_app_config):
        """Test that memory is properly managed during forward passes."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )

        import gc

        # Perform multiple forward passes
        for _ in range(10):
            x = torch.randn((4, input_channels, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE))
            with torch.no_grad():
                policy, value = model(x)

            # Clean up explicitly
            del x, policy, value
            gc.collect()

        # Model should still be functional
        x_test = torch.randn((1, input_channels, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE))
        policy_test, value_test = model(x_test)
        assert torch.isfinite(policy_test).all()
        assert torch.isfinite(value_test).all()

    def test_gradient_memory_efficiency(self, minimal_app_config):
        """Test memory efficiency during gradient computation."""
        config = minimal_app_config
        input_channels = config.env.input_channels
        num_actions = config.env.num_actions_total

        model = ActorCritic(
            input_channels=input_channels, num_actions_total=num_actions
        )

        # Perform gradient computation and cleanup
        x = torch.randn(
            (2, input_channels, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE), requires_grad=True
        )
        policy, value = model(x)

        # Compute simple loss and backward
        loss = policy.sum() + value.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

        # Clear gradients
        model.zero_grad()
        x.grad = None

        # Should be able to repeat
        policy2, value2 = model(x)
        loss2 = policy2.sum() + value2.sum()
        loss2.backward()

        assert x.grad is not None
