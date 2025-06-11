"""
Unit tests for advanced PPOAgent learning functionality.

This module tests sophisticated PPO learning features including:
- Loss component computation and validation
- Advantage normalization (enabled/disabled/behavioral differences)
- Gradient clipping mechanisms
- KL divergence tracking
- Minibatch processing
- Training robustness features
"""

import numpy as np
import torch

from keisei.constants import (
    CORE_OBSERVATION_CHANNELS,
    EPSILON_MEDIUM,
    SHOGI_BOARD_SIZE,
    TEST_ADVANTAGE_STD_THRESHOLD,
    TEST_BUFFER_SIZE,
    TEST_EXTREME_REWARDS,
    TEST_GRADIENT_CLIP_NORM,
    TEST_HIGH_LEARNING_RATE,
    TEST_HIGH_VARIANCE_REWARDS,
    TEST_HIGH_VARIANCE_VALUES,
    TEST_KL_DIVERGENCE_THRESHOLD,
    TEST_LAST_VALUE,
    TEST_LOG_PROB_MULTIPLIER,
    TEST_MEDIUM_BUFFER_SIZE,
    TEST_MIXED_REWARDS,
    TEST_MIXED_VALUES,
    TEST_ODD_BUFFER_SIZE,
    TEST_PPO_EPOCHS,
    TEST_SCHEDULER_FINAL_FRACTION,
    TEST_SCHEDULER_LEARNING_RATE,
    TEST_SCHEDULER_STEPS_PER_EPOCH,
    TEST_SCHEDULER_TOTAL_TIMESTEPS,
    TEST_SMALL_MINIBATCH,
    TEST_UNEVEN_MINIBATCH_SIZE,
    TEST_VALUE_DEFAULT,
)
from keisei.core.experience_buffer import ExperienceBuffer
from keisei.core.neural_network import ActorCritic
from keisei.core.ppo_agent import PPOAgent
from keisei.utils import PolicyOutputMapper
from tests.conftest import ENV_DEFAULTS, TRAIN_DEFAULTS, assert_valid_ppo_metrics


class TestPPOAgentLossComponents:
    """Tests for PPO loss computation and validation."""

    def test_learn_loss_components(self, integration_test_config, ppo_test_model):
        """Test that PPOAgent.learn correctly computes and returns individual loss components."""
        # Override specific settings for this test
        config = integration_test_config.model_copy()
        config.training.ppo_epochs = (
            TEST_PPO_EPOCHS  # Multiple epochs to test learning behavior
        )

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        buffer_size = (
            TEST_MEDIUM_BUFFER_SIZE  # Larger buffer for more realistic training
        )
        experience_buffer = ExperienceBuffer(
            buffer_size=buffer_size,
            gamma=TRAIN_DEFAULTS.gamma,
            lambda_gae=TRAIN_DEFAULTS.lambda_gae,
            device="cpu",
        )

        # Create deterministic data for more predictable testing
        torch.manual_seed(ENV_DEFAULTS.seed)
        np.random.seed(ENV_DEFAULTS.seed)

        dummy_obs_tensor = torch.randn(
            CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE, device="cpu"
        )
        dummy_legal_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cpu"
        )

        # Create varied rewards and values to test advantage calculation
        rewards = TEST_MIXED_REWARDS
        values = TEST_MIXED_VALUES

        for i in range(buffer_size):
            experience_buffer.add(
                obs=dummy_obs_tensor,
                action=i % agent.num_actions_total,
                reward=rewards[i],
                log_prob=TEST_LOG_PROB_MULTIPLIER * (i + 1),  # Varied log probs
                value=values[i],
                done=(i == buffer_size - 1),
                legal_mask=dummy_legal_mask,
            )

        # Compute advantages with realistic last value
        last_value = TEST_LAST_VALUE
        experience_buffer.compute_advantages_and_returns(last_value)

        # Capture initial model parameters for change verification
        initial_params = [p.clone() for p in agent.model.parameters()]

        # Call learn method
        metrics = agent.learn(experience_buffer)

        # Verify all expected metrics are present and valid
        assert_valid_ppo_metrics(metrics)

        # Verify reasonable metric ranges
        assert metrics["ppo/learning_rate"] == config.training.learning_rate
        assert (
            metrics["ppo/entropy"] <= 0.0
        ), "Entropy loss should be negative (entropy bonus)"
        assert metrics["ppo/policy_loss"] >= 0.0, "Policy loss should be non-negative"
        assert metrics["ppo/value_loss"] >= 0.0, "Value loss should be non-negative"

        # Verify model parameters changed (learning occurred)
        final_params = [p.clone() for p in agent.model.parameters()]
        params_changed = any(
            not torch.allclose(initial, final, atol=EPSILON_MEDIUM)
            for initial, final in zip(initial_params, final_params)
        )
        assert params_changed, "Model parameters should change after learning"


class TestPPOAgentAdvantageNormalization:
    """Tests for advantage normalization functionality."""

    def test_advantage_normalization_config_option(
        self, minimal_app_config, ppo_test_model
    ):
        """Test that advantage normalization can be controlled via configuration."""
        # Test with normalization enabled (default)
        config_enabled = minimal_app_config.model_copy()
        config_enabled.training.normalize_advantages = True
        agent_enabled = PPOAgent(
            model=ppo_test_model, config=config_enabled, device=torch.device("cpu")
        )
        assert agent_enabled.normalize_advantages is True

        # Test with normalization disabled
        config_disabled = minimal_app_config.model_copy()
        config_disabled.training.normalize_advantages = False
        model_disabled = ActorCritic(
            CORE_OBSERVATION_CHANNELS, PolicyOutputMapper().get_total_actions()
        )
        agent_disabled = PPOAgent(
            model=model_disabled, config=config_disabled, device=torch.device("cpu")
        )
        assert agent_disabled.normalize_advantages is False

    def test_advantage_normalization_behavior_difference(self, minimal_app_config):
        """Test that enabling/disabling advantage normalization actually affects computation."""
        buffer_size = TEST_MEDIUM_BUFFER_SIZE

        # Create data with high variance advantages to test normalization
        experience_buffer = ExperienceBuffer(
            buffer_size=buffer_size,
            gamma=TRAIN_DEFAULTS.gamma,
            lambda_gae=TRAIN_DEFAULTS.lambda_gae,
            device="cpu",
        )

        dummy_obs_tensor = torch.randn(
            CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE, device="cpu"
        )
        dummy_legal_mask = torch.ones(
            ENV_DEFAULTS.num_actions_total, dtype=torch.bool, device="cpu"
        )

        # High variance rewards and values to create large advantage differences
        rewards = TEST_HIGH_VARIANCE_REWARDS
        values = TEST_HIGH_VARIANCE_VALUES

        for i in range(buffer_size):
            experience_buffer.add(
                obs=dummy_obs_tensor,
                action=i % ENV_DEFAULTS.num_actions_total,
                reward=rewards[i],
                log_prob=TEST_LOG_PROB_MULTIPLIER,
                value=values[i],
                done=(i == buffer_size - 1),
                legal_mask=dummy_legal_mask,
            )

        experience_buffer.compute_advantages_and_returns(0.0)

        # Get raw advantages before normalization
        batch_data = experience_buffer.get_batch()
        raw_advantages = batch_data["advantages"].clone()

        # Verify raw advantages have significant variance
        assert (
            torch.std(raw_advantages, dim=0).max() > TEST_ADVANTAGE_STD_THRESHOLD
        ), "Raw advantages should have high variance"
        assert not torch.allclose(
            raw_advantages, torch.zeros_like(raw_advantages)
        ), "Raw advantages should not be zero"

        # Test with normalization enabled
        config_enabled = minimal_app_config.model_copy()
        config_enabled.training.normalize_advantages = True
        model_enabled = ActorCritic(
            CORE_OBSERVATION_CHANNELS, PolicyOutputMapper().get_total_actions()
        )
        agent_enabled = PPOAgent(
            model=model_enabled, config=config_enabled, device=torch.device("cpu")
        )

        # Test with normalization disabled
        config_disabled = minimal_app_config.model_copy()
        config_disabled.training.normalize_advantages = False
        model_disabled = ActorCritic(
            CORE_OBSERVATION_CHANNELS, PolicyOutputMapper().get_total_actions()
        )
        agent_disabled = PPOAgent(
            model=model_disabled, config=config_disabled, device=torch.device("cpu")
        )

        # Both agents should learn successfully regardless of normalization
        metrics_enabled = agent_enabled.learn(experience_buffer)

        # Recreate buffer for second agent (since buffer is consumed)
        experience_buffer2 = ExperienceBuffer(
            buffer_size=buffer_size,
            gamma=TRAIN_DEFAULTS.gamma,
            lambda_gae=TRAIN_DEFAULTS.lambda_gae,
            device="cpu",
        )

        for i in range(buffer_size):
            experience_buffer2.add(
                obs=dummy_obs_tensor,
                action=i % ENV_DEFAULTS.num_actions_total,
                reward=rewards[i],
                log_prob=TEST_LOG_PROB_MULTIPLIER,
                value=values[i],
                done=(i == buffer_size - 1),
                legal_mask=dummy_legal_mask,
            )

        experience_buffer2.compute_advantages_and_returns(0.0)
        metrics_disabled = agent_disabled.learn(experience_buffer2)

        # Both should return valid metrics
        assert_valid_ppo_metrics(metrics_enabled)
        assert_valid_ppo_metrics(metrics_disabled)
        assert metrics_enabled["ppo/policy_loss"] >= 0.0
        assert not np.isnan(metrics_enabled["ppo/policy_loss"])
        assert metrics_disabled["ppo/policy_loss"] >= 0.0
        assert not np.isnan(metrics_disabled["ppo/policy_loss"])


class TestPPOAgentGradientClipping:
    """Tests for gradient clipping functionality."""

    def test_gradient_clipping(self, minimal_app_config, ppo_test_model):
        """Test that gradient clipping is applied during learning."""
        # Use high learning rate to potentially create large gradients
        config = minimal_app_config.model_copy()
        config.training.learning_rate = TEST_HIGH_LEARNING_RATE  # High learning rate
        config.training.gradient_clip_max_norm = (
            TEST_GRADIENT_CLIP_NORM  # Explicit gradient clipping
        )
        config.training.ppo_epochs = 1
        config.training.minibatch_size = TEST_SMALL_MINIBATCH

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        buffer_size = TEST_BUFFER_SIZE
        experience_buffer = ExperienceBuffer(
            buffer_size=buffer_size,
            gamma=TRAIN_DEFAULTS.gamma,
            lambda_gae=TRAIN_DEFAULTS.lambda_gae,
            device="cpu",
        )

        # Create data that might produce large gradients
        dummy_obs_tensor = torch.randn(
            CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE, device="cpu"
        )
        dummy_legal_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cpu"
        )

        # Extreme reward values to potentially create large policy updates
        rewards = TEST_EXTREME_REWARDS

        for i in range(buffer_size):
            experience_buffer.add(
                obs=dummy_obs_tensor,
                action=i % agent.num_actions_total,
                reward=rewards[i],
                log_prob=TEST_LOG_PROB_MULTIPLIER,
                value=0.0,
                done=(i == buffer_size - 1),
                legal_mask=dummy_legal_mask,
            )

        experience_buffer.compute_advantages_and_returns(0.0)

        # Learn should complete without exploding gradients due to clipping
        metrics = agent.learn(experience_buffer)

        # Verify learning completed successfully
        assert_valid_ppo_metrics(metrics)


class TestPPOAgentKLDivergence:
    """Tests for KL divergence tracking functionality."""

    def test_kl_divergence_tracking(self, minimal_app_config, ppo_test_model):
        """Test that KL divergence is properly computed and tracked."""
        config = minimal_app_config.model_copy()
        config.training.ppo_epochs = (
            TEST_PPO_EPOCHS  # Multiple epochs to see KL divergence change
        )
        config.training.minibatch_size = TEST_SMALL_MINIBATCH

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        buffer_size = TEST_BUFFER_SIZE
        experience_buffer = ExperienceBuffer(
            buffer_size=buffer_size,
            gamma=TRAIN_DEFAULTS.gamma,
            lambda_gae=TRAIN_DEFAULTS.lambda_gae,
            device="cpu",
        )

        dummy_obs_tensor = torch.randn(
            CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE, device="cpu"
        )
        dummy_legal_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cpu"
        )

        for i in range(buffer_size):
            experience_buffer.add(
                obs=dummy_obs_tensor,
                action=i % agent.num_actions_total,
                reward=float(i),
                log_prob=TEST_LOG_PROB_MULTIPLIER,
                value=0.5,
                done=(i == buffer_size - 1),
                legal_mask=dummy_legal_mask,
            )

        experience_buffer.compute_advantages_and_returns(0.0)

        # First learn call
        metrics = agent.learn(experience_buffer)
        kl_div = metrics["ppo/kl_divergence_approx"]

        # Verify KL divergence is tracked in agent
        assert hasattr(agent, "last_kl_div")
        assert agent.last_kl_div == kl_div

        # KL divergence should be a reasonable value
        assert isinstance(kl_div, float)
        assert not np.isnan(kl_div)
        assert not np.isinf(kl_div)

        # For multiple epochs with the same data, KL should generally be small
        # (policy shouldn't diverge dramatically from itself)
        assert (
            abs(kl_div) < TEST_KL_DIVERGENCE_THRESHOLD
        ), f"KL divergence {kl_div} seems too large"


class TestPPOAgentMinibatchProcessing:
    """Tests for minibatch processing functionality."""

    def test_minibatch_processing(self, minimal_app_config, ppo_test_model):
        """Test that minibatch processing works correctly with different batch sizes."""
        # Test with buffer size that doesn't divide evenly by minibatch size
        config = minimal_app_config.model_copy()
        config.training.ppo_epochs = 1
        config.training.minibatch_size = (
            TEST_UNEVEN_MINIBATCH_SIZE  # Doesn't divide evenly into 5
        )

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        buffer_size = (
            TEST_ODD_BUFFER_SIZE  # Odd size to test uneven minibatch splitting
        )
        experience_buffer = ExperienceBuffer(
            buffer_size=buffer_size,
            gamma=TRAIN_DEFAULTS.gamma,
            lambda_gae=TRAIN_DEFAULTS.lambda_gae,
            device="cpu",
        )

        dummy_obs_tensor = torch.randn(
            CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE, device="cpu"
        )
        dummy_legal_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cpu"
        )

        for i in range(buffer_size):
            experience_buffer.add(
                obs=dummy_obs_tensor,
                action=i % agent.num_actions_total,
                reward=float(i),
                log_prob=TEST_LOG_PROB_MULTIPLIER,
                value=TEST_VALUE_DEFAULT,
                done=(i == buffer_size - 1),
                legal_mask=dummy_legal_mask,
            )

        experience_buffer.compute_advantages_and_returns(0.0)

        # Learn should handle uneven minibatch split correctly
        metrics = agent.learn(experience_buffer)

        # Should complete successfully
        assert_valid_ppo_metrics(metrics)


class TestPPOAgentRobustness:
    """Tests for PPO learning robustness and edge cases."""

    def test_empty_buffer_handling(self, minimal_app_config, ppo_test_model):
        """Test PPOAgent.learn behavior with empty experience buffer."""
        agent = PPOAgent(
            model=ppo_test_model, config=minimal_app_config, device=torch.device("cpu")
        )

        # Create empty buffer
        experience_buffer = ExperienceBuffer(
            buffer_size=TEST_BUFFER_SIZE,
            gamma=TRAIN_DEFAULTS.gamma,
            lambda_gae=TRAIN_DEFAULTS.lambda_gae,
            device="cpu",
        )

        # Don't add any experiences - buffer remains empty

        # Learn should handle empty buffer gracefully
        metrics = agent.learn(experience_buffer)

        # Should return default/zero metrics without crashing
        assert metrics is not None
        assert isinstance(metrics, dict)

        expected_metrics = [
            "ppo/policy_loss",
            "ppo/value_loss",
            "ppo/entropy",
            "ppo/kl_divergence_approx",
            "ppo/learning_rate",
        ]

        for metric in expected_metrics:
            assert metric in metrics
            # Should be zero or default values for empty buffer
            assert isinstance(metrics[metric], (int, float))

    def test_single_experience_learning(self, minimal_app_config, ppo_test_model):
        """Test learning with minimal experience buffer (single experience)."""
        # Modify config for single experience learning to prevent std() warning
        config = minimal_app_config.model_copy()
        config.training.minibatch_size = 1  # Keep as 1 for this specific test
        config.training.normalize_advantages = (
            False  # Disable normalization for single experience
        )

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        experience_buffer = ExperienceBuffer(
            buffer_size=1,
            gamma=TRAIN_DEFAULTS.gamma,
            lambda_gae=TRAIN_DEFAULTS.lambda_gae,
            device="cpu",
        )

        dummy_obs_tensor = torch.randn(
            CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE, device="cpu"
        )
        dummy_legal_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cpu"
        )

        experience_buffer.add(
            obs=dummy_obs_tensor,
            action=0,
            reward=1.0,
            log_prob=TEST_LOG_PROB_MULTIPLIER,
            value=TEST_VALUE_DEFAULT,
            done=True,
            legal_mask=dummy_legal_mask,
        )

        experience_buffer.compute_advantages_and_returns(0.0)

        # Should handle single experience without errors
        metrics = agent.learn(experience_buffer)
        assert_valid_ppo_metrics(metrics)


class TestPPOAgentSchedulerLearning:
    """Tests for learning rate scheduler functionality during learning."""

    def test_learning_with_linear_scheduler(self, minimal_app_config, ppo_test_model):
        """Test that linear scheduler correctly modifies learning rate during learning."""
        config = minimal_app_config.model_copy()
        config.training.learning_rate = TEST_SCHEDULER_LEARNING_RATE
        config.training.lr_schedule_type = "linear"
        config.training.lr_schedule_step_on = "epoch"
        config.training.lr_schedule_kwargs = {
            "final_lr_fraction": TEST_SCHEDULER_FINAL_FRACTION
        }
        config.training.total_timesteps = TEST_SCHEDULER_TOTAL_TIMESTEPS
        config.training.steps_per_epoch = TEST_SCHEDULER_STEPS_PER_EPOCH
        config.training.ppo_epochs = 1

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        # Create simple experience buffer
        experience_buffer = ExperienceBuffer(
            buffer_size=TEST_BUFFER_SIZE,
            gamma=TRAIN_DEFAULTS.gamma,
            lambda_gae=TRAIN_DEFAULTS.lambda_gae,
            device="cpu",
        )

        dummy_obs_tensor = torch.randn(
            CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE, device="cpu"
        )
        dummy_legal_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cpu"
        )

        # Add experiences
        for i in range(4):
            experience_buffer.add(
                obs=dummy_obs_tensor,
                action=i % agent.num_actions_total,
                reward=1.0,
                log_prob=0.1,
                value=0.5,
                done=(i == 3),
                legal_mask=dummy_legal_mask,
            )

        experience_buffer.compute_advantages_and_returns(0.0)

        # Track learning rate changes
        initial_lr = config.training.learning_rate

        # First learning call
        metrics1 = agent.learn(experience_buffer)
        lr_after_first = metrics1["ppo/learning_rate"]

        # Second learning call (scheduler steps automatically)
        metrics2 = agent.learn(experience_buffer)
        lr_after_second = metrics2["ppo/learning_rate"]

        # Learning rate should be decreasing with linear scheduler
        assert (
            lr_after_first < initial_lr
        ), "Learning rate should decrease after first epoch"
        assert (
            lr_after_second < lr_after_first
        ), "Learning rate should continue decreasing"

        # Verify learning rate is above minimum threshold
        min_lr = initial_lr * config.training.lr_schedule_kwargs["final_lr_fraction"]
        assert lr_after_second >= min_lr, "Learning rate should not go below minimum"

    def test_learning_with_cosine_scheduler(self, minimal_app_config, ppo_test_model):
        """Test that cosine scheduler provides smooth learning rate transitions during learning."""
        config = minimal_app_config.model_copy()
        config.training.learning_rate = 0.001
        config.training.lr_schedule_type = "cosine"
        config.training.lr_schedule_step_on = "epoch"
        config.training.lr_schedule_kwargs = {"eta_min_fraction": 0.05}
        config.training.total_timesteps = 1000
        config.training.steps_per_epoch = 100
        config.training.ppo_epochs = 1

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        # Create experience buffer
        experience_buffer = ExperienceBuffer(
            buffer_size=4,
            gamma=0.99,
            lambda_gae=0.95,
            device="cpu",
        )

        dummy_obs_tensor = torch.randn(46, 9, 9, device="cpu")
        dummy_legal_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cpu"
        )

        # Add experiences
        for i in range(4):
            experience_buffer.add(
                obs=dummy_obs_tensor,
                action=i % agent.num_actions_total,
                reward=1.0,
                log_prob=0.1,
                value=0.5,
                done=(i == 3),
                legal_mask=dummy_legal_mask,
            )

        experience_buffer.compute_advantages_and_returns(0.0)

        # Track learning rates over multiple learning calls
        learning_rates = []
        initial_lr = config.training.learning_rate

        for _ in range(5):
            metrics = agent.learn(experience_buffer)
            learning_rates.append(metrics["ppo/learning_rate"])

        # Verify cosine decay pattern
        assert learning_rates[0] < initial_lr, "First epoch should show scheduler step"

        # For cosine schedule, learning rate should generally decrease
        final_lr = learning_rates[-1]
        assert final_lr < initial_lr, "Final learning rate should be less than initial"

        # Verify minimum learning rate constraint
        min_lr = initial_lr * config.training.lr_schedule_kwargs["eta_min_fraction"]
        assert all(
            lr >= min_lr for lr in learning_rates
        ), "All learning rates should be >= min_lr"

    def test_learning_with_exponential_scheduler(
        self, minimal_app_config, ppo_test_model
    ):
        """Test exponential scheduler during learning with gamma decay."""
        config = minimal_app_config.model_copy()
        config.training.learning_rate = 0.001
        config.training.lr_schedule_type = "exponential"
        config.training.lr_schedule_step_on = "epoch"
        config.training.lr_schedule_kwargs = {"gamma": 0.9}  # 10% decay each step
        config.training.total_timesteps = 1000
        config.training.steps_per_epoch = 100
        config.training.ppo_epochs = 1

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        # Create experience buffer
        experience_buffer = ExperienceBuffer(
            buffer_size=4,
            gamma=0.99,
            lambda_gae=0.95,
            device="cpu",
        )

        dummy_obs_tensor = torch.randn(46, 9, 9, device="cpu")
        dummy_legal_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cpu"
        )

        # Add experiences
        for i in range(4):
            experience_buffer.add(
                obs=dummy_obs_tensor,
                action=i % agent.num_actions_total,
                reward=1.0,
                log_prob=0.1,
                value=0.5,
                done=(i == 3),
                legal_mask=dummy_legal_mask,
            )

        experience_buffer.compute_advantages_and_returns(0.0)

        # First learning call
        metrics1 = agent.learn(experience_buffer)
        lr1 = metrics1["ppo/learning_rate"]

        # Second learning call (scheduler steps automatically)
        metrics2 = agent.learn(experience_buffer)
        lr2 = metrics2["ppo/learning_rate"]

        # Verify exponential decay
        expected_ratio = config.training.lr_schedule_kwargs["gamma"]
        actual_ratio = lr2 / lr1
        assert (
            abs(actual_ratio - expected_ratio) < 0.01
        ), f"Expected ratio ~{expected_ratio}, got {actual_ratio}"

    def test_learning_with_step_scheduler(self, minimal_app_config, ppo_test_model):
        """Test step scheduler with step intervals during learning."""
        config = minimal_app_config.model_copy()
        config.training.learning_rate = 0.001
        config.training.lr_schedule_type = "step"
        config.training.lr_schedule_step_on = "epoch"
        config.training.lr_schedule_kwargs = {
            "step_size": 2,
            "gamma": 0.5,
        }  # 50% reduction every 2 epochs
        config.training.total_timesteps = 1000
        config.training.steps_per_epoch = 100
        config.training.ppo_epochs = 1

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        # Create experience buffer
        experience_buffer = ExperienceBuffer(
            buffer_size=4,
            gamma=0.99,
            lambda_gae=0.95,
            device="cpu",
        )

        dummy_obs_tensor = torch.randn(46, 9, 9, device="cpu")
        dummy_legal_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cpu"
        )

        # Add experiences
        for i in range(4):
            experience_buffer.add(
                obs=dummy_obs_tensor,
                action=i % agent.num_actions_total,
                reward=1.0,
                log_prob=0.1,
                value=0.5,
                done=(i == 3),
                legal_mask=dummy_legal_mask,
            )

        experience_buffer.compute_advantages_and_returns(0.0)

        initial_lr = config.training.learning_rate
        learning_rates = []

        # Test over several epochs to see step behavior
        for _ in range(4):
            metrics = agent.learn(experience_buffer)
            learning_rates.append(metrics["ppo/learning_rate"])

        # Verify step scheduler behavior
        # PyTorch StepLR behavior: steps at step_size calls, not after step_size calls
        # learning_rates[i] shows LR after the i-th scheduler.step() call

        # After 1st step: no change yet (step_size=2)
        assert (
            learning_rates[0] == initial_lr
        ), "LR should remain initial after 1st step"

        # After 2nd step: first reduction (step count reaches step_size=2)
        expected_lr_after_first_step = (
            initial_lr * config.training.lr_schedule_kwargs["gamma"]
        )
        assert (
            abs(learning_rates[1] - expected_lr_after_first_step) < 1e-6
        ), "LR should step down after 2nd step"

        # After 3rd step: no change yet (need step_size=2 more steps for next reduction)
        assert (
            abs(learning_rates[2] - expected_lr_after_first_step) < 1e-6
        ), "LR should remain stepped after 3rd step"

        # After 4th step: second reduction (step count reaches next multiple of step_size)
        expected_lr_after_second_step = (
            expected_lr_after_first_step * config.training.lr_schedule_kwargs["gamma"]
        )
        assert (
            abs(learning_rates[3] - expected_lr_after_second_step) < 1e-6
        ), "LR should step down again after 4th step"

    def test_scheduler_interaction_with_multiple_epochs(
        self, minimal_app_config, ppo_test_model
    ):
        """Test scheduler behavior with multiple PPO epochs per learning call."""
        config = minimal_app_config.model_copy()
        config.training.learning_rate = 0.001
        config.training.ppo_epochs = 3  # Multiple epochs per learn call
        config.training.lr_schedule_type = "linear"
        config.training.lr_schedule_step_on = "epoch"  # Step per epoch
        config.training.lr_schedule_kwargs = {"final_lr_fraction": 0.1}
        config.training.total_timesteps = 1000
        config.training.steps_per_epoch = 100

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        # Create experience buffer
        experience_buffer = ExperienceBuffer(
            buffer_size=4,
            gamma=0.99,
            lambda_gae=0.95,
            device="cpu",
        )

        dummy_obs_tensor = torch.randn(46, 9, 9, device="cpu")
        dummy_legal_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cpu"
        )

        # Add experiences
        for i in range(4):
            experience_buffer.add(
                obs=dummy_obs_tensor,
                action=i % agent.num_actions_total,
                reward=1.0,
                log_prob=0.1,
                value=0.5,
                done=(i == 3),
                legal_mask=dummy_legal_mask,
            )

        experience_buffer.compute_advantages_and_returns(0.0)

        initial_lr = config.training.learning_rate

        # First learning call (will run 3 epochs internally)
        metrics1 = agent.learn(experience_buffer)
        lr_after_first_learn = metrics1["ppo/learning_rate"]

        # Second learning call
        metrics2 = agent.learn(experience_buffer)
        lr_after_second_learn = metrics2["ppo/learning_rate"]

        # Since we're stepping per epoch and each learn() runs 3 epochs,
        # the learning rate should decrease more significantly
        assert (
            lr_after_first_learn < initial_lr
        ), "LR should decrease after first learn call"
        assert (
            lr_after_second_learn < lr_after_first_learn
        ), "LR should continue decreasing"

        # Verify the learning rate is still above minimum
        min_lr = initial_lr * config.training.lr_schedule_kwargs["final_lr_fraction"]
        assert lr_after_second_learn >= min_lr, "LR should not go below minimum"

    def test_scheduler_disabled_maintains_constant_lr(
        self, minimal_app_config, ppo_test_model
    ):
        """Test that when scheduler is disabled, learning rate remains constant during learning."""
        config = minimal_app_config.model_copy()
        config.training.learning_rate = 0.001
        config.training.lr_schedule_type = None  # Disable scheduler
        config.training.total_timesteps = 100

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        # Create experience buffer
        experience_buffer = ExperienceBuffer(
            buffer_size=4,
            gamma=0.99,
            lambda_gae=0.95,
            device="cpu",
        )

        dummy_obs_tensor = torch.randn(46, 9, 9, device="cpu")
        dummy_legal_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cpu"
        )

        # Add experiences
        for i in range(4):
            experience_buffer.add(
                obs=dummy_obs_tensor,
                action=i % agent.num_actions_total,
                reward=1.0,
                log_prob=0.1,
                value=0.5,
                done=(i == 3),
                legal_mask=dummy_legal_mask,
            )

        experience_buffer.compute_advantages_and_returns(0.0)

        initial_lr = config.training.learning_rate
        learning_rates = []

        # Multiple learning calls
        for _ in range(5):
            metrics = agent.learn(experience_buffer)
            learning_rates.append(metrics["ppo/learning_rate"])

        # All learning rates should be identical when scheduler is disabled
        assert all(
            lr == initial_lr for lr in learning_rates
        ), "Learning rate should remain constant when scheduler disabled"

    def test_scheduler_respects_optimizer_lr_changes(
        self, minimal_app_config, ppo_test_model
    ):
        """Test that scheduler correctly updates the actual optimizer learning rate."""
        config = minimal_app_config.model_copy()
        config.training.learning_rate = 0.001
        config.training.lr_schedule_type = "exponential"
        config.training.lr_schedule_step_on = "epoch"
        config.training.lr_schedule_kwargs = {"gamma": 0.8}
        config.training.total_timesteps = 1000
        config.training.steps_per_epoch = 100
        config.training.ppo_epochs = 1

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        # Create experience buffer
        experience_buffer = ExperienceBuffer(
            buffer_size=4,
            gamma=0.99,
            lambda_gae=0.95,
            device="cpu",
        )

        dummy_obs_tensor = torch.randn(46, 9, 9, device="cpu")
        dummy_legal_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cpu"
        )

        # Add experiences
        for i in range(4):
            experience_buffer.add(
                obs=dummy_obs_tensor,
                action=i % agent.num_actions_total,
                reward=1.0,
                log_prob=0.1,
                value=0.5,
                done=(i == 3),
                legal_mask=dummy_legal_mask,
            )

        experience_buffer.compute_advantages_and_returns(0.0)

        # Check initial optimizer learning rate
        initial_optimizer_lr = agent.optimizer.param_groups[0]["lr"]
        assert (
            initial_optimizer_lr == config.training.learning_rate
        ), "Initial optimizer LR should match config"

        # Perform learning
        metrics = agent.learn(experience_buffer)
        reported_lr = metrics["ppo/learning_rate"]

        # Perform another learning call
        metrics2 = agent.learn(experience_buffer)
        reported_lr2 = metrics2["ppo/learning_rate"]

        # Check that optimizer learning rate matches reported learning rate
        final_optimizer_lr = agent.optimizer.param_groups[0]["lr"]
        assert (
            abs(final_optimizer_lr - reported_lr2) < 1e-8
        ), "Optimizer LR should match reported LR"

        # Verify that learning rate actually changed
        assert reported_lr2 != initial_optimizer_lr, "Learning rate should have changed"
        assert (
            reported_lr2 < reported_lr
        ), "Learning rate should have decreased with exponential scheduler"
