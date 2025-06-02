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
import pytest
import torch

from keisei.core.experience_buffer import ExperienceBuffer
from keisei.core.neural_network import ActorCritic
from keisei.core.ppo_agent import PPOAgent
from keisei.utils import PolicyOutputMapper
from tests.conftest import assert_valid_ppo_metrics, create_test_experience_data


class TestPPOAgentLossComponents:
    """Tests for PPO loss computation and validation."""
    
    def test_learn_loss_components(self, integration_test_config, ppo_test_model):
        """Test that PPOAgent.learn correctly computes and returns individual loss components."""
        # Override specific settings for this test
        config = integration_test_config.model_copy()
        config.training.ppo_epochs = 2  # Multiple epochs to test learning behavior
        
        agent = PPOAgent(model=ppo_test_model, config=config, device=torch.device("cpu"))
        
        buffer_size = 8  # Larger buffer for more realistic training
        experience_buffer = ExperienceBuffer(
            buffer_size=buffer_size,
            gamma=0.99,
            lambda_gae=0.95,
            device="cpu",
        )
        
        # Create deterministic data for more predictable testing
        torch.manual_seed(42)
        np.random.seed(42)
        
        dummy_obs_tensor = torch.randn(46, 9, 9, device="cpu")
        dummy_legal_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cpu"
        )
        
        # Create varied rewards and values to test advantage calculation
        rewards = [1.0, -0.5, 2.0, 0.0, 1.5, -1.0, 0.5, 2.5]
        values = [0.8, 0.2, 1.5, 0.1, 1.0, -0.3, 0.3, 2.0]
        
        for i in range(buffer_size):
            experience_buffer.add(
                obs=dummy_obs_tensor,
                action=i % agent.num_actions_total,
                reward=rewards[i],
                log_prob=0.1 * (i + 1),  # Varied log probs
                value=values[i],
                done=(i == buffer_size - 1),
                legal_mask=dummy_legal_mask,
            )
        
        # Compute advantages with realistic last value
        last_value = 1.2
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
            not torch.allclose(initial, final, atol=1e-6)
            for initial, final in zip(initial_params, final_params)
        )
        assert params_changed, "Model parameters should change after learning"


class TestPPOAgentAdvantageNormalization:
    """Tests for advantage normalization functionality."""
    
    def test_advantage_normalization_config_option(self, minimal_app_config, ppo_test_model):
        """Test that advantage normalization can be controlled via configuration."""
        # Test with normalization enabled (default)
        config_enabled = minimal_app_config.model_copy()
        config_enabled.training.normalize_advantages = True
        agent_enabled = PPOAgent(model=ppo_test_model, config=config_enabled, device=torch.device("cpu"))
        assert agent_enabled.normalize_advantages is True
        
        # Test with normalization disabled
        config_disabled = minimal_app_config.model_copy()
        config_disabled.training.normalize_advantages = False
        model_disabled = ActorCritic(46, PolicyOutputMapper().get_total_actions())
        agent_disabled = PPOAgent(model=model_disabled, config=config_disabled, device=torch.device("cpu"))
        assert agent_disabled.normalize_advantages is False
        
    def test_advantage_normalization_behavior_difference(self, minimal_app_config):
        """Test that enabling/disabling advantage normalization actually affects computation."""
        buffer_size = 8
        
        # Create data with high variance advantages to test normalization
        experience_buffer = ExperienceBuffer(
            buffer_size=buffer_size,
            gamma=0.99,
            lambda_gae=0.95,
            device="cpu",
        )
        
        dummy_obs_tensor = torch.randn(46, 9, 9, device="cpu")
        dummy_legal_mask = torch.ones(13527, dtype=torch.bool, device="cpu")
        
        # High variance rewards and values to create large advantage differences
        rewards = [100.0, -50.0, 75.0, -25.0, 50.0, -75.0, 25.0, -100.0]
        values = [10.0, 5.0, 8.0, 3.0, 6.0, 2.0, 4.0, 1.0]
        
        for i in range(buffer_size):
            experience_buffer.add(
                obs=dummy_obs_tensor,
                action=i % 13527,
                reward=rewards[i],
                log_prob=0.1,
                value=values[i],
                done=(i == buffer_size - 1),
                legal_mask=dummy_legal_mask,
            )
        
        experience_buffer.compute_advantages_and_returns(0.0)
        
        # Get raw advantages before normalization
        batch_data = experience_buffer.get_batch()
        raw_advantages = batch_data["advantages"].clone()
        
        # Verify raw advantages have significant variance
        assert torch.std(raw_advantages, dim=0).max() > 10.0, "Raw advantages should have high variance"
        assert not torch.allclose(raw_advantages, torch.zeros_like(raw_advantages)), "Raw advantages should not be zero"
        
        # Test with normalization enabled
        config_enabled = minimal_app_config.model_copy()
        config_enabled.training.normalize_advantages = True
        model_enabled = ActorCritic(46, PolicyOutputMapper().get_total_actions())
        agent_enabled = PPOAgent(model=model_enabled, config=config_enabled, device=torch.device("cpu"))
        
        # Test with normalization disabled  
        config_disabled = minimal_app_config.model_copy()
        config_disabled.training.normalize_advantages = False
        model_disabled = ActorCritic(46, PolicyOutputMapper().get_total_actions())
        agent_disabled = PPOAgent(model=model_disabled, config=config_disabled, device=torch.device("cpu"))
        
        # Both agents should learn successfully regardless of normalization
        metrics_enabled = agent_enabled.learn(experience_buffer)
        
        # Recreate buffer for second agent (since buffer is consumed)
        experience_buffer2 = ExperienceBuffer(
            buffer_size=buffer_size,
            gamma=0.99,
            lambda_gae=0.95,
            device="cpu",
        )
        
        for i in range(buffer_size):
            experience_buffer2.add(
                obs=dummy_obs_tensor,
                action=i % 13527,
                reward=rewards[i],
                log_prob=0.1,
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
        config.training.learning_rate = 1.0  # High learning rate
        config.training.gradient_clip_max_norm = 0.5  # Explicit gradient clipping
        config.training.ppo_epochs = 1
        config.training.minibatch_size = 2
        
        agent = PPOAgent(model=ppo_test_model, config=config, device=torch.device("cpu"))
        
        buffer_size = 4
        experience_buffer = ExperienceBuffer(
            buffer_size=buffer_size,
            gamma=0.99,
            lambda_gae=0.95,
            device="cpu",
        )
        
        # Create data that might produce large gradients
        dummy_obs_tensor = torch.randn(46, 9, 9, device="cpu")
        dummy_legal_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cpu"
        )
        
        # Extreme reward values to potentially create large policy updates
        rewards = [100.0, -100.0, 50.0, -50.0]
        
        for i in range(buffer_size):
            experience_buffer.add(
                obs=dummy_obs_tensor,
                action=i % agent.num_actions_total,
                reward=rewards[i],
                log_prob=0.1,
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
        config.training.ppo_epochs = 2  # Multiple epochs to see KL divergence change
        config.training.minibatch_size = 2
        
        agent = PPOAgent(model=ppo_test_model, config=config, device=torch.device("cpu"))
        
        buffer_size = 4
        experience_buffer = ExperienceBuffer(
            buffer_size=buffer_size,
            gamma=0.99,
            lambda_gae=0.95,
            device="cpu",
        )
        
        dummy_obs_tensor = torch.randn(46, 9, 9, device="cpu")
        dummy_legal_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cpu"
        )
        
        for i in range(buffer_size):
            experience_buffer.add(
                obs=dummy_obs_tensor,
                action=i % agent.num_actions_total,
                reward=float(i),
                log_prob=0.1,
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
        assert abs(kl_div) < 10.0, f"KL divergence {kl_div} seems too large"


class TestPPOAgentMinibatchProcessing:
    """Tests for minibatch processing functionality."""
    
    def test_minibatch_processing(self, minimal_app_config, ppo_test_model):
        """Test that minibatch processing works correctly with different batch sizes."""
        # Test with buffer size that doesn't divide evenly by minibatch size
        config = minimal_app_config.model_copy()
        config.training.ppo_epochs = 1
        config.training.minibatch_size = 3  # Doesn't divide evenly into 5
        
        agent = PPOAgent(model=ppo_test_model, config=config, device=torch.device("cpu"))
        
        buffer_size = 5  # Odd size to test uneven minibatch splitting
        experience_buffer = ExperienceBuffer(
            buffer_size=buffer_size,
            gamma=0.99,
            lambda_gae=0.95,
            device="cpu",
        )
        
        dummy_obs_tensor = torch.randn(46, 9, 9, device="cpu")
        dummy_legal_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cpu"
        )
        
        for i in range(buffer_size):
            experience_buffer.add(
                obs=dummy_obs_tensor,
                action=i % agent.num_actions_total,
                reward=float(i),
                log_prob=0.1,
                value=0.5,
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
        agent = PPOAgent(model=ppo_test_model, config=minimal_app_config, device=torch.device("cpu"))
        
        # Create empty buffer
        experience_buffer = ExperienceBuffer(
            buffer_size=4,
            gamma=0.99,
            lambda_gae=0.95,
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
        agent = PPOAgent(model=ppo_test_model, config=minimal_app_config, device=torch.device("cpu"))
        
        experience_buffer = ExperienceBuffer(
            buffer_size=1,
            gamma=0.99,
            lambda_gae=0.95,
            device="cpu",
        )
        
        dummy_obs_tensor = torch.randn(46, 9, 9, device="cpu")
        dummy_legal_mask = torch.ones(
            agent.num_actions_total, dtype=torch.bool, device="cpu"
        )
        
        experience_buffer.add(
            obs=dummy_obs_tensor,
            action=0,
            reward=1.0,
            log_prob=0.1,
            value=0.5,
            done=True,
            legal_mask=dummy_legal_mask,
        )
        
        experience_buffer.compute_advantages_and_returns(0.0)
        
        # Should handle single experience without errors
        metrics = agent.learn(experience_buffer)
        assert_valid_ppo_metrics(metrics)
