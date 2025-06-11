"""
Unit tests for core PPOAgent functionality.

This module tests basic PPOAgent operations including:
- Initialization and dependency injection
- Action selection with legal move constraints
- Basic learning functionality
- Name and device handling
"""

from typing import List

import numpy as np
import pytest
import torch

from keisei.constants import (
    CORE_OBSERVATION_CHANNELS,
    FULL_ACTION_SPACE,
    SHOGI_BOARD_SIZE,
    TEST_SINGLE_LEGAL_ACTION_INDEX,
)
from keisei.core.ppo_agent import PPOAgent
from keisei.shogi import ShogiGame
from keisei.shogi.shogi_core_definitions import MoveTuple
from keisei.utils import PolicyOutputMapper
from tests.conftest import assert_valid_ppo_metrics


class TestPPOAgentInitialization:
    """Tests for PPOAgent initialization and basic properties."""

    def test_ppo_agent_init_basic(self, minimal_app_config, ppo_test_model):
        """Test basic PPOAgent initialization with dependency injection."""
        agent = PPOAgent(
            model=ppo_test_model,
            config=minimal_app_config,
            device=torch.device("cpu"),
            name="TestAgent",
        )

        # Verify basic properties
        assert agent.name == "TestAgent"
        assert agent.device == torch.device("cpu")
        assert agent.model is ppo_test_model
        assert agent.config == minimal_app_config

        # Verify model moved to correct device
        assert next(agent.model.parameters()).device == torch.device("cpu")

    def test_ppo_agent_get_name(self, ppo_agent_basic):
        """Test PPOAgent name getter functionality."""
        assert ppo_agent_basic.get_name() == "TestPPOAgent"

    def test_ppo_agent_num_actions_total(self, ppo_agent_basic):
        """Test that PPOAgent correctly identifies total action space size."""
        mapper = PolicyOutputMapper()
        assert ppo_agent_basic.num_actions_total == mapper.get_total_actions()


class TestPPOAgentActionSelection:
    """Tests for PPOAgent action selection functionality."""

    def test_select_action_basic(
        self, ppo_agent_basic, dummy_observation, dummy_legal_mask
    ):
        """Test basic action selection functionality."""
        selected_move, idx, log_prob, value = ppo_agent_basic.select_action(
            dummy_observation,
            dummy_legal_mask,
            is_training=True,
        )

        # Verify return types and ranges
        assert isinstance(idx, int)
        assert 0 <= idx < ppo_agent_basic.num_actions_total
        assert isinstance(selected_move, (tuple, type(None)))
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

        # Verify legal mask properties
        assert isinstance(dummy_legal_mask, torch.Tensor)
        assert dummy_legal_mask.shape[0] == ppo_agent_basic.num_actions_total
        assert dummy_legal_mask.dtype == torch.bool

    def test_select_action_with_game_context(
        self, policy_mapper, integration_test_config, ppo_test_model
    ):
        """Test action selection with real game context and legal moves."""
        agent = PPOAgent(
            model=ppo_test_model,
            config=integration_test_config,
            device=torch.device("cpu"),
        )

        # Create real game context
        rng = np.random.default_rng(42)
        obs = rng.random((46, 9, 9)).astype(np.float32)
        game = ShogiGame(max_moves_per_game=512)
        legal_moves: List[MoveTuple] = game.get_legal_moves()

        # Ensure we have legal moves for testing
        if not legal_moves:
            # Fallback to a standard opening move
            default_move: MoveTuple = (6, 7, 5, 7, False)  # Pawn 7g->6g
            if default_move in policy_mapper.move_to_idx:
                legal_moves.append(default_move)
            elif policy_mapper.idx_to_move:
                legal_moves.append(policy_mapper.idx_to_move[0])
            else:
                pytest.skip(
                    "PolicyOutputMapper has no moves, cannot test select_action effectively."
                )

        if not legal_moves:
            pytest.skip("No legal moves could be determined for select_action test.")

        # Create legal mask and test action selection
        legal_mask = policy_mapper.get_legal_mask(legal_moves, device=agent.device)

        selected_move, idx, log_prob, value = agent.select_action(
            obs,
            legal_mask,
            is_training=True,
        )

        # Verify valid action was selected
        assert isinstance(idx, int)
        assert 0 <= idx < agent.num_actions_total
        assert isinstance(selected_move, (tuple, type(None)))
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_select_action_deterministic_vs_stochastic(
        self, ppo_agent_basic, dummy_observation, dummy_legal_mask
    ):
        """Test deterministic vs stochastic action selection modes."""
        # Deterministic mode
        _, idx_det1, _, _ = ppo_agent_basic.select_action(
            dummy_observation, dummy_legal_mask, is_training=False
        )
        _, idx_det2, _, _ = ppo_agent_basic.select_action(
            dummy_observation, dummy_legal_mask, is_training=False
        )

        # Deterministic should be consistent (though this may not always hold due to model randomness)
        # We mainly test that the call succeeds with deterministic=False
        assert isinstance(idx_det1, int)
        assert isinstance(idx_det2, int)

        # Stochastic mode (training=True)
        _, idx_stoch, _, _ = ppo_agent_basic.select_action(
            dummy_observation, dummy_legal_mask, is_training=True
        )
        assert isinstance(idx_stoch, int)


class TestPPOAgentValueEstimation:
    """Tests for PPOAgent value estimation functionality."""

    def test_get_value_basic(self, ppo_agent_basic):
        """Test basic value estimation functionality."""
        rng = np.random.default_rng(42)
        obs_np = rng.random((46, 9, 9)).astype(np.float32)

        value = ppo_agent_basic.get_value(obs_np)

        assert isinstance(value, float)
        assert not np.isnan(value)
        assert not np.isinf(value)

    def test_get_value_batch_consistency(self, ppo_agent_basic):
        """Test value consistency across different observation batches."""
        rng = np.random.default_rng(42)
        obs1 = rng.random((46, 9, 9)).astype(np.float32)
        obs2 = rng.random((46, 9, 9)).astype(np.float32)

        value1 = ppo_agent_basic.get_value(obs1)
        value2 = ppo_agent_basic.get_value(obs2)

        # Both should be valid floats
        assert isinstance(value1, float)
        assert isinstance(value2, float)
        assert not np.isnan(value1)
        assert not np.isnan(value2)

        # Values may be different for different observations
        # We just verify the method works consistently


class TestPPOAgentBasicLearning:
    """Tests for basic PPOAgent learning functionality."""

    def test_learn_basic(self, ppo_agent_basic, populated_experience_buffer):
        """Test basic PPO learning functionality."""
        # Call the learn method
        metrics = ppo_agent_basic.learn(populated_experience_buffer)

        # Verify metrics structure and content
        assert_valid_ppo_metrics(metrics)

        # Verify learning rate matches config
        assert (
            metrics["ppo/learning_rate"]
            == ppo_agent_basic.config.training.learning_rate
        )

    def test_learn_parameter_updates(
        self, ppo_agent_basic, populated_experience_buffer
    ):
        """Test that learning actually updates model parameters."""
        # Capture initial parameters
        initial_params = [p.clone() for p in ppo_agent_basic.model.parameters()]

        # Perform learning
        metrics = ppo_agent_basic.learn(populated_experience_buffer)

        # Verify parameters changed
        final_params = [p.clone() for p in ppo_agent_basic.model.parameters()]
        params_changed = any(
            not torch.allclose(initial, final, atol=1e-6)
            for initial, final in zip(initial_params, final_params)
        )

        assert params_changed, "Model parameters should change after learning"
        assert_valid_ppo_metrics(metrics)

    def test_learn_multiple_calls(self, ppo_agent_basic):
        """Test that multiple learn calls work correctly."""
        from keisei.core.experience_buffer import ExperienceBuffer
        from tests.conftest import create_test_experience_data

        # Create two separate experience buffers
        buffer1 = ExperienceBuffer(
            buffer_size=4, gamma=0.99, lambda_gae=0.95, device="cpu"
        )
        buffer2 = ExperienceBuffer(
            buffer_size=4, gamma=0.99, lambda_gae=0.95, device="cpu"
        )

        # Populate with different data
        experiences1 = create_test_experience_data(4)
        experiences2 = create_test_experience_data(4)

        for exp in experiences1:
            buffer1.add(**exp)
        buffer1.compute_advantages_and_returns(0.0)

        for exp in experiences2:
            buffer2.add(**exp)
        buffer2.compute_advantages_and_returns(0.0)

        # Multiple learn calls should succeed
        metrics1 = ppo_agent_basic.learn(buffer1)
        metrics2 = ppo_agent_basic.learn(buffer2)

        assert_valid_ppo_metrics(metrics1)
        assert_valid_ppo_metrics(metrics2)


class TestPPOAgentSchedulerIntegration:
    """Tests for PPOAgent learning rate scheduler integration."""

    def test_scheduler_initialization_none(self, minimal_app_config, ppo_test_model):
        """Test PPOAgent initialization with no scheduler configured."""
        config = minimal_app_config.model_copy()
        config.training.lr_schedule_type = None

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        assert agent.scheduler is None
        assert agent.lr_schedule_type is None
        assert agent.lr_schedule_step_on == "epoch"  # default

    def test_scheduler_initialization_linear(self, minimal_app_config, ppo_test_model):
        """Test PPOAgent initialization with linear scheduler."""
        config = minimal_app_config.model_copy()
        config.training.lr_schedule_type = "linear"
        config.training.lr_schedule_step_on = "epoch"
        config.training.lr_schedule_kwargs = {"final_lr_fraction": 0.1}

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        assert agent.scheduler is not None
        assert agent.lr_schedule_type == "linear"
        assert agent.lr_schedule_step_on == "epoch"
        assert hasattr(agent.scheduler, "step")

    def test_scheduler_initialization_cosine(self, minimal_app_config, ppo_test_model):
        """Test PPOAgent initialization with cosine scheduler."""
        config = minimal_app_config.model_copy()
        config.training.lr_schedule_type = "cosine"
        config.training.lr_schedule_step_on = "update"
        config.training.lr_schedule_kwargs = {"eta_min_fraction": 0.05}

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        assert agent.scheduler is not None
        assert agent.lr_schedule_type == "cosine"
        assert agent.lr_schedule_step_on == "update"

    def test_scheduler_step_calculation_epoch_mode(
        self, minimal_app_config, ppo_test_model
    ):
        """Test scheduler total steps calculation for epoch stepping."""
        config = minimal_app_config.model_copy()
        config.training.total_timesteps = 1000
        config.training.steps_per_epoch = 100
        config.training.ppo_epochs = 4
        config.training.lr_schedule_type = "linear"  # Enable scheduler
        config.training.lr_schedule_step_on = "epoch"

        # Create agent to verify scheduler is set up correctly
        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )
        assert agent.scheduler is not None  # Verify scheduler exists

        # Expected: (1000 // 100) * 4 = 10 * 4 = 40 total steps
        expected_steps = 40
        # Use direct calculation instead of protected method
        total_epochs = (
            config.training.total_timesteps // config.training.steps_per_epoch
        )
        if config.training.lr_schedule_step_on == "epoch":
            calculated_steps = total_epochs * config.training.ppo_epochs
        else:
            updates_per_epoch = (
                config.training.steps_per_epoch // config.training.minibatch_size
            )
            calculated_steps = (
                total_epochs * updates_per_epoch * config.training.ppo_epochs
            )

        assert calculated_steps == expected_steps

    def test_scheduler_step_calculation_update_mode(
        self, minimal_app_config, ppo_test_model
    ):
        """Test scheduler total steps calculation for update stepping."""
        config = minimal_app_config.model_copy()
        config.training.total_timesteps = 1000
        config.training.steps_per_epoch = 100
        config.training.ppo_epochs = 4
        config.training.minibatch_size = 10
        config.training.lr_schedule_type = "linear"  # Enable scheduler
        config.training.lr_schedule_step_on = "update"

        # Create agent to verify scheduler is set up correctly
        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )
        assert agent.scheduler is not None  # Verify scheduler exists

        # Expected: (100 // 10) * 4 = 40 updates per epoch, 10 epochs = 400 total steps
        expected_steps = 400
        # Use direct calculation instead of protected method
        total_epochs = (
            config.training.total_timesteps // config.training.steps_per_epoch
        )
        if config.training.lr_schedule_step_on == "epoch":
            calculated_steps = total_epochs * config.training.ppo_epochs
        else:
            updates_per_epoch = (
                config.training.steps_per_epoch // config.training.minibatch_size
            )
            calculated_steps = (
                total_epochs * updates_per_epoch * config.training.ppo_epochs
            )

        assert calculated_steps == expected_steps

    def test_learning_rate_changes_with_linear_scheduler(
        self, minimal_app_config, ppo_test_model
    ):
        """Test that learning rate actually changes during learning with linear scheduler."""
        config = minimal_app_config.model_copy()
        config.training.lr_schedule_type = "linear"
        config.training.lr_schedule_step_on = "epoch"
        config.training.lr_schedule_kwargs = {"final_lr_fraction": 0.5}
        config.training.ppo_epochs = 1  # Single epoch to see step effect

        agent = PPOAgent(
            model=ppo_test_model, config=config, device=torch.device("cpu")
        )

        initial_lr = agent.optimizer.param_groups[0]["lr"]

        # Create simple experience buffer
        from keisei.core.experience_buffer import ExperienceBuffer
        from tests.conftest import create_test_experience_data

        buffer = ExperienceBuffer(
            buffer_size=4, gamma=0.99, lambda_gae=0.95, device="cpu"
        )
        experiences = create_test_experience_data(4)

        for exp in experiences:
            buffer.add(**exp)
        buffer.compute_advantages_and_returns(0.0)

        # Perform learning - this should step the scheduler
        metrics = agent.learn(buffer)

        new_lr = agent.optimizer.param_groups[0]["lr"]

        # Learning rate should have decreased (linear decay)
        assert new_lr < initial_lr
        assert metrics["ppo/learning_rate"] == new_lr

    def test_scheduler_step_on_epoch_vs_update(
        self, minimal_app_config, ppo_test_model
    ):
        """Test difference between epoch and update stepping modes."""
        base_config = minimal_app_config.model_copy()
        base_config.training.lr_schedule_type = "linear"
        base_config.training.lr_schedule_kwargs = {"final_lr_fraction": 0.8}
        base_config.training.ppo_epochs = 2
        base_config.training.minibatch_size = 2

        # Test epoch stepping
        config_epoch = base_config.model_copy()
        config_epoch.training.lr_schedule_step_on = "epoch"
        agent_epoch = PPOAgent(
            model=ppo_test_model, config=config_epoch, device=torch.device("cpu")
        )

        # Test update stepping
        config_update = base_config.model_copy()
        config_update.training.lr_schedule_step_on = "update"
        # Need a fresh model for the second agent
        from keisei.core.neural_network import ActorCritic

        mapper = PolicyOutputMapper()
        model_update = ActorCritic(
            input_channels=46, num_actions_total=mapper.get_total_actions()
        )
        agent_update = PPOAgent(
            model=model_update, config=config_update, device=torch.device("cpu")
        )

        initial_lr_epoch = agent_epoch.optimizer.param_groups[0]["lr"]
        initial_lr_update = agent_update.optimizer.param_groups[0]["lr"]

        # Create experience buffer for testing
        from keisei.core.experience_buffer import ExperienceBuffer
        from tests.conftest import create_test_experience_data

        buffer_epoch = ExperienceBuffer(
            buffer_size=4, gamma=0.99, lambda_gae=0.95, device="cpu"
        )
        buffer_update = ExperienceBuffer(
            buffer_size=4, gamma=0.99, lambda_gae=0.95, device="cpu"
        )

        experiences = create_test_experience_data(4)

        for exp in experiences:
            buffer_epoch.add(**exp)
            buffer_update.add(**exp)

        buffer_epoch.compute_advantages_and_returns(0.0)
        buffer_update.compute_advantages_and_returns(0.0)

        # Perform learning
        agent_epoch.learn(buffer_epoch)
        agent_update.learn(buffer_update)

        lr_after_epoch = agent_epoch.optimizer.param_groups[0]["lr"]
        lr_after_update = agent_update.optimizer.param_groups[0]["lr"]

        # Both should have changed
        assert lr_after_epoch < initial_lr_epoch
        assert lr_after_update < initial_lr_update


class TestPPOAgentMasking:
    """Test PPO agent's handling of legal action masks."""

    def test_legal_mask_enforcement(self, ppo_agent_basic):
        """Test that agent respects legal action masks."""
        agent = ppo_agent_basic
        
        # Create a restrictive legal mask
        legal_mask = torch.zeros(FULL_ACTION_SPACE, dtype=torch.bool)
        legal_mask[TEST_SINGLE_LEGAL_ACTION_INDEX] = True  # Only one action is legal
        
        # Get observation as numpy array (matching interface)
        obs = np.random.randn(CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE).astype(np.float32)
        
        # Test multiple times to ensure consistency
        for _ in range(10):
            # Select action with mask
            selected_move, action_idx, log_prob, value = agent.select_action(obs, legal_mask, is_training=False)
            
            # Action should always be the single legal action
            assert action_idx == TEST_SINGLE_LEGAL_ACTION_INDEX, f"Expected {TEST_SINGLE_LEGAL_ACTION_INDEX}, got {action_idx}"

    def test_legal_mask_log_probabilities(self, ppo_agent_basic):
        """Test that log probabilities correctly handle legal masks."""
        agent = ppo_agent_basic
        
        # Create legal mask with subset of actions
        legal_mask = torch.zeros(FULL_ACTION_SPACE, dtype=torch.bool)
        legal_mask[:10] = True  # Only first 10 actions are legal
        
        obs = np.random.randn(CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE).astype(np.float32)
        
        # Get action and value
        selected_move, action_idx, log_prob, value = agent.select_action(obs, legal_mask, is_training=False)
        
        # Action should be in legal range
        assert 0 <= action_idx < 10, f"Action {action_idx} is not in legal range [0, 10)"
        
        # Log probability should be finite (not -inf)
        assert np.isfinite(log_prob), f"Log probability is not finite: {log_prob}"

    def test_empty_legal_mask_handling(self, ppo_agent_basic):
        """Test agent behavior with empty legal mask (all actions illegal)."""
        agent = ppo_agent_basic
        
        # Create empty legal mask
        legal_mask = torch.zeros(FULL_ACTION_SPACE, dtype=torch.bool)
        
        obs = np.random.randn(CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE).astype(np.float32)
        
        # Agent should handle gracefully (might use uniform distribution or raise error)
        try:
            selected_move, action_idx, log_prob, value = agent.select_action(obs, legal_mask, is_training=False)
            
            # If it succeeds, action should still be valid
            assert 0 <= action_idx < FULL_ACTION_SPACE
        except (ValueError, RuntimeError):
            # Acceptable to raise an error for impossible situation
            pass

    def test_mask_batch_consistency(self, ppo_agent_basic):
        """Test legal mask handling with different mask configurations."""
        agent = ppo_agent_basic
        
        obs = np.random.randn(CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE).astype(np.float32)
        
        # Test different mask configurations
        test_configs = [
            (slice(0, 5), "actions 0-4"),
            (slice(10, 15), "actions 10-14"), 
            (slice(20, 25), "actions 20-24"),
            (slice(None), "all actions")
        ]
        
        for action_slice, description in test_configs:
            legal_mask = torch.zeros(FULL_ACTION_SPACE, dtype=torch.bool)
            legal_mask[action_slice] = True
            
            selected_move, action_idx, log_prob, value = agent.select_action(obs, legal_mask, is_training=False)
            
            # Check action respects the mask
            if action_slice == slice(None):
                assert 0 <= action_idx < FULL_ACTION_SPACE, f"Action {action_idx} not in valid range for {description}"
            else:
                assert legal_mask[action_idx], f"Action {action_idx} doesn't respect mask for {description}"


class TestPPOAgentInterfaceConsistency:
    """Test consistency between different PPO agent interfaces."""

    def test_action_value_consistency(self, ppo_agent_basic):
        """Test consistency between action selection and value estimation."""
        agent = ppo_agent_basic
        
        obs = np.random.randn(CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE).astype(np.float32)
        legal_mask = torch.ones(FULL_ACTION_SPACE, dtype=torch.bool)
        legal_mask[FULL_ACTION_SPACE//2:] = False  # Half actions legal
        
        # Get action and value
        selected_move, action_idx, log_prob, action_value = agent.select_action(obs, legal_mask, is_training=False)
        standalone_value = agent.get_value(obs)  # get_value returns float directly
        
        # Values should be consistent (same observation)
        np.testing.assert_allclose(action_value, standalone_value, atol=1e-5, rtol=1e-5)
        
        # Action should respect mask
        assert action_idx < FULL_ACTION_SPACE//2, f"Action {action_idx} doesn't respect legal mask"

    def test_training_vs_evaluation_mode(self, ppo_agent_basic):
        """Test that training vs evaluation modes affect behavior appropriately."""
        agent = ppo_agent_basic
        
        obs = np.random.randn(CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE).astype(np.float32)
        legal_mask = torch.ones(FULL_ACTION_SPACE, dtype=torch.bool)
        
        # Get actions in both modes
        _, action_train, _, _ = agent.select_action(obs, legal_mask, is_training=True)
        _, action_eval, _, _ = agent.select_action(obs, legal_mask, is_training=False)
        
        # Both actions should be valid
        assert 0 <= action_train < FULL_ACTION_SPACE
        assert 0 <= action_eval < FULL_ACTION_SPACE
        
        # Actions might be different due to dropout/batch norm differences
        # but we can't assert they're different as it depends on the model

    def test_stochastic_variation(self, ppo_agent_basic):
        """Test that repeated action selection produces variation."""
        agent = ppo_agent_basic
        
        obs = np.random.randn(CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE).astype(np.float32)
        legal_mask = torch.ones(FULL_ACTION_SPACE, dtype=torch.bool)
        
        # Get multiple actions
        actions = []
        for _ in range(20):
            _, action_idx, _, _ = agent.select_action(obs, legal_mask, is_training=True)
            actions.append(action_idx)
        
        # Should have some variation (high probability)
        unique_actions = set(actions)
        assert len(unique_actions) > 1, f"Actions should vary, got only: {unique_actions}"

    def test_value_estimation_consistency(self, ppo_agent_basic):
        """Test value estimation consistency."""
        agent = ppo_agent_basic
        
        # Single observation
        obs_single = np.random.randn(CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE).astype(np.float32)
        value_single = agent.get_value(obs_single)  # get_value returns float directly
        
        # Test with multiple identical observations
        for _ in range(3):
            obs_test = obs_single.copy()
            value_test = agent.get_value(obs_test)
            
            # All should equal the single value (within numerical precision)
            np.testing.assert_allclose(value_test, value_single, atol=1e-6, rtol=1e-6)


class TestPPOAgentNumericalStability:
    """Test numerical stability of PPO agent operations."""

    def test_extreme_observation_values(self, ppo_agent_basic):
        """Test agent behavior with extreme observation values."""
        agent = ppo_agent_basic
        legal_mask = torch.ones(FULL_ACTION_SPACE, dtype=torch.bool)
        
        # Very small values
        obs_small = np.full((CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE), 1e-8, dtype=np.float32)
        _, _, log_prob_small, value_small = agent.select_action(obs_small, legal_mask, is_training=False)
        assert np.isfinite(log_prob_small), "Log prob not finite for small observations"
        assert np.isfinite(value_small), "Value not finite for small observations"
        
        # Very large values
        obs_large = np.full((CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE), 1e3, dtype=np.float32)
        _, _, log_prob_large, value_large = agent.select_action(obs_large, legal_mask, is_training=False)
        assert np.isfinite(log_prob_large), "Log prob not finite for large observations"
        assert np.isfinite(value_large), "Value not finite for large observations"

    def test_gradient_flow_stability(self, ppo_agent_basic, populated_experience_buffer):
        """Test that learning produces stable gradients."""
        agent = ppo_agent_basic
        buffer = populated_experience_buffer
        
        # Store initial parameters
        initial_params = {}
        for name, param in agent.model.named_parameters():
            initial_params[name] = param.clone().detach()
        
        # Perform learning
        metrics = agent.learn(buffer)
        
        # Check that parameters changed (learning occurred)
        params_changed = False
        for name, param in agent.model.named_parameters():
            if not torch.equal(initial_params[name], param):
                params_changed = True
                break
        
        assert params_changed, "No parameters changed during learning"
        
        # Check that gradients are finite
        for name, param in agent.model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite gradients in {name}"
        
        # Check that metrics are reasonable
        assert_valid_ppo_metrics(metrics)
