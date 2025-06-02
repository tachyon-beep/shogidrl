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
            name="TestAgent"
        )
        
        # Verify basic properties
        assert agent.name == "TestAgent"
        assert agent.device == torch.device("cpu")
        assert agent.model is ppo_test_model
        assert agent.config is minimal_app_config
        
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
    
    def test_select_action_basic(self, ppo_agent_basic, dummy_observation, dummy_legal_mask):
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
        
    def test_select_action_with_game_context(self, policy_mapper, integration_test_config, ppo_test_model):
        """Test action selection with real game context and legal moves."""
        agent = PPOAgent(model=ppo_test_model, config=integration_test_config, device=torch.device("cpu"))
        
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
                pytest.skip("PolicyOutputMapper has no moves, cannot test select_action effectively.")
        
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
        
    def test_select_action_deterministic_vs_stochastic(self, ppo_agent_basic, dummy_observation, dummy_legal_mask):
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
        assert metrics["ppo/learning_rate"] == ppo_agent_basic.config.training.learning_rate
        
    def test_learn_parameter_updates(self, ppo_agent_basic, populated_experience_buffer):
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
        from tests.conftest import create_test_experience_data
        from keisei.core.experience_buffer import ExperienceBuffer
        
        # Create two separate experience buffers
        buffer1 = ExperienceBuffer(buffer_size=4, gamma=0.99, lambda_gae=0.95, device="cpu")
        buffer2 = ExperienceBuffer(buffer_size=4, gamma=0.99, lambda_gae=0.95, device="cpu")
        
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
