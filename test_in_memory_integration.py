#!/usr/bin/env python3
"""
Test script to verify that the in-memory evaluation integration is working.
"""

import torch
from pathlib import Path
import asyncio
import sys

# Add the keisei package to path
sys.path.insert(0, str(Path(__file__).parent))

from keisei.evaluation.core.model_manager import ModelWeightManager
from keisei.evaluation.strategies.single_opponent import SingleOpponentEvaluator
from keisei.evaluation.core import AgentInfo, OpponentInfo, EvaluationContext, EvaluationConfig


def create_mock_weights():
    """Create mock weights that match ActorCritic architecture."""
    return {
        'conv.weight': torch.randn(16, 46, 3, 3),
        'conv.bias': torch.randn(16),
        'policy_head.weight': torch.randn(4096, 1296),
        'policy_head.bias': torch.randn(4096),
        'value_head.weight': torch.randn(1, 1296),
        'value_head.bias': torch.randn(1),
    }


async def test_in_memory_integration():
    """Test that evaluators can use the ModelWeightManager to create agents from weights."""
    print("Testing in-memory evaluation integration...")
    
    # Create mock weights for agent and opponent
    agent_weights = create_mock_weights()
    opponent_weights = create_mock_weights()
    
    # Create evaluator with minimal config
    config = EvaluationConfig(
        num_games=1,
        opponent_type="ppo_agent",
        evaluation_interval_timesteps=50000,
        enable_periodic_evaluation=False,
        max_moves_per_game=500,
        log_file_path_eval="/tmp/eval.log",
        wandb_log_eval=False,
        elo_registry_path=None,
        agent_id=None,
        opponent_id=None,
        previous_model_pool_size=5,
    )
    
    evaluator = SingleOpponentEvaluator(config)
    
    # Set up the in-memory weights
    evaluator.agent_weights = agent_weights
    evaluator.opponent_weights = opponent_weights
    
    # Create agent and opponent info
    agent_info = AgentInfo(
        name="test_agent",
        model_type="PPOAgent",
        metadata={}
    )
    
    opponent_info = OpponentInfo(
        name="test_opponent",
        type="ppo_agent",
        checkpoint_path=None
    )
    
    # Test agent loading from in-memory weights
    try:
        agent = await evaluator._load_agent_in_memory(agent_info, "cpu", 46)
        print("✓ Successfully created agent from in-memory weights")
        print(f"  Agent type: {type(agent)}")
        print(f"  Agent has model: {hasattr(agent, 'model')}")
        if hasattr(agent, 'model') and agent.model is not None:
            print(f"  Model parameters: {sum(p.numel() for p in agent.model.parameters())}")
    except Exception as e:
        print(f"✗ Failed to create agent from in-memory weights: {e}")
        return False
    
    # Test opponent loading from in-memory weights  
    try:
        opponent = await evaluator._load_opponent_in_memory(opponent_info, "cpu", 46)
        print("✓ Successfully created opponent from in-memory weights")
        print(f"  Opponent type: {type(opponent)}")
        print(f"  Opponent has model: {hasattr(opponent, 'model')}")
        if hasattr(opponent, 'model') and opponent.model is not None:
            print(f"  Model parameters: {sum(p.numel() for p in opponent.model.parameters())}")
    except Exception as e:
        print(f"✗ Failed to create opponent from in-memory weights: {e}")
        return False
    
    print("\n🎉 In-memory evaluation integration test PASSED!")
    print("The evaluators can now successfully create agents from weight dictionaries!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_in_memory_integration())
    if not success:
        sys.exit(1)
