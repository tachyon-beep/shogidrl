"""
Unit tests for PPOAgent in ppo_agent.py
"""

import torch
import numpy as np
import pytest
from typing import List  # Add this import
from keisei.ppo_agent import PPOAgent
from keisei.utils import PolicyOutputMapper
from keisei.shogi import ShogiGame  # Corrected import for ShogiGame
from keisei.shogi.shogi_core_definitions import MoveTuple  # Ensure MoveTuple is imported


def test_ppo_agent_init_and_select_action():
    """Test PPOAgent initializes and select_action returns a valid index."""
    mapper = PolicyOutputMapper()
    agent = PPOAgent(
        input_channels=46, policy_output_mapper=mapper
    )
    obs = np.random.rand(46, 9, 9).astype(np.float32)  # Ensure correct dtype for tensor conversion
    game = ShogiGame()
    legal_moves: List[MoveTuple] = game.get_legal_moves()

    # Ensure there's at least one legal move for the test to proceed
    if not legal_moves:
        # If ShogiGame starts with no legal moves (e.g. before first player acts, or specific setup)
        # and PolicyOutputMapper is populated, we need a known valid move.
        # This is a fallback for test robustness.
        # A standard opening move for Black (Sente)
        default_move: MoveTuple = (6, 7, 5, 7, False)  # Example: Pawn 7g->6g
        if default_move in mapper.move_to_idx:  # Check if mapper knows this move
            legal_moves.append(default_move)
        else:
            # If even this default isn't in mapper, the mapper or test setup is problematic.
            # For now, try to find *any* move the mapper knows to avoid crashing select_action.
            if mapper.idx_to_move:
                legal_moves.append(mapper.idx_to_move[0])
            else:
                pytest.skip("PolicyOutputMapper has no moves, cannot test select_action effectively.")

    if not legal_moves:  # If still no legal_moves, skip test
        pytest.skip("No legal moves could be determined for select_action test.")

    selected_move, idx, log_prob, value = agent.select_action(obs, legal_shogi_moves=legal_moves)
    assert isinstance(idx, int)
    assert 0 <= idx < agent.num_actions_total
    assert isinstance(selected_move, tuple)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_ppo_agent_ppo_update_and_gae():
    """Test PPOAgent's ppo_update and compute_gae methods with dummy data."""
    mapper = PolicyOutputMapper()
    agent = PPOAgent(
        input_channels=46, policy_output_mapper=mapper
    )
    # Create dummy batch
    obs = [torch.zeros((46, 9, 9)) for _ in range(8)]
    actions = [0, 1, 2, 3, 4, 5, 6, 7]
    old_log_probs = [0.0] * 8
    rewards = [1.0] * 8
    values = [0.5] * 8
    dones = [0] * 8
    next_value = 0.0
    returns = agent.compute_gae(rewards, values, dones, next_value)
    advantages = [r - v for r, v in zip(returns, values)]
    # Should not raise error
    agent.ppo_update(
        obs, actions, old_log_probs, returns, advantages, epochs=1, batch_size=4
    )
    assert len(returns) == 8
    assert isinstance(returns[0], float)
