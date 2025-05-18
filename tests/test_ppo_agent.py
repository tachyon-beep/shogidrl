"""
Unit tests for PPOAgent in ppo_agent.py
"""

import torch
import numpy as np
import pytest
from typing import List  # Add this import
from keisei.ppo_agent import PPOAgent
from keisei.utils import PolicyOutputMapper
from keisei.experience_buffer import ExperienceBuffer  # Added import
from keisei.shogi import ShogiGame  # Corrected import for ShogiGame
from keisei.shogi.shogi_core_definitions import (
    MoveTuple,
)  # Ensure MoveTuple is imported


def test_ppo_agent_init_and_select_action():
    """Test PPOAgent initializes and select_action returns a valid index."""
    mapper = PolicyOutputMapper()
    agent = PPOAgent(input_channels=46, policy_output_mapper=mapper)
    obs = np.random.rand(46, 9, 9).astype(
        np.float32
    )  # Ensure correct dtype for tensor conversion
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
                pytest.skip(
                    "PolicyOutputMapper has no moves, cannot test select_action effectively."
                )

    if not legal_moves:  # If still no legal_moves, skip test
        pytest.skip("No legal moves could be determined for select_action test.")

    selected_move, idx, log_prob, value = agent.select_action(
        obs, legal_shogi_moves=legal_moves
    )
    assert isinstance(idx, int)
    assert 0 <= idx < agent.num_actions_total
    assert isinstance(selected_move, tuple)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_ppo_agent_learn():
    """Test PPOAgent's learn method with dummy data from an ExperienceBuffer."""
    mapper = PolicyOutputMapper()
    agent = PPOAgent(
        input_channels=46,
        policy_output_mapper=mapper,
        ppo_epochs=1,  # Keep epochs low for faster test
        minibatch_size=2,  # Keep minibatch size low
    )

    buffer_size = 4  # Small buffer for testing
    experience_buffer = ExperienceBuffer(
        buffer_size=buffer_size,
        gamma=0.99,
        lambda_gae=0.95,
        device="cpu",  # Use CPU for testing
    )

    # Populate buffer with some dummy data
    dummy_obs = np.random.rand(46, 9, 9).astype(np.float32)
    for i in range(buffer_size):
        experience_buffer.add(
            obs=dummy_obs,
            action=i % agent.num_actions_total,  # Cycle through some actions
            reward=float(i),
            log_prob=0.1 * i,
            value=0.5 * i,
            done=(i == buffer_size - 1),  # Last one is 'done'
        )

    assert len(experience_buffer) == buffer_size

    # Compute advantages and returns
    last_value = 0.0  # Assuming terminal state after buffer is full for simplicity
    experience_buffer.compute_advantages_and_returns(last_value)

    # Call the learn method
    try:
        avg_policy_loss, avg_value_loss, avg_entropy = agent.learn(experience_buffer)
        # Check if losses are returned and are floats (or can be zero)
        assert isinstance(avg_policy_loss, float)
        assert isinstance(avg_value_loss, float)
        assert isinstance(avg_entropy, float)
    except Exception as e:
        pytest.fail(f"agent.learn() raised an exception: {e}")

    # Optionally, check if buffer is cleared after learn (if that's the intended behavior of learn or a subsequent step)
    # For now, just ensuring it runs and returns losses.
    # If learn is supposed to clear the buffer, add:
    # assert len(experience_buffer) == 0
    # However, current PPO plan has clear after learn in train.py, not in agent.learn() itself.


# Further tests could include:
# - Testing select_action in eval mode (is_training=False)
# - Testing model saving and loading (if not covered elsewhere)
# - More specific checks on loss values if expected behavior is known for dummy data
#   (though this can be complex and brittle)
