"""
Unit tests for PPOAgent in ppo_agent.py
"""

from typing import List  # Add this import

import numpy as np
import pytest
import torch

from keisei.experience_buffer import ExperienceBuffer  # Added import
from keisei.ppo_agent import PPOAgent
from keisei.shogi import ShogiGame  # Corrected import for ShogiGame
from keisei.shogi.shogi_core_definitions import (  # Ensure MoveTuple is imported
    MoveTuple,
)
from keisei.utils import PolicyOutputMapper


def test_ppo_agent_init_and_select_action():
    """Test PPOAgent initializes and select_action returns a valid index."""
    mapper = PolicyOutputMapper()
    agent = PPOAgent(input_channels=46, policy_output_mapper=mapper)
    obs = np.random.rand(46, 9, 9).astype(
        np.float32
    )  # Ensure correct dtype for tensor conversion
    game = ShogiGame(max_moves_per_game=512)  # Added max_moves_per_game
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

    selected_move, idx, log_prob, value, legal_mask_returned = agent.select_action(
        obs, legal_shogi_moves=legal_moves
    )
    assert isinstance(idx, int)
    assert 0 <= idx < agent.num_actions_total
    assert isinstance(selected_move, tuple) or selected_move is None # select_action can return None if no legal moves (though guarded by caller)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    assert isinstance(legal_mask_returned, torch.Tensor) # Check type of returned legal_mask
    assert legal_mask_returned.shape[0] == agent.num_actions_total # Check shape
    assert legal_mask_returned.dtype == torch.bool # Check dtype


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
    dummy_obs_np = np.random.rand(46, 9, 9).astype(np.float32)
    dummy_obs_tensor = torch.from_numpy(dummy_obs_np).to(
        torch.device("cpu")
    )  # Convert to tensor on CPU

    # Create a dummy legal_mask. For this test, its content might not be critical,
    # but its shape should match num_actions_total.
    dummy_legal_mask = torch.ones(agent.num_actions_total, dtype=torch.bool, device="cpu")
    # Make at least one action illegal if num_actions_total > 0 to test masking, if desired
    if agent.num_actions_total > 0:
        dummy_legal_mask[0] = False

    for i in range(buffer_size):
        experience_buffer.add(
            obs=dummy_obs_tensor,  # <<< PASS THE TENSOR HERE
            action=i % agent.num_actions_total,
            reward=float(i),
            log_prob=0.1 * i,
            value=0.5 * i,
            done=(i == buffer_size - 1),
            legal_mask=dummy_legal_mask, # Added dummy_legal_mask
        )

    assert len(experience_buffer) == buffer_size

    # Compute advantages and returns
    last_value = 0.0  # Assuming terminal state after buffer is full for simplicity
    experience_buffer.compute_advantages_and_returns(last_value)

    # Call the learn method
    try:
        metrics = agent.learn(experience_buffer)
        assert (
            metrics is not None
        ), "learn() should return a metrics dictionary, not None"
        # Check if losses are returned and are floats (or can be zero)
        assert isinstance(metrics["ppo/policy_loss"], float)
        assert isinstance(metrics["ppo/value_loss"], float)
        assert isinstance(metrics["ppo/entropy"], float)
        assert isinstance(metrics["ppo/kl_divergence_approx"], float)
        assert isinstance(metrics["ppo/learning_rate"], float)
    except (
        RuntimeError
    ) as e:  # Catch a more specific exception if possible, or document why general Exception is needed.
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
