"""
Tests for opponent classes and initialization functions in evaluation system.
"""

import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from keisei.shogi.shogi_game import ShogiGame
from keisei.utils import PolicyOutputMapper
from keisei.utils.agent_loading import initialize_opponent, load_evaluation_agent
from keisei.utils.opponents import SimpleHeuristicOpponent, SimpleRandomOpponent

from .conftest import INPUT_CHANNELS, MockPPOAgent, make_test_config


# --- Tests for Opponent Classes ---


@pytest.mark.parametrize(
    "opponent_class,opponent_name",
    [
        (SimpleRandomOpponent, "SimpleRandomOpponent"),
        (SimpleHeuristicOpponent, "SimpleHeuristicOpponent"),
    ],
    ids=["random", "heuristic"],
)
def test_opponent_select_move(
    shogi_game_initial: ShogiGame, opponent_class, opponent_name
):
    """Test that opponents select legal moves from the game state."""
    opponent = opponent_class()
    legal_moves = shogi_game_initial.get_legal_moves()
    selected_move = opponent.select_move(shogi_game_initial)
    assert selected_move in legal_moves, f"{opponent_name} should select a legal move"


# --- Tests for Initialization Functions ---


@pytest.mark.parametrize(
    "opponent_type,expected_class",
    [
        ("random", SimpleRandomOpponent),
        ("heuristic", SimpleHeuristicOpponent),
    ],
    ids=["random", "heuristic"],
)
def test_initialize_opponent_types(policy_mapper, opponent_type, expected_class):
    """Test that initialize_opponent returns correct opponent types."""
    opponent = initialize_opponent(
        opponent_type, None, "cpu", policy_mapper, INPUT_CHANNELS
    )
    assert isinstance(opponent, expected_class)


def test_initialize_opponent_unknown_type(policy_mapper):
    """Test that initialize_opponent raises error for unknown opponent type."""
    with pytest.raises(ValueError, match="Unknown opponent type"):
        initialize_opponent("unknown", None, "cpu", policy_mapper, INPUT_CHANNELS)


@patch(
    "keisei.utils.agent_loading.load_evaluation_agent"  # Corrected patch target
)  # Mock load_evaluation_agent within evaluate.py
def test_initialize_opponent_ppo(mock_load_agent, policy_mapper):
    """Test that initialize_opponent returns a PPOAgent when type is 'ppo' and path is provided."""
    mock_ppo_instance = MockPPOAgent(
        config=make_test_config("cpu", INPUT_CHANNELS, PolicyOutputMapper()),
        device=torch.device("cpu"),
        name="MockPPOAgentForTest",
    )
    mock_load_agent.return_value = mock_ppo_instance

    opponent_ppo = initialize_opponent(
        "ppo", "dummy_path.pth", "cpu", policy_mapper, INPUT_CHANNELS
    )
    mock_load_agent.assert_called_once_with(
        "dummy_path.pth", "cpu", policy_mapper, INPUT_CHANNELS
    )
    assert opponent_ppo == mock_ppo_instance

    with pytest.raises(
        ValueError, match="Opponent path must be provided for PPO opponent type."
    ):
        initialize_opponent("ppo", None, "cpu", policy_mapper, INPUT_CHANNELS)


def test_initialize_opponent_invalid_type(policy_mapper):
    """
    Test that initialize_opponent raises ValueError for an invalid opponent type.

    This ensures that the evaluation pipeline is robust to user/configuration errors and
    provides a clear error message if an unsupported opponent type is specified.
    """
    with pytest.raises(ValueError, match="Unknown opponent type"):
        initialize_opponent("not_a_type", None, "cpu", policy_mapper, INPUT_CHANNELS)
