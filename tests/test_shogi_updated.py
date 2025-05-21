"""
Unit tests for ShogiGame class in shogi_game.py
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

# Add proper import path handling
if __name__ == "__main__":
    # Make the repo root directory available for imports
    REPO_ROOT = Path(__file__).parent.parent.absolute()
    sys.path.insert(0, str(REPO_ROOT))

# Import our mock setup function
# pylint: disable=wrong-import-position
from tests.mock_utilities import setup_pytorch_mock_environment

# Wrap the entire test module in the mock environment
# This allows importing modules that depend on PyTorch
with setup_pytorch_mock_environment():
    from keisei.shogi.shogi_core_definitions import (
        Color,
        PieceType,
    )
    from keisei.shogi.shogi_game import ShogiGame


@dataclass
class GameState:
    """Helper class to store a snapshot of the game state."""

    board_str: str
    current_player: Color
    move_count: int
    black_hand: Dict[PieceType, int] = field(default_factory=dict)
    white_hand: Dict[PieceType, int] = field(default_factory=dict)

    @classmethod
    def from_game(cls, game: ShogiGame) -> "GameState":
        """Creates a GameState snapshot from a ShogiGame instance."""
        return cls(
            board_str=game.to_string(),
            current_player=game.current_player,
            move_count=game.move_count,
            black_hand=game.hands[Color.BLACK.value].copy(),
            white_hand=game.hands[Color.WHITE.value].copy(),
        )


@pytest.fixture
def new_game() -> ShogiGame:
    """Returns a ShogiGame instance initialized to the starting position."""
    return ShogiGame(max_moves_per_game=512)


def test_get_observation_initial_state_dimensions(new_game: ShogiGame):
    """Test the dimensions of the observation from the initial state."""
    # Import inside test to avoid PyTorch import issues
    with setup_pytorch_mock_environment():
        # pylint: disable=import-outside-toplevel
        from keisei.shogi.shogi_game_io import generate_neural_network_observation

    obs = generate_neural_network_observation(new_game)

    # Check observation dimensions
    assert obs.shape == (46, 9, 9)

    # Check some expected values in the observation
    # Black's pieces are represented in planes 0-14 (unpromoted pieces)
    assert np.sum(obs[0:15]) > 0  # Black unpromoted pieces
    assert np.sum(obs[15:30]) > 0  # White unpromoted pieces
    assert np.sum(obs[30:38]) == 0  # No pieces in hand initially
