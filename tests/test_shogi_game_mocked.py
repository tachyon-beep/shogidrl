"""
Unit tests for ShogiGame class in shogi_game.py, using mocks for PyTorch dependencies.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import (  # Removed TYPE_CHECKING for simplicity if direct import works
    Dict,
    Optional,
)

import numpy as np

from keisei.shogi.shogi_core_definitions import Color, PieceType
from tests.mock_utilities import setup_pytorch_mock_environment

# Add proper import path handling
# This block is mainly for running the script directly.
# Test runners like pytest usually handle paths if run from the project root.
if __name__ == "__main__":
    REPO_ROOT = Path(__file__).parent.parent.absolute()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


@dataclass
class GameState:
    """Helper class to store a snapshot of the game state."""

    board_str: str
    current_player: Optional[Color] = None  # "Color" should now be resolved
    move_count: int = 0
    black_hand: Dict[PieceType, int] = field(
        default_factory=dict
    )  # "PieceType" should now be resolved
    white_hand: Dict[PieceType, int] = field(
        default_factory=dict
    )  # "PieceType" should now be resolved

    @classmethod
    def from_game(cls, game):  # 'game' will be an instance of ShogiGame
        """Creates a GameState snapshot from a ShogiGame instance."""
        # Ensure 'game.current_player' is an instance of your Color enum for these to work:
        # game.current_player.BLACK and game.current_player.WHITE
        # This assumes Color enum has BLACK and WHITE members.
        # If ShogiGame.current_player is already a Color instance, this is fine.
        # If game.hands keys are Color.BLACK.value and Color.WHITE.value, this logic is fine.

        # Accessing Color members directly if game.current_player is already a Color instance.
        # This structure assumes Color.BLACK and Color.WHITE are valid members of the Color enum.
        return cls(
            board_str=game.to_string(),
            current_player=game.current_player,
            move_count=game.move_count,
            # Assuming game.hands is keyed by integer values of the Color enum
            # and Color.BLACK / Color.WHITE are enum members.
            black_hand=game.hands[Color.BLACK.value].copy(),
            white_hand=game.hands[Color.WHITE.value].copy(),
        )


# Individual test cases start here
def test_get_observation_initial_state_dimensions():
    """Test the dimensions of the observation from the initial state."""
    with setup_pytorch_mock_environment():  # This should now work without type errors
        # pylint: disable=import-outside-toplevel
        # Color and PieceType are already imported globally, but ShogiGame and io are specific to this test
        from keisei.shogi.shogi_game import ShogiGame
        from keisei.shogi.shogi_game_io import generate_neural_network_observation

        game = ShogiGame(max_moves_per_game=512)
        obs = generate_neural_network_observation(game)

        assert obs.shape == (46, 9, 9)
        assert np.sum(obs[0:15]) > 0
        assert np.sum(obs[15:30]) > 0
        assert np.sum(obs[30:38]) == 0
