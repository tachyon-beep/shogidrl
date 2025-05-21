"""
Example test that demonstrates how to use mock utilities
to test Shogi modules without PyTorch dependencies.
"""

import sys
from pathlib import Path

import numpy as np

# Add proper import path handling
if __name__ == "__main__":
    # Make the repo root directory available for imports
    REPO_ROOT = Path(__file__).parent.parent.absolute()
    if str(REPO_ROOT) not in sys.path:  # Avoid adding duplicates if run multiple times
        sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from tests.mock_utilities import setup_pytorch_mock_environment


# Create a test that demonstrates the approach
def test_shogi_with_mocks():
    """
    This is a demonstration of how to test Shogi components
    by mocking the PyTorch dependencies.
    """
    # Use our convenience function to set up the mocked environment
    with setup_pytorch_mock_environment():
        # Now we can safely import the modules that depend on torch
        # pylint: disable=import-outside-toplevel
        from keisei.shogi.shogi_core_definitions import Color, PieceType
        from keisei.shogi.shogi_game import ShogiGame

        # Run our actual test code
        game = ShogiGame()

        # A simple assertion to make sure the game was initialized
        assert game.current_player == Color.BLACK

        # If we need to test observation generation:
        from keisei.shogi.shogi_game_io import generate_neural_network_observation

        obs = generate_neural_network_observation(game)
        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] >= 42  # At least the core channels

        # Test a basic game operation
        black_pawn_pos = (6, 4)  # Position of a black pawn in initial setup
        target_pos = (5, 4)  # Move one square forward

        # MODIFIED: Check piece exists before accessing .type
        piece_at_start = game.get_piece(*black_pawn_pos)
        assert (
            piece_at_start is not None
        ), f"Expected piece at {black_pawn_pos}, got None."
        assert piece_at_start.type == PieceType.PAWN

        # Make the move
        game.make_move(
            (black_pawn_pos[0], black_pawn_pos[1], target_pos[0], target_pos[1], False)
        )

        # Verify the move was made
        assert game.get_piece(*black_pawn_pos) is None

        # MODIFIED: Check piece exists before accessing .type
        piece_at_target = game.get_piece(*target_pos)
        assert (
            piece_at_target is not None
        ), f"Expected piece at {target_pos} after move, got None."
        assert piece_at_target.type == PieceType.PAWN
        assert game.current_player == Color.WHITE


# This won't run automatically when the file is imported
# but can be run manually or as part of a test suite
if __name__ == "__main__":
    test_shogi_with_mocks()
    print("All tests passed!")
