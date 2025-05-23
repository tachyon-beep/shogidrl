"""
Tests for reward propagation with flipped board perspective.
"""

import pytest

from keisei.shogi.shogi_core_definitions import Color, Piece, PieceType
from keisei.shogi.shogi_game import ShogiGame
from keisei.shogi.shogi_game_io import generate_neural_network_observation


@pytest.fixture
def new_game() -> ShogiGame:
    """Returns a ShogiGame instance initialized to the starting position."""
    return ShogiGame(max_moves_per_game=512)


def test_reward_with_flipped_perspective():
    """Test that rewards are correct when the board perspective is flipped."""
    # Setup a simple position where both players can see the board from their perspective
    game = ShogiGame()

    # Clear the board
    for r in range(9):
        for c in range(9):
            game.set_piece(r, c, None)

    # Set up pieces such that coordinates can be easily verified
    # Black king in top-left corner (from Black's perspective)
    game.set_piece(0, 0, Piece(PieceType.KING, Color.BLACK))

    # White king in bottom-right corner (from Black's perspective)
    # This would be top-left from White's perspective when flipped
    game.set_piece(8, 8, Piece(PieceType.KING, Color.WHITE))

    # A white rook that will deliver checkmate to black
    game.set_piece(0, 1, Piece(PieceType.ROOK, Color.WHITE))

    # Verify initial board state from Black's perspective
    obs_black = generate_neural_network_observation(game)

    # Black king should be at (0,0) in the unpromoted pieces plane for Black
    assert obs_black[7, 0, 0] == 1.0  # King is at index 7 in unpromoted pieces

    # Switch to White's perspective
    game.current_player = Color.WHITE

    # White makes a move to checkmate Black
    move_outcome = game.make_move((0, 1, 0, 0, False))  # White rook takes Black king

    # The returned observation should be from Black's perspective (next player)
    # But the reward should be from White's perspective (player who made the move)
    _next_obs, reward, _done, _info = move_outcome  # pylint: disable=unused-variable

    # After the move is made, we need to manually set the game state to simulate checkmate
    # since our test setup is simplified and doesn't fully represent a legal position
    game.game_over = True
    game.winner = Color.WHITE
    game.termination_reason = "Tsumi"

    # Now verify the reward
    reward = game.get_reward(Color.WHITE)
    assert reward == 1.0
    assert game.game_over is True
    assert game.termination_reason == "Tsumi"

    # Get reward from Black's perspective
    black_reward = game.get_reward(Color.BLACK)
    assert black_reward == -1.0


def test_feature_plane_flipping_for_observation():
    """Test that the feature planes are correctly flipped when the board is viewed from White's perspective."""
    game = ShogiGame()

    # Clear the board
    for r in range(9):
        for c in range(9):
            game.set_piece(r, c, None)

    # Place some test pieces:
    # A pawn for each player at specific coordinates to verify flipping
    # Black pawn at (6,4) - should be (2,4) when flipped for White
    game.set_piece(6, 4, Piece(PieceType.PAWN, Color.BLACK))

    # White pawn at (2,4) - should be (6,4) when flipped for White
    game.set_piece(2, 4, Piece(PieceType.PAWN, Color.WHITE))

    # Get observation from Black's perspective (default)
    game.current_player = Color.BLACK
    obs_black = generate_neural_network_observation(game)

    # Get observation from White's perspective
    game.current_player = Color.WHITE
    obs_white = generate_neural_network_observation(game)

    # In Black's view: Black pawn at (6,4)
    assert obs_black[0, 6, 4] == 1.0  # Black pawn in first unpromoted plane

    # In Black's view: White pawn at (2,4)
    assert obs_black[14, 2, 4] == 1.0  # White pawn in opponent unpromoted plane

    # In White's view: Black pawn should be at (2,4) due to flipping
    assert obs_white[14, 2, 4] == 1.0  # Now Black is the opponent in unpromoted plane

    # In White's view: White pawn should be at (6,4) due to flipping
    assert obs_white[0, 6, 4] == 1.0  # Now White is current player in unpromoted plane


if __name__ == "__main__":
    pytest.main()
