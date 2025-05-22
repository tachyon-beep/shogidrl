"""
Comprehensive unit tests for ShogiGame class in shogi_game.py using mock utilities.

This test file aims to significantly increase the test coverage of the ShogiGame class
by testing its functionality in various scenarios.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytest

# Add proper import path handling
if __name__ == "__main__":
    REPO_ROOT = Path(__file__).parent.parent.absolute()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from tests.mock_utilities import setup_pytorch_mock_environment
from keisei.shogi import ShogiGame, Color, PieceType
from keisei.shogi.shogi_core_definitions import Piece


@dataclass
class GameState:
    """Helper class to store a snapshot of the game state."""

    board_str: str
    current_player: Optional[str] = None  # Storing .name of Color enum
    move_count: int = 0
    black_hand: Dict[str, int] = field(
        default_factory=dict
    )  # Storing .name of PieceType
    white_hand: Dict[str, int] = field(
        default_factory=dict
    )  # Storing .name of PieceType

    @classmethod
    def from_game(cls, game):
        """Creates a GameState snapshot from a ShogiGame instance."""
        with setup_pytorch_mock_environment():
            # Convert PieceType objects in hands to their string names
            black_hand_str = {
                pt.name: count for pt, count in game.hands[Color.BLACK.value].items()
            }
            white_hand_str = {
                pt.name: count for pt, count in game.hands[Color.WHITE.value].items()
            }
            return cls(
                board_str=game.to_string(),
                current_player=game.current_player.name,  # Assuming Color enum has a .name attribute
                move_count=game.move_count,
                black_hand=black_hand_str,
                white_hand=white_hand_str,
            )


# --- Fixtures ---


@pytest.fixture
def new_game():
    """Fixture providing a fresh ShogiGame instance."""
    with setup_pytorch_mock_environment():
        return ShogiGame()


@pytest.fixture
def empty_game():
    """Fixture providing an empty board ShogiGame instance."""
    with setup_pytorch_mock_environment():
        game = ShogiGame()
        # Clear the board
        for row in range(9):
            for col in range(9):
                game.set_piece(row, col, None)

        return game


# --- Test Game Initialization and Reset ---


def test_game_initialization(new_game):  # Use fixture
    """Test that ShogiGame initializes correctly."""
    game = new_game  # Use the fixture

    assert game.current_player.value == Color.BLACK.value
    assert game.move_count == 0
    assert not game.game_over
    assert game.winner is None

    # Check initial board setup - test a few key pieces
    # Black pieces
    black_king = game.get_piece(8, 4)
    # FIX: Avoid isinstance due to possible duplicate Piece class from mocks
    assert hasattr(black_king, "type") and hasattr(
        black_king, "color"
    ), "Black King not found or not a Piece-like object."
    assert (
        black_king.type.name == "KING" and black_king.color.name == "BLACK"
    ), "Black King not found at (8,4) or has wrong type/color."

    # White pieces
    white_king = game.get_piece(0, 4)
    # FIX: Avoid isinstance due to possible duplicate Piece class from mocks
    assert hasattr(white_king, "type") and hasattr(
        white_king, "color"
    ), "White King not found or not a Piece-like object."
    assert (
        white_king.type.name == "KING" and white_king.color.name == "WHITE"
    ), "White King not found at (0,4) or has wrong type/color."

    # Pawns
    for col in range(9):
        black_pawn = game.get_piece(6, col)
        # FIX: Avoid isinstance due to possible duplicate Piece class from mocks
        assert hasattr(black_pawn, "type") and hasattr(
            black_pawn, "color"
        ), f"Black Pawn at (6,{col}) not found or not a Piece-like object."
        assert (
            black_pawn.type.name == "PAWN" and black_pawn.color.name == "BLACK"
        ), f"Black Pawn at (6,{col}) has wrong type/color."

        white_pawn = game.get_piece(2, col)
        # FIX: Avoid isinstance due to possible duplicate Piece class from mocks
        assert hasattr(white_pawn, "type") and hasattr(
            white_pawn, "color"
        ), f"White Pawn at (2,{col}) not found or not a Piece-like object."
        assert (
            white_pawn.type.name == "PAWN" and white_pawn.color.name == "WHITE"
        ), f"White Pawn at (2,{col}) has wrong type/color."


def test_game_reset(new_game):
    """Test that ShogiGame.reset() properly resets the game state."""
    new_game.make_move((6, 4, 5, 4, False))
    new_game.make_move((2, 4, 3, 4, False))

    assert new_game.current_player.value == Color.BLACK.value
    assert new_game.move_count == 2
    moved_black_pawn = new_game.get_piece(5, 4)
    assert moved_black_pawn is not None, "Black pawn missing after move."
    assert new_game.get_piece(6, 4) is None

    observation = new_game.reset()

    assert new_game.current_player.value == Color.BLACK.value
    assert new_game.move_count == 0
    assert not new_game.game_over
    assert new_game.winner is None

    reset_black_pawn = new_game.get_piece(6, 4)
    assert isinstance(reset_black_pawn, Piece), "Black pawn missing after reset."
    assert reset_black_pawn.type.value == PieceType.PAWN.value
    assert new_game.get_piece(5, 4) is None

    assert isinstance(observation, np.ndarray)
    assert observation.shape == (46, 9, 9)


# --- Test Board Manipulation ---


def test_get_set_piece(empty_game):
    """Test get_piece and set_piece methods."""
    for row in range(9):
        for col in range(9):
            assert empty_game.get_piece(row, col) is None

    test_piece = Piece(PieceType.ROOK, Color.BLACK)
    empty_game.set_piece(4, 4, test_piece)

    retrieved_piece = empty_game.get_piece(4, 4)
    assert retrieved_piece is not None, "Piece not set correctly."
    assert retrieved_piece.type.value == PieceType.ROOK.value
    assert retrieved_piece.color.value == Color.BLACK.value

    empty_game.set_piece(4, 4, None)
    assert empty_game.get_piece(4, 4) is None

    assert empty_game.get_piece(-1, 4) is None
    assert empty_game.get_piece(9, 4) is None
    assert empty_game.get_piece(4, -1) is None
    assert empty_game.get_piece(4, 9) is None


def test_is_on_board(new_game):
    """Test the is_on_board method."""
    for row in range(9):
        for col in range(9):
            assert new_game.is_on_board(row, col)

    assert not new_game.is_on_board(-1, 0)
    assert not new_game.is_on_board(0, -1)
    assert not new_game.is_on_board(9, 0)
    assert not new_game.is_on_board(0, 9)
    assert not new_game.is_on_board(9, 9)
    assert not new_game.is_on_board(-1, -1)


# --- Test Move Execution ---


def test_make_move_basic(new_game):
    """Test making a basic non-capturing move."""
    move = (6, 4, 5, 4, False)
    new_game.make_move(move)

    assert new_game.get_piece(6, 4) is None
    moved_pawn = new_game.get_piece(5, 4)
    assert isinstance(moved_pawn, Piece), "Pawn not at target after move."
    assert moved_pawn.type.value == PieceType.PAWN.value
    assert moved_pawn.color.value == Color.BLACK.value

    assert new_game.current_player.value == Color.WHITE.value
    assert new_game.move_count == 1
    assert len(new_game.move_history) == 1
    assert new_game.move_history[0]["move"] == move


def test_make_move_capture(new_game):
    """Test making a capturing move."""
    # Black pawn advance, White pawn reply, Black pawn advance (existing opening)
    new_game.make_move((6, 4, 5, 4, False))
    # new_game.make_move((2, 3, 3, 3, False))
    new_game.make_move((2, 6, 3, 6, False))
    new_game.make_move((5, 4, 4, 4, False))

    # --- clear the bishop’s diagonal properly ---
    # White pawn at (2,6) steps forward to (3,6); that frees squares (2,6) & (3,5)
    # new_game.make_move((2, 6, 3, 6, False))

    # Now the bishop capture is legal
    move = (1, 7, 4, 4, False)  # White bishop b9-e6 (1,7 ➜ 4,4)

    white_hand_before = new_game.hands[Color.WHITE.value].copy()

    new_game.make_move(move)

    # Assertions
    assert new_game.get_piece(1, 7) is None
    capturing_bishop = new_game.get_piece(4, 4)
    assert isinstance(capturing_bishop, Piece), "Bishop not at target after capture."
    assert capturing_bishop.type is PieceType.BISHOP
    assert capturing_bishop.color is Color.WHITE

    captured_type = PieceType.PAWN
    assert (
        new_game.hands[Color.WHITE.value].get(captured_type, 0)
        == white_hand_before.get(captured_type, 0) + 1
    )


def test_make_move_promotion(empty_game):
    """Test making a move with promotion."""
    pawn = Piece(PieceType.PAWN, Color.BLACK)
    empty_game.set_piece(3, 4, pawn)
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))
    empty_game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))
    empty_game.current_player = Color.BLACK

    move = (3, 4, 2, 4, True)
    empty_game.make_move(move)

    promoted_piece = empty_game.get_piece(2, 4)
    assert promoted_piece is not None, "Promoted piece not found."
    assert promoted_piece.type.value == PieceType.PROMOTED_PAWN.value
    assert promoted_piece.color.value == Color.BLACK.value


def test_make_move_piece_drop(empty_game):
    """Test dropping a piece from hand."""
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.current_player = Color.BLACK

    drop_move = (None, None, 5, 4, PieceType.PAWN)  # Using piece type for drop

    empty_game.make_move(drop_move)

    dropped_piece = empty_game.get_piece(5, 4)
    assert isinstance(dropped_piece, Piece), "Dropped piece not found."
    assert dropped_piece.type.value == PieceType.PAWN.value
    assert dropped_piece.color.value == Color.BLACK.value

    assert empty_game.hands[Color.BLACK.value].get(PieceType.PAWN, 0) == 0


# --- Test Undo Move ---


def test_undo_basic_move(new_game):
    """Test undoing a basic non-capturing move."""
    initial_state = GameState.from_game(new_game)

    new_game.make_move((6, 4, 5, 4, False))

    moved_pawn_before_undo = new_game.get_piece(5, 4)
    assert isinstance(moved_pawn_before_undo, Piece), "Pawn not at (5,4) before undo."
    assert moved_pawn_before_undo.type.value == PieceType.PAWN.value
    assert new_game.get_piece(6, 4) is None

    new_game.undo_move()

    pawn_at_origin_after_undo = new_game.get_piece(6, 4)
    assert isinstance(pawn_at_origin_after_undo, Piece), "Pawn not at (6,4) after undo."
    assert pawn_at_origin_after_undo.type.value == PieceType.PAWN.value
    assert new_game.get_piece(5, 4) is None

    assert new_game.current_player.value == Color.BLACK.value
    assert new_game.move_count == 0
    assert len(new_game.move_history) == 0

    final_state = GameState.from_game(new_game)
    assert initial_state.board_str == final_state.board_str
    assert initial_state.current_player == final_state.current_player
    assert initial_state.move_count == final_state.move_count


def test_undo_capture_move(new_game):
    """Test undoing a capturing move."""
    new_game.make_move((6, 4, 5, 4, False))
    new_game.make_move((2, 3, 3, 3, False))
    new_game.make_move((5, 4, 4, 4, False))  # Black pawn to (4,4)

    # Clear the path for the bishop by removing the pawn at (2,6)
    new_game.set_piece(2, 6, None)

    state_before_capture = GameState.from_game(new_game)
    initial_white_hand_pawn_count = new_game.hands[Color.WHITE.value].get(
        PieceType.PAWN, 0
    )

    new_game.make_move((1, 7, 4, 4, False))  # White bishop captures black pawn at (4,4)

    capturing_bishop = new_game.get_piece(4, 4)
    assert isinstance(capturing_bishop, Piece), "Bishop not at (4,4) after capture."
    assert capturing_bishop.type.value == PieceType.BISHOP.value
    assert capturing_bishop.color.value == Color.WHITE.value
    assert new_game.get_piece(1, 7) is None
    assert (
        new_game.hands[Color.WHITE.value].get(PieceType.PAWN, 0)
        == initial_white_hand_pawn_count + 1
    )

    new_game.undo_move()

    bishop_after_undo = new_game.get_piece(1, 7)
    assert isinstance(bishop_after_undo, Piece), "Bishop not at (1,7) after undo."
    assert bishop_after_undo.type.value == PieceType.BISHOP.value

    captured_pawn_restored = new_game.get_piece(4, 4)
    assert isinstance(
        captured_pawn_restored, Piece
    ), "Captured pawn not restored at (4,4)."
    assert captured_pawn_restored.type.value == PieceType.PAWN.value
    assert captured_pawn_restored.color.value == Color.BLACK.value
    assert (
        new_game.hands[Color.WHITE.value].get(PieceType.PAWN, 0)
        == initial_white_hand_pawn_count
    )

    state_after_undo = GameState.from_game(new_game)
    assert state_before_capture.board_str == state_after_undo.board_str
    assert state_before_capture.current_player == state_after_undo.current_player
    assert state_before_capture.move_count == state_after_undo.move_count
    assert (
        state_before_capture.white_hand == state_after_undo.white_hand
    )  # Compare hands


def test_undo_promotion_move(empty_game):
    """Test undoing a move with promotion."""
    pawn = Piece(PieceType.PAWN, Color.BLACK)
    empty_game.set_piece(3, 4, pawn)
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))
    empty_game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))
    empty_game.current_player = Color.BLACK

    state_before_promotion = GameState.from_game(empty_game)

    empty_game.make_move((3, 4, 2, 4, True))

    promoted_piece = empty_game.get_piece(2, 4)
    assert promoted_piece is not None, "Promoted piece not at (2,4)."
    assert promoted_piece.type.value == PieceType.PROMOTED_PAWN.value

    empty_game.undo_move()

    unpromoted_pawn = empty_game.get_piece(3, 4)
    assert isinstance(unpromoted_pawn, Piece), "Pawn not restored at (3,4)."
    assert unpromoted_pawn.type.value == PieceType.PAWN.value
    assert empty_game.get_piece(2, 4) is None

    state_after_undo = GameState.from_game(empty_game)
    assert state_before_promotion.board_str == state_after_undo.board_str
    assert state_before_promotion.current_player == state_after_undo.current_player


def test_undo_drop_move(empty_game):
    """Test undoing a drop move."""
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.current_player = Color.BLACK

    state_before_drop = GameState.from_game(empty_game)

    drop_move = (None, None, 5, 4, PieceType.PAWN)
    empty_game.make_move(drop_move)

    dropped_piece = empty_game.get_piece(5, 4)
    assert isinstance(dropped_piece, Piece), "Dropped piece not at (5,4)."
    assert dropped_piece.type.value == PieceType.PAWN.value
    assert empty_game.hands[Color.BLACK.value].get(PieceType.PAWN, 0) == 0

    empty_game.undo_move()

    assert empty_game.get_piece(5, 4) is None
    assert empty_game.hands[Color.BLACK.value].get(PieceType.PAWN, 0) == 1

    state_after_undo = GameState.from_game(empty_game)
    assert state_before_drop.current_player == state_after_undo.current_player
    assert state_before_drop.black_hand == state_after_undo.black_hand  # Compare hands


def test_undo_move_preserves_legal_moves_pinned_piece_in_check(empty_game):
    """
    Tests if make_move followed by undo_move correctly restores the game state
    such that get_legal_moves returns the same results, specifically in a scenario
    where a piece is pinned and the king is in check by the pinning piece.
    This is relevant to the bug observed in test_move_legality_pinned_piece.
    """
    game = empty_game

    # Setup: Black King at (8,4), Black Rook at (6,4) (pinned),
    # White Rook at (1,4) (pinner & checker). White King at (0,0).
    game.set_piece(8, 4, Piece(PieceType.KING, Color.BLACK))
    game.set_piece(6, 4, Piece(PieceType.ROOK, Color.BLACK))  # Pinned piece
    game.set_piece(
        1, 4, Piece(PieceType.ROOK, Color.WHITE)
    )  # Pinning and checking piece
    game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))  # Opponent king
    game.current_player = Color.BLACK
    game.move_count = 5  # Arbitrary starting move count

    initial_state_snapshot = GameState.from_game(game)
    initial_legal_moves = game.get_legal_moves()  # CORRECTED_CALL

    # Ensure there are legal moves to make (e.g., king escape, capture pinner)
    assert len(initial_legal_moves) > 0, "Test setup: No legal moves found for Black."

    move_to_make = initial_legal_moves[0]

    # Make and undo the move
    game.make_move(move_to_make)
    game.undo_move()

    final_legal_moves = game.get_legal_moves()  # CORRECTED_CALL
    final_state_snapshot = GameState.from_game(game)

    # Primary assertion: Legal moves should be identical
    assert set(initial_legal_moves) == set(
        final_legal_moves
    ), "Legal moves differ after make/undo cycle."

    # Secondary assertion: Full game state should be restored
    assert (
        initial_state_snapshot.board_str == final_state_snapshot.board_str
    ), "Board state differs after make/undo."
    assert (
        initial_state_snapshot.current_player == final_state_snapshot.current_player
    ), "Current player differs after make/undo."
    assert (
        initial_state_snapshot.move_count == final_state_snapshot.move_count
    ), "Move count differs after make/undo."
    assert (
        initial_state_snapshot.black_hand == final_state_snapshot.black_hand
    ), "Black's hand differs after make/undo."
    assert (
        initial_state_snapshot.white_hand == final_state_snapshot.white_hand
    ), "White's hand differs after make/undo."
    # Check all GameState fields for equality
    assert (
        initial_state_snapshot == final_state_snapshot
    ), "GameState object differs after make/undo."


def test_move_limit():
    """Test that the game enforces the move limit and ends the game appropriately."""
    # Set up a game with a low move limit for testing
    game = ShogiGame(max_moves_per_game=3)
    # Make 3 moves
    for _ in range(3):
        # Find a legal move (just move a pawn forward if possible)
        for r in range(9):
            for c in range(9):
                piece = game.get_piece(r, c)
                if (
                    piece
                    and piece.color == game.current_player
                    and piece.type == PieceType.PAWN
                ):
                    to_r = r - 1 if game.current_player == Color.BLACK else r + 1
                    if 0 <= to_r < 9 and game.get_piece(to_r, c) is None:
                        move = (r, c, to_r, c, False)
                        game.make_move(move)
                        break
            else:
                continue
            break
    # After 3 moves, the move limit should be reached
    assert game.move_count == 3
    assert game.game_over, "Game should be over after reaching move limit"


# DEPRECATED: This file is superseded by 'test_shogi_game_mock_comprehensive.py' and 'test_shogi_game_updated_with_mocks.py'.
# All core ShogiGame, undo move, and observation tests should be added to those files.


if __name__ == "__main__":
    # Running pytest this way ensures that fixtures are discovered and used correctly.
    pytest.main(["-xvs", __file__])
