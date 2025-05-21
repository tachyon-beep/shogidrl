"""
Comprehensive unit tests for ShogiGame class in shogi_game.py using mock utilities.

This test file aims to significantly increase the test coverage of the ShogiGame class
by testing its functionality in various scenarios.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple # Optional is used by GameState

import numpy as np
import pytest

# Add proper import path handling
if __name__ == "__main__":
    REPO_ROOT = Path(__file__).parent.parent.absolute()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from tests.mock_utilities import setup_pytorch_mock_environment
# Global imports for types used in GameState if not importing them locally each time
# from keisei.shogi.shogi_core_definitions import Color, PieceType # Example if needed globally

@dataclass
class GameState:
    """Helper class to store a snapshot of the game state."""
    
    board_str: str
    current_player: Optional[str] = None # Storing .name of Color enum
    move_count: int = 0
    black_hand: Dict[str, int] = field(default_factory=dict) # Storing .name of PieceType
    white_hand: Dict[str, int] = field(default_factory=dict) # Storing .name of PieceType
    
    @classmethod
    def from_game(cls, game):
        """Creates a GameState snapshot from a ShogiGame instance."""
        # This inner setup is redundant if from_game is called from an already mocked context,
        # but it makes from_game usable independently if needed.
        with setup_pytorch_mock_environment():
            # pylint: disable=import-outside-toplevel
            from keisei.shogi.shogi_core_definitions import Color, PieceType
            
            # Convert PieceType objects in hands to their string names
            black_hand_str = {
                pt.name: count for pt, count in game.hands[Color.BLACK.value].items()
            }
            white_hand_str = {
                pt.name: count for pt, count in game.hands[Color.WHITE.value].items()
            }
            
            return cls(
                board_str=game.to_string(),
                current_player=game.current_player.name, # Assuming Color enum has a .name attribute
                move_count=game.move_count,
                black_hand=black_hand_str,
                white_hand=white_hand_str,
            )


# --- Fixtures ---

@pytest.fixture
def new_game():
    """Fixture providing a fresh ShogiGame instance."""
    with setup_pytorch_mock_environment():
        # pylint: disable=import-outside-toplevel
        from keisei.shogi.shogi_game import ShogiGame
        return ShogiGame()


@pytest.fixture
def empty_game():
    """Fixture providing an empty board ShogiGame instance."""
    with setup_pytorch_mock_environment():
        # pylint: disable=import-outside-toplevel
        from keisei.shogi.shogi_game import ShogiGame
        # from keisei.shogi.shogi_core_definitions import Color # Not strictly needed if not used
        
        game = ShogiGame()
        # Clear the board
        for row in range(9):
            for col in range(9):
                game.set_piece(row, col, None)
        
        return game


# --- Test Game Initialization and Reset ---

def test_game_initialization(new_game): # Use fixture
    """Test that ShogiGame initializes correctly."""
    # pylint: disable=import-outside-toplevel
    from keisei.shogi.shogi_core_definitions import Color, Piece, PieceType
    
    game = new_game # Use the fixture
    
    assert game.current_player.value == Color.BLACK.value
    assert game.move_count == 0
    assert not game.game_over
    assert game.winner is None
    
    # Check initial board setup - test a few key pieces
    # Black pieces
    black_king = game.get_piece(8, 4)
    assert isinstance(black_king, Piece), "Black King not found or not a Piece instance."
    assert black_king.type.value == PieceType.KING.value
    assert black_king.color.value == Color.BLACK.value
    
    # White pieces
    white_king = game.get_piece(0, 4)
    assert isinstance(white_king, Piece), "White King not found or not a Piece instance."
    assert white_king.type.value == PieceType.KING.value
    assert white_king.color.value == Color.WHITE.value
    
    # Pawns
    for col in range(9):
        black_pawn = game.get_piece(6, col)
        assert isinstance(black_pawn, Piece), f"Black Pawn at (6,{col}) not found."
        assert black_pawn.type.value == PieceType.PAWN.value
        assert black_pawn.color.value == Color.BLACK.value
        
        white_pawn = game.get_piece(2, col)
        assert isinstance(white_pawn, Piece), f"White Pawn at (2,{col}) not found."
        assert white_pawn.type.value == PieceType.PAWN.value
        assert white_pawn.color.value == Color.WHITE.value


def test_game_reset(new_game):
    """Test that ShogiGame.reset() properly resets the game state."""
    # pylint: disable=import-outside-toplevel
    from keisei.shogi.shogi_core_definitions import Color, PieceType, Piece

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
    # pylint: disable=import-outside-toplevel
    from keisei.shogi.shogi_core_definitions import Color, Piece, PieceType
    
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
    # pylint: disable=import-outside-toplevel
    from keisei.shogi.shogi_core_definitions import Color, PieceType, Piece
    
    # initial_state = GameState.from_game(new_game) # Not used in this specific test
    
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
    # pylint: disable=import-outside-toplevel
    from keisei.shogi.shogi_core_definitions import Color, Piece, PieceType
    
    new_game.make_move((6, 4, 5, 4, False))
    new_game.make_move((2, 3, 3, 3, False))
    new_game.make_move((5, 4, 4, 4, False))
    
    move = (1, 7, 4, 4, False)
    
    white_hand_before = new_game.hands[Color.WHITE.value].copy()
    
    new_game.make_move(move)
    
    assert new_game.get_piece(1, 7) is None
    capturing_bishop = new_game.get_piece(4, 4)
    assert isinstance(capturing_bishop, Piece), "Bishop not at target after capture."
    assert capturing_bishop.type.value == PieceType.BISHOP.value
    assert capturing_bishop.color.value == Color.WHITE.value
    
    captured_type = PieceType.PAWN
    assert new_game.hands[Color.WHITE.value].get(captured_type, 0) == white_hand_before.get(captured_type, 0) + 1


def test_make_move_promotion(empty_game):
    """Test making a move with promotion."""
    # pylint: disable=import-outside-toplevel
    from keisei.shogi.shogi_core_definitions import Color, Piece, PieceType
    
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
    # pylint: disable=import-outside-toplevel
    from keisei.shogi.shogi_core_definitions import Color, PieceType, Piece
    # from keisei.shogi.shogi_core_definitions import DropMoveTuple # Not needed if tuple is constructed directly

    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.current_player = Color.BLACK
    
    drop_move = (None, None, 5, 4, PieceType.PAWN) # Using piece type for drop
    
    empty_game.make_move(drop_move)
    
    dropped_piece = empty_game.get_piece(5, 4)
    assert isinstance(dropped_piece, Piece), "Dropped piece not found."
    assert dropped_piece.type.value == PieceType.PAWN.value
    assert dropped_piece.color.value == Color.BLACK.value
    
    assert empty_game.hands[Color.BLACK.value].get(PieceType.PAWN, 0) == 0


# --- Test Undo Move ---

def test_undo_basic_move(new_game):
    """Test undoing a basic non-capturing move."""
    # pylint: disable=import-outside-toplevel
    from keisei.shogi.shogi_core_definitions import Color, PieceType, Piece
    
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
    # pylint: disable=import-outside-toplevel
    from keisei.shogi.shogi_core_definitions import Color, PieceType, Piece
    
    new_game.make_move((6, 4, 5, 4, False))
    new_game.make_move((2, 3, 3, 3, False))
    new_game.make_move((5, 4, 4, 4, False)) # Black pawn to (4,4)
    
    state_before_capture = GameState.from_game(new_game)
    initial_white_hand_pawn_count = new_game.hands[Color.WHITE.value].get(PieceType.PAWN, 0)

    new_game.make_move((1, 7, 4, 4, False)) # White bishop captures black pawn at (4,4)
    
    capturing_bishop = new_game.get_piece(4, 4)
    assert isinstance(capturing_bishop, Piece), "Bishop not at (4,4) after capture."
    assert capturing_bishop.type.value == PieceType.BISHOP.value
    assert capturing_bishop.color.value == Color.WHITE.value
    assert new_game.get_piece(1, 7) is None
    assert new_game.hands[Color.WHITE.value].get(PieceType.PAWN, 0) == initial_white_hand_pawn_count + 1
    
    new_game.undo_move()
    
    bishop_after_undo = new_game.get_piece(1, 7)
    assert isinstance(bishop_after_undo, Piece), "Bishop not at (1,7) after undo."
    assert bishop_after_undo.type.value == PieceType.BISHOP.value

    captured_pawn_restored = new_game.get_piece(4, 4)
    assert isinstance(captured_pawn_restored, Piece), "Captured pawn not restored at (4,4)."
    assert captured_pawn_restored.type.value == PieceType.PAWN.value
    assert captured_pawn_restored.color.value == Color.BLACK.value
    assert new_game.hands[Color.WHITE.value].get(PieceType.PAWN, 0) == initial_white_hand_pawn_count
    
    state_after_undo = GameState.from_game(new_game)
    assert state_before_capture.board_str == state_after_undo.board_str
    assert state_before_capture.current_player == state_after_undo.current_player
    assert state_before_capture.move_count == state_after_undo.move_count
    assert state_before_capture.white_hand == state_after_undo.white_hand # Compare hands


def test_undo_promotion_move(empty_game):
    """Test undoing a move with promotion."""
    # pylint: disable=import-outside-toplevel
    from keisei.shogi.shogi_core_definitions import Color, Piece, PieceType
    
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
    # pylint: disable=import-outside-toplevel
    from keisei.shogi.shogi_core_definitions import Color, PieceType, Piece
    
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
    assert state_before_drop.black_hand == state_after_undo.black_hand # Compare hands


def test_move_limit():
    """Test game termination by move limit."""
    # This test creates its own game, so it needs its own mock environment setup
    with setup_pytorch_mock_environment():
        # pylint: disable=import-outside-toplevel
        from keisei.shogi.shogi_game import ShogiGame
        # from keisei.shogi.shogi_core_definitions import Color # Not directly used

        game = ShogiGame(max_moves_per_game=2)
        
        game.make_move((6, 4, 5, 4, False))
        game.make_move((2, 4, 3, 4, False))
        
        assert game.game_over
        assert game.winner is None # Draw by move limit
        
        prev_board_str = game.to_string() # Simpler than full GameState for this check
        prev_move_count = game.move_count

        game.make_move((5, 4, 4, 4, False)) # Try to make another move
        
        assert game.to_string() == prev_board_str # Board state should not change
        assert game.move_count == prev_move_count # Move count should not change


if __name__ == "__main__":
    # Running pytest this way ensures that fixtures are discovered and used correctly.
    pytest.main(["-xvs", __file__])