"""
Unit tests for Shogi game logic functions in shogi_rules_logic.py
"""
import pytest
from keisei.shogi.shogi_core_definitions import Piece, PieceType, Color, MoveTuple
from keisei.shogi.shogi_game import ShogiGame
from keisei.shogi.shogi_rules_logic import (
    can_drop_specific_piece,
    generate_all_legal_moves,
    check_for_nifu,
    check_for_uchi_fu_zume
)

@pytest.fixture
def empty_game() -> ShogiGame:
    """Returns a ShogiGame instance with an empty board and empty hands."""
    game = ShogiGame()
    for r in range(9):
        for c in range(9):
            game.set_piece(r, c, None)
    # Initialize hands for both players to be empty for all droppable piece types
    # Corrected method name from get_hand_piece_types to get_unpromoted_types
    game.hands[Color.BLACK.value] = {pt: 0 for pt in PieceType.get_unpromoted_types()}
    game.hands[Color.WHITE.value] = {pt: 0 for pt in PieceType.get_unpromoted_types()}
    game.current_player = Color.BLACK # Default to Black's turn
    return game

# Tests for can_drop_specific_piece
def test_can_drop_piece_empty_square(empty_game: ShogiGame):
    """Test can drop piece on an empty square."""
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    assert can_drop_specific_piece(empty_game, PieceType.PAWN, 4, 4, Color.BLACK)

def test_cannot_drop_piece_occupied_square(empty_game: ShogiGame):
    """Test cannot drop piece on an occupied square."""
    empty_game.set_piece(4, 4, Piece(PieceType.PAWN, Color.WHITE))
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    assert not can_drop_specific_piece(empty_game, PieceType.PAWN, 4, 4, Color.BLACK)

def test_can_drop_pawn_nifu_false(empty_game: ShogiGame):
    """Test can drop pawn when nifu condition is false."""
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    assert can_drop_specific_piece(empty_game, PieceType.PAWN, 3, 3, Color.BLACK)

def test_cannot_drop_pawn_nifu_true(empty_game: ShogiGame):
    """Test cannot drop pawn when nifu condition is true."""
    empty_game.set_piece(6, 3, Piece(PieceType.PAWN, Color.BLACK)) # Existing black pawn on file 3
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    assert not can_drop_specific_piece(empty_game, PieceType.PAWN, 3, 3, Color.BLACK)

def test_cannot_drop_pawn_last_rank_black(empty_game: ShogiGame):
    """Test cannot drop pawn on last rank for Black (rank 0)."""
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    assert not can_drop_specific_piece(empty_game, PieceType.PAWN, 0, 4, Color.BLACK)

def test_cannot_drop_pawn_last_rank_white(empty_game: ShogiGame):
    """Test cannot drop pawn on last rank for White (rank 8)."""
    empty_game.hands[Color.WHITE.value][PieceType.PAWN] = 1
    assert not can_drop_specific_piece(empty_game, PieceType.PAWN, 8, 4, Color.WHITE)

def test_cannot_drop_lance_last_rank_black(empty_game: ShogiGame):
    """Test cannot drop lance on last rank for Black (rank 0)."""
    empty_game.hands[Color.BLACK.value][PieceType.LANCE] = 1
    assert not can_drop_specific_piece(empty_game, PieceType.LANCE, 0, 4, Color.BLACK)

def test_cannot_drop_lance_last_rank_white(empty_game: ShogiGame):
    """Test cannot drop lance on last rank for White (rank 8)."""
    empty_game.hands[Color.WHITE.value][PieceType.LANCE] = 1
    assert not can_drop_specific_piece(empty_game, PieceType.LANCE, 8, 4, Color.WHITE)

def test_cannot_drop_knight_last_two_ranks_black(empty_game: ShogiGame):
    """Test cannot drop knight on last two ranks for Black (ranks 0, 1)."""
    empty_game.hands[Color.BLACK.value][PieceType.KNIGHT] = 1
    assert not can_drop_specific_piece(empty_game, PieceType.KNIGHT, 0, 4, Color.BLACK)
    assert not can_drop_specific_piece(empty_game, PieceType.KNIGHT, 1, 4, Color.BLACK)

def test_cannot_drop_knight_last_two_ranks_white(empty_game: ShogiGame):
    """Test cannot drop knight on last two ranks for White (ranks 8, 7)."""
    empty_game.hands[Color.WHITE.value][PieceType.KNIGHT] = 1
    assert not can_drop_specific_piece(empty_game, PieceType.KNIGHT, 8, 4, Color.WHITE)
    assert not can_drop_specific_piece(empty_game, PieceType.KNIGHT, 7, 4, Color.WHITE)

def test_can_drop_gold_any_rank(empty_game: ShogiGame):
    """Test can drop Gold on any valid empty rank."""
    empty_game.hands[Color.BLACK.value][PieceType.GOLD] = 1
    assert can_drop_specific_piece(empty_game, PieceType.GOLD, 0, 4, Color.BLACK)
    assert can_drop_specific_piece(empty_game, PieceType.GOLD, 8, 4, Color.BLACK)

def test_cannot_drop_pawn_uchi_fu_zume(empty_game: ShogiGame):
    """Test cannot drop pawn if it results in Uchi Fu Zume."""
    # Setup Uchi Fu Zume scenario (from test_shogi_engine.py)
    empty_game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))  # White king
    empty_game.set_piece(0, 3, Piece(PieceType.GOLD, Color.BLACK))
    empty_game.set_piece(0, 5, Piece(PieceType.GOLD, Color.BLACK))
    empty_game.set_piece(1, 3, Piece(PieceType.GOLD, Color.BLACK))
    empty_game.set_piece(1, 5, Piece(PieceType.GOLD, Color.BLACK))
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.current_player = Color.BLACK # Important for uchi_fu_zume check context

    assert not can_drop_specific_piece(empty_game, PieceType.PAWN, 1, 4, Color.BLACK)

def test_can_drop_pawn_not_uchi_fu_zume_escape_possible(empty_game: ShogiGame):
    """Test can drop pawn if it's check but not Uchi Fu Zume (king can escape)."""
    empty_game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))
    # King can escape to (0,3)
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.current_player = Color.BLACK
    assert can_drop_specific_piece(empty_game, PieceType.PAWN, 1, 4, Color.BLACK)


# Tests for generate_all_legal_moves (focusing on drops)

def test_generate_legal_moves_includes_valid_pawn_drop(empty_game: ShogiGame):
    """Test generate_all_legal_moves includes a valid pawn drop."""
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.set_piece(8, 4, Piece(PieceType.KING, Color.BLACK)) # Own king
    empty_game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE)) # Opponent king
    empty_game.current_player = Color.BLACK

    legal_moves = generate_all_legal_moves(empty_game)
    expected_drop: MoveTuple = (None, None, 4, 4, PieceType.PAWN)
    assert expected_drop in legal_moves

def test_generate_legal_moves_excludes_nifu_pawn_drop(empty_game: ShogiGame):
    """Test generate_all_legal_moves excludes a pawn drop that would cause Nifu."""
    empty_game.set_piece(6, 4, Piece(PieceType.PAWN, Color.BLACK)) # Existing black pawn
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))
    empty_game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))
    empty_game.current_player = Color.BLACK

    legal_moves = generate_all_legal_moves(empty_game)
    drop_on_same_file: MoveTuple = (None, None, 4, 4, PieceType.PAWN)
    assert drop_on_same_file not in legal_moves

def test_generate_legal_moves_excludes_pawn_drop_last_rank(empty_game: ShogiGame):
    """Test generate_all_legal_moves excludes pawn drop on its last rank."""
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))
    empty_game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))
    empty_game.current_player = Color.BLACK

    legal_moves = generate_all_legal_moves(empty_game)
    drop_on_last_rank: MoveTuple = (None, None, 0, 4, PieceType.PAWN) # Black's last rank
    assert drop_on_last_rank not in legal_moves

def test_generate_legal_moves_excludes_knight_drop_last_two_ranks(empty_game: ShogiGame):
    """Test generate_all_legal_moves excludes knight drop on its last two ranks."""
    empty_game.hands[Color.BLACK.value][PieceType.KNIGHT] = 1
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))
    empty_game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))
    empty_game.current_player = Color.BLACK
    legal_moves = generate_all_legal_moves(empty_game)
    drop_on_last_rank: MoveTuple = (None, None, 0, 4, PieceType.KNIGHT)
    drop_on_second_last_rank: MoveTuple = (None, None, 1, 4, PieceType.KNIGHT)
    assert drop_on_last_rank not in legal_moves
    assert drop_on_second_last_rank not in legal_moves

def test_generate_legal_moves_excludes_drop_leaving_king_in_check(empty_game: ShogiGame):
    """Test generate_all_legal_moves excludes a drop that leaves own king in check."""
    empty_game.set_piece(8, 4, Piece(PieceType.KING, Color.BLACK)) # Black King
    empty_game.set_piece(7, 4, Piece(PieceType.ROOK, Color.WHITE)) # White Rook attacking King if pawn is dropped elsewhere
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.current_player = Color.BLACK

    legal_moves = generate_all_legal_moves(empty_game)
    # Dropping a pawn at (5,5) is valid by itself, but would leave king in check from Rook at (7,4)
    # if the king was at (8,4) and the pawn drop didn't block the check.
    # This test needs a scenario where the drop itself doesn't block an existing check,
    # but reveals one or fails to resolve one.

    # Simpler: King at (8,4), opponent rook at (0,4) (attacks along file 4)
    # If black drops pawn at (5,5), king is still in check.
    empty_game.set_piece(0,4, Piece(PieceType.ROOK, Color.WHITE))
    empty_game.set_piece(7,4, None) # remove previous rook

    legal_moves = generate_all_legal_moves(empty_game)
    drop_elsewhere: MoveTuple = (None, None, 5, 5, PieceType.PAWN)
    assert drop_elsewhere not in legal_moves

    # A valid drop would be to block the check
    blocking_drop: MoveTuple = (None, None, 7, 4, PieceType.PAWN) # Drop pawn between king and rook
    if not check_for_nifu(empty_game, Color.BLACK, 4): # Ensure no nifu
         assert blocking_drop in legal_moves


def test_generate_legal_moves_includes_drop_giving_check(empty_game: ShogiGame):
    """Test generate_all_legal_moves includes a pawn drop that gives check (not mate)."""
    empty_game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE)) # Opponent king
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK)) # Own king
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.current_player = Color.BLACK

    # Dropping pawn at (1,4) checks white king at (0,4). King can escape to (0,3) or (0,5).
    legal_moves = generate_all_legal_moves(empty_game)
    checking_drop: MoveTuple = (None, None, 1, 4, PieceType.PAWN)
    assert checking_drop in legal_moves

def test_generate_legal_moves_no_drops_if_hand_empty(empty_game: ShogiGame):
    """Test generate_all_legal_moves produces no drop moves if hand is empty."""
    empty_game.set_piece(8, 4, Piece(PieceType.KING, Color.BLACK))
    empty_game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))
    empty_game.current_player = Color.BLACK
    # Ensure hands are empty (fixture does this, but double check for clarity)
    empty_game.hands[Color.BLACK.value] = {pt: 0 for pt in PieceType.get_unpromoted_types()}

    legal_moves = generate_all_legal_moves(empty_game)
    for move in legal_moves:
        assert move[0] is not None # Ensure it's a board move, not a drop

def test_generate_legal_moves_board_moves_and_drop_moves(empty_game: ShogiGame):
    """Test that legal moves can contain both board moves and drop moves."""
    game = ShogiGame() # Start with a standard board
    game.current_player = Color.BLACK
    # Make a few moves to get a pawn in hand for black
    game.make_move((6, 7, 5, 7, False)) # Black pawn P-5g
    game.make_move((2, 2, 3, 2, False)) # White pawn P-3c
    game.make_move((7, 7, 6, 7, False)) # Black rook R-6g (move rook so pawn can be captured)
    game.make_move((0, 1, 2, 2, False)) # White knight N-2c (captures pawn at 3c, then moves to 2c)
                                        # This is an illegal move in shogi, let's simplify
    game.reset()
    game.current_player = Color.BLACK
    # Black moves P-7f (6,6) -> (5,6)
    game.make_move((6,6,5,6, False))
    # White moves P-3d (2,3) -> (3,3)
    game.make_move((2,3,3,3, False))
    # Black moves P-2f (6,1) -> (5,1)
    game.make_move((6,1,5,1, False))
    # White captures P-7d (2,6) x P-7f (5,6) -> White gets pawn in hand
    # For white to capture, black must have moved a piece to (2,6) or white moves to (5,6)
    # Let's set up a capture:
    game.reset() # Easier to set up manually
    empty_game.set_piece(8,8, Piece(PieceType.KING, Color.BLACK))
    empty_game.set_piece(0,0, Piece(PieceType.KING, Color.WHITE))
    empty_game.set_piece(6,0, Piece(PieceType.PAWN, Color.BLACK)) # Black pawn at 7i
    empty_game.set_piece(2,1, Piece(PieceType.PAWN, Color.WHITE)) # White pawn at 3b
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1 # Black has a pawn in hand
    empty_game.current_player = Color.BLACK

    legal_moves = generate_all_legal_moves(empty_game)

    has_board_move = any(m[0] is not None for m in legal_moves)
    has_drop_move = any(m[0] is None and m[4] == PieceType.PAWN for m in legal_moves)

    assert has_board_move, "Should have legal board moves"
    assert has_drop_move, "Should have legal drop moves"

def test_can_drop_specific_piece_pawn_no_nifu_uchi_fu_zume_valid_rank(empty_game: ShogiGame):
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE)) # Opponent king
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK)) # Own king
    # Ensure no nifu on column 4
    assert not check_for_nifu(empty_game, Color.BLACK, 4)
    # Ensure no uchi_fu_zume for a drop at (3,4)
    # This requires setting up a scenario where dropping at (3,4) is not mate
    # For simplicity, assume it's not uchi_fu_zume if the king has any escape.
    # King at (0,0) can escape to (0,1), (1,0), (1,1) if a pawn is dropped at (3,4)
    assert not check_for_uchi_fu_zume(empty_game, 3, 4, Color.BLACK)
    assert can_drop_specific_piece(empty_game, PieceType.PAWN, 3, 4, Color.BLACK)

def test_generate_all_legal_moves_promotion_options(empty_game: ShogiGame):
    # Black pawn at 2,2 (rank 7 for black), can move to 1,2 (promotion zone)
    empty_game.set_piece(2, 2, Piece(PieceType.PAWN, Color.BLACK))
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))
    empty_game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))
    empty_game.current_player = Color.BLACK

    legal_moves = generate_all_legal_moves(empty_game)
    
    move_no_promote: MoveTuple = (2,2,1,2,False)
    move_promote: MoveTuple = (2,2,1,2,True)

    assert move_no_promote in legal_moves
    assert move_promote in legal_moves

def test_generate_all_legal_moves_forced_promotion(empty_game: ShogiGame):
    # Black pawn at 1,2 (rank 8 for black), MUST move to 0,2 (last rank) and promote
    empty_game.set_piece(1, 2, Piece(PieceType.PAWN, Color.BLACK))
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))
    empty_game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))
    empty_game.current_player = Color.BLACK

    legal_moves = generate_all_legal_moves(empty_game)
    
    move_no_promote: MoveTuple = (1,2,0,2,False) # This should be illegal
    move_promote: MoveTuple = (1,2,0,2,True)

    assert move_no_promote not in legal_moves
    assert move_promote in legal_moves

