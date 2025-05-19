"""
Unit tests for Shogi game logic functions in shogi_rules_logic.py
"""

import pytest

from keisei.shogi.shogi_core_definitions import Color, MoveTuple, Piece, PieceType, get_unpromoted_types
from keisei.shogi.shogi_game import ShogiGame
from keisei.shogi.shogi_rules_logic import (
    can_drop_specific_piece,
    check_for_nifu,
    check_for_uchi_fu_zume,
    generate_all_legal_moves,
)


@pytest.fixture
def empty_game() -> ShogiGame:
    """Returns a ShogiGame instance with an empty board and empty hands."""
    game = ShogiGame(max_moves_per_game=512)
    for r in range(9):
        for c in range(9):
            game.set_piece(r, c, None)
    # Initialize hands for both players to be empty for all droppable piece types
    # Corrected method name from get_hand_piece_types to get_unpromoted_types
    game.hands[Color.BLACK.value] = {pt: 0 for pt in get_unpromoted_types()}
    game.hands[Color.WHITE.value] = {pt: 0 for pt in get_unpromoted_types()}
    game.current_player = Color.BLACK  # Default to Black's turn
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
    empty_game.set_piece(
        6, 3, Piece(PieceType.PAWN, Color.BLACK)
    )  # Existing black pawn on file 3
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    assert not can_drop_specific_piece(empty_game, PieceType.PAWN, 3, 3, Color.BLACK)


def test_nifu_with_promoted_pawn_on_file_is_legal(empty_game: ShogiGame):
    """Test can drop pawn if a *promoted* pawn (Tokin) of the same color is on the file."""
    empty_game.set_piece(
        6, 3, Piece(PieceType.PROMOTED_PAWN, Color.BLACK)
    )  # Existing black Tokin on file 3
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.current_player = Color.BLACK

    # Test with can_drop_specific_piece
    assert can_drop_specific_piece(empty_game, PieceType.PAWN, 3, 3, Color.BLACK)

    # Test with generate_all_legal_moves
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))  # Own king
    empty_game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))  # Opponent king

    legal_moves = generate_all_legal_moves(empty_game)
    expected_drop: MoveTuple = (None, None, 3, 3, PieceType.PAWN)
    assert expected_drop in legal_moves


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
    empty_game.current_player = Color.BLACK  # Important for uchi_fu_zume check context

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
    empty_game.set_piece(8, 4, Piece(PieceType.KING, Color.BLACK))  # Own king
    empty_game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))  # Opponent king
    empty_game.current_player = Color.BLACK

    legal_moves = generate_all_legal_moves(empty_game)
    expected_drop: MoveTuple = (None, None, 4, 4, PieceType.PAWN)
    assert expected_drop in legal_moves


def test_generate_legal_moves_excludes_nifu_pawn_drop(empty_game: ShogiGame):
    """Test generate_all_legal_moves excludes a pawn drop that would cause Nifu."""
    empty_game.set_piece(
        6, 4, Piece(PieceType.PAWN, Color.BLACK)
    )  # Existing black pawn
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
    drop_on_last_rank: MoveTuple = (
        None,
        None,
        0,
        4,
        PieceType.PAWN,
    )  # Black's last rank
    assert drop_on_last_rank not in legal_moves


def test_generate_legal_moves_excludes_knight_drop_last_two_ranks(
    empty_game: ShogiGame,
):
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


def test_generate_legal_moves_excludes_drop_leaving_king_in_check(
    empty_game: ShogiGame,
):
    """Test generate_all_legal_moves excludes a drop that leaves own king in check."""
    empty_game.set_piece(8, 4, Piece(PieceType.KING, Color.BLACK))  # Black King
    empty_game.set_piece(
        7, 4, Piece(PieceType.ROOK, Color.WHITE)
    )  # White Rook attacking King if pawn is dropped elsewhere
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.current_player = Color.BLACK

    legal_moves = generate_all_legal_moves(empty_game)
    # Dropping a pawn at (5,5) is valid by itself, but would leave king in check from Rook at (7,4)
    # if the king was at (8,4) and the pawn drop didn't block the check.
    # This test needs a scenario where the drop itself doesn't block an existing check,
    # but reveals one or fails to resolve one.

    # Simpler: King at (8,4), opponent rook at (0,4) (attacks along file 4)
    # If black drops pawn at (5,5), king is still in check.
    empty_game.set_piece(0, 4, Piece(PieceType.ROOK, Color.WHITE))
    empty_game.set_piece(7, 4, None)  # remove previous rook

    legal_moves = generate_all_legal_moves(empty_game)
    drop_elsewhere: MoveTuple = (None, None, 5, 5, PieceType.PAWN)
    assert drop_elsewhere not in legal_moves

    # A valid drop would be to block the check
    blocking_drop: MoveTuple = (
        None,
        None,
        7,
        4,
        PieceType.PAWN,
    )  # Drop pawn between king and rook
    if not check_for_nifu(empty_game, Color.BLACK, 4):  # Ensure no nifu
        assert blocking_drop in legal_moves


def test_generate_legal_moves_includes_drop_giving_check(empty_game: ShogiGame):
    """Test generate_all_legal_moves includes a pawn drop that gives check (not mate)."""
    empty_game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))  # Opponent king
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))  # Own king
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
    empty_game.hands[Color.BLACK.value] = {
        pt: 0 for pt in get_unpromoted_types()
    }

    legal_moves = generate_all_legal_moves(empty_game)
    for move in legal_moves:
        assert move[0] is not None  # Ensure it's a board move, not a drop


def test_generate_legal_moves_board_moves_and_drop_moves(empty_game: ShogiGame):
    """Test that legal moves can contain both board moves and drop moves."""
    game = ShogiGame(max_moves_per_game=100)  # Provide max_moves_per_game
    game.current_player = Color.BLACK
    # Make a few moves to get a pawn in hand for black
    game.make_move((6, 7, 5, 7, False))  # Black pawn P-5g
    game.make_move((2, 2, 3, 2, False))  # White pawn P-3c
    game.make_move(
        (7, 7, 6, 7, False)
    )  # Black rook R-6g (move rook so pawn can be captured)
    game.make_move(
        (0, 1, 2, 2, False)
    )  # White knight N-2c (captures pawn at 3c, then moves to 2c)
    # This is an illegal move in shogi, let's simplify
    game.reset()
    game.current_player = Color.BLACK
    # Black moves P-7f (6,6) -> (5,6)
    game.make_move((6, 6, 5, 6, False))
    # White moves P-3d (2,3) -> (3,3)
    game.make_move((2, 3, 3, 3, False))
    # Black moves P-2f (6,1) -> (5,1)
    game.make_move((6, 1, 5, 1, False))
    # White captures P-7d (2,6) x P-7f (5,6) -> White gets pawn in hand
    # For white to capture, black must have moved a piece to (2,6) or white moves to (5,6)
    # Let's set up a capture:
    game.reset()  # Easier to set up manually
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))
    empty_game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))
    empty_game.set_piece(6, 0, Piece(PieceType.PAWN, Color.BLACK))  # Black pawn at 7i
    empty_game.set_piece(2, 1, Piece(PieceType.PAWN, Color.WHITE))  # White pawn at 3b
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1  # Black has a pawn in hand
    empty_game.current_player = Color.BLACK

    legal_moves = generate_all_legal_moves(empty_game)

    has_board_move = any(m[0] is not None for m in legal_moves)
    has_drop_move = any(m[0] is None and m[4] == PieceType.PAWN for m in legal_moves)

    assert has_board_move, "Should have legal board moves"
    assert has_drop_move, "Should have legal drop moves"


def test_can_drop_specific_piece_pawn_no_nifu_uchi_fu_zume_valid_rank(
    empty_game: ShogiGame,
):
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))  # Opponent king
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))  # Own king
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

    move_no_promote: MoveTuple = (2, 2, 1, 2, False)
    move_promote: MoveTuple = (2, 2, 1, 2, True)

    assert move_no_promote in legal_moves
    assert move_promote in legal_moves


def test_generate_all_legal_moves_forced_promotion(empty_game: ShogiGame):
    # Black pawn at 1,2 (rank 8 for black), MUST move to 0,2 (last rank) and promote
    empty_game.set_piece(1, 2, Piece(PieceType.PAWN, Color.BLACK))
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))
    empty_game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))
    empty_game.current_player = Color.BLACK

    legal_moves = generate_all_legal_moves(empty_game)

    move_no_promote: MoveTuple = (1, 2, 0, 2, False)  # This should be illegal
    move_promote: MoveTuple = (1, 2, 0, 2, True)

    assert move_no_promote not in legal_moves
    assert move_promote in legal_moves


# New tests for drop logic specifically related to generate_all_legal_moves


def test_drop_pawn_checkmate_is_legal_not_uchifuzume(empty_game: ShogiGame):
    """Test dropping a pawn to achieve checkmate is legal if not uchi_fu_zume."""
    # Scenario: White King at (0,4), Black has Gold at (1,3) and (1,5)
    # Dropping a Black Pawn at (1,4) is checkmate.
    # This is NOT uchi_fu_zume because the mate is delivered by a pawn
    # that is not the one causing uchi_fu_zume (which is a pawn drop directly).
    # This tests if a pawn drop that results in mate is allowed.
    empty_game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))
    empty_game.set_piece(
        1, 3, Piece(PieceType.GOLD, Color.BLACK)
    )  # Covers (0,3), (0,4) escape
    empty_game.set_piece(
        1, 5, Piece(PieceType.GOLD, Color.BLACK)
    )  # Covers (0,5) escape
    # King has no moves if pawn dropped at (1,4)
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.current_player = Color.BLACK
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))  # Own king far away

    legal_moves = generate_all_legal_moves(empty_game)
    expected_move: MoveTuple = (None, None, 1, 4, PieceType.PAWN)
    assert expected_move in legal_moves


def test_drop_non_pawn_checkmate_is_legal(empty_game: ShogiGame):
    """Test dropping a non-pawn (e.g., Gold) to achieve checkmate is legal."""
    # Scenario: White King at (0,4). Black has pieces controlling escapes.
    # Dropping a Gold at (1,4) is checkmate.
    empty_game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))
    empty_game.set_piece(0, 3, Piece(PieceType.SILVER, Color.BLACK))  # Controls (0,3)
    empty_game.set_piece(0, 5, Piece(PieceType.SILVER, Color.BLACK))  # Controls (0,5)
    # King has no moves if Gold dropped at (1,4) as it covers (0,4) and (1,4) itself
    empty_game.hands[Color.BLACK.value][PieceType.GOLD] = 1
    empty_game.current_player = Color.BLACK
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))  # Own king

    legal_moves = generate_all_legal_moves(empty_game)
    expected_move: MoveTuple = (None, None, 1, 4, PieceType.GOLD)
    assert expected_move in legal_moves


def test_cannot_drop_piece_if_no_piece_in_hand(empty_game: ShogiGame):
    """Test that a piece cannot be dropped if it's not in hand, even if otherwise legal."""
    empty_game.current_player = Color.BLACK
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 0  # No pawns in hand
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))
    empty_game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))

    legal_moves = generate_all_legal_moves(empty_game)
    potential_drop: MoveTuple = (None, None, 4, 4, PieceType.PAWN)
    assert potential_drop not in legal_moves


def test_drop_multiple_piece_types_available(empty_game: ShogiGame):
    """Test that all valid drops for different pieces in hand are generated."""
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.hands[Color.BLACK.value][PieceType.GOLD] = 1
    empty_game.hands[Color.BLACK.value][PieceType.SILVER] = 1
    empty_game.current_player = Color.BLACK
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))
    empty_game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))

    legal_moves = generate_all_legal_moves(empty_game)

    # Check for a valid pawn drop (e.g., to 4,4)
    # Ensure no nifu for pawn drop at (4,4)
    if not check_for_nifu(empty_game, Color.BLACK, 4):
        pawn_drop: MoveTuple = (None, None, 4, 4, PieceType.PAWN)
        assert pawn_drop in legal_moves

    # Check for a valid gold drop (e.g., to 5,5)
    gold_drop: MoveTuple = (None, None, 5, 5, PieceType.GOLD)
    assert gold_drop in legal_moves

    # Check for a valid silver drop (e.g., to 6,6)
    silver_drop: MoveTuple = (None, None, 6, 6, PieceType.SILVER)
    assert silver_drop in legal_moves


def test_drop_pawn_respects_uchi_fu_zume_in_generate_all_legal_moves(
    empty_game: ShogiGame,
):
    """Test generate_all_legal_moves correctly excludes uchi_fu_zume pawn drops."""
    # Setup Uchi Fu Zume: White King at (0,4), surrounded by Black pieces
    # such that a pawn drop at (1,4) would be mate AND uchi_fu_zume.
    empty_game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))
    empty_game.set_piece(
        0, 3, Piece(PieceType.GOLD, Color.BLACK)
    )  # Covers (0,3) escape
    empty_game.set_piece(
        0, 5, Piece(PieceType.GOLD, Color.BLACK)
    )  # Covers (0,5) escape
    empty_game.set_piece(
        1, 3, Piece(PieceType.GOLD, Color.BLACK)
    )  # Covers (1,3) escape for King if it could move to (1,4)
    empty_game.set_piece(
        1, 5, Piece(PieceType.GOLD, Color.BLACK)
    )  # Covers (1,5) escape for King if it could move to (1,4)
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.current_player = Color.BLACK
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))

    # Verify uchi_fu_zume condition is met for this drop
    assert check_for_uchi_fu_zume(empty_game, 1, 4, Color.BLACK)

    legal_moves = generate_all_legal_moves(empty_game)
    illegal_uchi_fu_zume_drop: MoveTuple = (None, None, 1, 4, PieceType.PAWN)
    assert illegal_uchi_fu_zume_drop not in legal_moves


# Tests for Uchi Fu Zume (Illegal Pawn Drop Checkmate) Edge Cases


def test_uchifuzume_king_can_escape_diagonally_not_uchifuzume(empty_game: ShogiGame):
    """Uchi Fu Zume: King can escape diagonally, so pawn drop is legal."""
    # White King at (0,4). Black drops pawn at (1,4) for check.
    # King can escape to (0,3) or (0,5) if unattacked.
    empty_game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))
    # No other black pieces attacking (0,3) or (0,5)
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.current_player = Color.BLACK
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))  # Own king

    assert not check_for_uchi_fu_zume(empty_game, 1, 4, Color.BLACK)
    legal_moves = generate_all_legal_moves(empty_game)
    assert (None, None, 1, 4, PieceType.PAWN) in legal_moves


def test_uchifuzume_king_can_capture_checking_pawn_not_uchifuzume(
    empty_game: ShogiGame,
):
    """Uchi Fu Zume: King can capture the checking pawn, so pawn drop is legal."""
    empty_game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))  # WK at (0,4)
    # Black drops pawn at (1,4). WK can capture it.
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.current_player = Color.BLACK
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))

    assert not check_for_uchi_fu_zume(empty_game, 1, 4, Color.BLACK)
    legal_moves = generate_all_legal_moves(empty_game)
    assert (None, None, 1, 4, PieceType.PAWN) in legal_moves


def test_uchifuzume_another_piece_can_capture_checking_pawn_not_uchifuzume(
    empty_game: ShogiGame,
):
    """Uchi Fu Zume: Another white piece can capture the checking pawn, so pawn drop is legal."""
    empty_game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))  # WK at (0,4)
    empty_game.set_piece(
        1, 0, Piece(PieceType.ROOK, Color.WHITE)
    )  # WR at (1,0), can move to (1,4)
    # Black drops pawn at (1,4). WR can capture it.
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.current_player = Color.BLACK
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))

    assert not check_for_uchi_fu_zume(empty_game, 1, 4, Color.BLACK)
    legal_moves = generate_all_legal_moves(empty_game)
    assert (None, None, 1, 4, PieceType.PAWN) in legal_moves


def test_uchifuzume_pawn_drop_is_mate_and_king_has_no_legal_moves_IS_uchifuzume(
    empty_game: ShogiGame,
):
    """Uchi Fu Zume: Pawn drop is mate, king has no moves, no piece can take pawn. IS Uchi Fu Zume."""
    # WK at (0,4). Black Golds at (0,3), (0,5), (1,3), (1,5) block all escapes.
    empty_game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))
    empty_game.set_piece(0, 3, Piece(PieceType.GOLD, Color.BLACK))
    empty_game.set_piece(0, 5, Piece(PieceType.GOLD, Color.BLACK))
    empty_game.set_piece(1, 3, Piece(PieceType.GOLD, Color.BLACK))
    empty_game.set_piece(1, 5, Piece(PieceType.GOLD, Color.BLACK))
    # No white pieces can capture a pawn dropped at (1,4).
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.current_player = Color.BLACK
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))

    assert check_for_uchi_fu_zume(empty_game, 1, 4, Color.BLACK)
    legal_moves = generate_all_legal_moves(empty_game)
    assert (None, None, 1, 4, PieceType.PAWN) not in legal_moves


def test_uchifuzume_pawn_drop_check_but_not_mate_due_to_block_not_uchifuzume(
    empty_game: ShogiGame,
):
    """Uchi Fu Zume: Pawn drop is check, but not mate because another piece can block. Not Uchi Fu Zume."""
    # WK at (0,4). Black drops pawn at (1,4) for check.
    # White has a Rook at (2,4) that can move to (1,4) to block (but not capture pawn).
    # This is subtle: uchi_fu_zume is about the *mate* itself being inescapable by the king,
    # or the pawn being uncapturable. If the check can be blocked, it's not mate.
    empty_game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))
    empty_game.set_piece(
        2, 4, Piece(PieceType.ROOK, Color.WHITE)
    )  # WR can move to (1,4) to block
    empty_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    empty_game.current_player = Color.BLACK
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))

    # The check_for_uchi_fu_zume function simulates the drop and checks for opponent legal moves.
    # If White Rook can move to (1,4) to block, then it's not mate, so not uchi_fu_zume.
    # However, the current implementation of check_for_uchi_fu_zume in shogi_rules_logic.py
    # primarily checks king escapes and pawn captures, not general blocks by other pieces.
    # This test might expose a need to enhance check_for_uchi_fu_zume if it fails.
    # For now, assuming the current logic: if king cannot move and pawn cannot be taken, it's uchi_fu_zume.
    # If the rook at (2,4) cannot take the pawn at (1,4), and king has no moves, it would be uchi_fu_zume.
    # Let's adjust: White Rook at (0,0), White Gold at (2,4) can move to (1,4) to block.
    empty_game.set_piece(
        2, 4, Piece(PieceType.GOLD, Color.WHITE)
    )  # WG can move to (1,4)
    empty_game.set_piece(0, 0, Piece(PieceType.KING, Color.BLACK))  # Black king

    assert not check_for_uchi_fu_zume(empty_game, 1, 4, Color.BLACK)
    legal_moves = generate_all_legal_moves(empty_game)
    assert (None, None, 1, 4, PieceType.PAWN) in legal_moves


# Placeholder for tests for get_observation with hand pieces
# These will likely need more complex setup and assertions on the numpy array
# @pytest.mark.skip(reason="Not yet implemented")
# def test_get_observation_with_hand_pieces_black():
#     pass

# @pytest.mark.skip(reason="Not yet implemented")
# def test_get_observation_with_hand_pieces_white():
#     pass

# @pytest.mark.skip(reason="Not yet implemented")
# def test_get_observation_empty_hands():
#     pass

# Placeholder for tests for undo_move with drops and promotions
# These will require making a move and then undoing it, checking board state, hand state, current player etc.
# @pytest.mark.skip(reason="Not yet implemented")
# def test_undo_move_simple_board_move():
#     pass

# @pytest.mark.skip(reason="Not yet implemented")
# def test_undo_move_capture():
#     pass

# @pytest.mark.skip(reason="Not yet implemented")
# def test_undo_move_drop():
#     pass

# @pytest.mark.skip(reason="Not yet implemented")
# def test_undo_move_promotion_no_capture():
#     pass

# @pytest.mark.skip(reason="Not yet implemented")
# def test_undo_move_promotion_with_capture():
#     pass

# @pytest.mark.skip(reason="Not yet implemented")
# def test_undo_move_forced_promotion():
#     pass

# Tests for advanced move legality: Pinned Pieces and King Safety


def test_gamelm_pinned_rook_cannot_expose_king(empty_game: ShogiGame):
    """Test pinned rook cannot make moves exposing the king."""
    empty_game.set_piece(8, 4, Piece(PieceType.KING, Color.BLACK))  # Black King
    empty_game.set_piece(
        4, 4, Piece(PieceType.ROOK, Color.BLACK)
    )  # Black Rook at (4,4)
    empty_game.set_piece(
        0, 4, Piece(PieceType.ROOK, Color.WHITE)
    )  # White Rook at (0,4) pinning BR
    empty_game.set_piece(
        0, 0, Piece(PieceType.KING, Color.WHITE)
    )  # White King for completeness
    empty_game.current_player = Color.BLACK

    legal_moves = generate_all_legal_moves(empty_game)

    # Illegal: BR (4,4) -> (4,3) (sideways, exposes king)
    illegal_move_sideways: MoveTuple = (4, 4, 4, 3, False)
    assert illegal_move_sideways not in legal_moves

    # Legal: BR (4,4) -> (3,4) (towards pinner, along pin line)
    # (3,4) is not in Black's promotion zone (rows 0,1,2)
    legal_move_towards_pinner: MoveTuple = (4, 4, 3, 4, False)
    assert legal_move_towards_pinner in legal_moves

    # Legal: BR (4,4) -> (0,4) (capture pinner at (0,4))
    # (0,4) is in Black's promotion zone. Rook promotion is optional.
    capture_pinner_promoted: MoveTuple = (4, 4, 0, 4, True)
    capture_pinner_not_promoted: MoveTuple = (4, 4, 0, 4, False)
    assert capture_pinner_promoted in legal_moves
    assert capture_pinner_not_promoted in legal_moves

    # Legal: BR (4,4) -> (5,4) (towards own king, along pin line)
    legal_move_towards_king: MoveTuple = (4, 4, 5, 4, False)
    assert legal_move_towards_king in legal_moves


def test_gamelm_pinned_bishop_cannot_expose_king(empty_game: ShogiGame):
    """Test pinned bishop cannot make moves exposing the king."""
    empty_game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))  # Black King
    empty_game.set_piece(
        4, 4, Piece(PieceType.BISHOP, Color.BLACK)
    )  # Black Bishop at (4,4)
    empty_game.set_piece(
        0, 0, Piece(PieceType.BISHOP, Color.WHITE)
    )  # White Bishop at (0,0) pinning BB
    empty_game.set_piece(
        0, 8, Piece(PieceType.KING, Color.WHITE)
    )  # White King for completeness
    empty_game.current_player = Color.BLACK

    legal_moves = generate_all_legal_moves(empty_game)

    # Illegal: BB (4,4) -> (3,5) (valid bishop move direction, but off pin line)
    illegal_move_off_pin: MoveTuple = (4, 4, 3, 5, False)
    assert illegal_move_off_pin not in legal_moves

    # Legal: BB (4,4) -> (3,3) (towards pinner, along pin line)
    # (3,3) is not in Black's promotion zone.
    legal_move_towards_pinner: MoveTuple = (4, 4, 3, 3, False)
    assert legal_move_towards_pinner in legal_moves

    # Legal: BB (4,4) -> (0,0) (capture pinner at (0,0))
    # (0,0) is in Black's promotion zone. Bishop promotion is optional.
    capture_pinner_promoted: MoveTuple = (4, 4, 0, 0, True)
    capture_pinner_not_promoted: MoveTuple = (4, 4, 0, 0, False)
    assert capture_pinner_promoted in legal_moves
    assert capture_pinner_not_promoted in legal_moves

    # Legal: BB (4,4) -> (5,5) (towards own king, along pin line)
    legal_move_towards_king: MoveTuple = (4, 4, 5, 5, False)
    assert legal_move_towards_king in legal_moves


def test_gamelm_king_cannot_move_into_check(empty_game: ShogiGame):
    """Test king cannot move into a square attacked by the opponent."""
    empty_game.set_piece(4, 4, Piece(PieceType.KING, Color.BLACK))  # Black King
    empty_game.set_piece(
        4, 0, Piece(PieceType.ROOK, Color.WHITE)
    )  # White Rook attacking row 4
    empty_game.set_piece(
        0, 4, Piece(PieceType.ROOK, Color.WHITE)
    )  # White Rook attacking col 4
    empty_game.set_piece(0, 8, Piece(PieceType.KING, Color.WHITE))  # Opponent King
    empty_game.current_player = Color.BLACK

    legal_moves = generate_all_legal_moves(empty_game)

    # King at (4,4) is attacked by WR at (4,0) and WR at (0,4).
    # Illegal moves for the King:
    assert (4, 4, 3, 4, False) not in legal_moves  # Into WR(0,4) attack
    assert (4, 4, 5, 4, False) not in legal_moves  # Into WR(0,4) attack
    assert (4, 4, 4, 3, False) not in legal_moves  # Into WR(4,0) attack
    assert (4, 4, 4, 5, False) not in legal_moves  # Into WR(4,0) attack

    # Legal moves for the King (to unattacked diagonal squares):
    assert (4, 4, 3, 3, False) in legal_moves
    assert (4, 4, 3, 5, False) in legal_moves
    assert (4, 4, 5, 3, False) in legal_moves
    assert (4, 4, 5, 5, False) in legal_moves

    # Clear board and hands for Scenario 2 to ensure a clean slate
    for r_idx in range(9):
        for c_idx in range(9):
            empty_game.set_piece(r_idx, c_idx, None)
    empty_game.hands[Color.BLACK.value] = {
        pt: 0 for pt in get_unpromoted_types()
    }
    empty_game.hands[Color.WHITE.value] = {
        pt: 0 for pt in get_unpromoted_types()
    }

    # Scenario 2: King is on edge, only one escape square, but it's attacked.
    empty_game.current_player = Color.BLACK
    empty_game.set_piece(7, 7, Piece(PieceType.KING, Color.BLACK))  # BK
    empty_game.set_piece(0, 0, Piece(PieceType.ROOK, Color.WHITE))  # WR
    empty_game.set_piece(8, 0, Piece(PieceType.KING, Color.WHITE))  # WK

    legal_moves = generate_all_legal_moves(empty_game)
    # King at (7,7) can move to (7,6), (6,7), (6,6) etc.
    # Cannot move to (7,0) (attacked by WR(0,0)) or (0,7) (attacked by WR(0,0))
    assert (7, 7, 7, 0, False) not in legal_moves
    assert (7, 7, 0, 7, False) not in legal_moves
    assert (7, 7, 6, 6, False) in legal_moves  # Example of a safe move
