"""
Unit tests for ShogiGame class in shogi_game.py
"""

import pytest
import numpy as np
from keisei.shogi.shogi_game import ShogiGame
from keisei.shogi.shogi_core_definitions import (
    Piece,
    PieceType,
    Color,
    OBS_UNPROMOTED_ORDER,
    OBS_PROMOTED_ORDER,
)


@pytest.fixture
def base_game() -> ShogiGame:
    """Returns a ShogiGame instance initialized to the starting position."""
    return ShogiGame()


def test_get_observation_initial_state_dimensions(game_instance: ShogiGame):
    """Test the dimensions of the observation from the initial state."""
    obs = game_instance.get_observation()
    assert isinstance(obs, np.ndarray), "Observation should be a numpy array"
    assert obs.shape == (
        46,
        9,
        9,
    ), "Observation shape is incorrect based on shogi_game_io.py"


@pytest.fixture
def game_with_black_pawn_in_hand(_base_game: ShogiGame) -> ShogiGame:
    """Game instance with Black having one pawn in hand."""
    _base_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    return _base_game


def test_get_observation_hand_pieces_black_one_pawn(
    game_with_black_pawn_in_hand_fixture: ShogiGame,
):
    """Test observation when Black has one pawn in hand."""
    obs = game_with_black_pawn_in_hand_fixture.get_observation()
    # OBS_UNPROMOTED_ORDER is used in shogi_core_definitions for hand piece order in observation
    # but shogi_game_io.py uses PieceType.get_unpromoted_types() directly.
    # Let's align with shogi_game_io.py for hand channel indexing.
    hand_types_order = PieceType.get_unpromoted_types()
    pawn_hand_channel_index = 28 + hand_types_order.index(PieceType.PAWN)
    expected_value = 1 / 18.0
    assert np.allclose(
        obs[pawn_hand_channel_index], expected_value
    ), f"Black pawn hand plane incorrect. Expected {expected_value}, got {obs[pawn_hand_channel_index][0][0]}"
    # Check other hand planes for black are zero
    for i, pt in enumerate(hand_types_order):
        if pt != PieceType.PAWN:
            channel_idx = 28 + i
            assert np.all(
                obs[channel_idx] == 0.0
            ), f"Black hand plane for {pt.name} should be 0. Got {obs[channel_idx][0][0]}"


@pytest.fixture
def game_with_white_rook_in_hand(_base_game: ShogiGame) -> ShogiGame:
    """Game instance with White having one rook in hand."""
    _base_game.hands[Color.WHITE.value][PieceType.ROOK] = 1
    return _base_game


def test_get_observation_hand_pieces_white_one_rook(
    game_with_white_rook_in_hand_fixture: ShogiGame,
):
    """Test observation when White has one rook in hand."""
    obs = game_with_white_rook_in_hand_fixture.get_observation()
    hand_types_order = PieceType.get_unpromoted_types()
    rook_hand_channel_index = 35 + hand_types_order.index(PieceType.ROOK)
    expected_value = 1 / 18.0
    assert np.allclose(
        obs[rook_hand_channel_index], expected_value
    ), f"White rook hand plane incorrect. Expected {expected_value}, got {obs[rook_hand_channel_index][0][0]}"
    # Check other hand planes for white are zero
    for i, pt in enumerate(hand_types_order):
        if pt != PieceType.ROOK:
            channel_idx = 35 + i
            assert np.all(
                obs[channel_idx] == 0.0
            ), f"White hand plane for {pt.name} should be 0. Got {obs[channel_idx][0][0]}"


@pytest.fixture
def game_with_mixed_hands(_base_game: ShogiGame) -> ShogiGame:
    """Game instance with multiple pieces in hand for both players."""
    _base_game.hands[Color.BLACK.value][PieceType.PAWN] = 3
    _base_game.hands[Color.BLACK.value][PieceType.GOLD] = 1
    _base_game.hands[Color.WHITE.value][PieceType.BISHOP] = 2
    _base_game.hands[Color.WHITE.value][PieceType.SILVER] = 1
    return _base_game


def test_get_observation_multiple_hand_pieces_mixed_players(
    game_with_mixed_hands_fixture: ShogiGame,
):
    """Test observation with multiple pieces in hand for both players."""
    obs = game_with_mixed_hands_fixture.get_observation()
    hand_types_order = PieceType.get_unpromoted_types()

    # Black's hand
    pawn_idx_black = 28 + hand_types_order.index(PieceType.PAWN)
    gold_idx_black = 28 + hand_types_order.index(PieceType.GOLD)
    expected_pawn_black = 3 / 18.0
    expected_gold_black = 1 / 18.0
    assert np.allclose(
        obs[pawn_idx_black], expected_pawn_black
    ), f"Black 3 pawns hand plane incorrect. Expected {expected_pawn_black}, got {obs[pawn_idx_black][0][0]}"
    assert np.allclose(
        obs[gold_idx_black], expected_gold_black
    ), f"Black 1 gold hand plane incorrect. Expected {expected_gold_black}, got {obs[gold_idx_black][0][0]}"

    # White's hand
    bishop_idx_white = 35 + hand_types_order.index(PieceType.BISHOP)
    silver_idx_white = 35 + hand_types_order.index(PieceType.SILVER)
    expected_bishop_white = 2 / 18.0
    expected_silver_white = 1 / 18.0
    assert np.allclose(
        obs[bishop_idx_white], expected_bishop_white
    ), f"White 2 bishops hand plane incorrect. Expected {expected_bishop_white}, got {obs[bishop_idx_white][0][0]}"
    assert np.allclose(
        obs[silver_idx_white], expected_silver_white
    ), f"White 1 silver hand plane incorrect. Expected {expected_silver_white}, got {obs[silver_idx_white][0][0]}"

    # Check a piece not in hand for Black
    lance_idx_black = 28 + OBS_UNPROMOTED_ORDER.index(PieceType.LANCE)
    assert np.all(obs[lance_idx_black] == 0.0), "Black lance hand plane should be 0"

    # Check a piece not in hand for White
    rook_idx_white = 28 + 7 + OBS_UNPROMOTED_ORDER.index(PieceType.ROOK)
    assert np.all(obs[rook_idx_white] == 0.0), "White rook hand plane should be 0"


def test_get_observation_empty_hands(game_instance: ShogiGame):
    """Test observation when both players have empty hands (initial state)."""
    # base_game fixture already has empty hands initially
    obs = game_instance.get_observation()

    # All hand planes (28 through 28 + 7 + 7 - 1 = 41) should be 0
    for i in range(14):  # 7 for black, 7 for white
        hand_channel_index = 28 + i
        assert np.all(
            obs[hand_channel_index] == 0.0
        ), f"Hand channel {hand_channel_index} should be 0 for empty hands"


def test_get_observation_current_player_plane_black_turn(game_instance: ShogiGame):
    """Test current player plane when it's Black's turn."""
    game_instance.current_player = Color.BLACK
    obs = game_instance.get_observation()
    current_player_plane_index = 42
    assert np.all(
        obs[current_player_plane_index] == 1.0
    ), "Current player plane incorrect for Black's turn (should be 1.0)"


def test_get_observation_current_player_plane_white_turn(game_instance: ShogiGame):
    """Test current player plane when it's White's turn."""
    game_instance.current_player = Color.WHITE
    obs = game_instance.get_observation()
    current_player_plane_index = 42
    assert np.all(
        obs[current_player_plane_index] == 0.0
    ), "Current player plane incorrect for White's turn (should be 0.0)"


@pytest.fixture
def game_with_move_count_5(_base_game: ShogiGame) -> ShogiGame:
    """Game instance with move count set to 5."""
    _base_game.move_count = 5
    return _base_game


def test_get_observation_move_count_plane(game_with_move_count_5_fixture: ShogiGame):
    """Test move count plane."""
    obs = game_with_move_count_5_fixture.get_observation()
    move_count_plane_index = 43
    expected_value = 5 / 512.0
    assert np.allclose(
        obs[move_count_plane_index], expected_value
    ), f"Move count plane incorrect. Expected {expected_value}, got {obs[move_count_plane_index][0][0]}"


def test_get_observation_board_pieces_consistency_after_reset(game: ShogiGame):
    """Test that board piece planes are correctly set after a game reset (initial position)."""
    # game is already reset
    obs = game.get_observation()

    # Check a few key pieces for Black (current player perspective)
    # Black's Pawn at (6,0) (row 6, col 0)
    # Pawn is OBS_UNPROMOTED_ORDER[0]
    black_pawn_plane = OBS_UNPROMOTED_ORDER.index(PieceType.PAWN)
    assert (
        obs[black_pawn_plane, 6, 0] == 1.0
    ), "Black pawn at (6,0) not found in observation"

    # Black's Rook at (7,7)
    black_rook_plane = OBS_UNPROMOTED_ORDER.index(PieceType.ROOK)
    assert (
        obs[black_rook_plane, 7, 7] == 1.0
    ), "Black rook at (7,7) not found in observation"

    # Check a few key pieces for White (opponent perspective)
    # White's Pawn at (2,0)
    # Opponent planes start after current player's 14 planes (i.e., at index 14)
    white_pawn_plane = 14 + OBS_UNPROMOTED_ORDER.index(PieceType.PAWN)
    assert (
        obs[white_pawn_plane, 2, 0] == 1.0
    ), "White pawn at (2,0) not found in observation"

    # White's King at (0,4)
    white_king_plane = 14 + OBS_UNPROMOTED_ORDER.index(PieceType.KING)
    assert (
        obs[white_king_plane, 0, 4] == 1.0
    ), "White king at (0,4) not found in observation"

    # Ensure a square that should be empty for a piece type is 0
    assert (
        obs[black_pawn_plane, 0, 0] == 0.0
    ), "Square (0,0) should be empty of black pawns"
    assert (
        obs[white_pawn_plane, 6, 0] == 0.0
    ), "Square (6,0) should be empty of white pawns"


def test_get_observation_promoted_piece_on_board(game_with_promoted_piece: ShogiGame):
    """Test observation when a promoted piece is on the board."""
    # Place a promoted pawn (Tokin) for Black at (2,2)
    game_with_promoted_piece.set_piece(2, 2, Piece(PieceType.PROMOTED_PAWN, Color.BLACK))
    game_with_promoted_piece.current_player = Color.BLACK  # Ensure perspective is Black's
    obs = game_with_promoted_piece.get_observation()

    # Promoted pawn for current player (Black)
    # Promoted planes start after unpromoted planes (index 8 for current player)
    promoted_pawn_plane = 8 + OBS_PROMOTED_ORDER.index(PieceType.PROMOTED_PAWN)
    assert (
        obs[promoted_pawn_plane, 2, 2] == 1.0
    ), "Black Promoted Pawn at (2,2) not found"

    # Ensure the unpromoted pawn plane is 0 at that location for Black
    unpromoted_pawn_plane = OBS_UNPROMOTED_ORDER.index(PieceType.PAWN)
    assert (
        obs[unpromoted_pawn_plane, 2, 2] == 0.0
    ), "Unpromoted Black Pawn should not be at (2,2)"

    # Place a promoted rook (Dragon) for White at (5,5)
    game_with_promoted_piece.set_piece(5, 5, Piece(PieceType.PROMOTED_ROOK, Color.WHITE))
    obs = game_with_promoted_piece.get_observation()  # Re-get obs after change

    # Promoted rook for opponent (White)
    # Opponent planes start at 14. Promoted opponent planes start at 14 + 8 = 22.
    promoted_rook_plane_opponent = (
        14 + 8 + OBS_PROMOTED_ORDER.index(PieceType.PROMOTED_ROOK)
    )
    assert (
        obs[promoted_rook_plane_opponent, 5, 5] == 1.0
    ), "White Promoted Rook at (5,5) not found"

    # Ensure unpromoted rook plane is 0 at that location for White
    unpromoted_rook_plane_opponent = 14 + OBS_UNPROMOTED_ORDER.index(PieceType.ROOK)
    assert (
        obs[unpromoted_rook_plane_opponent, 5, 5] == 0.0
    ), "Unpromoted White Rook should not be at (5,5)"


# --- Tests for undo_move ---


def test_undo_move_simple_board_move(game: ShogiGame):
    """Test undoing a simple pawn move."""
    initial_board_str = game.to_string()
    initial_player = game.current_player
    initial_move_count = game.move_count

    # Black pawn P-7f (6,6) -> (5,6)
    move: tuple = (6, 6, 5, 6, False)  # Using tuple directly as MoveTuple is Union
    game.make_move(move)

    piece_on_target = game.get_piece(5, 6)
    assert piece_on_target is not None
    assert piece_on_target.type == PieceType.PAWN
    assert game.get_piece(6, 6) is None
    assert game.current_player != initial_player
    assert game.move_count == initial_move_count + 1

    game.undo_move()

    assert game.to_string() == initial_board_str, "Board state not restored after undo"
    assert (
        game.current_player == initial_player
    ), "Current player not restored after undo"
    assert game.move_count == initial_move_count, "Move count not restored after undo"
    piece_on_source = game.get_piece(6, 6)
    assert piece_on_source is not None
    assert piece_on_source.type == PieceType.PAWN
    assert game.get_piece(5, 6) is None


def test_undo_move_capture(game_with_capture_setup: ShogiGame):
    """Test undoing a move that involves a capture."""
    game = game_with_capture_setup
    # Setup: Black pawn at (6,6), White pawn at (5,6)
    game.set_piece(6, 6, Piece(PieceType.PAWN, Color.BLACK))
    game.set_piece(5, 6, Piece(PieceType.PAWN, Color.WHITE))
    game.set_piece(2, 2, None)  # Clear white pawn from initial setup for this test
    game.current_player = Color.BLACK

    initial_black_hand = game.get_pieces_in_hand(Color.BLACK).copy()
    initial_white_hand = game.get_pieces_in_hand(Color.WHITE).copy()
    initial_board_str = game.to_string()  # Get string before modifying board for test
    initial_player = game.current_player
    initial_move_count = game.move_count

    # Black pawn (6,6) captures White pawn (5,6)
    move: tuple = (6, 6, 5, 6, False)
    game.make_move(move)

    moved_piece = game.get_piece(5, 6)
    assert moved_piece is not None
    assert moved_piece.type == PieceType.PAWN
    assert moved_piece.color == Color.BLACK
    assert (
        game.hands[Color.BLACK.value][PieceType.PAWN]
        == initial_black_hand.get(PieceType.PAWN, 0) + 1
    )

    game.undo_move()

    assert game.to_string() == initial_board_str, "Board state not restored"
    assert game.current_player == initial_player, "Current player not restored"
    assert game.move_count == initial_move_count, "Move count not restored"
    assert (
        game.hands[Color.BLACK.value] == initial_black_hand
    ), "Black's hand not restored"
    assert (
        game.hands[Color.WHITE.value] == initial_white_hand
    ), "White's hand not restored (should be unchanged)"

    restored_source_piece = game.get_piece(6, 6)
    assert restored_source_piece is not None
    assert restored_source_piece.type == PieceType.PAWN

    restored_captured_piece = game.get_piece(5, 6)
    assert restored_captured_piece is not None
    assert restored_captured_piece.type == PieceType.PAWN
    assert restored_captured_piece.color == Color.WHITE


def test_undo_move_drop(game: ShogiGame):
    """Test undoing a drop move."""
    game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    game.current_player = Color.BLACK

    # Clear a square for dropping
    game.set_piece(4, 4, None)

    initial_black_hand = game.get_pieces_in_hand(Color.BLACK).copy()
    initial_board_str = game.to_string()
    initial_player = game.current_player
    initial_move_count = game.move_count

    # Black drops pawn at (4,4)
    drop_move: tuple = (None, None, 4, 4, PieceType.PAWN)
    game.make_move(drop_move)

    dropped_piece = game.get_piece(4, 4)
    assert dropped_piece is not None
    assert dropped_piece.type == PieceType.PAWN
    assert dropped_piece.color == Color.BLACK
    assert (
        game.hands[Color.BLACK.value][PieceType.PAWN]
        == initial_black_hand.get(PieceType.PAWN, 0) - 1
    )

    game.undo_move()

    assert game.to_string() == initial_board_str, "Board state not restored"
    assert game.current_player == initial_player, "Current player not restored"
    assert game.move_count == initial_move_count, "Move count not restored"
    assert (
        game.hands[Color.BLACK.value] == initial_black_hand
    ), "Black's hand not restored"
    assert (
        game.get_piece(4, 4) is None
    ), "Dropped piece not removed from board after undo"


def test_undo_move_promotion_no_capture(game: ShogiGame):
    """Test undoing a promotion without a capture."""
    # Black pawn at (2,2), moves to (1,2) and promotes
    game.set_piece(2, 2, Piece(PieceType.PAWN, Color.BLACK))
    game.set_piece(1, 2, None)  # Ensure target is empty
    # Clear other pieces that might interfere with this specific test
    for r in range(9):
        for c in range(9):
            if (r, c) != (2, 2):
                game.set_piece(r, c, None)
    game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))  # Own king
    game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))  # Opponent king

    game.current_player = Color.BLACK
    initial_board_str = game.to_string()
    initial_player = game.current_player
    initial_move_count = game.move_count

    move: tuple = (2, 2, 1, 2, True)  # Promote
    game.make_move(move)

    promoted_piece = game.get_piece(1, 2)
    assert promoted_piece is not None
    assert promoted_piece.type == PieceType.PROMOTED_PAWN
    assert promoted_piece.color == Color.BLACK

    game.undo_move()

    assert game.to_string() == initial_board_str, "Board state not restored"
    assert game.current_player == initial_player, "Current player not restored"
    assert game.move_count == initial_move_count, "Move count not restored"

    original_piece = game.get_piece(2, 2)
    assert original_piece is not None
    assert original_piece.type == PieceType.PAWN
    assert original_piece.color == Color.BLACK
    assert game.get_piece(1, 2) is None


def test_undo_move_promotion_with_capture(game_with_promotion_capture: ShogiGame):
    """Test undoing a promotion with a capture."""
    game = game_with_promotion_capture
    # Black pawn at (2,2), White piece at (1,2). Black captures and promotes.
    game.set_piece(2, 2, Piece(PieceType.PAWN, Color.BLACK))
    game.set_piece(1, 2, Piece(PieceType.LANCE, Color.WHITE))
    # Clear other pieces
    for r in range(9):
        for c in range(9):
            if (r, c) not in [(2, 2), (1, 2)]:
                game.set_piece(r, c, None)
    game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))
    game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))

    game.current_player = Color.BLACK
    initial_black_hand = game.get_pieces_in_hand(Color.BLACK).copy()
    initial_white_hand = game.get_pieces_in_hand(Color.WHITE).copy()
    initial_board_str = game.to_string()
    initial_player = game.current_player
    initial_move_count = game.move_count

    move: tuple = (2, 2, 1, 2, True)  # Capture and promote
    game.make_move(move)

    promoted_capturing_piece = game.get_piece(1, 2)
    assert promoted_capturing_piece is not None
    assert promoted_capturing_piece.type == PieceType.PROMOTED_PAWN
    assert promoted_capturing_piece.color == Color.BLACK
    assert (
        game.hands[Color.BLACK.value][PieceType.LANCE]
        == initial_black_hand.get(PieceType.LANCE, 0) + 1
    )

    game.undo_move()

    assert game.to_string() == initial_board_str, "Board state not restored"
    assert game.current_player == initial_player, "Current player not restored"
    assert game.move_count == initial_move_count, "Move count not restored"
    assert (
        game.hands[Color.BLACK.value] == initial_black_hand
    ), "Black's hand not restored"
    assert (
        game.hands[Color.WHITE.value] == initial_white_hand
    ), "White's hand not restored"

    original_moved_piece = game.get_piece(2, 2)
    assert original_moved_piece is not None
    assert original_moved_piece.type == PieceType.PAWN

    original_captured_piece = game.get_piece(1, 2)
    assert original_captured_piece is not None
    assert original_captured_piece.type == PieceType.LANCE
    assert original_captured_piece.color == Color.WHITE


def test_undo_move_forced_promotion(game: ShogiGame):
    """Test undoing a forced promotion (e.g., pawn to last rank)."""
    # Black pawn at (1,2), moves to (0,2) and must promote
    game.set_piece(1, 2, Piece(PieceType.PAWN, Color.BLACK))
    game.set_piece(0, 2, None)
    for r in range(9):
        for c in range(9):
            if (r, c) not in [(1, 2), (0, 2)]:
                game.set_piece(r, c, None)
    game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))
    game.set_piece(
        0, 8, Piece(PieceType.KING, Color.WHITE)
    )  # King away from promotion square

    game.current_player = Color.BLACK
    initial_board_str = game.to_string()
    initial_player = game.current_player
    initial_move_count = game.move_count

    move: tuple = (1, 2, 0, 2, True)  # Forced promotion
    game.make_move(move)

    forced_promoted_piece = game.get_piece(0, 2)
    assert forced_promoted_piece is not None
    assert forced_promoted_piece.type == PieceType.PROMOTED_PAWN
    assert forced_promoted_piece.color == Color.BLACK

    game.undo_move()

    assert game.to_string() == initial_board_str, "Board state not restored"
    assert game.current_player == initial_player, "Current player not restored"
    assert game.move_count == initial_move_count, "Move count not restored"

    original_piece_before_forced_promo = game.get_piece(1, 2)
    assert original_piece_before_forced_promo is not None
    assert original_piece_before_forced_promo.type == PieceType.PAWN
    assert original_piece_before_forced_promo.color == Color.BLACK
    assert game.get_piece(0, 2) is None


def test_undo_move_multiple_moves(game: ShogiGame):
    """Test undoing multiple moves sequentially."""
    initial_board_str = game.to_string()
    initial_player = game.current_player
    initial_move_count = game.move_count
    initial_black_hand = game.hands[Color.BLACK.value].copy()
    initial_white_hand = game.hands[Color.WHITE.value].copy()

    # 1. Black P-7f (6,6) -> (5,6)
    move1: tuple = (6, 6, 5, 6, False)
    game.make_move(move1)
    # 2. White P-3d (2,3) -> (3,3)
    move2: tuple = (2, 3, 3, 3, False)
    game.make_move(move2)
    # 3. Black P-2f (6,1) -> P-2e (5,1) (capture, promote)
    # Setup: place a white pawn at (5,1) for capture
    # IMPORTANT: Store this piece to manually remove it if undo doesn't handle it.
    # However, a robust undo should handle this. Let's assume it does for now.
    game.board[5][1] = Piece(PieceType.PAWN, Color.WHITE)
    move3: tuple = (6, 1, 5, 1, True)  # (from_x, from_y, to_x, to_y, promote)
    game.make_move(move3)

    # Check state after 3 moves
    promoted_pawn_at_5_1 = game.get_piece(5, 1)
    assert promoted_pawn_at_5_1 is not None
    assert promoted_pawn_at_5_1.type == PieceType.PROMOTED_PAWN
    assert promoted_pawn_at_5_1.color == Color.BLACK
    assert (
        game.hands[Color.BLACK.value].get(PieceType.PAWN, 0)
        == initial_black_hand.get(PieceType.PAWN, 0) + 1
    )  # Captured white pawn

    # Undo move 3 (Black P-2e (promoted) captures P@2e -> P-2f)
    game.undo_move()
    pawn_at_6_1_after_undo1 = game.get_piece(6, 1)
    assert pawn_at_6_1_after_undo1 is not None
    assert pawn_at_6_1_after_undo1.type == PieceType.PAWN
    assert pawn_at_6_1_after_undo1.color == Color.BLACK

    captured_pawn_restored_at_5_1 = game.get_piece(5, 1)
    assert captured_pawn_restored_at_5_1 is not None
    assert captured_pawn_restored_at_5_1.type == PieceType.PAWN
    assert captured_pawn_restored_at_5_1.color == Color.WHITE
    assert (
        game.current_player == Color.BLACK
    )  # Should be Black's turn again (was White's before undo)
    assert game.hands[Color.BLACK.value].get(
        PieceType.PAWN, 0
    ) == initial_black_hand.get(PieceType.PAWN, 0)

    # Undo move 2 (White P-3d -> P-4d)
    game.undo_move()
    pawn_at_2_3_after_undo2 = game.get_piece(2, 3)
    assert pawn_at_2_3_after_undo2 is not None
    assert pawn_at_2_3_after_undo2.type == PieceType.PAWN
    assert pawn_at_2_3_after_undo2.color == Color.WHITE
    assert game.get_piece(3, 3) is None
    assert (
        game.current_player == Color.WHITE
    )  # Should be White's turn (was Black's before undo)

    # Undo move 1 (Black P-7f -> P-6f)
    game.undo_move()
    pawn_at_6_6_after_undo3 = game.get_piece(6, 6)
    assert pawn_at_6_6_after_undo3 is not None
    assert pawn_at_6_6_after_undo3.type == PieceType.PAWN
    assert pawn_at_6_6_after_undo3.color == Color.BLACK
    assert game.get_piece(5, 6) is None
    assert (
        game.current_player == Color.BLACK
    )  # Should be Black's turn (was White's before undo)

    # Manually clear the square that had the test-specific piece for move3
    # This piece was manually placed on the board for the capture in move3.
    # Standard undo logic would restore it after undoing move3 (correctly),
    # but it wouldn't be removed by undoing earlier, unrelated moves (move1, move2).
    # For this test to assert equality with the absolute initial board state,
    # this test-specific artifact must be cleared.
    game.board[5][1] = None

    # Critical check: board string representation
    current_board_str = game.to_string()
    assert (
        current_board_str == initial_board_str
    ), f"Board state not fully restored after multiple undos\nExpected:\n{initial_board_str}\nGot:\n{current_board_str}"

    assert (
        game.current_player == initial_player
    ), "Current player not restored after multiple undos"
    assert (
        game.move_count == initial_move_count
    ), "Move count not restored after multiple undos"
    assert (
        game.hands[Color.BLACK.value] == initial_black_hand
    ), "Black's hand not restored after multiple undos"
    assert (
        game.hands[Color.WHITE.value] == initial_white_hand
    ), "White's hand not restored after multiple undos"
