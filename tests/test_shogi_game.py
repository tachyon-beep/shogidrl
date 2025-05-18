"""
Unit tests for ShogiGame class in shogi_game.py
"""
import pytest
import numpy as np
from keisei.shogi.shogi_game import ShogiGame
from keisei.shogi.shogi_core_definitions import Piece, PieceType, Color, OBS_UNPROMOTED_ORDER, OBS_PROMOTED_ORDER

@pytest.fixture
def new_game() -> ShogiGame:
    """Returns a ShogiGame instance initialized to the starting position."""
    return ShogiGame()

def test_get_observation_initial_state_dimensions(new_game: ShogiGame):
    """Test the dimensions of the observation from the initial state."""
    obs = new_game.get_observation()
    assert isinstance(obs, np.ndarray), "Observation should be a numpy array"
    # Expected channels: 14 for current player pieces (8 unpromoted, 6 promoted)
    #                    14 for opponent pieces (8 unpromoted, 6 promoted)
    #                    14 for current player hand (7 types)
    #                    14 for opponent hand (7 types) - wait, this is not how it's structured.
    # Current structure:
    # Player pieces (14 planes: 8 unpromoted, 6 promoted)
    # Opponent pieces (14 planes: 8 unpromoted, 6 promoted)
    # Player 1 hand (7 planes: P, L, N, S, G, B, R)
    # Player 2 hand (7 planes: P, L, N, S, G, B, R)
    # Color to play (1 plane)
    # Total: 14 + 14 + 7 + 7 + 1 = 43.
    # The original code had 46, with 2 reserved. Let's stick to 43 for now based on current logic.
    # If shogi_game_io.py uses 46, this test will need adjustment.
    # From shogi_core_definitions.py:
    # Total expected channels for current obs structure: 28 (board) + 14 (hands) + 1 (meta: current player) = 43
    # The comment in shogi_core_definitions.py says:
    # Total board piece planes = 14 * 2 = 28
    # Hands: 7 piece types * 2 players = 14 planes
    # Other: current player, move count = 2 planes -> This implies 44.
    # Let's re-check shogi_game_io.py for the exact channel count.
    # Assuming the 44 channel structure (28 board, 14 hands, 2 meta)
    assert obs.shape == (44, 9, 9), "Observation shape is incorrect"

def test_get_observation_hand_pieces_black_one_pawn(new_game: ShogiGame):
    """Test observation when Black has one pawn in hand."""
    new_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    obs = new_game.get_observation()

    # Hand piece channels for Black start after board piece channels (28)
    # Black Pawn is the first piece in OBS_UNPROMOTED_ORDER
    pawn_hand_channel_index = 28 + OBS_UNPROMOTED_ORDER.index(PieceType.PAWN)
    
    # The entire plane for this hand piece should be 1.0
    assert np.all(obs[pawn_hand_channel_index] == 1.0), "Black pawn hand plane incorrect"
    # Ensure other Black hand planes are 0
    for i, pt in enumerate(OBS_UNPROMOTED_ORDER):
        if pt != PieceType.PAWN:
            assert np.all(obs[28 + i] == 0.0), f"Black {pt.name} hand plane should be 0"

def test_get_observation_hand_pieces_white_one_rook(new_game: ShogiGame):
    """Test observation when White has one rook in hand."""
    new_game.hands[Color.WHITE.value][PieceType.ROOK] = 1
    obs = new_game.get_observation()

    # Hand piece channels for White start after Black's hand channels (28 + 7)
    rook_hand_channel_index = 28 + 7 + OBS_UNPROMOTED_ORDER.index(PieceType.ROOK)
    
    assert np.all(obs[rook_hand_channel_index] == 1.0), "White rook hand plane incorrect"
    # Ensure other White hand planes are 0
    for i, pt in enumerate(OBS_UNPROMOTED_ORDER):
        if pt != PieceType.ROOK:
            assert np.all(obs[28 + 7 + i] == 0.0), f"White {pt.name} hand plane should be 0"


def test_get_observation_multiple_hand_pieces_mixed_players(new_game: ShogiGame):
    """Test observation with multiple pieces in hand for both players."""
    new_game.hands[Color.BLACK.value][PieceType.PAWN] = 3
    new_game.hands[Color.BLACK.value][PieceType.GOLD] = 1
    new_game.hands[Color.WHITE.value][PieceType.BISHOP] = 2
    new_game.hands[Color.WHITE.value][PieceType.SILVER] = 1
    
    obs = new_game.get_observation()

    # Black's hand
    pawn_idx_black = 28 + OBS_UNPROMOTED_ORDER.index(PieceType.PAWN)
    gold_idx_black = 28 + OBS_UNPROMOTED_ORDER.index(PieceType.GOLD)
    assert np.all(obs[pawn_idx_black] == 1.0), "Black 3 pawns hand plane incorrect" # Should be 1 if any
    assert np.all(obs[gold_idx_black] == 1.0), "Black 1 gold hand plane incorrect"

    # White's hand
    bishop_idx_white = 28 + 7 + OBS_UNPROMOTED_ORDER.index(PieceType.BISHOP)
    silver_idx_white = 28 + 7 + OBS_UNPROMOTED_ORDER.index(PieceType.SILVER)
    assert np.all(obs[bishop_idx_white] == 1.0), "White 2 bishops hand plane incorrect"
    assert np.all(obs[silver_idx_white] == 1.0), "White 1 silver hand plane incorrect"

    # Check a piece not in hand for Black
    lance_idx_black = 28 + OBS_UNPROMOTED_ORDER.index(PieceType.LANCE)
    assert np.all(obs[lance_idx_black] == 0.0), "Black lance hand plane should be 0"

    # Check a piece not in hand for White
    rook_idx_white = 28 + 7 + OBS_UNPROMOTED_ORDER.index(PieceType.ROOK)
    assert np.all(obs[rook_idx_white] == 0.0), "White rook hand plane should be 0"

def test_get_observation_empty_hands(new_game: ShogiGame):
    """Test observation when both players have empty hands (initial state)."""
    # new_game fixture already has empty hands initially
    obs = new_game.get_observation()

    # All hand planes (28 through 28 + 7 + 7 - 1 = 41) should be 0
    for i in range(14): # 7 for black, 7 for white
        hand_channel_index = 28 + i
        assert np.all(obs[hand_channel_index] == 0.0), f"Hand channel {hand_channel_index} should be 0 for empty hands"

def test_get_observation_current_player_plane_black_turn(new_game: ShogiGame):
    """Test current player plane when it's Black's turn."""
    new_game.current_player = Color.BLACK
    obs = new_game.get_observation()
    # Current player plane is the second to last one (index 42 if 44 total channels)
    # If black is current player, this plane is all 0s.
    current_player_plane_index = 42 # 28 board + 14 hand = 42. This is the first meta plane.
    assert np.all(obs[current_player_plane_index] == 0.0), "Current player plane incorrect for Black's turn"

def test_get_observation_current_player_plane_white_turn(new_game: ShogiGame):
    """Test current player plane when it's White's turn."""
    new_game.current_player = Color.WHITE
    obs = new_game.get_observation()
    # If white is current player, this plane is all 1s.
    current_player_plane_index = 42
    assert np.all(obs[current_player_plane_index] == 1.0), "Current player plane incorrect for White's turn"

def test_get_observation_move_count_plane(new_game: ShogiGame):
    """Test move count plane."""
    new_game.move_count = 5
    obs = new_game.get_observation()
    # Move count plane is the last one (index 43 if 44 total channels)
    # Value should be move_count / MAX_GAME_MOVES (assume MAX_GAME_MOVES = 300 for now, as in shogi_game_io.py)
    # This needs to be confirmed from shogi_game_io.py or config
    # For now, let's assume shogi_game_io.py uses a MAX_GAME_MOVES.
    # If MAX_GAME_MOVES is, for example, 300, then 5/300.
    # The actual value depends on MAX_GAME_MOVES in shogi_game_io.py
    # Let's check if shogi_game_io.py is accessible or if we need to mock/assume
    # From shogi_game_io.py: MAX_GAME_MOVES = 300
    move_count_plane_index = 43
    expected_value = 5 / 300.0
    assert np.all(obs[move_count_plane_index] == expected_value), "Move count plane incorrect"

def test_get_observation_board_pieces_consistency_after_reset(new_game: ShogiGame):
    """Test that board piece planes are correctly set after a game reset (initial position)."""
    # new_game is already reset
    obs = new_game.get_observation()

    # Check a few key pieces for Black (current player perspective)
    # Black's Pawn at (6,0) (row 6, col 0)
    # Pawn is OBS_UNPROMOTED_ORDER[0]
    black_pawn_plane = OBS_UNPROMOTED_ORDER.index(PieceType.PAWN)
    assert obs[black_pawn_plane, 6, 0] == 1.0, "Black pawn at (6,0) not found in observation"
    
    # Black's Rook at (7,7)
    black_rook_plane = OBS_UNPROMOTED_ORDER.index(PieceType.ROOK)
    assert obs[black_rook_plane, 7, 7] == 1.0, "Black rook at (7,7) not found in observation"

    # Check a few key pieces for White (opponent perspective)
    # White's Pawn at (2,0)
    # Opponent planes start after current player's 14 planes (i.e., at index 14)
    white_pawn_plane = 14 + OBS_UNPROMOTED_ORDER.index(PieceType.PAWN)
    assert obs[white_pawn_plane, 2, 0] == 1.0, "White pawn at (2,0) not found in observation"

    # White's King at (0,4)
    white_king_plane = 14 + OBS_UNPROMOTED_ORDER.index(PieceType.KING)
    assert obs[white_king_plane, 0, 4] == 1.0, "White king at (0,4) not found in observation"

    # Ensure a square that should be empty for a piece type is 0
    assert obs[black_pawn_plane, 0, 0] == 0.0, "Square (0,0) should be empty of black pawns"
    assert obs[white_pawn_plane, 6, 0] == 0.0, "Square (6,0) should be empty of white pawns"

def test_get_observation_promoted_piece_on_board(new_game: ShogiGame):
    """Test observation when a promoted piece is on the board."""
    # Place a promoted pawn (Tokin) for Black at (2,2)
    new_game.set_piece(2, 2, Piece(PieceType.PROMOTED_PAWN, Color.BLACK))
    new_game.current_player = Color.BLACK # Ensure perspective is Black's
    obs = new_game.get_observation()

    # Promoted pawn for current player (Black)
    # Promoted planes start after unpromoted planes (index 8 for current player)
    promoted_pawn_plane = 8 + OBS_PROMOTED_ORDER.index(PieceType.PROMOTED_PAWN)
    assert obs[promoted_pawn_plane, 2, 2] == 1.0, "Black Promoted Pawn at (2,2) not found"
    
    # Ensure the unpromoted pawn plane is 0 at that location for Black
    unpromoted_pawn_plane = OBS_UNPROMOTED_ORDER.index(PieceType.PAWN)
    assert obs[unpromoted_pawn_plane, 2, 2] == 0.0, "Unpromoted Black Pawn should not be at (2,2)"

    # Place a promoted rook (Dragon) for White at (5,5)
    new_game.set_piece(5, 5, Piece(PieceType.PROMOTED_ROOK, Color.WHITE))
    obs = new_game.get_observation() # Re-get obs after change

    # Promoted rook for opponent (White)
    # Opponent planes start at 14. Promoted opponent planes start at 14 + 8 = 22.
    promoted_rook_plane_opponent = 14 + 8 + OBS_PROMOTED_ORDER.index(PieceType.PROMOTED_ROOK)
    assert obs[promoted_rook_plane_opponent, 5, 5] == 1.0, "White Promoted Rook at (5,5) not found"

    # Ensure unpromoted rook plane is 0 at that location for White
    unpromoted_rook_plane_opponent = 14 + OBS_UNPROMOTED_ORDER.index(PieceType.ROOK)
    assert obs[unpromoted_rook_plane_opponent, 5, 5] == 0.0, "Unpromoted White Rook should not be at (5,5)"

