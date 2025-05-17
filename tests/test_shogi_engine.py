"""
Unit tests for the Piece class in shogi_engine.py
"""

import numpy as np
from keisei.shogi_engine import Piece, ShogiGame


def test_piece_init():
    """Test Piece initialization and attributes."""
    p = Piece(0, 0)
    assert p.type == 0
    assert p.color == 0
    assert not p.is_promoted
    p2 = Piece(5, 1, True)
    assert p2.type == 5
    assert p2.color == 1
    assert p2.is_promoted


def test_piece_symbol():
    """Test Piece.symbol() returns correct string for type and color."""
    p = Piece(0, 0)
    assert p.symbol() == "P"
    p2 = Piece(0, 1)
    assert p2.symbol() == "p"
    p3 = Piece(8, 0)
    assert p3.symbol() == "+P"
    p4 = Piece(8, 1)
    assert p4.symbol() == "+p"
    p5 = Piece(7, 0)
    assert p5.symbol() == "K"
    p6 = Piece(7, 1)
    assert p6.symbol() == "k"


def test_shogigame_init_and_reset():
    """Test ShogiGame initialization and reset sets up the correct starting board."""
    game = ShogiGame()
    expected_types = [1, 2, 3, 4, 7, 4, 3, 2, 1]
    for c, t in enumerate(expected_types):
        p = game.get_piece(0, c)
        assert p is not None
        assert p.type == t
        assert p.color == 1
    p_r = game.get_piece(1, 1)
    assert p_r is not None
    assert p_r.type == 6 and p_r.color == 1
    p_b = game.get_piece(1, 7)
    assert p_b is not None
    assert p_b.type == 5 and p_b.color == 1
    for c in range(9):
        p = game.get_piece(2, c)
        assert p is not None
        assert p.type == 0
        assert p.color == 1
    for c in range(9):
        p = game.get_piece(6, c)
        assert p is not None
        assert p.type == 0
        assert p.color == 0
    p_b = game.get_piece(7, 1)
    assert p_b is not None
    assert p_b.type == 5 and p_b.color == 0
    p_r = game.get_piece(7, 7)
    assert p_r is not None
    assert p_r.type == 6 and p_r.color == 0
    for c, t in enumerate(expected_types):
        p = game.get_piece(8, c)
        assert p is not None
        assert p.type == t
        assert p.color == 0
    for r in range(3, 6):
        for c in range(9):
            assert game.get_piece(r, c) is None


def test_shogigame_to_string():
    """Test ShogiGame.to_string() returns a correct board string."""
    game = ShogiGame()
    board_str = game.to_string()
    assert isinstance(board_str, str)
    lines = board_str.split("\n")
    assert len(lines) == 9
    assert lines[0].replace(" ", "") == "lnsgkgsnl"
    assert lines[2].replace(" ", "") == "ppppppppp"
    assert lines[6].replace(" ", "") == "PPPPPPPPP"
    assert lines[8].replace(" ", "") == "LNSGKGSNL"


def test_shogigame_is_on_board():
    """Test ShogiGame.is_on_board for valid and invalid coordinates."""
    game = ShogiGame()
    assert game.is_on_board(0, 0)
    assert game.is_on_board(8, 8)
    assert game.is_on_board(4, 5)
    assert not game.is_on_board(-1, 0)
    assert not game.is_on_board(0, -1)
    assert not game.is_on_board(9, 0)
    assert not game.is_on_board(0, 9)
    assert not game.is_on_board(10, 10)


def test_get_individual_piece_moves_pawn():
    """Test get_individual_piece_moves for pawn (unpromoted and promoted)."""
    game = ShogiGame()
    pawn = Piece(0, 0)
    moves = game.get_individual_piece_moves(pawn, 4, 4)
    assert (3, 4) in moves
    assert len(moves) == 1
    pawn_w = Piece(0, 1)
    moves_w = game.get_individual_piece_moves(pawn_w, 4, 4)
    assert (5, 4) in moves_w
    assert len(moves_w) == 1
    prom_pawn = Piece(8, 0)
    prom_pawn.is_promoted = True
    moves_prom = game.get_individual_piece_moves(prom_pawn, 4, 4)
    expected = [(3, 4), (5, 4), (4, 3), (4, 5), (3, 3), (3, 5)]
    for m in expected:
        assert m in moves_prom
    king = Piece(7, 0)
    moves_king = game.get_individual_piece_moves(king, 4, 4)
    assert (3, 3) in moves_king and (5, 5) in moves_king and (4, 5) in moves_king
    assert len(moves_king) == 8


def test_get_individual_piece_moves_lance_knight():
    """Test get_individual_piece_moves for lance and knight (unpromoted and promoted)."""
    game = ShogiGame()
    # Clear the board for pure move generation
    for r in range(9):
        for c in range(9):
            game.set_piece(r, c, None)
    lance = Piece(1, 0)
    moves = game.get_individual_piece_moves(lance, 4, 4)
    expected = [(3, 4), (2, 4), (1, 4), (0, 4)]
    for m in expected:
        assert m in moves
    lance_w = Piece(1, 1)
    moves_w = game.get_individual_piece_moves(lance_w, 4, 4)
    expected_w = [(5, 4), (6, 4), (7, 4), (8, 4)]
    for m in expected_w:
        assert m in moves_w
    prom_lance = Piece(9, 0)
    prom_lance.is_promoted = True
    moves_prom = game.get_individual_piece_moves(prom_lance, 4, 4)
    expected_gold = [(3, 4), (5, 4), (4, 3), (4, 5), (3, 3), (3, 5)]
    for m in expected_gold:
        assert m in moves_prom
    knight = Piece(2, 0)
    moves_k = game.get_individual_piece_moves(knight, 4, 4)
    assert (2, 3) in moves_k and (2, 5) in moves_k
    assert len(moves_k) == 2
    knight_w = Piece(2, 1)
    moves_kw = game.get_individual_piece_moves(knight_w, 4, 4)
    assert (6, 3) in moves_kw and (6, 5) in moves_kw
    assert len(moves_kw) == 2
    prom_knight = Piece(10, 0)
    prom_knight.is_promoted = True
    moves_promk = game.get_individual_piece_moves(prom_knight, 4, 4)
    expected_gold = [(3, 4), (5, 4), (4, 3), (4, 5), (3, 3), (3, 5)]
    for m in expected_gold:
        assert m in moves_promk


def test_get_individual_piece_moves_silver_gold():
    """Test get_individual_piece_moves for silver and gold (unpromoted and promoted)."""
    game = ShogiGame()
    silver = Piece(3, 0)
    moves = game.get_individual_piece_moves(silver, 4, 4)
    expected = [(3, 4), (3, 3), (3, 5), (5, 3), (5, 5)]
    for m in expected:
        assert m in moves
    assert len(moves) == 5
    silver_w = Piece(3, 1)
    moves_w = game.get_individual_piece_moves(silver_w, 4, 4)
    expected_w = [(5, 4), (5, 3), (5, 5), (3, 3), (3, 5)]
    for m in expected_w:
        assert m in moves_w
    assert len(moves_w) == 5
    prom_silver = Piece(11, 0)
    prom_silver.is_promoted = True
    moves_prom = game.get_individual_piece_moves(prom_silver, 4, 4)
    expected_gold = [(3, 4), (5, 4), (4, 3), (4, 5), (3, 3), (3, 5)]
    for m in expected_gold:
        assert m in moves_prom
    gold = Piece(4, 0)
    moves_gold = game.get_individual_piece_moves(gold, 4, 4)
    for m in expected_gold:
        assert m in moves_gold
    gold_w = Piece(4, 1)
    expected_gold_w = [(5, 4), (3, 4), (4, 3), (4, 5), (5, 3), (5, 5)]
    moves_gold_w = game.get_individual_piece_moves(gold_w, 4, 4)
    for m in expected_gold_w:
        assert m in moves_gold_w


def test_get_individual_piece_moves_bishop_rook():
    """Test get_individual_piece_moves for bishop and rook (unpromoted and promoted)."""
    game = ShogiGame()
    # Clear the board for pure move generation
    for r in range(9):
        for c in range(9):
            game.set_piece(r, c, None)
    bishop = Piece(5, 0)
    moves = game.get_individual_piece_moves(bishop, 4, 4)
    for d in range(1, 5):
        assert (4 - d, 4 - d) in moves
        assert (4 - d, 4 + d) in moves
        assert (4 + d, 4 - d) in moves
        assert (4 + d, 4 + d) in moves
    prom_bishop = Piece(12, 0)
    prom_bishop.is_promoted = True
    moves_prom = game.get_individual_piece_moves(prom_bishop, 4, 4)
    for d in range(1, 5):
        assert (4 - d, 4 - d) in moves_prom
        assert (4 - d, 4 + d) in moves_prom
        assert (4 + d, 4 - d) in moves_prom
        assert (4 + d, 4 + d) in moves_prom
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        assert (4 + dr, 4 + dc) in moves_prom
    rook = Piece(6, 0)
    moves_r = game.get_individual_piece_moves(rook, 4, 4)
    for d in range(1, 5):
        assert (4 - d, 4) in moves_r
        assert (4 + d, 4) in moves_r
        assert (4, 4 - d) in moves_r
        assert (4, 4 + d) in moves_r
    prom_rook = Piece(13, 0)
    prom_rook.is_promoted = True
    moves_promr = game.get_individual_piece_moves(prom_rook, 4, 4)
    for d in range(1, 5):
        assert (4 - d, 4) in moves_promr
        assert (4 + d, 4) in moves_promr
        assert (4, 4 - d) in moves_promr
        assert (4, 4 + d) in moves_promr
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        assert (4 + dr, 4 + dc) in moves_promr


def test_shogigame_get_observation():
    """Test ShogiGame.get_observation() returns correct shape and encodes board state."""
    game = ShogiGame()
    obs = game.get_observation()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (46, 9, 9)
    assert np.all(obs[42] == 1.0)
    assert np.all(obs[43] == 0.0)
    assert np.all(obs[44] == 0.0)
    assert np.all(obs[45] == 0.0)
    assert np.all(obs[0, 6, :] == 1.0)
    assert np.all(obs[14, 2, :] == 1.0)


def test_nifu_detection():
    """Test ShogiGame.is_nifu detects Nifu (double pawn) correctly."""
    game = ShogiGame()
    # Black has pawns on all files at row 6
    for col in range(9):
        assert game.is_nifu(0, col)
    # Remove pawn from file 4
    game.set_piece(6, 4, None)
    assert not game.is_nifu(0, 4)
    # Add a promoted pawn (should not count for Nifu)
    game.set_piece(5, 4, Piece(8, 0, True))
    assert not game.is_nifu(0, 4)
    # Add an unpromoted black pawn back
    game.set_piece(3, 4, Piece(0, 0, False))
    assert game.is_nifu(0, 4)
    # White pawns
    for col in range(9):
        assert game.is_nifu(1, col)
    # Remove white pawn from file 2
    game.set_piece(2, 2, None)
    assert not game.is_nifu(1, 2)


def test_uchi_fu_zume():
    """Test ShogiGame.is_uchi_fu_zume detects illegal pawn drop mate (Uchi Fu Zume)."""
    game = ShogiGame()
    # Set up a true mate: white king at (0,4), black pawn can drop at (1,4), king's escape squares blocked
    for r in range(9):
        for c in range(9):
            game.set_piece(r, c, None)
    game.set_piece(0, 4, Piece(7, 1))  # White king
    # Block king's escape squares
    game.set_piece(0, 3, Piece(4, 0))  # Black gold
    game.set_piece(0, 5, Piece(4, 0))  # Black gold
    game.set_piece(1, 3, Piece(4, 0))  # Black gold
    game.set_piece(1, 5, Piece(4, 0))  # Black gold
    # Black to drop pawn at (1,4) for mate
    assert game.is_uchi_fu_zume(1, 4, 0)
    # If king can escape, not mate
    game.set_piece(0, 3, None)
    assert not game.is_uchi_fu_zume(1, 4, 0)
    # If not a pawn drop, not mate (simulate with a gold drop)
    game.set_piece(1, 4, None)
    game.set_piece(1, 4, Piece(4, 0))
    assert not game.is_uchi_fu_zume(1, 4, 0)


def test_sfen_encode_move_stub():
    """Test sfen_encode_move NotImplementedError."""
    game = ShogiGame()
    try:
        game.sfen_encode_move((6, 6, 6, 5, 0))
    except NotImplementedError:
        pass
    else:
        assert False, "sfen_encode_move should raise NotImplementedError initially"


def test_get_legal_moves_initial_position():
    """Test get_legal_moves returns plausible legal moves for initial position (Black to move)."""
    game = ShogiGame()
    moves = game.get_legal_moves()
    # All Black pawns should be able to move forward one square (row 6 to 5)
    pawn_moves = [(6, c, 5, c, 0) for c in range(9)]
    for m in pawn_moves:
        assert m in moves
    # No move should capture own piece
    for m in moves:
        r_from, c_from, r_to, c_to, _ = m
        piece = game.get_piece(r_from, c_from)
        error_message = (
            f"Move {m} originates from an empty square ({r_from},{c_from}). "
            "This implies a bug in get_legal_moves."
        )
        assert piece is not None, error_message
        target = game.get_piece(r_to, c_to)
        if target:
            assert target.color != piece.color


def test_make_move_pawn_forward():
    """Test make_move moves a black pawn forward and updates state."""
    game = ShogiGame()
    move = (6, 0, 5, 0, 0)  # Black pawn at (6,0) moves to (5,0)
    game.make_move(move)
    assert game.get_piece(6, 0) is None
    p = game.get_piece(5, 0)
    assert p is not None and p.type == 0 and p.color == 0
    assert game.current_player == 1  # White's turn
    assert game.move_count == 1


def test_make_move_and_undo_simple_pawn():
    """Test make_move and undo_move for a simple pawn move (no capture, no promotion)."""
    game = ShogiGame()
    # Save initial state
    initial_board = [[game.get_piece(r, c) for c in range(9)] for r in range(9)]
    initial_player = game.current_player
    initial_move_count = game.move_count
    initial_history_len = len(game.move_history)

    move = (6, 0, 5, 0, 0)  # Black pawn at (6,0) moves to (5,0)
    game.make_move(move)
    # After move, pawn should be at (5,0), (6,0) empty, player switched, move_count incremented
    assert game.get_piece(6, 0) is None
    p = game.get_piece(5, 0)
    assert p is not None and p.type == 0 and p.color == 0
    assert game.current_player != initial_player
    assert game.move_count == initial_move_count + 1
    assert len(game.move_history) == initial_history_len + 1

    # Undo move
    game.undo_move()
    # Board should be restored
    for r in range(9):
        for c in range(9):
            orig = initial_board[r][c]
            curr = game.get_piece(r, c)
            if orig is None:
                assert curr is None
            else:
                assert curr is not None
                assert curr.type == orig.type
                assert curr.color == orig.color
                assert curr.is_promoted == orig.is_promoted
    assert game.current_player == initial_player
    assert game.move_count == initial_move_count
    assert len(game.move_history) == initial_history_len


def test_is_in_check_basic():
    """Test _is_in_check for both players in simple scenarios."""
    game = ShogiGame()
    # Clear the board for pure check logic
    for r in range(9):
        for c in range(9):
            game.set_piece(r, c, None)
    # Place black king at (8,4), white king at (0,4)
    game.set_piece(8, 4, Piece(7, 0))
    game.set_piece(0, 4, Piece(7, 1))
    # Neither king is in check
    assert not game.is_in_check(0)
    assert not game.is_in_check(1)
    # Place a white rook to check black king
    game.set_piece(5, 4, Piece(6, 1))  # White rook at (5,4)
    assert game.is_in_check(0)
    assert not game.is_in_check(1)
    # Remove white rook, place black rook to check white king
    game.set_piece(5, 4, None)
    game.set_piece(3, 4, Piece(6, 0))  # Black rook at (3,4)
    assert game.is_in_check(1)
    assert not game.is_in_check(0)


def test_sennichite_detection():
    """Test ShogiGame detects Sennichite (fourfold repetition) and declares a draw."""
    game = ShogiGame()
    # Clear the board for a simple repetition test
    for r in range(9):
        for c in range(9):
            game.set_piece(r, c, None)
    # Place kings only
    game.set_piece(8, 4, Piece(7, 0))
    game.set_piece(0, 4, Piece(7, 1))
    # Repeat a simple move back and forth 4 times
    for _ in range(4):
        move1 = (8, 4, 7, 4, 0)  # Black king up
        move2 = (0, 4, 1, 4, 0)  # White king down
        move3 = (7, 4, 8, 4, 0)  # Black king back
        move4 = (1, 4, 0, 4, 0)  # White king back
        game.make_move(move1)
        game.make_move(move2)
        game.make_move(move3)
        game.make_move(move4)
    # After 4 repetitions, Sennichite should be detected
    assert game.game_over, "Game should be over due to Sennichite."
    assert game.winner is None, "Sennichite should be a draw (winner=None)."
