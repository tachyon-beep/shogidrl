"""
Unit tests for the Piece class in shogi_engine.py
"""

# pylint: disable=protected-access
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
    # Test top row (White major pieces)
    expected_types = [1, 2, 3, 4, 7, 4, 3, 2, 1]
    for c, t in enumerate(expected_types):
        p = game.get_piece(0, c)
        assert p is not None
        assert p.type == t
        assert p.color == 1
    # Test 2nd row (White rook and bishop)
    p_r = game.get_piece(1, 1)
    assert p_r is not None
    assert p_r.type == 6 and p_r.color == 1
    p_b = game.get_piece(1, 7)
    assert p_b is not None
    assert p_b.type == 5 and p_b.color == 1
    # Test 3rd row (White pawns)
    for c in range(9):
        p = game.get_piece(2, c)
        assert p is not None
        assert p.type == 0
        assert p.color == 1
    # Test 7th row (Black pawns)
    for c in range(9):
        p = game.get_piece(6, c)
        assert p is not None
        assert p.type == 0
        assert p.color == 0
    # Test 8th row (Black bishop and rook)
    p_b = game.get_piece(7, 1)
    assert p_b is not None
    assert p_b.type == 5 and p_b.color == 0
    p_r = game.get_piece(7, 7)
    assert p_r is not None
    assert p_r.type == 6 and p_r.color == 0
    # Test 9th row (Black major pieces)
    for c, t in enumerate(expected_types):
        p = game.get_piece(8, c)
        assert p is not None
        assert p.type == t
        assert p.color == 0
    # Test empty squares
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
    """Test ShogiGame._is_on_board for valid and invalid coordinates (allowed for unit test)."""
    game = ShogiGame()
    # Valid
    assert game._is_on_board(0, 0)
    assert game._is_on_board(8, 8)
    assert game._is_on_board(4, 5)
    # Invalid
    assert not game._is_on_board(-1, 0)
    assert not game._is_on_board(0, -1)
    assert not game._is_on_board(9, 0)
    assert not game._is_on_board(0, 9)
    assert not game._is_on_board(10, 10)


def test_get_individual_piece_moves_pawn():
    """Test _get_individual_piece_moves for pawn (unpromoted and promoted)."""
    game = ShogiGame()
    # Black pawn at (4, 4)
    pawn = Piece(0, 0)
    moves = game._get_individual_piece_moves(pawn, 4, 4)
    assert (3, 4) in moves
    assert len(moves) == 1
    # White pawn at (4, 4)
    pawn_w = Piece(0, 1)
    moves_w = game._get_individual_piece_moves(pawn_w, 4, 4)
    assert (5, 4) in moves_w
    assert len(moves_w) == 1
    # Promoted pawn (moves as gold)
    prom_pawn = Piece(8, 0)
    prom_pawn.is_promoted = True
    moves_prom = game._get_individual_piece_moves(prom_pawn, 4, 4)
    # Gold moves for black: forward, left, right, and three backward-diagonals
    expected = [(3, 4), (4, 3), (4, 5), (5, 3), (5, 5)]
    for m in expected:
        assert m in moves_prom
    # King
    king = Piece(7, 0)
    moves_king = game._get_individual_piece_moves(king, 4, 4)
    assert (3, 3) in moves_king and (5, 5) in moves_king and (4, 5) in moves_king
    assert len(moves_king) == 8


def test_get_individual_piece_moves_lance_knight():
    """Test _get_individual_piece_moves for lance and knight (unpromoted and promoted)."""
    game = ShogiGame()
    # Black lance at (4, 4)
    lance = Piece(1, 0)
    moves = game._get_individual_piece_moves(lance, 4, 4)
    # Should be all squares above (3,4), (2,4), (1,4), (0,4)
    expected = [(3, 4), (2, 4), (1, 4), (0, 4)]
    for m in expected:
        assert m in moves
    # White lance at (4, 4)
    lance_w = Piece(1, 1)
    moves_w = game._get_individual_piece_moves(lance_w, 4, 4)
    expected_w = [(5, 4), (6, 4), (7, 4), (8, 4)]
    for m in expected_w:
        assert m in moves_w
    # Promoted lance (moves as gold)
    prom_lance = Piece(9, 0)
    prom_lance.is_promoted = True
    moves_prom = game._get_individual_piece_moves(prom_lance, 4, 4)
    # Gold moves for black: forward, left, right, and three backward-diagonals
    expected_gold = [(3, 4), (4, 3), (4, 5), (5, 3), (5, 5)]
    for m in expected_gold:
        assert m in moves_prom
    # Black knight at (4, 4)
    knight = Piece(2, 0)
    moves_k = game._get_individual_piece_moves(knight, 4, 4)
    assert (2, 3) in moves_k and (2, 5) in moves_k
    assert len(moves_k) == 2
    # White knight at (4, 4)
    knight_w = Piece(2, 1)
    moves_kw = game._get_individual_piece_moves(knight_w, 4, 4)
    assert (6, 3) in moves_kw and (6, 5) in moves_kw
    assert len(moves_kw) == 2
    # Promoted knight (moves as gold)
    prom_knight = Piece(10, 0)
    prom_knight.is_promoted = True
    moves_promk = game._get_individual_piece_moves(prom_knight, 4, 4)
    for m in expected_gold:
        assert m in moves_promk


def test_get_individual_piece_moves_silver_gold():
    """Test _get_individual_piece_moves for silver and gold (unpromoted and promoted)."""
    game = ShogiGame()
    # Black silver at (4, 4)
    silver = Piece(3, 0)
    moves = game._get_individual_piece_moves(silver, 4, 4)
    expected = [(3, 4), (3, 3), (3, 5), (5, 3), (5, 5)]
    for m in expected:
        assert m in moves
    assert len(moves) == 5
    # White silver at (4, 4)
    silver_w = Piece(3, 1)
    moves_w = game._get_individual_piece_moves(silver_w, 4, 4)
    expected_w = [(5, 4), (5, 3), (5, 5), (3, 3), (3, 5)]
    for m in expected_w:
        assert m in moves_w
    assert len(moves_w) == 5
    # Promoted silver (moves as gold)
    prom_silver = Piece(11, 0)
    prom_silver.is_promoted = True
    moves_prom = game._get_individual_piece_moves(prom_silver, 4, 4)
    expected_gold = [(3, 4), (4, 3), (4, 5), (5, 3), (5, 5)]
    for m in expected_gold:
        assert m in moves_prom
    # Black gold at (4, 4)
    gold = Piece(4, 0)
    moves_gold = game._get_individual_piece_moves(gold, 4, 4)
    for m in expected_gold:
        assert m in moves_gold
    # White gold at (4, 4)
    gold_w = Piece(4, 1)
    expected_gold_w = [(5, 4), (4, 3), (4, 5), (3, 3), (3, 5)]
    moves_gold_w = game._get_individual_piece_moves(gold_w, 4, 4)
    for m in expected_gold_w:
        assert m in moves_gold_w


def test_get_individual_piece_moves_bishop_rook():
    """Test _get_individual_piece_moves for bishop and rook (unpromoted and promoted)."""
    game = ShogiGame()
    # Bishop at (4, 4)
    bishop = Piece(5, 0)
    moves = game._get_individual_piece_moves(bishop, 4, 4)
    # Should include all diagonal squares
    for d in range(1, 5):
        assert (4 - d, 4 - d) in moves
        assert (4 - d, 4 + d) in moves
        assert (4 + d, 4 - d) in moves
        assert (4 + d, 4 + d) in moves
    # Promoted bishop (adds king's orthogonal moves)
    prom_bishop = Piece(12, 0)
    prom_bishop.is_promoted = True
    moves_prom = game._get_individual_piece_moves(prom_bishop, 4, 4)
    for d in range(1, 5):
        assert (4 - d, 4 - d) in moves_prom
        assert (4 - d, 4 + d) in moves_prom
        assert (4 + d, 4 - d) in moves_prom
        assert (4 + d, 4 + d) in moves_prom
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        assert (4 + dr, 4 + dc) in moves_prom
    # Rook at (4, 4)
    rook = Piece(6, 0)
    moves_r = game._get_individual_piece_moves(rook, 4, 4)
    for d in range(1, 5):
        assert (4 - d, 4) in moves_r
        assert (4 + d, 4) in moves_r
        assert (4, 4 - d) in moves_r
        assert (4, 4 + d) in moves_r
    # Promoted rook (adds king's diagonal moves)
    prom_rook = Piece(13, 0)
    prom_rook.is_promoted = True
    moves_promr = game._get_individual_piece_moves(prom_rook, 4, 4)
    for d in range(1, 5):
        assert (4 - d, 4) in moves_promr
        assert (4 + d, 4) in moves_promr
        assert (4, 4 - d) in moves_promr
        assert (4, 4 + d) in moves_promr
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        assert (4 + dr, 4 + dc) in moves_promr


def test_shogigame_get_observation():
    """Test ShogiGame.get_observation() returns correct shape and encodes board state."""
    import numpy as np
    game = ShogiGame()
    obs = game.get_observation()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (44, 9, 9)
    # Check that the current player plane is all 1 for Black at start
    assert np.all(obs[42] == 1.0)
    # Check that the move count plane is all zeros at start
    assert np.all(obs[43] == 0.0)
    # Check that there are 9 black pawns and 9 white pawns in the correct planes
    # Black pawns: channel 0, row 6
    assert np.all(obs[0, 6, :] == 1.0)
    # White pawns: channel 14, row 2
    assert np.all(obs[14, 2, :] == 1.0)
