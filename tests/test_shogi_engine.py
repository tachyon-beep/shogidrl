"""
Unit tests for the Piece class in shogi_engine.py
"""

import numpy as np
from keisei.shogi.shogi_core_definitions import Piece, PieceType, Color
from keisei.shogi.shogi_game import ShogiGame


def test_piece_init():
    """Test Piece initialization and attributes."""
    p = Piece(PieceType.PAWN, Color.BLACK)
    assert p.type == PieceType.PAWN
    assert p.color == Color.BLACK
    assert not p.is_promoted
    p2 = Piece(PieceType.PROMOTED_BISHOP, Color.WHITE)
    assert p2.type == PieceType.PROMOTED_BISHOP
    assert p2.color == Color.WHITE
    assert p2.is_promoted


def test_piece_symbol():
    """Test Piece.symbol() returns correct string for type and color."""
    p = Piece(PieceType.PAWN, Color.BLACK)
    assert p.symbol() == "P"
    p2 = Piece(PieceType.PAWN, Color.WHITE)
    assert p2.symbol() == "p"
    p3 = Piece(PieceType.PROMOTED_PAWN, Color.BLACK)
    assert p3.symbol() == "+P"
    p4 = Piece(PieceType.PROMOTED_PAWN, Color.WHITE)
    assert p4.symbol() == "+p"
    p5 = Piece(PieceType.KING, Color.BLACK)
    assert p5.symbol() == "K"
    p6 = Piece(PieceType.KING, Color.WHITE)
    assert p6.symbol() == "k"


def test_shogigame_init_and_reset():
    """Test ShogiGame initialization and reset sets up the correct starting board."""
    game = ShogiGame()
    expected_types = [
        PieceType.LANCE,
        PieceType.KNIGHT,
        PieceType.SILVER,
        PieceType.GOLD,
        PieceType.KING,
        PieceType.GOLD,
        PieceType.SILVER,
        PieceType.KNIGHT,
        PieceType.LANCE,
    ]
    for c, t in enumerate(expected_types):
        p = game.get_piece(0, c)
        assert p is not None
        assert p.type == t
        assert p.color == Color.WHITE
    p_r = game.get_piece(1, 1)
    assert p_r is not None
    assert p_r.type == PieceType.ROOK and p_r.color == Color.WHITE
    p_b = game.get_piece(1, 7)
    assert p_b is not None
    assert p_b.type == PieceType.BISHOP and p_b.color == Color.WHITE
    for c in range(9):
        p = game.get_piece(2, c)
        assert p is not None
        assert p.type == PieceType.PAWN
        assert p.color == Color.WHITE
    for c in range(9):
        p = game.get_piece(6, c)
        assert p is not None
        assert p.type == PieceType.PAWN
        assert p.color == Color.BLACK
    p_b = game.get_piece(7, 1)
    assert p_b is not None
    assert p_b.type == PieceType.BISHOP and p_b.color == Color.BLACK
    p_r = game.get_piece(7, 7)
    assert p_r is not None
    assert p_r.type == PieceType.ROOK and p_r.color == Color.BLACK
    for c, t in enumerate(expected_types):
        p = game.get_piece(8, c)
        assert p is not None
        assert p.type == t
        assert p.color == Color.BLACK
    for r in range(3, 6):
        for c in range(9):
            assert game.get_piece(r, c) is None


def test_shogigame_to_string():
    """Test ShogiGame.to_string() returns a correct board string."""
    game = ShogiGame()
    board_str = game.to_string()
    assert isinstance(board_str, str)
    lines = board_str.split("\n")
    # Expected lines: 9 for board, 1 for file letters, 2 for hands, 1 for current player
    assert len(lines) == 13 

    # Helper to extract piece characters from a board line string
    def get_pieces_from_line(line_str):
        return "".join(line_str.split()[1:])

    assert get_pieces_from_line(lines[0]) == "lnsgkgsnl"  # White back rank (Rank 9)
    # lines[1] is White's bishop/rook rank
    assert get_pieces_from_line(lines[2]) == "ppppppppp"  # White pawn rank (Rank 7)
    # lines[3-5] are empty middle ranks
    assert get_pieces_from_line(lines[6]) == "PPPPPPPPP"  # Black pawn rank (Rank 3)
    # lines[7] is Black's bishop/rook rank
    assert get_pieces_from_line(lines[8]) == "LNSGKGSNL"  # Black back rank (Rank 1)


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
    pawn = Piece(PieceType.PAWN, Color.BLACK)
    moves = game.get_individual_piece_moves(pawn, 4, 4)
    assert (3, 4) in moves
    assert len(moves) == 1
    pawn_w = Piece(PieceType.PAWN, Color.WHITE)
    moves_w = game.get_individual_piece_moves(pawn_w, 4, 4)
    assert (5, 4) in moves_w
    assert len(moves_w) == 1
    prom_pawn = Piece(PieceType.PROMOTED_PAWN, Color.BLACK)
    moves_prom = game.get_individual_piece_moves(prom_pawn, 4, 4)
    expected = [(3, 4), (5, 4), (4, 3), (4, 5), (3, 3), (3, 5)]
    for m in expected:
        assert m in moves_prom
    king = Piece(PieceType.KING, Color.BLACK)
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
    lance = Piece(PieceType.LANCE, Color.BLACK)
    moves = game.get_individual_piece_moves(lance, 4, 4)
    expected = [(3, 4), (2, 4), (1, 4), (0, 4)]
    for m in expected:
        assert m in moves
    lance_w = Piece(PieceType.LANCE, Color.WHITE)
    moves_w = game.get_individual_piece_moves(lance_w, 4, 4)
    expected_w = [(5, 4), (6, 4), (7, 4), (8, 4)]
    for m in expected_w:
        assert m in moves_w
    prom_lance = Piece(PieceType.PROMOTED_LANCE, Color.BLACK)
    moves_prom = game.get_individual_piece_moves(prom_lance, 4, 4)
    expected_gold = [(3, 4), (5, 4), (4, 3), (4, 5), (3, 3), (3, 5)]
    for m in expected_gold:
        assert m in moves_prom
    knight = Piece(PieceType.KNIGHT, Color.BLACK)
    moves_k = game.get_individual_piece_moves(knight, 4, 4)
    assert (2, 3) in moves_k and (2, 5) in moves_k
    assert len(moves_k) == 2
    knight_w = Piece(PieceType.KNIGHT, Color.WHITE)
    moves_kw = game.get_individual_piece_moves(knight_w, 4, 4)
    assert (6, 3) in moves_kw and (6, 5) in moves_kw
    assert len(moves_kw) == 2
    prom_knight = Piece(PieceType.PROMOTED_KNIGHT, Color.BLACK)
    moves_promk = game.get_individual_piece_moves(prom_knight, 4, 4)
    expected_gold = [(3, 4), (5, 4), (4, 3), (4, 5), (3, 3), (3, 5)]
    for m in expected_gold:
        assert m in moves_promk


def test_get_individual_piece_moves_silver_gold():
    """Test get_individual_piece_moves for silver and gold (unpromoted and promoted)."""
    game = ShogiGame()
    silver = Piece(PieceType.SILVER, Color.BLACK)
    moves = game.get_individual_piece_moves(silver, 4, 4)
    expected = [(3, 4), (3, 3), (3, 5), (5, 3), (5, 5)]
    for m in expected:
        assert m in moves
    assert len(moves) == 5
    silver_w = Piece(PieceType.SILVER, Color.WHITE)
    moves_w = game.get_individual_piece_moves(silver_w, 4, 4)
    expected_w = [(5, 4), (5, 3), (5, 5), (3, 3), (3, 5)]
    for m in expected_w:
        assert m in moves_w
    assert len(moves_w) == 5
    prom_silver = Piece(PieceType.PROMOTED_SILVER, Color.BLACK)
    moves_prom = game.get_individual_piece_moves(prom_silver, 4, 4)
    expected_gold = [(3, 4), (5, 4), (4, 3), (4, 5), (3, 3), (3, 5)]
    for m in expected_gold:
        assert m in moves_prom
    gold = Piece(PieceType.GOLD, Color.BLACK)
    moves_gold = game.get_individual_piece_moves(gold, 4, 4)
    for m in expected_gold:
        assert m in moves_gold
    gold_w = Piece(PieceType.GOLD, Color.WHITE)
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
    bishop = Piece(PieceType.BISHOP, Color.BLACK)
    moves = game.get_individual_piece_moves(bishop, 4, 4)
    for d in range(1, 5):
        assert (4 - d, 4 - d) in moves
        assert (4 - d, 4 + d) in moves
        assert (4 + d, 4 - d) in moves
        assert (4 + d, 4 + d) in moves
    prom_bishop = Piece(PieceType.PROMOTED_BISHOP, Color.BLACK)
    moves_prom = game.get_individual_piece_moves(prom_bishop, 4, 4)
    for d in range(1, 5):
        assert (4 - d, 4 - d) in moves_prom
        assert (4 - d, 4 + d) in moves_prom
        assert (4 + d, 4 - d) in moves_prom
        assert (4 + d, 4 + d) in moves_prom
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        assert (4 + dr, 4 + dc) in moves_prom
    rook = Piece(PieceType.ROOK, Color.BLACK)
    moves_r = game.get_individual_piece_moves(rook, 4, 4)
    for d in range(1, 5):
        assert (4 - d, 4) in moves_r
        assert (4 + d, 4) in moves_r
        assert (4, 4 - d) in moves_r
        assert (4, 4 + d) in moves_r
    prom_rook = Piece(PieceType.PROMOTED_ROOK, Color.BLACK)
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
        assert game.is_nifu(Color.BLACK, col)
    # Remove pawn from file 4
    game.set_piece(6, 4, None)
    assert not game.is_nifu(Color.BLACK, 4)
    # Add a promoted pawn (should not count for Nifu)
    game.set_piece(5, 4, Piece(PieceType.PROMOTED_PAWN, Color.BLACK))
    assert not game.is_nifu(Color.BLACK, 4)
    # Add an unpromoted black pawn back
    game.set_piece(3, 4, Piece(PieceType.PAWN, Color.BLACK))
    assert game.is_nifu(Color.BLACK, 4)
    # White pawns
    for col in range(9):
        assert game.is_nifu(Color.WHITE, col)
    # Remove white pawn from file 2
    game.set_piece(2, 2, None)
    assert not game.is_nifu(Color.WHITE, 2)


def test_nifu_promoted_pawn_does_not_count():
    """Promoted pawns do not count for Nifu."""
    game = ShogiGame()
    for c in range(9):
        game.set_piece(6, c, None)
    game.set_piece(4, 4, Piece(PieceType.PROMOTED_PAWN, Color.BLACK))  # Promoted pawn
    assert not game.is_nifu(Color.BLACK, 4)
    game.set_piece(5, 4, Piece(PieceType.PAWN, Color.BLACK))
    assert game.is_nifu(Color.BLACK, 4)


def test_nifu_after_capture_and_drop():
    """Nifu after pawn is captured and dropped again."""
    game = ShogiGame()
    game.set_piece(6, 0, None)
    assert not game.is_nifu(Color.BLACK, 0)
    game.set_piece(3, 0, Piece(PieceType.PAWN, Color.BLACK))
    assert game.is_nifu(Color.BLACK, 0)


def test_nifu_promote_and_drop():
    """Nifu after pawn is promoted and a new pawn is dropped."""
    game = ShogiGame()
    game.set_piece(6, 1, None)
    game.set_piece(2, 1, Piece(PieceType.PROMOTED_PAWN, Color.BLACK))
    assert not game.is_nifu(Color.BLACK, 1)
    game.set_piece(4, 1, Piece(PieceType.PAWN, Color.BLACK))
    assert game.is_nifu(Color.BLACK, 1)


def test_uchi_fu_zume():
    """Test ShogiGame.is_uchi_fu_zume detects illegal pawn drop mate (Uchi Fu Zume)."""
    game = ShogiGame()
    # Set up a true mate: white king at (0,4), black pawn can drop at (1,4), king's escape squares blocked
    for r in range(9):
        for c in range(9):
            game.set_piece(r, c, None)
    game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))  # White king
    # Block king's escape squares
    game.set_piece(0, 3, Piece(PieceType.GOLD, Color.BLACK))  # Black gold
    game.set_piece(0, 5, Piece(PieceType.GOLD, Color.BLACK))  # Black gold
    game.set_piece(1, 3, Piece(PieceType.GOLD, Color.BLACK))  # Black gold
    game.set_piece(1, 5, Piece(PieceType.GOLD, Color.BLACK))  # Black gold
    # Black to drop pawn at (1,4) for mate
    assert game.is_uchi_fu_zume(1, 4, Color.BLACK)
    # If king can escape, not mate
    game.set_piece(0, 3, None)
    assert not game.is_uchi_fu_zume(1, 4, Color.BLACK)
    # If not a pawn drop, not mate (simulate with a gold drop)
    game.set_piece(1, 4, None)
    game.set_piece(1, 4, Piece(PieceType.GOLD, Color.BLACK))
    assert not game.is_uchi_fu_zume(1, 4, Color.BLACK)


def test_uchi_fu_zume_complex_escape():
    """Uchi Fu Zume: king's escape squares are all attacked or occupied by opponent pieces, leading to mate."""
    game = ShogiGame()
    for r in range(9):
        for c in range(9):
            game.set_piece(r, c, None)
    # Set up white king at (0,4)
    game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))
    # Black Golds at (1,3) and (1,5) cover some escape squares.
    # If King captures them, the landing square is attacked by Lances.
    game.set_piece(1, 3, Piece(PieceType.GOLD, Color.BLACK))
    game.set_piece(1, 5, Piece(PieceType.GOLD, Color.BLACK))

    # Add pieces to block other escape routes
    # Top left
    game.set_piece(0, 3, Piece(PieceType.SILVER, Color.BLACK))
    # Top right
    game.set_piece(0, 5, Piece(PieceType.SILVER, Color.BLACK))
    # Add Black Lances to cover squares (0,3), (1,3), (0,5), (1,5)
    # This makes capturing the Silvers or Golds unsafe for the King.
    game.set_piece(2, 3, Piece(PieceType.LANCE, Color.BLACK))
    game.set_piece(2, 5, Piece(PieceType.LANCE, Color.BLACK))

    # A pawn drop at (1,4) by Black checks the White King.
    # The King has no legal moves (all escape squares/captures lead to check).
    # This is checkmate by pawn drop, hence uchi-fu-zume.
    assert game.is_uchi_fu_zume(1, 4, Color.BLACK)


def test_uchi_fu_zume_non_pawn_drop():
    """Non-pawn drops do not trigger Uchi Fu Zume."""
    game = ShogiGame()
    for r in range(9):
        for c in range(9):
            game.set_piece(r, c, None)
    game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))
    game.set_piece(1, 4, Piece(PieceType.GOLD, Color.BLACK))  # Gold drop
    assert not game.is_uchi_fu_zume(1, 4, Color.BLACK)


def test_uchi_fu_zume_king_in_check():
    """Uchi Fu Zume: king is in check from another piece."""
    game = ShogiGame()
    for r in range(9):
        for c in range(9):
            game.set_piece(r, c, None)
    game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))
    game.set_piece(2, 4, Piece(PieceType.ROOK, Color.BLACK))  # Black rook gives check
    assert not game.is_uchi_fu_zume(1, 4, Color.BLACK)


def test_sennichite_detection():
    """Test ShogiGame detects Sennichite (fourfold repetition) and declares a draw."""
    game = ShogiGame()
    # Clear the board for a simple repetition test
    for r in range(9):
        for c in range(9):
            game.set_piece(r, c, None)
    # Place kings only
    game.set_piece(8, 4, Piece(PieceType.KING, Color.BLACK))
    game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))
    # Repeat a simple move back and forth 4 times
    for _ in range(4):
        move1 = (8, 4, 7, 4, False)  # Black king up
        move2 = (0, 4, 1, 4, False)  # White king down
        move3 = (7, 4, 8, 4, False)  # Black king back
        move4 = (1, 4, 0, 4, False)  # White king back
        game.make_move(move1)
        game.make_move(move2)
        game.make_move(move3)
        game.make_move(move4)
    # After 4 repetitions, Sennichite should be detected
    assert game.game_over, "Game should be over due to Sennichite."
    assert game.winner is None, "Sennichite should be a draw (winner=None)."


def test_sennichite_with_drops():
    """Sennichite with repetition involving drops."""
    game = ShogiGame()
    for r in range(9):
        for c in range(9):
            game.set_piece(r, c, None)
    game.set_piece(8, 4, Piece(PieceType.KING, Color.BLACK))
    game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))

    # Add pawns to both players' hands
    game.hands[Color.BLACK.value][PieceType.PAWN] = 10
    game.hands[Color.WHITE.value][PieceType.PAWN] = 10

    for _ in range(4):
        game.make_move((8, 4, 7, 4, False))
        game.make_move((0, 4, 1, 4, False))
        game.make_move((None, None, 8, 3, PieceType.PAWN))
        game.make_move((None, None, 0, 3, PieceType.PAWN))

        # Move the kings back
        game.make_move((7, 4, 8, 4, False))
        game.make_move((1, 4, 0, 4, False))

        # Capture the pawns to return them to hand
        piece = game.get_piece(8, 3)
        if piece:
            game.set_piece(8, 3, None)
            game.hands[Color.BLACK.value][
                PieceType.PAWN
            ] += 1  # Add back to black's hand

        piece = game.get_piece(0, 3)
        if piece:
            game.set_piece(0, 3, None)
            game.hands[Color.WHITE.value][
                PieceType.PAWN
            ] += 1  # Add back to white's hand

    assert game.is_sennichite()
    assert game.game_over
    assert game.winner is None


def test_sennichite_with_captures():
    """Sennichite with repetition involving captures."""
    # This test simplifies the capture repetition due to complexity,
    # focusing instead on validating that a fourfold repetition causes a draw
    game = ShogiGame()
    # Clear the board for a simple repetition test
    for r in range(9):
        for c in range(9):
            game.set_piece(r, c, None)
    # Place kings only
    game.set_piece(8, 4, Piece(PieceType.KING, Color.BLACK))
    game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))
    # Repeat a simple move back and forth 4 times
    for _ in range(4):
        move1 = (8, 4, 7, 4, False)  # Black king up
        move2 = (0, 4, 1, 4, False)  # White king down
        move3 = (7, 4, 8, 4, False)  # Black king back
        move4 = (1, 4, 0, 4, False)  # White king back
        game.make_move(move1)
        game.make_move(move2)
        game.make_move(move3)
        game.make_move(move4)

    # The sennichite detection is tested in test_sennichite_detection
    # So just validate that we are indeed getting 4 identical positions
    assert game.is_sennichite(), "Sennichite (fourfold repetition) should be detected."

    # After 4 repetitions, Sennichite should be detected and game marked as over
    assert game.game_over, "Game should be over due to Sennichite."
    assert game.winner is None, "Sennichite should be a draw (winner=None)."


def test_illegal_pawn_drop_last_rank():
    """Illegal pawn drop on last rank."""
    game = ShogiGame()
    for c in range(9):
        game.set_piece(0, c, None)
    # Add a pawn to black's hand
    game.hands[Color.BLACK.value][PieceType.PAWN] = 1
    # Attempt to drop a pawn on the last rank (row 0) for black
    assert not game.can_drop_piece(
        PieceType.PAWN, 0, 4, Color.BLACK
    )  # Should be illegal


def test_illegal_knight_drop_last_two_ranks():
    """Illegal knight drop on last two ranks."""
    game = ShogiGame()
    for c in range(9):
        game.set_piece(0, c, None)
        game.set_piece(1, c, None)
    # Add a knight to black's hand
    game.hands[Color.BLACK.value][PieceType.KNIGHT] = 1
    # Attempt to drop a knight on the last two ranks (row 0 and 1) for black
    assert not game.can_drop_piece(PieceType.KNIGHT, 0, 4, Color.BLACK)  # Last rank
    assert not game.can_drop_piece(
        PieceType.KNIGHT, 1, 4, Color.BLACK
    )  # Second-to-last rank


def test_illegal_lance_drop_last_rank():
    """Illegal lance drop on last rank."""
    game = ShogiGame()
    for c in range(9):
        game.set_piece(0, c, None)
    # Add a lance to black's hand
    game.hands[Color.BLACK.value][PieceType.LANCE] = 1
    # Attempt to drop a lance on the last rank (row 0) for black
    assert not game.can_drop_piece(
        PieceType.LANCE, 0, 4, Color.BLACK
    )  # Should be illegal


def test_checkmate_minimal():
    """Minimal checkmate scenario."""
    game = ShogiGame()
    # Clear board and hands
    for r_idx in range(9):
        for c_idx in range(9):
            game.set_piece(r_idx, c_idx, None)
    game.hands[Color.BLACK.value] = {}
    game.hands[Color.WHITE.value] = {}
    for piece_type in PieceType.get_unpromoted_types():
        game.hands[Color.BLACK.value][piece_type] = 0
        game.hands[Color.WHITE.value][piece_type] = 0
    game.move_history = []
    # Accessing the sennichite_history attribute which is initialized in shogi_rules_logic.check_for_sennichite
    # For a clean test setup, we ensure it's empty or reset if it were publicly settable.
    # Since it's not directly settable, we rely on a fresh game instance or ensure no prior moves led to its population.
    # For the purpose of this test, a fresh ShogiGame() instance handles this.
    # If sennichite_history were a public attribute on ShogiGame, it would be: game.sennichite_history = {}
    game.game_over = False
    game.winner = None
    game.current_player = Color.WHITE # White to make the checkmating move

    # Setup:
    # Black King at (8,4)
    # White Gold at (6,4) -> moves to (7,4) for checkmate
    # White Gold at (7,3) (covers Black King's escape to (8,3))
    # White Gold at (7,5) (covers Black King's escape to (8,5))
    game.set_piece(8, 4, Piece(PieceType.KING, Color.BLACK))
    game.set_piece(6, 4, Piece(PieceType.GOLD, Color.WHITE)) # The moving piece
    game.set_piece(7, 3, Piece(PieceType.GOLD, Color.WHITE))
    game.set_piece(7, 5, Piece(PieceType.GOLD, Color.WHITE))

    # White makes the checkmating move: Gold (6,4) -> (7,4)
    checkmating_move = (6, 4, 7, 4, False) # (r_from, c_from, r_to, c_to, promote)
    
    assert not game.game_over, "Game should not be over before the checkmating move."
    
    game.make_move(checkmating_move)

    assert game.game_over, "Game should be over after checkmate."
    assert game.winner == Color.WHITE, f"Winner should be White, but got {game.winner}"

def test_stalemate_minimal():
    """Minimal stalemate scenario."""
    game = ShogiGame()
    # Clear board and hands
    for r_idx in range(9):
        for c_idx in range(9):
            game.set_piece(r_idx, c_idx, None)
    game.hands[Color.BLACK.value] = {}
    game.hands[Color.WHITE.value] = {}
    for piece_type in PieceType.get_unpromoted_types():
        game.hands[Color.BLACK.value][piece_type] = 0
        game.hands[Color.WHITE.value][piece_type] = 0
    game.move_history = []
    # Similar to checkmate test, sennichite_history is managed internally.
    # A fresh game instance is sufficient for a clean state.
    game.game_over = False
    game.winner = None
    game.current_player = Color.WHITE # White to make the move leading to stalemate for Black

    # Setup:
    # Black King at (8,8)
    # White Gold at (6,8) (covers Black King's escape to (7,8))
    # White Gold at (8,6) (covers Black King's escape to (8,7))
    # White King at (6,6) (covers Black King's escape to (7,7))
    # White Pawn at (0,0) for White to make a non-disruptive move.
    game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))
    game.set_piece(6, 8, Piece(PieceType.GOLD, Color.WHITE))
    game.set_piece(8, 6, Piece(PieceType.GOLD, Color.WHITE))
    game.set_piece(6, 6, Piece(PieceType.KING, Color.WHITE))
    game.set_piece(0, 0, Piece(PieceType.PAWN, Color.WHITE)) # White's moving piece

    # White makes a move that doesn't affect the stalemate net: Pawn (0,0) -> (1,0)
    stalemating_move = (0, 0, 1, 0, False) # (r_from, c_from, r_to, c_to, promote)

    assert not game.game_over, "Game should not be over before the stalemating move."
    
    game.make_move(stalemating_move)

    assert game.game_over, "Game should be over after stalemate."
    assert game.winner is None, f"Winner should be None for stalemate, but got {game.winner}"
