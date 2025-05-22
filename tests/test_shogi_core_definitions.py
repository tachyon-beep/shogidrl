"""
Unit tests for shogi_core_definitions.py
"""

import copy
import pytest
from keisei.shogi.shogi_core_definitions import (
    Piece,
    PieceType,
    Color,
    KIF_PIECE_SYMBOLS,
    SYMBOL_TO_PIECE_TYPE,
    BASE_TO_PROMOTED_TYPE,
    PROMOTED_TO_BASE_TYPE,
    PIECE_TYPE_TO_HAND_TYPE,
    get_piece_type_from_symbol,
    PROMOTED_TYPES_SET,
    OBS_UNPROMOTED_ORDER,
    OBS_PROMOTED_ORDER,
)

# Tests for Piece class


def test_piece_creation_and_attributes():
    p_pawn_black = Piece(PieceType.PAWN, Color.BLACK)
    assert p_pawn_black.type == PieceType.PAWN
    assert p_pawn_black.color == Color.BLACK
    assert not p_pawn_black.is_promoted

    p_prom_lance_white = Piece(PieceType.PROMOTED_LANCE, Color.WHITE)
    assert p_prom_lance_white.type == PieceType.PROMOTED_LANCE
    assert p_prom_lance_white.color == Color.WHITE
    assert p_prom_lance_white.is_promoted

    with pytest.raises(TypeError):
        Piece("PAWN", Color.BLACK)  # type: ignore
    with pytest.raises(TypeError):
        Piece(PieceType.PAWN, "BLACK")  # type: ignore


def test_piece_promote():
    p_pawn = Piece(PieceType.PAWN, Color.BLACK)
    p_pawn.promote()
    assert p_pawn.type == PieceType.PROMOTED_PAWN
    assert p_pawn.is_promoted

    p_gold = Piece(PieceType.GOLD, Color.BLACK)
    p_gold.promote()  # Should have no effect
    assert p_gold.type == PieceType.GOLD
    assert not p_gold.is_promoted

    p_king = Piece(PieceType.KING, Color.WHITE)
    p_king.promote()  # Should have no effect
    assert p_king.type == PieceType.KING
    assert not p_king.is_promoted

    p_prom_rook = Piece(PieceType.PROMOTED_ROOK, Color.BLACK)
    p_prom_rook.promote()  # Should have no effect
    assert p_prom_rook.type == PieceType.PROMOTED_ROOK
    assert p_prom_rook.is_promoted


def test_piece_unpromote():
    p_prom_pawn = Piece(PieceType.PROMOTED_PAWN, Color.BLACK)
    p_prom_pawn.unpromote()
    assert p_prom_pawn.type == PieceType.PAWN
    assert not p_prom_pawn.is_promoted

    p_pawn = Piece(PieceType.PAWN, Color.BLACK)
    p_pawn.unpromote()  # Should have no effect
    assert p_pawn.type == PieceType.PAWN
    assert not p_pawn.is_promoted

    p_gold = Piece(PieceType.GOLD, Color.BLACK)
    p_gold.unpromote()  # Should have no effect
    assert p_gold.type == PieceType.GOLD
    assert not p_gold.is_promoted


def test_piece_symbol():
    assert Piece(PieceType.PAWN, Color.BLACK).symbol() == "P"
    assert Piece(PieceType.PAWN, Color.WHITE).symbol() == "p"
    assert Piece(PieceType.PROMOTED_ROOK, Color.BLACK).symbol() == "+R"
    assert Piece(PieceType.PROMOTED_ROOK, Color.WHITE).symbol() == "+r"
    assert Piece(PieceType.KING, Color.BLACK).symbol() == "K"
    assert Piece(PieceType.KING, Color.WHITE).symbol() == "k"
    # Test a piece type that might not have a direct symbol but should raise error or default
    # Based on current implementation, all PieceTypes in the enum are handled.


def test_piece_repr():
    p = Piece(PieceType.SILVER, Color.BLACK)
    assert repr(p) == "Piece(SILVER, BLACK)"
    p_prom = Piece(PieceType.PROMOTED_SILVER, Color.WHITE)
    assert repr(p_prom) == "Piece(PROMOTED_SILVER, WHITE)"


def test_piece_eq_and_hash():
    p1 = Piece(PieceType.ROOK, Color.BLACK)
    p2 = Piece(PieceType.ROOK, Color.BLACK)
    p3 = Piece(PieceType.ROOK, Color.WHITE)
    p4 = Piece(PieceType.BISHOP, Color.BLACK)
    p5 = Piece(PieceType.PROMOTED_ROOK, Color.BLACK)

    assert p1 == p2
    assert p1 != p3
    assert p1 != p4
    assert p1 != p5
    assert p1 != "Rook"

    assert hash(p1) == hash(p2)
    assert hash(p1) != hash(p3)
    assert hash(p1) != hash(p4)
    assert hash(p1) != hash(p5)

    s = {p1, p2, p3, p4, p5}
    assert len(s) == 4


def test_piece_deepcopy():
    p1 = Piece(PieceType.BISHOP, Color.WHITE)
    p1.promote()
    p2 = copy.deepcopy(p1)

    assert p1 == p2
    assert p1 is not p2
    assert p2.type == PieceType.PROMOTED_BISHOP
    assert p2.color == Color.WHITE
    assert p2.is_promoted

    # Modify original, ensure copy is not affected
    p1.unpromote()
    assert p1.type == PieceType.BISHOP
    assert p2.type == PieceType.PROMOTED_BISHOP


# Tests for get_piece_type_from_symbol


@pytest.mark.parametrize(
    "symbol, expected_type",
    [
        ("P", PieceType.PAWN),
        ("L", PieceType.LANCE),
        ("N", PieceType.KNIGHT),
        ("S", PieceType.SILVER),
        ("G", PieceType.GOLD),
        ("B", PieceType.BISHOP),
        ("R", PieceType.ROOK),
        ("K", PieceType.KING),
        ("+P", PieceType.PROMOTED_PAWN),
        ("+L", PieceType.PROMOTED_LANCE),
        ("+N", PieceType.PROMOTED_KNIGHT),
        ("+S", PieceType.PROMOTED_SILVER),
        ("+B", PieceType.PROMOTED_BISHOP),
        ("+R", PieceType.PROMOTED_ROOK),
        # Lowercase variants (should be handled by the function)
        ("p", PieceType.PAWN),
        ("l", PieceType.LANCE),
        ("+p", PieceType.PROMOTED_PAWN),
        ("+r", PieceType.PROMOTED_ROOK),
    ],
)
def test_get_piece_type_from_symbol_valid(symbol, expected_type):
    assert get_piece_type_from_symbol(symbol) == expected_type


@pytest.mark.parametrize(
    "invalid_symbol", [("X"), (""), ("PP"), ("+K"), ("+G"), ("++P"), (" P"), ("P ")]
)
def test_get_piece_type_from_symbol_invalid(invalid_symbol):
    with pytest.raises(ValueError):
        get_piece_type_from_symbol(invalid_symbol)


# Tests for dictionary constants (existence and basic content)


def test_dictionary_constants():
    assert KIF_PIECE_SYMBOLS[PieceType.PAWN] == "FU"
    assert KIF_PIECE_SYMBOLS[PieceType.PROMOTED_ROOK] == "RY"
    assert (
        len(KIF_PIECE_SYMBOLS) == 14
    )  # Number of distinct piece types with KIF symbols

    assert SYMBOL_TO_PIECE_TYPE["P"] == PieceType.PAWN
    assert SYMBOL_TO_PIECE_TYPE["+R"] == PieceType.PROMOTED_ROOK
    assert len(SYMBOL_TO_PIECE_TYPE) == 14  # All base and promoted types except none

    assert BASE_TO_PROMOTED_TYPE[PieceType.PAWN] == PieceType.PROMOTED_PAWN
    assert len(BASE_TO_PROMOTED_TYPE) == 6  # PAWN, LANCE, KNIGHT, SILVER, BISHOP, ROOK

    assert PROMOTED_TO_BASE_TYPE[PieceType.PROMOTED_PAWN] == PieceType.PAWN
    assert len(PROMOTED_TO_BASE_TYPE) == 6

    assert PIECE_TYPE_TO_HAND_TYPE[PieceType.PROMOTED_BISHOP] == PieceType.BISHOP
    assert PIECE_TYPE_TO_HAND_TYPE[PieceType.GOLD] == PieceType.GOLD
    assert (
        len(PIECE_TYPE_TO_HAND_TYPE) == 13
    )  # All promotable and base types that go to hand

    assert PieceType.PROMOTED_PAWN in PROMOTED_TYPES_SET
    assert PieceType.PAWN not in PROMOTED_TYPES_SET
    assert len(PROMOTED_TYPES_SET) == 6

    assert PieceType.PAWN in OBS_UNPROMOTED_ORDER
    assert PieceType.KING in OBS_UNPROMOTED_ORDER
    assert len(OBS_UNPROMOTED_ORDER) == 8

    assert PieceType.PROMOTED_LANCE in OBS_PROMOTED_ORDER
    assert len(OBS_PROMOTED_ORDER) == 6
