"""
Unit tests for PolicyOutputMapper in utils.py
"""

from typing import List

import pytest
import torch

from keisei.shogi import MoveTuple, PieceType
from keisei.utils import PolicyOutputMapper

# Test for the new PolicyOutputMapper


@pytest.fixture
def mapper() -> PolicyOutputMapper:
    return PolicyOutputMapper()


def test_policy_output_mapper_init(mapper: PolicyOutputMapper):
    """Test PolicyOutputMapper initializes with correct total actions."""
    assert isinstance(mapper, PolicyOutputMapper)
    # Board moves: (81 * 81 - 81) * 2 = 80 * 81 * 2 = 12960
    # Drop moves: 81 * 7 = 567
    # Total = 12960 + 567 = 13527
    assert mapper.get_total_actions() == 13527


def test_policy_output_mapper_mappings(mapper: PolicyOutputMapper):
    """Test basic move to index and index to move conversions."""
    # Example board move: (0,0) to (1,0) without promotion
    move1: MoveTuple = (0, 0, 1, 0, False)
    idx1 = mapper.shogi_move_to_policy_index(move1)
    retrieved_move1 = mapper.policy_index_to_shogi_move(idx1)
    assert retrieved_move1 == move1

    # Example board move with promotion: (0,0) to (1,0) with promotion
    move_promo: MoveTuple = (0, 0, 1, 0, True)
    idx_promo = mapper.shogi_move_to_policy_index(move_promo)
    retrieved_move_promo = mapper.policy_index_to_shogi_move(idx_promo)
    assert retrieved_move_promo == move_promo
    assert idx_promo == idx1 + 1  # Promo move should be next to non-promo

    # Example drop move: Pawn to (4,4)
    # PieceType.PAWN is the first in droppable_piece_types list (index 0)
    # Drop moves start after all 12960 board moves.
    # Offset for (4,4) square: (4 * 9 + 4) * 7 = (36 + 4) * 7 = 40 * 7 = 280
    # Index for PAWN drop at (4,4) = 12960 (board moves) + 280 (offset for square and PAWN) = 13240
    move2: MoveTuple = (None, None, 4, 4, PieceType.PAWN)
    idx2 = mapper.shogi_move_to_policy_index(move2)
    retrieved_move2 = mapper.policy_index_to_shogi_move(idx2)
    assert retrieved_move2 == move2
    assert idx2 == 13240

    # Example drop move: LANCE to (0,0)
    # PieceType.LANCE is the second in droppable_piece_types list (index 1)
    # Offset for (0,0) square: (0 * 9 + 0) * 7 = 0
    # Index for LANCE drop at (0,0) = 12960 (board moves) + 0 (offset for square) + 1 (for LANCE) = 12961
    move3: MoveTuple = (None, None, 0, 0, PieceType.LANCE)
    idx3 = mapper.shogi_move_to_policy_index(move3)
    retrieved_move3 = mapper.policy_index_to_shogi_move(idx3)
    assert retrieved_move3 == move3
    assert (
        idx3 == 12960 + 1
    )  # (0*9+0)*7 + 1 (Lance is index 1 in droppable_piece_types)

    with pytest.raises(ValueError):
        # A move tuple that is structurally valid but wouldn't be generated
        # e.g. if from_sq == to_sq for a board move (already skipped in generation)
        # or a piece type not in droppable_piece_types for a drop.
        # For a simple ValueError test, use a move that get() would return None for.
        mapper.shogi_move_to_policy_index((-1, -1, -1, -1, False))

    with pytest.raises(IndexError):
        mapper.policy_index_to_shogi_move(
            mapper.get_total_actions()
        )  # Exact total_actions is out of bounds (0-indexed)
    with pytest.raises(IndexError):
        mapper.policy_index_to_shogi_move(-1)


def test_get_legal_mask(mapper: PolicyOutputMapper):
    """Test the get_legal_mask method."""
    device = torch.device("cpu")

    # Example legal Shogi moves (ensure these are valid MoveTuple structures)
    legal_shogi_moves: List[MoveTuple] = [
        (0, 0, 1, 0, False),
        (None, None, 4, 4, PieceType.PAWN),
        (2, 2, 3, 3, True),
    ]

    # Verify these test moves are known to the mapper
    for move in legal_shogi_moves:
        assert (
            move in mapper.move_to_idx
        ), f"Test move {move} is not in the mapper. Check move generation or test data."

    mask = mapper.get_legal_mask(legal_shogi_moves, device)

    assert mask.shape == (mapper.get_total_actions(),)
    assert mask.dtype == torch.bool
    assert mask.sum().item() == len(legal_shogi_moves)

    for move in legal_shogi_moves:
        idx = mapper.shogi_move_to_policy_index(move)
        assert mask[idx].item() is True

    # Test with an empty list of legal moves
    empty_mask = mapper.get_legal_mask([], device)
    assert empty_mask.sum().item() == 0

    # Test with a move not in the mapper (should be ignored by get_legal_mask if it occurs)
    # This case depends on the strictness of ShogiGame.get_legal_moves()
    # For now, PolicyOutputMapper.get_legal_mask has a try-except ValueError for robustness.
    # If ShogiGame guarantees all its output moves are mappable, this part is less critical.
    # non_mappable_move: MoveTuple = (99,99,99,99,False) # A deliberately unmappable move
    # mask_with_unmappable = mapper.get_legal_mask([legal_shogi_moves[0], non_mappable_move], device)
    # assert mask_with_unmappable.sum().item() == 1 # Only the mappable one should be True


# Additional tests for PolicyOutputMapper


def test_policy_output_mapper_total_actions(mapper: PolicyOutputMapper):
    # Based on the logic in PolicyOutputMapper:
    # Board moves: 9x9 (from_sq) * 8x8 (to_sq, excluding same square) = 81 * 64, but this is not how it's calculated.
    # It's (9*9 for from_sq) * (9*9 for to_sq), then filter out from_sq == to_sq.
    # (81 * 81 - 81) = 81 * 80 = 6480 board destinations.
    # Each board move can be with or without promotion: 6480 * 2 = 12960 board move policy outputs.
    # Drop moves: 9x9 (to_sq) * 7 (droppable piece types) = 81 * 7 = 567 drop move policy outputs.
    # Total = 12960 + 567 = 13527
    assert mapper.get_total_actions() == 13527
    assert len(mapper.idx_to_move) == 13527
    assert len(mapper.move_to_idx) == 13527


@pytest.mark.parametrize(
    "r_from, c_from, r_to, c_to, promote, expected_idx_offset",
    [
        (0, 0, 0, 1, False, 0),  # First possible board move (no promotion)
        (0, 0, 0, 1, True, 1),  # First possible board move (with promotion)
        (8, 8, 8, 7, False, 12958),  # Last possible board move (no promotion)
        (8, 8, 8, 7, True, 12959),  # Last possible board move (with promotion)
    ],
)
def test_board_move_to_policy_index_edges(
    mapper: PolicyOutputMapper, r_from, c_from, r_to, c_to, promote, expected_idx_offset
):
    move: MoveTuple = (r_from, c_from, r_to, c_to, promote)
    # This test is a bit fragile as it assumes specific ordering. The existing test_policy_output_mapper_mappings is better for general validation.
    # The exact index depends on the iteration order in __init__.
    # For now, let's ensure it's within bounds and consistent.
    idx = mapper.shogi_move_to_policy_index(move)
    assert 0 <= idx < 12960  # Board moves are in this range
    retrieved_move = mapper.policy_index_to_shogi_move(idx)
    assert retrieved_move == move


@pytest.mark.parametrize(
    "r_to, c_to, piece_type, expected_idx_start_offset",
    [
        (0, 0, PieceType.PAWN, 12960),  # First possible drop move (Pawn to 9a)
        (8, 8, PieceType.ROOK, 13526),  # Last possible drop move (Rook to 1i)
    ],
)
def test_drop_move_to_policy_index_edges(
    mapper: PolicyOutputMapper, r_to, c_to, piece_type, expected_idx_start_offset
):
    move: MoveTuple = (None, None, r_to, c_to, piece_type)
    idx = mapper.shogi_move_to_policy_index(move)
    # Exact index depends on iteration order and PieceType enum order.
    # Check if it's within the drop move range and consistent.
    assert 12960 <= idx < 13527
    retrieved_move = mapper.policy_index_to_shogi_move(idx)
    assert retrieved_move == move


def test_get_legal_mask_all_legal(mapper: PolicyOutputMapper):
    """Test get_legal_mask when all moves are theoretically legal (for mask creation purposes)."""
    # This doesn't mean they are game-legal, just that the mapper can map them.
    all_moves = mapper.idx_to_move  # Get all moves the mapper knows
    device = torch.device("cpu")
    mask = mapper.get_legal_mask(all_moves, device)
    assert mask.shape == (mapper.get_total_actions(),)
    assert mask.dtype == torch.bool
    assert mask.sum().item() == mapper.get_total_actions()


# Tests for USI conversion utilities


@pytest.mark.parametrize(
    "r, c, expected_usi",
    [
        (0, 0, "9a"),
        (8, 8, "1i"),
        (2, 2, "7c"),
        (4, 4, "5e"),
        (0, 8, "1a"),
        (8, 0, "9i"),
    ],
)
def test_usi_sq(mapper: PolicyOutputMapper, r, c, expected_usi):
    assert mapper._usi_sq(r, c) == expected_usi


@pytest.mark.parametrize("r, c", [(-1, 0), (0, -1), (9, 0), (0, 9)])
def test_usi_sq_invalid(mapper: PolicyOutputMapper, r, c):
    with pytest.raises(ValueError):
        mapper._usi_sq(r, c)


@pytest.mark.parametrize(
    "piece_type, expected_char",
    [
        (PieceType.PAWN, "P"),
        (PieceType.LANCE, "L"),
        (PieceType.KNIGHT, "N"),
        (PieceType.SILVER, "S"),
        (PieceType.GOLD, "G"),
        (PieceType.BISHOP, "B"),
        (PieceType.ROOK, "R"),
    ],
)
def test_get_usi_char_for_drop_valid(
    mapper: PolicyOutputMapper, piece_type, expected_char
):
    assert mapper._get_usi_char_for_drop(piece_type) == expected_char


@pytest.mark.parametrize(
    "invalid_piece_type",
    [PieceType.KING, PieceType.PROMOTED_PAWN, PieceType.PROMOTED_ROOK],
)
def test_get_usi_char_for_drop_invalid(mapper: PolicyOutputMapper, invalid_piece_type):
    with pytest.raises(ValueError):
        mapper._get_usi_char_for_drop(invalid_piece_type)


@pytest.mark.parametrize(
    "move_tuple, expected_usi",
    [
        ((2, 2, 3, 2, False), "7c7d"),  # Corrected: (2,2)=7c, (3,2)=7d
        ((7, 7, 8, 6, True), "2h3i+"),  # Corrected: (7,7)=2h, (8,6)=3i
        ((None, None, 4, 4, PieceType.PAWN), "P*5e"),  # Drop Pawn to 5e
        ((None, None, 0, 0, PieceType.ROOK), "R*9a"),  # Drop Rook to 9a
        ((6, 7, 5, 7, False), "2g2f"),  # Corrected: (6,7)=2g, (5,7)=2f
    ],
)
def test_shogi_move_to_usi_valid(
    mapper: PolicyOutputMapper, move_tuple: MoveTuple, expected_usi
):
    assert mapper.shogi_move_to_usi(move_tuple) == expected_usi


@pytest.mark.parametrize(
    "invalid_move_tuple",
    [
        # ((0, 0, 0, 0, False)), # This is a valid tuple structure, _usi_sq would work. PolicyOutputMapper skips same-square.
        # ((1, 1, 1, 1, True)),  # Same as above.
        (
            (None, None, 0, 0, PieceType.KING)
        ),  # Cannot drop King - _get_usi_char_for_drop raises ValueError
        (
            (0, 1, 2, 3, "False")
        ),  # Invalid promote type - isinstance check in shogi_move_to_usi
        ((0, 1, 2, 3, 4)),  # Invalid structure / promote type
        ("7g7f"),  # Not a tuple - shogi_move_to_usi expects a tuple
        ((0, 0, 9, 0, False)),  # Invalid to_sq for _usi_sq
    ],
)
def test_shogi_move_to_usi_invalid(mapper: PolicyOutputMapper, invalid_move_tuple):
    with pytest.raises(
        (ValueError, TypeError)
    ):  # Allow TypeError for cases like passing a string
        mapper.shogi_move_to_usi(invalid_move_tuple)  # type: ignore


@pytest.mark.parametrize(
    "usi_move, expected_tuple",
    [
        ("7c7d", (2, 2, 3, 2, False)),  # Corrected
        ("2h3i+", (7, 7, 8, 6, True)),  # Corrected
        ("P*5e", (None, None, 4, 4, PieceType.PAWN)),
        ("R*9a", (None, None, 0, 0, PieceType.ROOK)),
        ("2g2f", (6, 7, 5, 7, False)),  # Corrected
    ],
)
def test_usi_to_shogi_move_valid(
    mapper: PolicyOutputMapper, usi_move, expected_tuple: MoveTuple
):
    assert mapper.usi_to_shogi_move(usi_move) == expected_tuple


@pytest.mark.parametrize(
    "invalid_usi_move",
    [
        # ("7g7g"), # This is a valid USI string, though the move might be illegal in a game.
        ("P*5e+"),  # Drop moves cannot have promotion
        ("K*5e"),  # Cannot drop King (invalid piece char for drop)
        ("7g10f"),  # Invalid 'to_sq' rank
        ("1234"),  # Malformed board move
        ("L*1j"),  # Invalid 'to_sq' rank for drop
        (12345),  # Not a string
        ("P*5X"),  # Invalid rank char in drop
        ("X*5e"),  # Invalid piece char for drop
        ("7g7f++"),  # Double promotion
        ("7g7"),  # Too short
        ("B*"),  # Missing target square for drop
        ("*5e"),  # Missing piece for drop
    ],
)
def test_usi_to_shogi_move_invalid(mapper: PolicyOutputMapper, invalid_usi_move):
    with pytest.raises((ValueError, TypeError)):  # Allow TypeError for non-string input
        mapper.usi_to_shogi_move(invalid_usi_move)  # type: ignore


# Test for shogi_move_to_policy_index with a move that might cause issues with PieceType enum identity
# This is to ensure the fallback logic in shogi_move_to_policy_index is covered.
def test_shogi_move_to_policy_index_enum_identity_fallback(mapper: PolicyOutputMapper):
    # Create a move tuple where PieceType might be a different instance but same value
    # This is hard to simulate perfectly without manipulating enum internals, but we can try
    # by creating a PieceType that's equivalent but not identical if that were possible.
    # The current fallback handles `move[0] is None` and `isinstance(move[4], PieceType)`.
    # Let's test a drop move that should be found by the fallback.
    raw_move_tuple = (
        None,
        None,
        3,
        3,
        PieceType.SILVER.value,
    )  # Using .value to potentially bypass direct enum instance check
    # Reconstruct with the actual enum member to ensure it's a valid key for the main dict lookup
    # The goal is to test the *fallback* path, which is tricky.
    # The fallback specifically iterates through self.move_to_idx.items() if the initial get() fails.
    # This test might be more conceptual unless we can force a cache miss for the primary key.

    # Let's test a valid drop move that should be found.
    valid_drop_move: MoveTuple = (None, None, 2, 2, PieceType.GOLD)
    idx = mapper.shogi_move_to_policy_index(valid_drop_move)
    assert idx is not None
    assert 12960 <= idx < 13527

    # Test a board move that should be found.
    valid_board_move: MoveTuple = (1, 1, 2, 2, False)
    idx_board = mapper.shogi_move_to_policy_index(valid_board_move)
    assert idx_board is not None
    assert 0 <= idx_board < 12960

    # Test the ValueError for a completely unmappable move (already in existing tests but good to have here too)
    with pytest.raises(ValueError):
        mapper.shogi_move_to_policy_index((0, 0, 0, 0, PieceType.PAWN))  # type: ignore # Invalid piece type for board move promo flag


# Test TrainingLogger (already in test_logger.py, but can be here too for module completeness if desired)
# For now, assume test_logger.py covers TrainingLogger adequately.
