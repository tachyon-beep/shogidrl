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


# The TrainingLogger tests can remain as they are, assuming no changes to TrainingLogger itself.
# ... existing TrainingLogger tests ...
