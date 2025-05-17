"""
Unit tests for PolicyOutputMapper in utils.py
"""

import pytest
from keisei.utils import PolicyOutputMapper


def test_policy_output_mapper_init():
    """Test PolicyOutputMapper initializes with correct total actions."""
    mapper = PolicyOutputMapper()
    assert isinstance(mapper, PolicyOutputMapper)
    assert mapper.get_total_actions() == 3159
    assert isinstance(mapper.idx_to_shogi_move_spec, list)
    assert isinstance(mapper.shogi_move_spec_to_idx, dict)


def test_policy_output_mapper_not_implemented():
    """Test NotImplementedError for mapping methods."""
    mapper = PolicyOutputMapper()
    # Only the pawn move (6,0,5,0,False) is mapped; (0,0,0,0,False) is not
    with pytest.raises(NotImplementedError):
        mapper.shogi_move_to_policy_index((0, 0, 0, 0, False))
    # The only valid indices are 0-9; index 10 should raise
    with pytest.raises(NotImplementedError):
        mapper.policy_index_to_shogi_move(10)


def test_policy_output_mapper_pawn_move():
    """Test PolicyOutputMapper maps a simple pawn move to index and back."""
    mapper = PolicyOutputMapper()
    move = (6, 0, 5, 0, False)
    idx = mapper.shogi_move_to_policy_index(move)
    assert idx == 0
    move_back = mapper.policy_index_to_shogi_move(idx)
    assert move_back == (6, 0, 5, 0, False)


def test_policy_output_mapper_expanded():
    """Test PolicyOutputMapper maps several moves and drops correctly."""
    mapper = PolicyOutputMapper()
    # Black pawn forward
    move = (6, 1, 5, 1, False)
    idx = mapper.shogi_move_to_policy_index(move)
    assert idx == 1
    assert mapper.policy_index_to_shogi_move(idx) == move
    # White pawn forward
    move_w = (2, 2, 3, 2, False)
    idx_w = mapper.shogi_move_to_policy_index(move_w)
    assert idx_w == 6
    assert mapper.policy_index_to_shogi_move(idx_w) == move_w
    # Black pawn drop
    drop = (None, None, 4, 4, "drop_pawn_black")
    idx_drop = mapper.shogi_move_to_policy_index(drop)
    assert idx_drop == 8
    assert mapper.policy_index_to_shogi_move(idx_drop) == drop
    # White pawn drop
    drop_w = (None, None, 4, 4, "drop_pawn_white")
    idx_drop_w = mapper.shogi_move_to_policy_index(drop_w)
    assert idx_drop_w == 9
    assert mapper.policy_index_to_shogi_move(idx_drop_w) == drop_w
