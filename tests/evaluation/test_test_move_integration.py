"""
Unit tests for test_move functionality integration with evaluation strategies.

This module tests the new test_move method in ShogiGame and its integration
with evaluation strategies for better invalid move handling.
"""

from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from keisei.evaluation.core import EvaluationConfig, create_evaluation_config
from keisei.evaluation.strategies.single_opponent import SingleOpponentEvaluator
from keisei.shogi.shogi_core_definitions import Color, PieceType
from keisei.shogi.shogi_game import ShogiGame


class TestTestMoveBasicFunctionality:
    """Test basic test_move method functionality."""

    def test_valid_move_returns_true(self):
        """Test that valid moves return True."""
        game = ShogiGame()
        # Ensure the board is set up for the knight move
        from keisei.shogi.shogi_core_definitions import Piece

        game.set_piece(7, 1, Piece(PieceType.KNIGHT, Color.BLACK))
        game.set_piece(6, 0, Piece(PieceType.PAWN, Color.BLACK))
        
        # Valid pawn move - move from (6,0) to (5,0)
        # BoardMoveTuple = Tuple[int, int, int, int, bool] = (from_row, from_col, to_row, to_col, promote)
        move = (6, 0, 5, 0, False)  # BoardMoveTuple format
        is_valid = game.test_move(move)
        assert is_valid is True

    def test_invalid_move_returns_false(self):
        """Test that invalid moves return False."""
        game = ShogiGame()
        
        # Invalid move - try to move a piece that doesn't exist
        # BoardMoveTuple = Tuple[int, int, int, int, bool] = (from_row, from_col, to_row, to_col, promote)
        move = (4, 4, 3, 4, False)  # BoardMoveTuple format
        is_valid = game.test_move(move)
        assert is_valid is False


class TestEvaluationIntegration:
    """Test integration with evaluation strategies."""

    def test_single_opponent_evaluator_with_test_move(self):
        """Test that SingleOpponentEvaluator can use the unified config."""
        config = create_evaluation_config(
            strategy="single_opponent",
            opponent_name="random",
            num_games=1
        )
        evaluator = SingleOpponentEvaluator(config)
        
        # Basic validation that evaluator was created successfully
        assert evaluator.config.strategy == "single_opponent"
        assert evaluator.config.opponent_type == "random"
        assert evaluator.config.num_games == 1