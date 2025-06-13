"""
Unit tests for test_move functionality integration with evaluation strategies.

This module tests the new test_move method in ShogiGame and its integration
with evaluation strategies for better invalid move handling.
"""

from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from keisei.evaluation.core import SingleOpponentConfig
from keisei.evaluation.strategies.single_opponent import SingleOpponentEvaluator
from keisei.shogi.shogi_core_definitions import Color, MoveTuple, PieceType
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
        # Valid pawn move
        valid_move: MoveTuple = (6, 0, 5, 0, False)
        assert game.test_move(valid_move) is True
        # Valid knight move
        valid_knight_move: MoveTuple = (7, 1, 5, 2, False)
        assert game.test_move(valid_knight_move) is True

    def test_invalid_move_returns_false(self):
        """Test that invalid moves return False."""
        game = ShogiGame()

        # Invalid pawn move (can't move 2 squares)
        invalid_move: MoveTuple = (6, 0, 4, 0, False)
        assert game.test_move(invalid_move) is False

        # Move from empty square
        empty_square_move: MoveTuple = (5, 5, 4, 5, False)
        assert game.test_move(empty_square_move) is False

        # Move opponent's piece
        opponent_piece_move: MoveTuple = (2, 0, 3, 0, False)
        assert game.test_move(opponent_piece_move) is False

    def test_drop_move_validation(self):
        """Test drop move validation."""
        game = ShogiGame()
        # Clear the board to ensure the drop square is empty
        for r in range(9):
            for c in range(9):
                game.set_piece(r, c, None)
        # Invalid drop (no pieces in hand)
        drop_move: MoveTuple = (None, None, 5, 5, PieceType.PAWN)
        assert game.test_move(drop_move) is False
        # Add a piece to hand and test valid drop
        game.hands[Color.BLACK.value][PieceType.PAWN] = 1
        assert game.test_move(drop_move) is True
        # Invalid drop on occupied square
        from keisei.shogi.shogi_core_definitions import Piece

        game.set_piece(5, 5, Piece(PieceType.PAWN, Color.BLACK))
        assert game.test_move(drop_move) is False

    def test_game_over_returns_false(self):
        """Test that test_move returns False when game is over."""
        game = ShogiGame()
        game.game_over = True

        valid_move: MoveTuple = (6, 0, 5, 0, False)
        assert game.test_move(valid_move) is False

    def test_malformed_move_tuple_returns_false(self):
        """Test that malformed move tuples return False."""
        game = ShogiGame()

        # Wrong tuple length
        assert game.test_move((6, 0, 5)) is False  # type: ignore

        # Wrong types
        assert game.test_move(("a", 0, 5, 0, False)) is False  # type: ignore

        # None where it shouldn't be
        assert game.test_move((None, 0, 5, 0, False)) is False  # type: ignore


class TestTestMoveVsMakeMove:
    """Test that test_move matches make_move validation without side effects."""

    def test_test_move_matches_make_move_validation(self):
        """Test that test_move results match make_move validation."""
        from keisei.shogi.shogi_core_definitions import Piece

        test_moves = [
            (6, 0, 5, 0, False),  # Valid pawn move
            (6, 0, 4, 0, False),  # Invalid pawn move (too far)
            (7, 1, 5, 2, False),  # Valid knight move
            (5, 5, 4, 5, False),  # Invalid move from empty square
            (None, None, 5, 5, PieceType.PAWN),  # Pawn drop (should set up hand)
        ]
        for move in test_moves:
            # Create a fresh game for each test
            test_game = ShogiGame()
            original_game = ShogiGame()
            # Setup for pawn drop
            if move[0] is None and move[4] == PieceType.PAWN:
                for r in range(9):
                    for c in range(9):
                        test_game.set_piece(r, c, None)
                        original_game.set_piece(r, c, None)
                test_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
                original_game.hands[Color.BLACK.value][PieceType.PAWN] = 1
            # Test test_move result
            test_result = test_game.test_move(move)
            # Test make_move result
            make_move_result = True
            try:
                original_game.make_move(move)
            except (ValueError, AssertionError):
                make_move_result = False
            assert (
                test_result == make_move_result
            ), f"Mismatch for move {move}: test_move={test_result}, make_move={make_move_result}"

    def test_test_move_has_no_side_effects(self):
        """Test that test_move doesn't modify game state."""
        game = ShogiGame()
        original_board = [row[:] for row in game.board]  # Deep copy
        original_hands = {k: v.copy() for k, v in game.hands.items()}
        original_current_player = game.current_player
        original_move_count = game.move_count

        # Test various moves
        moves_to_test = [
            (6, 0, 5, 0, False),  # Valid move
            (6, 0, 4, 0, False),  # Invalid move
            (None, None, 5, 5, PieceType.PAWN),  # Drop move
        ]

        for move in moves_to_test:
            game.test_move(move)

            # Verify no state changes
            assert game.board == original_board
            assert game.hands == original_hands
            assert game.current_player == original_current_player
            assert game.move_count == original_move_count


class TestEvaluationIntegration:
    """Test integration of test_move with evaluation strategies."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock SingleOpponentConfig."""
        config = MagicMock(spec=SingleOpponentConfig)
        config.num_games = 1
        config.max_moves_per_game = 10
        config.log_level = "INFO"  # Add log_level to fix the logging setup
        config.opponent_config = MagicMock()
        config.opponent_config.type = "mock"
        return config

    @pytest.fixture
    def evaluator(self, mock_config):
        """Create a SingleOpponentEvaluator instance."""
        return SingleOpponentEvaluator(mock_config)

    @pytest.mark.asyncio
    async def test_validate_and_make_move_uses_test_move(self, evaluator):
        """Test that _validate_and_make_move uses test_move for validation."""
        game = ShogiGame()
        legal_moves = [(6, 0, 5, 0, False), (6, 1, 5, 1, False)]

        # Test valid move that passes test_move
        valid_move = (6, 0, 5, 0, False)
        with patch.object(game, "test_move", return_value=True) as mock_test_move:
            with patch.object(game, "make_move") as mock_make_move:
                result = await evaluator._validate_and_make_move(
                    game, valid_move, legal_moves, 0, "TestPlayer"
                )

                # Verify test_move was called
                mock_test_move.assert_called_once_with(valid_move)
                # Verify make_move was called since test_move returned True
                mock_make_move.assert_called_once_with(valid_move)
                assert result is True

    @pytest.mark.asyncio
    async def test_validate_and_make_move_rejects_invalid_test_move(self, evaluator):
        """Test that _validate_and_make_move rejects moves that fail test_move."""
        game = ShogiGame()
        legal_moves = [(6, 0, 5, 0, False), (6, 1, 5, 1, False)]

        # Test move that fails test_move validation
        invalid_move = (6, 0, 5, 0, False)
        with patch.object(game, "test_move", return_value=False) as mock_test_move:
            with patch.object(game, "make_move") as mock_make_move:
                result = await evaluator._validate_and_make_move(
                    game, invalid_move, legal_moves, 0, "TestPlayer"
                )

                # Verify test_move was called
                mock_test_move.assert_called_once_with(invalid_move)
                # Verify make_move was NOT called since test_move returned False
                mock_make_move.assert_not_called()
                # Verify game is marked as over with correct winner
                assert result is False
                assert game.game_over is True
                assert game.winner == Color.WHITE  # Opponent wins
                assert game.termination_reason == "Invalid move"

    @pytest.mark.asyncio
    async def test_invalid_move_creates_learning_signal(self, evaluator):
        """Test that invalid moves create immediate learning signals."""
        game = ShogiGame()

        # Create a move that's in legal_moves but fails test_move validation
        # This simulates a corrupted move that passes basic checks but fails detailed validation
        invalid_move = (6, 0, 5, 0, False)  # This should be valid normally
        legal_moves = [invalid_move]  # Include it in legal moves

        # Mock test_move to return False for this specific move
        with patch.object(game, "test_move", return_value=False):
            result = await evaluator._validate_and_make_move(
                game, invalid_move, legal_moves, 0, "PPOAgent"
            )

        # Verify the game ended immediately
        assert result is False
        assert game.game_over is True
        assert game.winner == Color.WHITE  # Opponent wins (player 1)
        assert game.termination_reason is not None
        assert "Invalid move" in game.termination_reason

        # Verify this creates a strong learning signal
        # The agent that made the invalid move loses immediately
        assert game.winner.value == 1  # Opponent's value

    def test_performance_comparison(self):
        """Test that test_move is faster than make_move for invalid moves."""
        import time

        game = ShogiGame()
        invalid_move = (6, 0, 4, 0, False)  # Invalid pawn move

        # Time test_move
        start = time.time()
        for _ in range(1000):
            game.test_move(invalid_move)
        test_move_time = time.time() - start

        # Time make_move with exception handling
        start = time.time()
        for _ in range(1000):
            try:
                temp_game = ShogiGame()
                temp_game.make_move(invalid_move)
            except (ValueError, AssertionError):
                pass
        make_move_time = time.time() - start

        # test_move should be faster (no exception creation/handling overhead)
        print(f"test_move time: {test_move_time:.4f}s")
        print(f"make_move time: {make_move_time:.4f}s")

        # Allow some variance, but test_move should be significantly faster
        assert test_move_time < make_move_time * 0.8


class TestAdvancedScenarios:
    """Test advanced scenarios and edge cases."""

    def test_complex_board_position(self):
        """Test test_move with complex board positions."""
        from keisei.shogi.shogi_core_definitions import Piece

        # Moves to set up the position
        setup_moves = [
            (6, 4, 5, 4, False),  # Black pawn move
            (2, 4, 3, 4, False),  # White pawn move
            (7, 3, 6, 4, False),  # Black gold move
        ]
        # Test moves to try in this position
        test_moves = [
            (5, 4, 4, 4, False),  # Continue pawn advance
            (6, 4, 5, 4, False),  # Gold move
            (8, 4, 7, 4, False),  # King move
        ]
        for move in test_moves:
            # Clear and set up both games identically for each test
            game = ShogiGame()
            temp_game = ShogiGame()
            for r in range(9):
                for c in range(9):
                    game.set_piece(r, c, None)
                    temp_game.set_piece(r, c, None)
            # Place required pieces
            game.set_piece(6, 4, Piece(PieceType.PAWN, Color.BLACK))
            temp_game.set_piece(6, 4, Piece(PieceType.PAWN, Color.BLACK))
            game.set_piece(2, 4, Piece(PieceType.PAWN, Color.WHITE))
            temp_game.set_piece(2, 4, Piece(PieceType.PAWN, Color.WHITE))
            game.set_piece(7, 3, Piece(PieceType.GOLD, Color.BLACK))
            temp_game.set_piece(7, 3, Piece(PieceType.GOLD, Color.BLACK))
            # Place both kings (required for test_move logic)
            game.set_piece(8, 4, Piece(PieceType.KING, Color.BLACK))
            temp_game.set_piece(8, 4, Piece(PieceType.KING, Color.BLACK))
            game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))
            temp_game.set_piece(0, 4, Piece(PieceType.KING, Color.WHITE))
            # Apply setup moves
            game.make_move(setup_moves[0])
            temp_game.make_move(setup_moves[0])
            game.current_player = Color.WHITE
            temp_game.current_player = Color.WHITE
            game.make_move(setup_moves[1])
            temp_game.make_move(setup_moves[1])
            game.current_player = Color.BLACK
            temp_game.current_player = Color.BLACK
            game.make_move(setup_moves[2])
            temp_game.make_move(setup_moves[2])
            # Test move
            test_result = game.test_move(move)
            make_move_result = True
            try:
                temp_game.make_move(move)
            except (ValueError, AssertionError):
                make_move_result = False
            assert test_result == make_move_result

    def test_promotion_validation(self):
        """Test promotion move validation."""
        game = ShogiGame()

        # Set up a position where promotion is possible
        # Move pawn to near promotion zone
        game.make_move((6, 0, 5, 0, False))  # Black pawn
        game.make_move((2, 8, 3, 8, False))  # White pawn
        game.make_move((5, 0, 4, 0, False))  # Black pawn
        game.make_move((3, 8, 4, 8, False))  # White pawn
        game.make_move((4, 0, 3, 0, False))  # Black pawn
        game.make_move((4, 8, 5, 8, False))  # White pawn

        # Test promotion move
        promotion_move = (3, 0, 2, 0, True)  # Promote pawn
        no_promotion_move = (3, 0, 2, 0, False)  # Don't promote

        # Both should be valid (promotion is optional in most cases)
        assert game.test_move(promotion_move) is True
        assert game.test_move(no_promotion_move) is True

    def test_drop_rule_validation(self):
        """Test specific drop rule validation (nifu, etc.)."""
        from keisei.shogi.shogi_core_definitions import Piece

        # --- Test 1: Drop pawn in empty file ---
        game = ShogiGame()
        for r in range(9):
            for c in range(9):
                game.set_piece(r, c, None)
        game.hands[Color.BLACK.value][PieceType.PAWN] = 1
        first_drop = (None, None, 5, 0, PieceType.PAWN)
        assert game.test_move(first_drop) is True
        # --- Test 2: Nifu (double pawn in file) ---
        game = ShogiGame()
        for r in range(9):
            for c in range(9):
                game.set_piece(r, c, None)
        game.set_piece(
            6, 0, Piece(PieceType.PAWN, Color.BLACK)
        )  # Existing pawn in file
        game.hands[Color.BLACK.value][PieceType.PAWN] = 1
        nifu_drop = (None, None, 3, 0, PieceType.PAWN)
        assert game.test_move(nifu_drop) is False
        # --- Test 3: Drop in different file should be OK ---
        game = ShogiGame()
        for r in range(9):
            for c in range(9):
                game.set_piece(r, c, None)
        game.set_piece(6, 0, Piece(PieceType.PAWN, Color.BLACK))
        game.hands[Color.BLACK.value][PieceType.PAWN] = 1
        different_file_drop = (None, None, 4, 1, PieceType.PAWN)
        assert game.test_move(different_file_drop) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
