"""
Test suite for move formatting functionality.

This module contains comprehensive tests for the move formatting system,
including basic move formatting, enhanced formatting with piece information,
and Japanese piece naming.
"""

from types import SimpleNamespace

import pytest

from keisei.utils import PolicyOutputMapper
from keisei.train import (
    format_move_with_description,
    format_move_with_description_enhanced,
    _get_piece_name,
    _coords_to_square_name,
)
from keisei.shogi.shogi_core_definitions import PieceType


class TestBasicMoveFormatting:
    """Test basic move formatting functionality."""

    @pytest.fixture
    def policy_mapper(self):
        """Initialize policy output mapper for tests."""
        return PolicyOutputMapper()

    def test_board_move_without_game_context(self, policy_mapper):
        """Test formatting board moves without game context."""
        move = (6, 6, 5, 6, False)  # Simple move
        result = format_move_with_description(move, policy_mapper, game=None)

        assert "3g3f" in result
        assert "piece moving from 3g to 3f" in result
        assert result.endswith(".")

    def test_promoting_move_without_game_context(self, policy_mapper):
        """Test formatting promoting moves without game context."""
        move = (6, 6, 5, 6, True)  # Promoting move
        result = format_move_with_description(move, policy_mapper, game=None)

        assert "3g3f+" in result
        assert "piece promoting moving from 3g to 3f" in result
        assert result.endswith(".")

    def test_drop_move_pawn(self, policy_mapper):
        """Test formatting pawn drop moves."""
        move = (None, None, 4, 4, PieceType.PAWN)
        result = format_move_with_description(move, policy_mapper, game=None)

        assert "P*5e" in result
        assert "Fuhyō (Pawn) drop to 5e" in result
        assert result.endswith(".")

    def test_drop_move_rook(self, policy_mapper):
        """Test formatting rook drop moves."""
        move = (None, None, 2, 6, PieceType.ROOK)
        result = format_move_with_description(move, policy_mapper, game=None)

        assert "R*3c" in result
        assert "Hisha (Rook) drop to 3c" in result
        assert result.endswith(".")

    def test_drop_move_knight(self, policy_mapper):
        """Test formatting knight drop moves."""
        move = (None, None, 5, 3, PieceType.KNIGHT)
        result = format_move_with_description(move, policy_mapper, game=None)

        assert "N*6f" in result
        assert "Keima (Knight) drop to 6f" in result
        assert result.endswith(".")

    def test_none_move(self, policy_mapper):
        """Test formatting None move."""
        result = format_move_with_description(None, policy_mapper, game=None)
        assert result == "None"

    def test_move_formatting_error_handling(self, policy_mapper):
        """Test error handling in move formatting."""
        # Invalid move tuple
        invalid_move = (1, 2, 3)  # Too few elements
        result = format_move_with_description(invalid_move, policy_mapper, game=None)

        assert "format error" in result


class TestEnhancedMoveFormatting:
    """Test enhanced move formatting with piece information."""

    @pytest.fixture
    def policy_mapper(self):
        """Initialize policy output mapper for tests."""
        return PolicyOutputMapper()

    @pytest.fixture
    def mock_pawn_piece(self):
        """Create a mock pawn piece."""
        piece = SimpleNamespace()
        piece.type = PieceType.PAWN
        return piece

    @pytest.fixture
    def mock_rook_piece(self):
        """Create a mock rook piece."""
        piece = SimpleNamespace()
        piece.type = PieceType.ROOK
        return piece

    def test_enhanced_board_move_with_piece_info(self, policy_mapper, mock_pawn_piece):
        """Test enhanced formatting with piece information."""
        move = (6, 6, 5, 6, False)
        result = format_move_with_description_enhanced(
            move, policy_mapper, mock_pawn_piece
        )

        assert "3g3f" in result
        assert "Fuhyō (Pawn) moving from 3g to 3f" in result
        assert result.endswith(".")

    def test_enhanced_promoting_move_with_piece_info(
        self, policy_mapper, mock_pawn_piece
    ):
        """Test enhanced formatting with promoting piece."""
        move = (6, 6, 5, 6, True)
        result = format_move_with_description_enhanced(
            move, policy_mapper, mock_pawn_piece
        )

        assert "3g3f+" in result
        assert "Fuhyō (Pawn) → Tokin (Promoted Pawn) moving from 3g to 3f" in result
        assert result.endswith(".")

    def test_enhanced_board_move_without_piece_info(self, policy_mapper):
        """Test enhanced formatting without piece information."""
        move = (8, 8, 7, 7, False)
        result = format_move_with_description_enhanced(
            move, policy_mapper, piece_info=None
        )

        assert "1i2h" in result
        assert "piece moving from 1i to 2h" in result
        assert result.endswith(".")

    def test_enhanced_drop_move(self, policy_mapper):
        """Test enhanced formatting for drop moves (piece_info ignored)."""
        move = (None, None, 4, 4, PieceType.ROOK)
        result = format_move_with_description_enhanced(
            move, policy_mapper, piece_info=None
        )

        assert "R*5e" in result
        assert "Hisha (Rook) drop to 5e" in result
        assert result.endswith(".")

    def test_enhanced_none_move(self, policy_mapper):
        """Test enhanced formatting for None move."""
        result = format_move_with_description_enhanced(
            None, policy_mapper, piece_info=None
        )
        assert result == "None"


class TestPieceNaming:
    """Test Japanese piece naming functionality."""

    def test_regular_piece_names(self):
        """Test regular piece name generation."""
        test_cases = [
            (PieceType.PAWN, "Fuhyō (Pawn)"),
            (PieceType.ROOK, "Hisha (Rook)"),
            (PieceType.BISHOP, "Kakugyō (Bishop)"),
            (PieceType.KING, "Ōshō (King)"),
            (PieceType.KNIGHT, "Keima (Knight)"),
            (PieceType.LANCE, "Kyōsha (Lance)"),
            (PieceType.SILVER, "Ginsho (Silver General)"),
            (PieceType.GOLD, "Kinshō (Gold General)"),
        ]

        for piece_type, expected_name in test_cases:
            result = _get_piece_name(piece_type, is_promoting=False)
            assert result == expected_name

    def test_promoted_piece_names(self):
        """Test promoted piece name generation."""
        test_cases = [
            (PieceType.PROMOTED_PAWN, "Tokin (Promoted Pawn)"),
            (PieceType.PROMOTED_ROOK, "Ryūō (Dragon King)"),
            (PieceType.PROMOTED_BISHOP, "Ryūma (Dragon Horse)"),
            (PieceType.PROMOTED_LANCE, "Narikyo (Promoted Lance)"),
            (PieceType.PROMOTED_KNIGHT, "Narikei (Promoted Knight)"),
            (PieceType.PROMOTED_SILVER, "Narigin (Promoted Silver)"),
        ]

        for piece_type, expected_name in test_cases:
            result = _get_piece_name(piece_type, is_promoting=False)
            assert result == expected_name

    def test_promotion_transformations(self):
        """Test piece promotion transformations."""
        test_cases = [
            (PieceType.PAWN, "Fuhyō (Pawn) → Tokin (Promoted Pawn)"),
            (PieceType.ROOK, "Hisha (Rook) → Ryūō (Dragon King)"),
            (PieceType.BISHOP, "Kakugyō (Bishop) → Ryūma (Dragon Horse)"),
            (PieceType.LANCE, "Kyōsha (Lance) → Narikyo (Promoted Lance)"),
            (PieceType.KNIGHT, "Keima (Knight) → Narikei (Promoted Knight)"),
            (PieceType.SILVER, "Ginsho (Silver General) → Narigin (Promoted Silver)"),
        ]

        for piece_type, expected_name in test_cases:
            result = _get_piece_name(piece_type, is_promoting=True)
            assert result == expected_name

    def test_unknown_piece_type(self):
        """Test handling of unknown piece types."""

        # Create a mock unknown piece type
        class UnknownPieceType:
            def __str__(self):
                return "UNKNOWN_PIECE"

        unknown_piece = UnknownPieceType()
        result = _get_piece_name(unknown_piece, is_promoting=False)
        assert "UNKNOWN_PIECE" in result


class TestCoordinateConversion:
    """Test coordinate to square name conversion."""

    def test_coordinate_conversion(self):
        """Test coordinate to square name conversion."""
        test_cases = [
            ((0, 0), "9a"),  # Top-left corner
            ((0, 8), "1a"),  # Top-right corner
            ((8, 0), "9i"),  # Bottom-left corner
            ((8, 8), "1i"),  # Bottom-right corner
            ((4, 4), "5e"),  # Center
        ]

        for (row, col), expected_square in test_cases:
            result = _coords_to_square_name(row, col)
            assert result == expected_square

    def test_coordinate_bounds(self):
        """Test coordinate conversion at boundaries."""
        # Test all corners and edges
        corners_and_edges = [
            (0, 0),
            (0, 4),
            (0, 8),  # Top row
            (4, 0),
            (4, 8),  # Middle row edges
            (8, 0),
            (8, 4),
            (8, 8),  # Bottom row
        ]

        for row, col in corners_and_edges:
            result = _coords_to_square_name(row, col)
            # Should produce valid square names
            assert len(result) == 2
            assert result[0] in "123456789"
            assert result[1] in "abcdefghi"


class TestIntegrationMoveFormatting:
    """Integration tests combining all formatting components."""

    @pytest.fixture
    def policy_mapper(self):
        """Initialize policy output mapper for tests."""
        return PolicyOutputMapper()

    def test_comprehensive_move_set(self, policy_mapper):
        """Test a comprehensive set of different move types."""
        test_moves = [
            # (move_tuple, expected_usi_pattern, expected_description_pattern)
            ((6, 6, 5, 6, False), "3g3f", "piece moving from 3g to 3f"),
            ((8, 8, 7, 7, False), "1i2h", "piece moving from 1i to 2h"),
            ((6, 6, 5, 6, True), "3g3f+", "piece promoting moving from 3g to 3f"),
            ((None, None, 4, 4, PieceType.PAWN), "P*5e", "Fuhyō (Pawn) drop to 5e"),
            ((None, None, 2, 6, PieceType.ROOK), "R*3c", "Hisha (Rook) drop to 3c"),
            ((None, None, 5, 3, PieceType.KNIGHT), "N*6f", "Keima (Knight) drop to 6f"),
        ]

        for move, expected_usi, expected_desc in test_moves:
            result = format_move_with_description(move, policy_mapper, game=None)

            assert expected_usi in result
            assert expected_desc in result
            assert " - " in result
            assert result.endswith(".")

    def test_format_consistency(self, policy_mapper):
        """Test that all formatted moves follow consistent format."""
        test_moves = [
            (6, 6, 5, 6, False),
            (None, None, 4, 4, PieceType.PAWN),
            (8, 8, 7, 7, True),
        ]

        for move in test_moves:
            result = format_move_with_description(move, policy_mapper, game=None)

            # All results should follow: "USI - Description."
            parts = result.split(" - ")
            assert len(parts) == 2
            assert parts[1].endswith(".")
            assert len(parts[0]) > 0  # USI part should not be empty
            assert len(parts[1]) > 1  # Description part should not be just "."
