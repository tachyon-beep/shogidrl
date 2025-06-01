"""
Test suite for move formatting functionality.

This module contains comprehensive tests for the move formatting system,
including basic move formatting, enhanced formatting with piece information,
and Japanese piece naming.
"""

from types import SimpleNamespace

import pytest

from keisei.shogi.shogi_core_definitions import PieceType
from keisei.utils import (
    PolicyOutputMapper,
    _coords_to_square_name,
    _get_piece_name,
    format_move_with_description,
    format_move_with_description_enhanced,
)


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

    @pytest.mark.parametrize(
        "piece_type,col,row,expected_notation,expected_description",
        [
            (PieceType.PAWN, 4, 4, "P*5e", "Fuhyō (Pawn) drop to 5e"),
            (PieceType.ROOK, 2, 6, "R*3c", "Hisha (Rook) drop to 3c"),
            (PieceType.KNIGHT, 5, 3, "N*6f", "Keima (Knight) drop to 6f"),
        ],
        ids=["pawn", "rook", "knight"],
    )
    def test_drop_moves(self, policy_mapper, piece_type, col, row, expected_notation, expected_description):
        """Test formatting drop moves for different pieces."""
        move = (None, None, col, row, piece_type)
        result = format_move_with_description(move, policy_mapper, game=None)

        assert expected_notation in result
        assert expected_description in result
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

    @pytest.mark.parametrize(
        "piece_type,expected_name",
        [
            (PieceType.PAWN, "Fuhyō (Pawn)"),
            (PieceType.ROOK, "Hisha (Rook)"),
            (PieceType.BISHOP, "Kakugyō (Bishop)"),
            (PieceType.KING, "Ōshō (King)"),
            (PieceType.KNIGHT, "Keima (Knight)"),
            (PieceType.LANCE, "Kyōsha (Lance)"),
            (PieceType.SILVER, "Ginsho (Silver General)"),
            (PieceType.GOLD, "Kinshō (Gold General)"),
        ],
        ids=["pawn", "rook", "bishop", "king", "knight", "lance", "silver", "gold"],
    )
    def test_regular_piece_names(self, piece_type, expected_name):
        """Test regular piece name generation."""
        result = _get_piece_name(piece_type, is_promoting=False)
        assert result == expected_name

    @pytest.mark.parametrize(
        "piece_type,expected_name",
        [
            (PieceType.PROMOTED_PAWN, "Tokin (Promoted Pawn)"),
            (PieceType.PROMOTED_ROOK, "Ryūō (Dragon King)"),
            (PieceType.PROMOTED_BISHOP, "Ryūma (Dragon Horse)"),
            (PieceType.PROMOTED_LANCE, "Narikyo (Promoted Lance)"),
            (PieceType.PROMOTED_KNIGHT, "Narikei (Promoted Knight)"),
            (PieceType.PROMOTED_SILVER, "Narigin (Promoted Silver)"),
        ],
        ids=["promoted_pawn", "promoted_rook", "promoted_bishop", "promoted_lance", "promoted_knight", "promoted_silver"],
    )
    def test_promoted_piece_names(self, piece_type, expected_name):
        """Test promoted piece name generation."""
        result = _get_piece_name(piece_type, is_promoting=False)
        assert result == expected_name

    @pytest.mark.parametrize(
        "piece_type,expected_name",
        [
            (PieceType.PAWN, "Fuhyō (Pawn) → Tokin (Promoted Pawn)"),
            (PieceType.ROOK, "Hisha (Rook) → Ryūō (Dragon King)"),
            (PieceType.BISHOP, "Kakugyō (Bishop) → Ryūma (Dragon Horse)"),
            (PieceType.LANCE, "Kyōsha (Lance) → Narikyo (Promoted Lance)"),
            (PieceType.KNIGHT, "Keima (Knight) → Narikei (Promoted Knight)"),
            (PieceType.SILVER, "Ginsho (Silver General) → Narigin (Promoted Silver)"),
        ],
        ids=["pawn_promotion", "rook_promotion", "bishop_promotion", "lance_promotion", "knight_promotion", "silver_promotion"],
    )
    def test_promotion_transformations(self, piece_type, expected_name):
        """Test piece promotion transformations."""
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

    @pytest.mark.parametrize(
        "row,col,expected_square",
        [
            (0, 0, "9a"),  # Top-left corner
            (0, 8, "1a"),  # Top-right corner
            (8, 0, "9i"),  # Bottom-left corner
            (8, 8, "1i"),  # Bottom-right corner
            (4, 4, "5e"),  # Center
        ],
        ids=["top_left", "top_right", "bottom_left", "bottom_right", "center"],
    )
    def test_coordinate_conversion(self, row, col, expected_square):
        """Test coordinate to square name conversion."""
        result = _coords_to_square_name(row, col)
        assert result == expected_square

    @pytest.mark.parametrize(
        "row,col",
        [
            (0, 0), (0, 4), (0, 8),  # Top row
            (4, 0), (4, 8),          # Middle row edges
            (8, 0), (8, 4), (8, 8),  # Bottom row
        ],
        ids=["corner_0_0", "edge_0_4", "corner_0_8", "edge_4_0", "edge_4_8", "corner_8_0", "edge_8_4", "corner_8_8"],
    )
    def test_coordinate_bounds(self, row, col):
        """Test coordinate conversion at boundaries."""
        result = _coords_to_square_name(row, col)
        # Should not raise an exception and should return a valid string
        assert isinstance(result, str)
        assert len(result) == 2  # Format: digit + letter
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
