"""
This module re-exports the main classes from the refactored Shogi engine components
for backward compatibility.

This file is maintained for backward compatibility with code that imports directly from 
keisei.shogi_engine. New code should import from keisei.shogi.
"""

# Re-export from the new location
from .shogi.shogi_core_definitions import Color, PieceType, Piece, MoveTuple
from .shogi.shogi_game import ShogiGame

# These exports allow code that previously imported from shogi_engine.py
# to continue working without changes
__all__ = ['Color', 'PieceType', 'Piece', 'ShogiGame', 'MoveTuple']