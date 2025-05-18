"""
DRL Shogi Client - A reinforcement learning agent for Japanese chess (Shogi).

This package contains:
- Shogi game engine and rules
- Reinforcement learning agents
- Neural network models
- Training utilities
"""

# Re-export the main components for easy access
from .shogi.shogi_core_definitions import Color, PieceType, Piece, MoveTuple
from .shogi.shogi_game import ShogiGame

__all__ = [
    # Shogi core types
    'Color', 'PieceType', 'Piece', 'MoveTuple', 'ShogiGame',

    # Let the other modules be imported explicitly
]
