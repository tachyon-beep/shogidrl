"""Opponent implementations and management utilities."""

from .opponent_pool import OpponentPool

__all__ = ["OpponentPool"]

# Enhanced features (optional)
try:
    from .enhanced_manager import EnhancedOpponentManager, SelectionStrategy, OpponentPerformanceData
    __all__.extend(['EnhancedOpponentManager', 'SelectionStrategy', 'OpponentPerformanceData'])
except ImportError:
    pass
