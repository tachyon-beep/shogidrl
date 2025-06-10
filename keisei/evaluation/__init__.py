# keisei/evaluation/__init__.py
"""Public interface for the evaluation package."""

from .manager import EvaluationManager

__all__ = ["EvaluationManager"]

# Enhanced features (optional)
try:
    from .enhanced_manager import EnhancedEvaluationManager
    __all__.append('EnhancedEvaluationManager')
except ImportError:
    pass
