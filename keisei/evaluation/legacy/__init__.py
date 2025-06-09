"""
Legacy evaluation components for backward compatibility.

This module contains the original evaluation files moved here during the refactor
to maintain backward compatibility while the new system is being developed.
"""

# Legacy components - maintain imports for backward compatibility
try:
    from .elo_registry import EloRegistry
    from .evaluate import Evaluator
    from .loop import run_evaluation_loop

    __all__ = [
        "Evaluator",
        "run_evaluation_loop",
        "EloRegistry",
    ]
except ImportError:
    # Files may not be present yet during migration
    __all__ = []
