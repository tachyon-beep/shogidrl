"""Compatibility wrapper for evaluation entrypoints."""

from .legacy.evaluate import Evaluator, execute_full_evaluation_run

__all__ = [
    "Evaluator",
    "execute_full_evaluation_run",
]
