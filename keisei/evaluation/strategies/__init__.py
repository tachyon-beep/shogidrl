"""
Evaluation strategies for Keisei Shogi.

This package contains different evaluation strategy implementations, such as:
- Single Opponent: Evaluate against one fixed opponent.
- Tournament: Round-robin evaluation against a pool of opponents.
- Ladder: ELO-based ladder progression against adaptive opponents.
- Benchmark: Evaluate against a fixed suite of benchmark scenarios.
"""

from .benchmark import BenchmarkEvaluator
from .ladder import LadderEvaluator
from .single_opponent import SingleOpponentEvaluator
from .tournament import TournamentEvaluator

__all__ = [
    "SingleOpponentEvaluator",
    "TournamentEvaluator",
    "LadderEvaluator",
    "BenchmarkEvaluator",
]
