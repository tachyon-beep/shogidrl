"""
Core evaluation system components.

This module contains the foundational classes and abstractions for the
evaluation system, including contexts, results, and base evaluator classes.
"""

from .base_evaluator import (
    BaseEvaluator,
    EvaluatorFactory,
    create_agent_info,
    evaluate_agent,
)
from .evaluation_config import (
    BenchmarkConfig,
    EvaluationConfig,
    EvaluationStrategy,
    LadderConfig,
    SingleOpponentConfig,
    TournamentConfig,
    create_evaluation_config,
    from_legacy_config,
)
from .evaluation_context import AgentInfo, EvaluationContext, OpponentInfo
from .evaluation_result import (
    EvaluationResult,
    GameResult,
    SummaryStats,
    create_game_result,
)

__all__ = [
    # Base classes
    "BaseEvaluator",
    "EvaluatorFactory",
    "evaluate_agent",
    "create_agent_info",
    # Configuration
    "EvaluationConfig",
    "EvaluationStrategy",
    "SingleOpponentConfig",
    "TournamentConfig",
    "LadderConfig",
    "BenchmarkConfig",
    "create_evaluation_config",
    "from_legacy_config",
    # Context and metadata
    "EvaluationContext",
    "AgentInfo",
    "OpponentInfo",
    # Results
    "EvaluationResult",
    "GameResult",
    "SummaryStats",
    "create_game_result",
]
