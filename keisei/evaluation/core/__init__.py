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
)
from .evaluation_context import AgentInfo, EvaluationContext, OpponentInfo
from .evaluation_result import (
    EvaluationResult,
    GameResult,
    SummaryStats,
    create_game_result,
)
from .model_manager import ModelWeightManager
from .parallel_executor import (
    BatchGameExecutor,
    ParallelGameExecutor,
    ParallelGameTask,
    create_parallel_game_tasks,
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
    # Context and metadata
    "EvaluationContext",
    "AgentInfo",
    "OpponentInfo",
    # Results
    "EvaluationResult",
    "GameResult",
    "SummaryStats",
    "create_game_result",
    # Model management
    "ModelWeightManager",
    # Parallel execution
    "ParallelGameExecutor",
    "BatchGameExecutor",
    "ParallelGameTask",
    "create_parallel_game_tasks",
]

# Enhanced features (optional)
try:
    from .background_tournament import (
        BackgroundTournamentManager,
        TournamentProgress,
        TournamentStatus,
    )

    __all__.extend(
        ["BackgroundTournamentManager", "TournamentProgress", "TournamentStatus"]
    )
except ImportError:
    # Background tournament features not available
    pass
