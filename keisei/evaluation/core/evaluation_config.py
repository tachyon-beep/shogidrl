"""
Configuration classes for evaluation strategies and parameters.

This module defines the configuration structures for different evaluation strategies,
including strategy types, parameters, and validation logic.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class EvaluationStrategy(Enum):
    """Enumeration of available evaluation strategies."""

    SINGLE_OPPONENT = "single_opponent"
    TOURNAMENT = "tournament"
    LADDER = "ladder"
    BENCHMARK = "benchmark"
    CUSTOM = "custom"


@dataclass
class EvaluationConfig:
    """Base configuration for evaluation runs."""

    # Core parameters
    strategy: EvaluationStrategy
    num_games: int = 100
    max_concurrent_games: int = 4
    timeout_per_game: Optional[float] = None

    # Randomization
    randomize_positions: bool = True
    random_seed: Optional[int] = None

    # Output and logging
    save_games: bool = True
    save_path: Optional[str] = None
    log_level: str = "INFO"

    # Integration
    wandb_logging: bool = True
    update_elo: bool = True

    # Performance optimization settings
    enable_in_memory_evaluation: bool = True
    model_weight_cache_size: int = 5
    enable_parallel_execution: bool = True
    process_restart_threshold: int = 100
    temp_agent_device: str = "cpu"
    clear_cache_after_evaluation: bool = True

    # Additional parameters (strategy-specific)
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_games <= 0:
            raise ValueError("num_games must be positive")

        if self.max_concurrent_games <= 0:
            raise ValueError("max_concurrent_games must be positive")

        if self.timeout_per_game is not None and self.timeout_per_game <= 0:
            raise ValueError("timeout_per_game must be positive if specified")

        # Validate log level
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "strategy": self.strategy.value,
            "num_games": self.num_games,
            "max_concurrent_games": self.max_concurrent_games,
            "timeout_per_game": self.timeout_per_game,
            "randomize_positions": self.randomize_positions,
            "random_seed": self.random_seed,
            "save_games": self.save_games,
            "save_path": self.save_path,
            "log_level": self.log_level,
            "wandb_logging": self.wandb_logging,
            "update_elo": self.update_elo,
            "strategy_params": self.strategy_params.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationConfig":
        """Create configuration from dictionary."""
        # Convert strategy string to enum
        if isinstance(data.get("strategy"), str):
            data = data.copy()
            data["strategy"] = EvaluationStrategy(data["strategy"])

        return cls(**data)


@dataclass
class SingleOpponentConfig(EvaluationConfig):
    """Configuration for single opponent evaluation."""

    strategy: EvaluationStrategy = field(
        default=EvaluationStrategy.SINGLE_OPPONENT, init=False
    )
    opponent_name: str = "default_opponent"
    opponent_path: Optional[str] = None  # Path to opponent model/config
    opponent_params: Dict[str, Any] = field(
        default_factory=dict
    )  # Additional params for opponent
    play_as_both_colors: bool = True  # Play games as Sente and Gote
    color_balance_tolerance: float = 0.1  # Max allowed imbalance (e.g. 0.1 for 10%)


@dataclass
class TournamentConfig(EvaluationConfig):
    """Configuration for tournament evaluation."""

    strategy: EvaluationStrategy = field(
        default=EvaluationStrategy.TOURNAMENT, init=False
    )
    opponent_pool_config: List[Dict[str, Any]] = field(
        default_factory=list
    )  # List of opponent configurations
    # Example opponent_pool_config entry: {"name": "opp1", "type": "ppo", "checkpoint_path": "/path/to/model"}
    num_games_per_opponent: Optional[int] = (
        None  # If None, num_games is divided among opponents
    )


@dataclass
class LadderConfig(EvaluationConfig):
    """Configuration for ladder (ELO) evaluation."""

    strategy: EvaluationStrategy = field(default=EvaluationStrategy.LADDER, init=False)
    opponent_pool_config: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Initial pool of opponents for the ladder
    elo_config: Dict[str, Any] = field(
        default_factory=dict
    )  # Configuration for EloTracker (e.g., K-factor, initial ratings)
    num_games_per_match: int = (
        2  # Number of games to play against each selected ladder opponent
    )
    num_opponents_per_evaluation: int = (
        3  # Number of opponents to select from ladder for one evaluation run
    )
    rating_match_range: int = 200  # ELO points range for selecting opponents


@dataclass
class BenchmarkConfig(EvaluationConfig):
    """Configuration for benchmark evaluation."""

    strategy: EvaluationStrategy = field(
        default=EvaluationStrategy.BENCHMARK, init=False
    )
    suite_config: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Defines the benchmark cases
    # Example suite_config entry: {"name": "case1", "type": "ppo", "checkpoint_path": "/path/to/model1.ptk", "metadata": {"description": "vs strong bot"}}
    # Or: {"name": "opening_puzzle_X", "type": "scenario", "fen": "start_fen_string...", "metadata": {"objective": "win in 10 moves"}}
    num_games_per_benchmark_case: int = 1


# Mapping from strategy enum to config class
STRATEGY_CONFIG_MAP: Dict[EvaluationStrategy, type[EvaluationConfig]] = {
    EvaluationStrategy.SINGLE_OPPONENT: SingleOpponentConfig,
    EvaluationStrategy.TOURNAMENT: TournamentConfig,
    EvaluationStrategy.LADDER: LadderConfig,
    EvaluationStrategy.BENCHMARK: BenchmarkConfig,
    # EvaluationStrategy.CUSTOM: CustomConfig, # If you have custom configs
}


def get_config_class(
    strategy: Union[EvaluationStrategy, str],
) -> type[EvaluationConfig]:
    """Get the specific config class for a given strategy."""
    if isinstance(strategy, str):
        try:
            strategy = EvaluationStrategy(strategy)
        except ValueError:
            logger.error(f"Unknown evaluation strategy string: {strategy}")
            raise

    config_class = STRATEGY_CONFIG_MAP.get(strategy)
    if config_class is None:
        logger.warning(
            f"No specific config class found for strategy {strategy}, using base EvaluationConfig."
        )
        return EvaluationConfig
    return config_class


def create_evaluation_config(
    strategy: Union[str, EvaluationStrategy], **kwargs
) -> EvaluationConfig:
    """
    Factory function to create appropriate evaluation configuration.

    Args:
        strategy: Evaluation strategy type
        **kwargs: Strategy-specific parameters

    Returns:
        Configured evaluation config object
    """
    if isinstance(strategy, str):
        strategy = EvaluationStrategy(strategy)

    config_classes = {
        EvaluationStrategy.SINGLE_OPPONENT: SingleOpponentConfig,
        EvaluationStrategy.TOURNAMENT: TournamentConfig,
        EvaluationStrategy.LADDER: LadderConfig,
        EvaluationStrategy.BENCHMARK: BenchmarkConfig,
    }

    config_class = config_classes.get(strategy, EvaluationConfig)

    try:
        # Don't pass strategy to specific config classes since they have init=False
        if config_class != EvaluationConfig:
            return config_class(**kwargs)
        else:
            # Base config class can accept strategy parameter
            return config_class(strategy=strategy, **kwargs)
    except TypeError as e:
        logger.error("Invalid parameters for %s config: %s", strategy.value, e)
        # Fall back to base config with strategy params
        strategy_params = kwargs.copy()
        base_params = {
            k: v
            for k, v in kwargs.items()
            if k in EvaluationConfig.__dataclass_fields__
        }
        strategy_params = {
            k: v
            for k, v in kwargs.items()
            if k not in EvaluationConfig.__dataclass_fields__
        }
        base_params["strategy_params"] = strategy_params
        return EvaluationConfig(strategy=strategy, **base_params)
