"""
Test utilities and factories for evaluation testing.

This module provides reusable test components, factories, and utilities
to support better testing patterns and reduce test code duplication.
"""

import random
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import torch

from keisei.evaluation.core import (
    EvaluationContext,
    EvaluationResult,
    GameResult,
    SummaryStats,
    EvaluationConfig,
    create_evaluation_config,
)


class EvaluationTestFactory:
    """Factory for creating test evaluation objects."""

    @staticmethod
    def create_test_agent(model_complexity: str = "simple"):
        """Create a test agent with specified complexity."""
        from tests.evaluation.conftest import make_test_config
        from keisei.core.ppo_agent import PPOAgent
        from keisei.training.models.resnet_tower import ActorCriticResTower

        if model_complexity == "simple":
            num_blocks = 2
        elif model_complexity == "medium":
            num_blocks = 4
        elif model_complexity == "complex":
            num_blocks = 8
        else:
            num_blocks = 2

        model = ActorCriticResTower(
            input_channels=46,
            num_actions_total=13527,
            tower_depth=num_blocks,
            tower_width=32,
            se_ratio=0.25,
        )

        config = make_test_config()
        return PPOAgent(model=model, config=config, device=torch.device("cpu"))

    @staticmethod
    def create_game_result(
        winner: str = "agent",
        game_length: int = 50,
        elo_change: float = 10.0,
        timestamp: Optional[datetime] = None,
    ) -> GameResult:
        """Create a game result for testing."""
        if timestamp is None:
            timestamp = datetime.now()

        from keisei.evaluation.core import AgentInfo, OpponentInfo
        
        return GameResult(
            game_id=f"test_game_{random.randint(1000, 9999)}",
            winner=0 if winner == "agent" else (1 if winner == "opponent" else None),
            moves_count=game_length,
            duration_seconds=game_length * 2.0,  # Approximate timing
            agent_info=AgentInfo(name="TestAgent"),
            opponent_info=OpponentInfo(name="TestOpponent", type="random"),
        )

    @staticmethod
    def create_game_series(
        num_games: int = 10,
        win_rate: float = 0.6,
        avg_game_length: int = 50,
        base_timestamp: Optional[datetime] = None,
    ) -> List[GameResult]:
        """Create a series of game results with specified characteristics."""
        if base_timestamp is None:
            base_timestamp = datetime.now() - timedelta(hours=1)

        games = []
        num_wins = int(num_games * win_rate)
        num_losses = num_games - num_wins

        # Create wins
        for i in range(num_wins):
            games.append(
                EvaluationTestFactory.create_game_result(
                    winner="agent",
                    game_length=avg_game_length + random.randint(-10, 10),
                    elo_change=random.uniform(5.0, 15.0),
                    timestamp=base_timestamp + timedelta(minutes=i * 5),
                )
            )

        # Create losses
        for i in range(num_losses):
            games.append(
                EvaluationTestFactory.create_game_result(
                    winner="opponent",
                    game_length=avg_game_length + random.randint(-10, 10),
                    elo_change=random.uniform(-15.0, -5.0),
                    timestamp=base_timestamp + timedelta(minutes=(num_wins + i) * 5),
                )
            )

        # Shuffle to randomize order
        random.shuffle(games)
        return games

    @staticmethod
    def create_summary_stats(
        wins: int = 6,
        losses: int = 4,
        draws: int = 0,
        avg_game_length: float = 50.0,
        total_time: float = 300.0,
    ) -> SummaryStats:
        """Create summary statistics for testing."""
        total_games = wins + losses + draws
        win_rate = wins / total_games if total_games > 0 else 0.0
        loss_rate = losses / total_games if total_games > 0 else 0.0
        draw_rate = draws / total_games if total_games > 0 else 0.0

        return SummaryStats(
            total_games=total_games,
            agent_wins=wins,
            opponent_wins=losses,
            draws=draws,
            win_rate=win_rate,
            loss_rate=loss_rate,
            draw_rate=draw_rate,
            avg_game_length=avg_game_length,
            total_moves=int(avg_game_length * total_games),
            avg_duration_seconds=total_time / total_games if total_games > 0 else 0.0,
        )

    @staticmethod
    def create_evaluation_result(
        games: Optional[List[GameResult]] = None,
        summary_stats: Optional[SummaryStats] = None,
        analytics: Optional[Dict[str, Any]] = None,
        context: Optional[EvaluationContext] = None,
    ) -> EvaluationResult:
        """Create a complete evaluation result for testing."""
        if games is None:
            games = EvaluationTestFactory.create_game_series()

        if summary_stats is None:
            # Calculate from games
            wins = sum(1 for g in games if g.winner == 0)
            losses = sum(1 for g in games if g.winner == 1)
            draws = sum(1 for g in games if g.winner is None)
            avg_length = sum(g.moves_count for g in games) / len(games) if games else 0

            summary_stats = EvaluationTestFactory.create_summary_stats(
                wins=wins, losses=losses, draws=draws, avg_game_length=avg_length
            )

        if context is None:
            context = EvaluationTestFactory.create_evaluation_context()

        return EvaluationResult(
            context=context,
            games=games,
            summary_stats=summary_stats,
        )

    @staticmethod
    def create_evaluation_context(
        strategy: str = "single_opponent",
        opponent_name: str = "test_opponent",
        run_name: str = "test_run",
    ) -> EvaluationContext:
        """Create evaluation context for testing."""
        from keisei.evaluation.core import AgentInfo
        
        config = create_evaluation_config(
            strategy=strategy,
            opponent_name=opponent_name
        )
        
        return EvaluationContext(
            session_id="test_session",
            timestamp=datetime.now(),
            agent_info=AgentInfo(name="TestAgent"),
            configuration=config,
            environment_info={"device": "cpu"},
            metadata={"run_name": run_name},
        )


class ConfigurationFactory:
    """Factory for creating test configurations."""

    @staticmethod
    def create_single_opponent_config(
        opponent_name: str = "test_opponent",
        num_games: int = 10,
        play_as_both_colors: bool = True,
        max_concurrent_games: int = 1,
    ) -> EvaluationConfig:
        """Create single opponent configuration."""
        return create_evaluation_config(
            strategy="single_opponent",
            opponent_name=opponent_name,
            num_games=num_games,
            max_concurrent_games=max_concurrent_games,
            strategy_params={"play_as_both_colors": play_as_both_colors}
        )

    @staticmethod
    def create_tournament_config(
        opponent_pool: Optional[List[str]] = None,
        num_games_per_opponent: int = 5,
        max_concurrent_games: int = 1,
    ) -> EvaluationConfig:
        """Create tournament configuration."""
        if opponent_pool is None:
            opponent_pool = ["opponent1", "opponent2", "opponent3"]

        return create_evaluation_config(
            strategy="tournament",
            num_games=num_games_per_opponent * len(opponent_pool),
            max_concurrent_games=max_concurrent_games,
            strategy_params={
                "opponent_pool": opponent_pool,
                "num_games_per_opponent": num_games_per_opponent
            }
        )

    @staticmethod
    def create_ladder_config(
        initial_opponents: Optional[List[str]] = None,
        games_per_evaluation: int = 10,
        elo_k_factor: float = 32.0,
    ) -> EvaluationConfig:
        """Create ladder configuration."""
        if initial_opponents is None:
            initial_opponents = ["ladder_opponent1", "ladder_opponent2"]

        return create_evaluation_config(
            strategy="ladder",
            num_games=games_per_evaluation,
            strategy_params={
                "initial_opponents": initial_opponents,
                "elo_k_factor": elo_k_factor
            }
        )

    @staticmethod
    def create_benchmark_config(
        benchmark_opponents: Optional[List[str]] = None, 
        games_per_benchmark: int = 20
    ) -> EvaluationConfig:
        """Create benchmark configuration."""
        if benchmark_opponents is None:
            benchmark_opponents = ["benchmark1", "benchmark2"]

        return create_evaluation_config(
            strategy="benchmark",
            num_games=games_per_benchmark * len(benchmark_opponents),
            strategy_params={
                "benchmark_opponents": benchmark_opponents,
                "num_games_per_benchmark_case": games_per_benchmark,
                "suite_config": [{"name": name, "type": "random"} for name in benchmark_opponents]
            }
        )


class MockFactory:
    """Factory for creating properly configured mocks."""

    @staticmethod
    def create_mock_evaluator(
        evaluation_result: Optional[EvaluationResult] = None,
    ) -> MagicMock:
        """Create a mock evaluator with realistic behavior."""
        mock_evaluator = MagicMock()

        if evaluation_result is None:
            evaluation_result = EvaluationTestFactory.create_evaluation_result()

        mock_evaluator.evaluate.return_value = evaluation_result
        mock_evaluator.validate_config.return_value = True

        return mock_evaluator

    @staticmethod
    def create_mock_agent(with_model: bool = True) -> MagicMock:
        """Create a mock agent with optional model."""
        mock_agent = MagicMock()

        if with_model:
            mock_model = MagicMock()
            mock_model.state_dict.return_value = {
                "layer1.weight": torch.randn(10, 5),
                "layer1.bias": torch.randn(10),
                "layer2.weight": torch.randn(1, 10),
                "layer2.bias": torch.randn(1),
            }
            mock_agent.model = mock_model

        return mock_agent


# Convenience functions for common test patterns
def create_test_evaluation_manager(
    config_type: str = "single_opponent", **config_kwargs
):
    """Create a test evaluation manager with specified configuration."""
    from keisei.evaluation.core_manager import EvaluationManager

    if config_type == "single_opponent":
        config = ConfigurationFactory.create_single_opponent_config(**config_kwargs)
    elif config_type == "tournament":
        config = ConfigurationFactory.create_tournament_config(**config_kwargs)
    elif config_type == "ladder":
        config = ConfigurationFactory.create_ladder_config(**config_kwargs)
    elif config_type == "benchmark":
        config = ConfigurationFactory.create_benchmark_config(**config_kwargs)
    else:
        raise ValueError(f"Unknown config type: {config_type}")

    return EvaluationManager(config, "test_run")


def create_test_evaluation_scenario(
    num_games: int = 10, win_rate: float = 0.6, config_type: str = "single_opponent"
):
    """Create a complete test scenario with manager, agent, and expected results."""
    manager = create_test_evaluation_manager(config_type)
    agent = EvaluationTestFactory.create_test_agent()
    games = EvaluationTestFactory.create_game_series(
        num_games=num_games, win_rate=win_rate
    )
    expected_result = EvaluationTestFactory.create_evaluation_result(games=games)

    return {
        "manager": manager,
        "agent": agent,
        "expected_games": games,
        "expected_result": expected_result,
    }


# Pytest fixtures for common test objects
def pytest_test_agent():
    """Pytest fixture for test agent."""
    return EvaluationTestFactory.create_test_agent()


def pytest_test_config():
    """Pytest fixture for test configuration."""
    return ConfigurationFactory.create_single_opponent_config()


def pytest_sample_games():
    """Pytest fixture for sample game results."""
    return EvaluationTestFactory.create_game_series()


def pytest_evaluation_result():
    """Pytest fixture for evaluation result."""
    return EvaluationTestFactory.create_evaluation_result()