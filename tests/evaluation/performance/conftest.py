"""Shared fixtures and utilities for performance tests."""

import gc
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import pytest
import torch

from keisei.config_schema import (
    AppConfig,
    DisplayConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    ParallelConfig,
    TrainingConfig,
    WandBConfig,
)
from keisei.evaluation.core import (
    AgentInfo,
    EvaluationStrategy,
    GameResult,
    OpponentInfo,
    SummaryStats,
)
from keisei.evaluation.core_manager import EvaluationManager


class PerformanceMonitor:
    """Monitor system performance during tests."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = None
        self.start_time = None

    def start_monitoring(self):
        """Start performance monitoring."""
        gc.collect()  # Force garbage collection before measurement
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_time = time.perf_counter()

    def stop_monitoring(self):
        """Stop monitoring and return performance metrics."""
        end_time = time.perf_counter()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        return {
            "execution_time": end_time - self.start_time,
            "memory_used": end_memory - self.start_memory,
            "peak_memory": end_memory,
        }

    def get_current_memory_mb(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024


class ConfigurationFactory:
    """Factory for creating test configurations."""

    @staticmethod
    def create_performance_test_config(
        num_games: int = 20,
        enable_enhanced_features: bool = False,
        parallel_execution: bool = False,
    ):
        """Create evaluation configuration optimized for performance testing."""
        # Import the correct factory function
        from keisei.evaluation.core import (
            create_evaluation_config as create_eval_config,
        )

        return create_eval_config(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            num_games=num_games,
            max_concurrent_games=4 if parallel_execution else 1,
            timeout_per_game=30.0,
            save_games=False,
            wandb_logging=False,
            opponent_name="test_opponent",
            enable_in_memory_evaluation=True,
            enable_parallel_execution=parallel_execution,
            temp_agent_device="cpu",
            strategy_params={
                "enable_enhanced_features": enable_enhanced_features,
                "enhanced_analytics": enable_enhanced_features,
                "detailed_logging": False,
            },
        )

    @staticmethod
    def create_minimal_test_config():
        """Create minimal evaluation configuration for testing."""
        # Import the correct factory function
        from keisei.evaluation.core import (
            create_evaluation_config as create_eval_config,
        )

        return create_eval_config(
            strategy=EvaluationStrategy.SINGLE_OPPONENT,
            num_games=1,
            max_concurrent_games=1,
            timeout_per_game=10.0,
            save_games=False,
            wandb_logging=False,
            opponent_name="test_opponent",
            enable_in_memory_evaluation=True,
            enable_parallel_execution=False,
            temp_agent_device="cpu",
        )

    @staticmethod
    def create_base_config(
        strategy: EvaluationStrategy = EvaluationStrategy.SINGLE_OPPONENT,
        num_games: int = 10,
        max_concurrent_games: int = 4,
        timeout_per_game: float = 60.0,
        wandb_logging: bool = False,
        save_games: bool = False,
        **kwargs,
    ):
        """Create evaluation configuration for testing."""
        # Import the correct factory function
        from keisei.evaluation.core import (
            create_evaluation_config as create_eval_config,
        )

        return create_eval_config(
            strategy=strategy,
            num_games=num_games,
            max_concurrent_games=max_concurrent_games,
            timeout_per_game=timeout_per_game,
            save_games=save_games,
            wandb_logging=wandb_logging,
            opponent_name=kwargs.get("opponent_name", "test_opponent"),
            enable_in_memory_evaluation=kwargs.get("enable_in_memory_evaluation", True),
            enable_parallel_execution=kwargs.get("enable_parallel_execution", False),
            temp_agent_device=kwargs.get("temp_agent_device", "cpu"),
            strategy_params=kwargs.get("strategy_params", {}),
        )


def create_evaluation_config(
    strategy: EvaluationStrategy = EvaluationStrategy.SINGLE_OPPONENT,
    num_games: int = 10,
    max_concurrent_games: int = 4,
    timeout_per_game: float = 60.0,
    opponent_name: str = "test_opponent",
    wandb_logging: bool = False,
    save_games: bool = False,
    **kwargs,
) -> EvaluationConfig:
    """Create evaluation configuration for testing."""
    # Import the correct factory function
    from keisei.evaluation.core import create_evaluation_config as create_eval_config

    return create_eval_config(
        strategy=strategy,
        num_games=num_games,
        max_concurrent_games=max_concurrent_games,
        timeout_per_game=timeout_per_game,
        save_games=save_games,
        opponent_name=opponent_name,
        wandb_logging=wandb_logging,
        enable_in_memory_evaluation=kwargs.get("enable_in_memory_evaluation", True),
        enable_parallel_execution=kwargs.get("enable_parallel_execution", False),
        temp_agent_device=kwargs.get("temp_agent_device", "cpu"),
        strategy_params=kwargs.get("strategy_params", {}),
    )


class TestAgentFactory:
    """Factory for creating test agents."""

    @staticmethod
    def create_test_agent(config, checkpoint_path: Optional[str] = None) -> AgentInfo:
        """Create a test agent for performance testing."""
        if checkpoint_path is None:
            # Create a temporary checkpoint file
            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
                # Save a dummy checkpoint
                torch.save(
                    {
                        "model_state_dict": {},
                        "optimizer_state_dict": {},
                        "epoch": 1,
                        "loss": 0.5,
                    },
                    tmp_file.name,
                )
                checkpoint_path = tmp_file.name

        return AgentInfo(
            name="test_agent",
            checkpoint_path=checkpoint_path,
            model_type="test",
            training_timesteps=100,
            metadata={
                "config": (
                    config.model_dump()
                    if hasattr(config, "model_dump")
                    else str(config)
                ),
                "performance_baseline": True,
            },
        )


class MockGameResultFactory:
    """Factory for creating mock game results."""

    @staticmethod
    def create_successful_game_result(
        game_id: str = "test_game",
        winner: str = "agent",
        game_length: int = 100,
        duration: float = 1.0,
    ) -> GameResult:
        """Create a successful game result for testing."""
        # Create mock agent and opponent info
        from keisei.evaluation.core import AgentInfo, OpponentInfo

        agent_info = AgentInfo(
            name="test_agent", model_type="test", metadata={"test": True}
        )

        opponent_info = OpponentInfo(
            name="test_opponent", type="built_in", metadata={"test": True}
        )

        return GameResult(
            game_id=game_id,
            winner=0 if winner == "agent" else 1 if winner == "opponent" else None,
            moves_count=game_length,
            duration_seconds=duration,
            agent_info=agent_info,
            opponent_info=opponent_info,
            metadata={
                "performance_test": True,
                "mock_result": True,
            },
        )

    @staticmethod
    def create_game_result_batch(
        count: int = 10,
        agent_win_rate: float = 0.6,
        avg_game_length: int = 100,
        avg_duration: float = 1.0,
    ) -> List[GameResult]:
        """Create a batch of game results for testing."""
        results = []
        for i in range(count):
            # Determine winner based on win rate
            if i / count < agent_win_rate:
                winner = "agent"
            elif i / count < agent_win_rate + (1 - agent_win_rate) * 0.5:
                winner = "opponent"
            else:
                winner = "draw"

            result = MockGameResultFactory.create_successful_game_result(
                game_id=f"batch_game_{i}",
                winner=winner,
                game_length=avg_game_length + (i % 20 - 10),
                duration=avg_duration + (i % 10 - 5) * 0.1,
            )
            results.append(result)

        return results


@pytest.fixture
def performance_monitor():
    """Provide a performance monitor for tests."""
    return PerformanceMonitor()


@pytest.fixture
def minimal_config():
    """Provide minimal test configuration."""
    return ConfigurationFactory.create_minimal_test_config()


@pytest.fixture
def performance_config():
    """Provide performance test configuration."""
    return ConfigurationFactory.create_performance_test_config()


@pytest.fixture
def test_agent():
    """Provide a test agent."""
    config = ConfigurationFactory.create_minimal_test_config()
    return TestAgentFactory.create_test_agent(config)


@pytest.fixture
def mock_game_results():
    """Provide mock game results for testing."""
    return MockGameResultFactory.create_game_result_batch(count=10)
