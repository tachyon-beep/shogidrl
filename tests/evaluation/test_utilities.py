"""
Test utilities and factories for evaluation testing.

This module provides reusable test components, factories, and utilities
to support better testing patterns and reduce test code duplication.
"""

import random
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from unittest.mock import MagicMock

import torch

from keisei.evaluation.core import (
    EvaluationResult, GameResult, SummaryStats, EvaluationContext
)
from keisei.evaluation.strategies.single_opponent import SingleOpponentConfig
from keisei.evaluation.strategies.tournament import TournamentConfig
from keisei.evaluation.strategies.ladder import LadderConfig
from keisei.evaluation.strategies.benchmark import BenchmarkConfig


class EvaluationTestFactory:
    """Factory for creating test evaluation objects."""
    
    @staticmethod
    def create_test_agent(model_complexity: str = "simple"):
        """Create a test agent with specified complexity."""
        from keisei.training.core.model.resnet_tower import ResNetTower
        from keisei.training.core.agent.ppo_agent import PPOAgent
        from keisei.config import CONFIG
        
        if model_complexity == "simple":
            num_blocks = 2
        elif model_complexity == "medium":
            num_blocks = 4
        elif model_complexity == "complex":
            num_blocks = 8
        else:
            num_blocks = 2
            
        model = ResNetTower(
            board_size=CONFIG.GAME.BOARD_SIZE,
            num_channels=CONFIG.MODEL.NUM_CHANNELS,
            num_blocks=num_blocks
        )
        
        return PPOAgent(model=model, config=CONFIG)
    
    @staticmethod
    def create_game_result(
        winner: str = "agent",
        game_length: int = 50,
        elo_change: float = 10.0,
        timestamp: Optional[datetime] = None
    ) -> GameResult:
        """Create a game result for testing."""
        if timestamp is None:
            timestamp = datetime.now()
            
        return GameResult(
            winner=winner,
            game_length=game_length,
            elo_change=elo_change,
            timestamp=timestamp
        )
    
    @staticmethod
    def create_game_series(
        num_games: int = 10,
        win_rate: float = 0.6,
        avg_game_length: int = 50,
        base_timestamp: Optional[datetime] = None
    ) -> List[GameResult]:
        """Create a series of game results with specified characteristics."""
        if base_timestamp is None:
            base_timestamp = datetime.now() - timedelta(hours=1)
        
        games = []
        num_wins = int(num_games * win_rate)
        num_losses = num_games - num_wins
        
        # Create wins
        for i in range(num_wins):
            games.append(EvaluationTestFactory.create_game_result(
                winner="agent",
                game_length=avg_game_length + random.randint(-10, 10),
                elo_change=random.uniform(5.0, 15.0),
                timestamp=base_timestamp + timedelta(minutes=i*5)
            ))
        
        # Create losses
        for i in range(num_losses):
            games.append(EvaluationTestFactory.create_game_result(
                winner="opponent",
                game_length=avg_game_length + random.randint(-10, 10),
                elo_change=random.uniform(-15.0, -5.0),
                timestamp=base_timestamp + timedelta(minutes=(num_wins + i)*5)
            ))
        
        # Shuffle to randomize order
        random.shuffle(games)
        return games
    
    @staticmethod
    def create_summary_stats(
        wins: int = 6,
        losses: int = 4,
        draws: int = 0,
        avg_game_length: float = 50.0,
        total_time: float = 300.0
    ) -> SummaryStats:
        """Create summary statistics for testing."""
        total_games = wins + losses + draws
        win_rate = wins / total_games if total_games > 0 else 0.0
        
        return SummaryStats(
            wins=wins,
            losses=losses,
            draws=draws,
            total_games=total_games,
            win_rate=win_rate,
            avg_game_length=avg_game_length,
            total_time=total_time
        )
    
    @staticmethod
    def create_evaluation_result(
        games: Optional[List[GameResult]] = None,
        summary_stats: Optional[SummaryStats] = None,
        analytics: Optional[Dict[str, Any]] = None,
        context: Optional[EvaluationContext] = None
    ) -> EvaluationResult:
        """Create a complete evaluation result for testing."""
        if games is None:
            games = EvaluationTestFactory.create_game_series()
        
        if summary_stats is None:
            # Calculate from games
            wins = sum(1 for g in games if g.winner == "agent")
            losses = sum(1 for g in games if g.winner == "opponent")
            draws = sum(1 for g in games if g.winner == "draw")
            avg_length = sum(g.game_length for g in games) / len(games) if games else 0
            
            summary_stats = EvaluationTestFactory.create_summary_stats(
                wins=wins,
                losses=losses,
                draws=draws,
                avg_game_length=avg_length
            )
        
        if analytics is None:
            analytics = {
                "first_player_win_rate": 0.6,
                "second_player_win_rate": 0.4,
                "avg_elo_change": 5.0
            }
        
        if context is None:
            context = EvaluationTestFactory.create_evaluation_context()
        
        return EvaluationResult(
            context=context,
            games=games,
            summary_stats=summary_stats,
            analytics=analytics
        )
    
    @staticmethod
    def create_evaluation_context(
        strategy: str = "single_opponent",
        opponent_name: str = "test_opponent",
        run_name: str = "test_run"
    ) -> EvaluationContext:
        """Create evaluation context for testing."""
        return EvaluationContext(
            strategy=strategy,
            opponent_name=opponent_name,
            run_name=run_name,
            timestamp=datetime.now()
        )


class ConfigurationFactory:
    """Factory for creating test configurations."""
    
    @staticmethod
    def create_single_opponent_config(
        opponent_name: str = "test_opponent",
        num_games: int = 10,
        play_as_both_colors: bool = True,
        max_concurrent_games: int = 1
    ) -> SingleOpponentConfig:
        """Create single opponent configuration."""
        return SingleOpponentConfig(
            opponent_name=opponent_name,
            num_games=num_games,
            play_as_both_colors=play_as_both_colors,
            max_concurrent_games=max_concurrent_games
        )
    
    @staticmethod
    def create_tournament_config(
        opponent_pool: Optional[List[str]] = None,
        num_games_per_opponent: int = 5,
        max_concurrent_games: int = 1
    ) -> TournamentConfig:
        """Create tournament configuration."""
        if opponent_pool is None:
            opponent_pool = ["opponent1", "opponent2", "opponent3"]
        
        return TournamentConfig(
            opponent_pool=opponent_pool,
            num_games_per_opponent=num_games_per_opponent,
            max_concurrent_games=max_concurrent_games
        )
    
    @staticmethod
    def create_ladder_config(
        initial_opponents: Optional[List[str]] = None,
        games_per_evaluation: int = 10,
        elo_k_factor: float = 32.0
    ) -> LadderConfig:
        """Create ladder configuration."""
        if initial_opponents is None:
            initial_opponents = ["ladder_opponent1", "ladder_opponent2"]
        
        return LadderConfig(
            initial_opponents=initial_opponents,
            games_per_evaluation=games_per_evaluation,
            elo_k_factor=elo_k_factor
        )
    
    @staticmethod
    def create_benchmark_config(
        benchmark_opponents: Optional[List[str]] = None,
        games_per_benchmark: int = 20
    ) -> BenchmarkConfig:
        """Create benchmark configuration."""
        if benchmark_opponents is None:
            benchmark_opponents = ["benchmark1", "benchmark2"]
        
        return BenchmarkConfig(
            benchmark_opponents=benchmark_opponents,
            games_per_benchmark=games_per_benchmark
        )


class MockFactory:
    """Factory for creating properly configured mocks."""
    
    @staticmethod
    def create_mock_evaluator(
        evaluation_result: Optional[EvaluationResult] = None
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
                'layer1.weight': torch.randn(10, 5),
                'layer1.bias': torch.randn(10),
                'layer2.weight': torch.randn(1, 10),
                'layer2.bias': torch.randn(1)
            }
            mock_agent.model = mock_model
        
        return mock_agent


class TestDataGenerator:
    """Generator for creating realistic test data patterns."""
    
    @staticmethod
    def generate_performance_trend(
        num_sessions: int = 10,
        trend_type: str = "improving",
        noise_level: float = 0.1
    ) -> List[tuple]:
        """Generate evaluation sessions with performance trends."""
        sessions = []
        base_time = datetime.now() - timedelta(days=num_sessions)
        
        for i in range(num_sessions):
            if trend_type == "improving":
                base_win_rate = 0.3 + (i / num_sessions) * 0.4  # 0.3 to 0.7
            elif trend_type == "declining":
                base_win_rate = 0.7 - (i / num_sessions) * 0.4  # 0.7 to 0.3
            elif trend_type == "stable":
                base_win_rate = 0.5
            else:
                base_win_rate = 0.5
            
            # Add noise
            win_rate = max(0.0, min(1.0, base_win_rate + random.uniform(-noise_level, noise_level)))
            
            # Generate games for this session
            games = EvaluationTestFactory.create_game_series(
                num_games=10,
                win_rate=win_rate,
                base_timestamp=base_time + timedelta(days=i)
            )
            
            result = EvaluationTestFactory.create_evaluation_result(games=games)
            sessions.append((base_time + timedelta(days=i), result))
        
        return sessions
    
    @staticmethod
    def generate_realistic_game_lengths(
        num_games: int = 100,
        game_type: str = "normal"
    ) -> List[int]:
        """Generate realistic game length distributions."""
        if game_type == "normal":
            # Normal distribution around 50 moves
            return [max(5, int(random.gauss(50, 15))) for _ in range(num_games)]
        elif game_type == "quick":
            # Shorter games
            return [max(5, int(random.gauss(25, 8))) for _ in range(num_games)]
        elif game_type == "long":
            # Longer games
            return [max(10, int(random.gauss(100, 30))) for _ in range(num_games)]
        else:
            return [50] * num_games
    
    @staticmethod
    def generate_elo_progression(
        num_games: int = 100,
        starting_elo: float = 1500.0,
        skill_change_rate: float = 0.1
    ) -> List[float]:
        """Generate realistic ELO progression over games."""
        elo_changes = []
        current_skill = 0.0  # Skill relative to opponents
        
        for i in range(num_games):
            # Gradually improve skill
            current_skill += skill_change_rate * random.uniform(-0.1, 0.2)
            
            # Win probability based on skill difference
            win_prob = 1 / (1 + 10 ** (-current_skill / 400))
            
            # Determine game outcome
            if random.random() < win_prob:
                # Win: positive ELO change
                elo_change = random.uniform(5.0, 20.0)
            else:
                # Loss: negative ELO change
                elo_change = random.uniform(-20.0, -5.0)
            
            elo_changes.append(elo_change)
        
        return elo_changes


class PerformanceTestUtils:
    """Utilities for performance testing."""
    
    @staticmethod
    def measure_memory_usage(func, *args, **kwargs):
        """Measure memory usage of a function call."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        result = func(*args, **kwargs)
        
        final_memory = process.memory_info().rss
        memory_used = final_memory - initial_memory
        
        return result, memory_used
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs):
        """Measure execution time of a function call."""
        import time
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        
        return result, execution_time
    
    @staticmethod
    def create_temporary_checkpoint(agent, checkpoint_dir: Optional[Path] = None):
        """Create a temporary checkpoint file for testing."""
        if checkpoint_dir is None:
            checkpoint_dir = Path(tempfile.mkdtemp())
        
        checkpoint_path = checkpoint_dir / "test_checkpoint.pth"
        
        # Save agent state
        torch.save({
            'model_state_dict': agent.model.state_dict(),
            'agent_type': type(agent).__name__,
            'config': getattr(agent, 'config', None)
        }, checkpoint_path)
        
        return checkpoint_path


class ValidationUtils:
    """Utilities for validating test results."""
    
    @staticmethod
    def validate_evaluation_result(result: EvaluationResult) -> bool:
        """Validate that an evaluation result is well-formed."""
        try:
            # Check basic structure
            assert result.games is not None
            assert result.summary_stats is not None
            assert result.analytics is not None
            
            # Check summary stats consistency
            stats = result.summary_stats
            assert stats.total_games == len(result.games)
            assert stats.wins + stats.losses + stats.draws == stats.total_games
            
            if stats.total_games > 0:
                expected_win_rate = stats.wins / stats.total_games
                assert abs(stats.win_rate - expected_win_rate) < 0.001
            
            # Check game results
            for game in result.games:
                assert game.winner in ["agent", "opponent", "draw"]
                assert game.game_length > 0
                assert isinstance(game.elo_change, (int, float))
            
            return True
            
        except AssertionError:
            return False
    
    @staticmethod
    def validate_performance_metrics(
        execution_time: float,
        memory_usage: int,
        max_time: float = 10.0,
        max_memory: int = 100_000_000  # 100MB
    ) -> bool:
        """Validate performance metrics against thresholds."""
        return execution_time < max_time and memory_usage < max_memory
    
    @staticmethod
    def compare_evaluation_results(
        result1: EvaluationResult,
        result2: EvaluationResult,
        tolerance: float = 0.01
    ) -> bool:
        """Compare two evaluation results for similarity."""
        stats1, stats2 = result1.summary_stats, result2.summary_stats
        
        # Compare win rates
        if abs(stats1.win_rate - stats2.win_rate) > tolerance:
            return False
        
        # Compare game counts
        if stats1.total_games != stats2.total_games:
            return False
        
        # Compare average game lengths
        if abs(stats1.avg_game_length - stats2.avg_game_length) > tolerance * 100:
            return False
        
        return True


# Convenience functions for common test patterns
def create_test_evaluation_manager(config_type: str = "single_opponent", **config_kwargs):
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
    num_games: int = 10,
    win_rate: float = 0.6,
    config_type: str = "single_opponent"
):
    """Create a complete test scenario with manager, agent, and expected results."""
    manager = create_test_evaluation_manager(config_type)
    agent = EvaluationTestFactory.create_test_agent()
    games = EvaluationTestFactory.create_game_series(num_games=num_games, win_rate=win_rate)
    expected_result = EvaluationTestFactory.create_evaluation_result(games=games)
    
    return {
        'manager': manager,
        'agent': agent,
        'expected_games': games,
        'expected_result': expected_result
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
