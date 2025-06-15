"""Shared fixtures and utilities for analytics tests."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from keisei.evaluation.analytics.advanced_analytics import (
    AdvancedAnalytics,
    PerformanceComparison,
    StatisticalTest,
    TrendAnalysis,
)
from keisei.evaluation.core import EvaluationResult, GameResult
from keisei.evaluation.core.evaluation_result import SummaryStats


@pytest.fixture
def analytics():
    """Create an AdvancedAnalytics instance for testing."""
    return AdvancedAnalytics(
        significance_level=0.05,
        min_practical_difference=0.05,
        trend_window_days=30,
    )


@pytest.fixture
def sample_game_results():
    """Create sample game results for testing."""
    results = []

    # Create variety of game results with different outcomes
    for i in range(10):
        game = Mock(spec=GameResult)
        game.is_agent_win = i < 6  # 60% win rate
        game.agent_player = "agent"
        game.winner = "agent" if game.is_agent_win else "opponent"
        game.result = "win" if game.is_agent_win else "loss"
        game.moves_count = 50 + (i * 5)  # Varying game lengths
        game.duration_seconds = 30.0 + (i * 2.0)  # CHANGED
        game.timestamp = datetime.now() - timedelta(hours=i)
        results.append(game)

    return results


@pytest.fixture
def sample_evaluation_result(sample_game_results):
    """Create a sample EvaluationResult for testing."""
    context = Mock()
    context.session_id = "test_session"
    context.timestamp = datetime.now()

    # Create summary stats
    total_games = len(sample_game_results)
    agent_wins = sum(1 for game in sample_game_results if game.is_agent_win)
    opponent_wins = total_games - agent_wins

    summary_stats = SummaryStats(
        total_games=total_games,
        agent_wins=agent_wins,
        opponent_wins=opponent_wins,
        draws=0,
        win_rate=agent_wins / total_games,
        loss_rate=opponent_wins / total_games,
        draw_rate=0.0,
        avg_game_length=sum(game.moves_count for game in sample_game_results)
        / total_games,
        total_moves=sum(game.moves_count for game in sample_game_results),
        avg_duration_seconds=sum(
            game.duration_seconds for game in sample_game_results  # CHANGED
        )
        / total_games,
    )

    return EvaluationResult(
        context=context,
        games=sample_game_results,
        summary_stats=summary_stats,
        analytics_data={},
    )


@pytest.fixture
def baseline_game_results():
    """Create baseline game results with lower performance."""
    results = []

    # Create baseline with 40% win rate
    for i in range(10):
        game = Mock(spec=GameResult)
        game.is_agent_win = i < 4  # 40% win rate
        game.agent_player = "agent"
        game.winner = "agent" if game.is_agent_win else "opponent"
        game.result = "win" if game.is_agent_win else "loss"
        game.moves_count = 60 + (i * 3)  # Different game lengths
        game.duration_seconds = 35.0 + (i * 1.5)  # CHANGED
        game.timestamp = datetime.now() - timedelta(hours=i + 20)
        results.append(game)

    return results


@pytest.fixture
def baseline_evaluation_result(baseline_game_results):
    """Create a baseline EvaluationResult for comparison testing."""
    context = Mock()
    context.session_id = "baseline_session"
    context.timestamp = datetime.now() - timedelta(days=1)

    # Create summary stats for baseline
    total_games = len(baseline_game_results)
    agent_wins = sum(1 for game in baseline_game_results if game.is_agent_win)
    opponent_wins = total_games - agent_wins

    summary_stats = SummaryStats(
        total_games=total_games,
        agent_wins=agent_wins,
        opponent_wins=opponent_wins,
        draws=0,
        win_rate=agent_wins / total_games,
        loss_rate=opponent_wins / total_games,
        draw_rate=0.0,
        avg_game_length=sum(game.moves_count for game in baseline_game_results)
        / total_games,
        total_moves=sum(game.moves_count for game in baseline_game_results),
        avg_duration_seconds=sum(
            game.duration_seconds for game in baseline_game_results  # CHANGED
        )
        / total_games,
    )

    return EvaluationResult(
        context=context,
        games=baseline_game_results,
        summary_stats=summary_stats,
        analytics_data={},
    )


@pytest.fixture
def historical_evaluation_data():
    """Create historical evaluation data for trend analysis."""
    historical_data = []
    base_time = datetime.now() - timedelta(days=60)

    for i in range(15):
        timestamp = base_time + timedelta(days=i * 4)

        # Create a copy of the evaluation result with modified stats
        context = Mock()
        context.session_id = f"session_{i}"
        context.timestamp = timestamp

        # Simulate improving performance over time
        win_rate = 0.3 + (i * 0.03)  # Gradual improvement from 30% to 72%
        total_games = 20
        agent_wins = int(total_games * win_rate)
        opponent_wins = total_games - agent_wins

        summary_stats = SummaryStats(
            total_games=total_games,
            agent_wins=agent_wins,
            opponent_wins=opponent_wins,
            draws=0,
            win_rate=win_rate,
            loss_rate=opponent_wins / total_games,
            draw_rate=0.0,
            avg_game_length=50 + (i * 2),  # Slightly increasing game length
            total_moves=total_games * (50 + (i * 2)),
            avg_duration_seconds=(600 + (i * 30)) / total_games,
        )

        # Create mock games for this result
        games = []
        for j in range(total_games):
            game = Mock(spec=GameResult)
            game.is_agent_win = j < agent_wins
            game.moves_count = summary_stats.avg_game_length + (j % 10)
            game.duration_seconds = 30.0 + (j % 15)  # CHANGED
            games.append(game)

        result = EvaluationResult(
            context=context,
            games=games,
            summary_stats=summary_stats,
            analytics_data={},
        )

        historical_data.append((timestamp, result))

    return historical_data


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_mock_evaluation_result(
    win_rate: float = 0.6,
    total_games: int = 10,
    avg_game_length: int = 50,
    session_id: str = "test_session",
    timestamp: datetime = None,
) -> EvaluationResult:
    """Helper function to create mock evaluation results with specified parameters."""
    if timestamp is None:
        timestamp = datetime.now()

    context = Mock()
    context.session_id = session_id
    context.timestamp = timestamp

    agent_wins = int(total_games * win_rate)
    opponent_wins = total_games - agent_wins

    # Create mock games
    games = []
    for i in range(total_games):
        game = Mock(spec=GameResult)
        game.is_agent_win = i < agent_wins
        game.moves_count = avg_game_length + (i % 10)
        game.duration_seconds = 30.0 + (i % 15)  # CHANGED
        game.timestamp = timestamp - timedelta(minutes=i)
        games.append(game)

    summary_stats = SummaryStats(
        total_games=total_games,
        agent_wins=agent_wins,
        opponent_wins=opponent_wins,
        draws=0,
        win_rate=win_rate,
        loss_rate=opponent_wins / total_games,
        draw_rate=0.0,
        avg_game_length=avg_game_length,
        total_moves=total_games * avg_game_length,
        avg_duration_seconds=30.0,
    )

    return EvaluationResult(
        context=context,
        games=games,
        summary_stats=summary_stats,
        analytics_data={},
    )
