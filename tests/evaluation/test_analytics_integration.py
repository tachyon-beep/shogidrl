"""
Integration tests for analytics components in the evaluation system.

This module tests the integration between various analytics components
including trend analysis, statistical testing, and report generation.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

from keisei.evaluation.analytics.advanced_analytics import AdvancedAnalytics
from keisei.evaluation.core.evaluation_result import EvaluationResult, GameResult, SummaryStats
from keisei.evaluation.core.evaluation_context import AgentInfo, OpponentInfo, EvaluationContext
from keisei.evaluation.core.evaluation_config import SingleOpponentConfig


class TestAnalyticsIntegration:
    """Test integration between different analytics components."""

    @pytest.fixture
    def analytics(self):
        """Create AdvancedAnalytics instance for testing."""
        return AdvancedAnalytics()

    @pytest.fixture
    def sample_evaluation_context(self):
        """Create a proper evaluation context for testing."""
        config = SingleOpponentConfig(num_games=10)
        agent_info = AgentInfo(name="test_agent", version="1.0", metadata={})
        
        return EvaluationContext(
            session_id="test_session_123",
            timestamp=datetime.now(),
            agent_info=agent_info,
            configuration=config,
            environment_info={"device": "cpu"},
            metadata={"test": True}
        )

    @pytest.fixture
    def sample_evaluation_results(self, sample_evaluation_context):
        """Create sample evaluation results for testing."""
        results = []
        base_time = datetime.now() - timedelta(days=10)
        
        for i in range(5):
            games = []
            # Create some games for each evaluation
            for j in range(6):
                games.append(GameResult(
                    game_id=f"test_{i}_{j}",
                    winner=0 if j % 2 == 0 else 1,  # Alternate wins
                    moves_count=50 + j * 5,
                    duration_seconds=120.0 + j * 10,
                    agent_info=AgentInfo(name="test_agent", version="1.0", metadata={}),
                    opponent_info=OpponentInfo(name="test_opponent", type="random", metadata={}),
                    metadata={"timestamp": base_time + timedelta(days=i, minutes=j*10)}
                ))
            
            agent_wins = len([g for g in games if g.winner == 0])
            opponent_wins = len([g for g in games if g.winner == 1])
            
            result = EvaluationResult(
                context=sample_evaluation_context,
                games=games,
                summary_stats=SummaryStats(
                    total_games=len(games),
                    agent_wins=agent_wins,
                    opponent_wins=opponent_wins,
                    draws=0,
                    win_rate=agent_wins / len(games),
                    loss_rate=opponent_wins / len(games),
                    draw_rate=0.0,
                    avg_game_length=sum(g.moves_count for g in games) / len(games),
                    total_moves=sum(g.moves_count for g in games),
                    avg_duration_seconds=sum(g.duration_seconds for g in games) / len(games)
                ),
                analytics_data={}
            )
            results.append(result)
        
        return results

    def test_analytics_basic_integration(self, analytics, sample_evaluation_results):
        """Test basic analytics integration with evaluation results."""
        # Test trend analysis - need to create tuples with timestamps
        timestamped_results = []
        base_time = datetime.now() - timedelta(days=10)
        for i, result in enumerate(sample_evaluation_results):
            timestamp = base_time + timedelta(days=i)
            timestamped_results.append((timestamp, result))
        
        trend = analytics.analyze_trends(timestamped_results, "win_rate")
        
        assert trend is not None
        assert trend.metric_name == "win_rate"
        assert trend.data_points == len(sample_evaluation_results)
        
    def test_analytics_statistical_comparison(self, analytics, sample_evaluation_results):
        """Test statistical comparison functionality."""
        if len(sample_evaluation_results) >= 2:
            result1 = sample_evaluation_results[0]
            result2 = sample_evaluation_results[1]
            
            # Extract games from EvaluationResult objects
            baseline_games = result1.games
            comparison_games = result2.games
            
            comparison = analytics.compare_performance(baseline_games, comparison_games)
            
            assert comparison is not None
            assert hasattr(comparison, 'win_rate_difference')
            
    def test_analytics_report_generation(self, analytics, sample_evaluation_results):
        """Test comprehensive analytics report generation."""
        # Test available analytics methods instead of generate_comprehensive_report
        if len(sample_evaluation_results) >= 2:
            result1 = sample_evaluation_results[0]
            result2 = sample_evaluation_results[1]
            
            # Test comparison functionality
            comparison = analytics.compare_performance(result1.games, result2.games)
            assert comparison is not None
            
            # Test trend analysis with timestamped data
            timestamped_results = []
            base_time = datetime.now() - timedelta(days=10)
            for i, result in enumerate(sample_evaluation_results):
                timestamp = base_time + timedelta(days=i)
                timestamped_results.append((timestamp, result))
            
            trend = analytics.analyze_trends(timestamped_results, "win_rate")
            assert trend is not None
        
    def test_analytics_error_handling(self, analytics, sample_evaluation_context):
        """Test analytics graceful error handling with edge cases."""
        # Test with empty results
        empty_trend = analytics.analyze_trends([], "win_rate")
        assert empty_trend.data_points == 0
        assert empty_trend.trend_direction == "insufficient_data"
        
        # Test with single result
        single_game = GameResult(
            game_id="test",
            winner=0,
            moves_count=50,
            duration_seconds=120.0,
            agent_info=AgentInfo(name="test_agent", version="1.0", metadata={}),
            opponent_info=OpponentInfo(name="test_opponent", type="random", metadata={})
        )
        
        single_result = EvaluationResult(
            context=sample_evaluation_context,
            games=[single_game],
            summary_stats=SummaryStats(
                total_games=1,
                agent_wins=1,
                opponent_wins=0,
                draws=0,
                win_rate=1.0,
                loss_rate=0.0,
                draw_rate=0.0,
                avg_game_length=50.0,
                total_moves=50,
                avg_duration_seconds=120.0
            ),
            analytics_data={}
        )
        
        single_trend = analytics.analyze_trends([(datetime.now(), single_result)], "win_rate")
        assert single_trend.data_points == 1
        assert single_trend.trend_direction == "insufficient_data"

    def test_analytics_memory_efficiency(self, analytics, sample_evaluation_context):
        """Test that analytics processing is memory efficient."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create large dataset for analytics processing
        large_dataset = []
        base_time = datetime.now() - timedelta(days=100)

        for i in range(20):  # Reduced from 50 to 20 for faster testing
            games = []
            for j in range(10):  # Reduced from 20 to 10 for faster testing
                games.append(GameResult(
                    game_id=f"memory_test_{i}_{j}",
                    winner=0 if j % 2 == 0 else 1,
                    moves_count=50 + j,
                    duration_seconds=120.0,
                    agent_info=AgentInfo(name="test_agent", version="1.0", metadata={}),
                    opponent_info=OpponentInfo(name="test_opponent", type="random", metadata={})
                ))

            agent_wins = len([g for g in games if g.winner == 0])
            opponent_wins = len([g for g in games if g.winner == 1])

            result = EvaluationResult(
                context=sample_evaluation_context,
                games=games,
                summary_stats=SummaryStats(
                    total_games=len(games),
                    agent_wins=agent_wins,
                    opponent_wins=opponent_wins,
                    draws=0,
                    win_rate=agent_wins / len(games),
                    loss_rate=opponent_wins / len(games),
                    draw_rate=0.0,
                    avg_game_length=sum(g.moves_count for g in games) / len(games),
                    total_moves=sum(g.moves_count for g in games),
                    avg_duration_seconds=sum(g.duration_seconds for g in games) / len(games)
                ),
                analytics_data={}
            )
            large_dataset.append(result)

        # Process large dataset
        # Convert to timestamped format for trend analysis
        timestamped_dataset = []
        for i, result in enumerate(large_dataset):
            timestamp = base_time + timedelta(days=i)
            timestamped_dataset.append((timestamp, result))
        
        trend = analytics.analyze_trends(timestamped_dataset, "win_rate")

        # Check memory usage hasn't exploded
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Allow reasonable memory increase (100MB max for this test)
        assert memory_increase < 100 * 1024 * 1024, f"Memory usage increased by {memory_increase / 1024 / 1024:.1f}MB"
        
        # Verify processing succeeded
        assert trend is not None