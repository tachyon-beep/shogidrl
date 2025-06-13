"""
Integration tests for advanced analytics pipeline.

This module provides comprehensive testing of the analytics integration
with the evaluation system, focusing on real-world scenarios and data flow.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from keisei.evaluation.analytics.advanced_analytics import AdvancedAnalytics
from keisei.evaluation.core import EvaluationResult, GameResult, SummaryStats
from keisei.evaluation.core.evaluation_context import AgentInfo, OpponentInfo
from keisei.evaluation.core_manager import EvaluationManager
from keisei.evaluation.enhanced_manager import EnhancedEvaluationManager
from keisei.evaluation.strategies.single_opponent import SingleOpponentConfig


class TestAnalyticsIntegration:
    """Test analytics integration with evaluation system."""

    @pytest.fixture
    def evaluation_results_series(self):
        """Create a series of evaluation results for trend analysis."""
        results = []
        base_time = datetime.now() - timedelta(days=30)

        for i in range(10):
            # Create realistic game results with varying performance
            games = []
            wins = 3 + (i % 3)  # Vary wins: 3, 4, 5, 3, 4, 5...
            losses = 6 - wins

            # Create winning games
            for j in range(wins):
                games.append(GameResult(
                    game_id=f"test_{i}_{j}",
                    winner=0,  # 0=agent win
                    moves_count=50 + j * 10,
                    duration_seconds=120.0,
                    agent_info=AgentInfo(name="test_agent", version="1.0", metadata={}),
                    opponent_info=OpponentInfo(name="test_opponent", type="random", metadata={}),
                    metadata={"timestamp": base_time + timedelta(days=i, minutes=j*5)}
                ))

            # Create losing games
            for j in range(losses):
                games.append(GameResult(
                    game_id=f"test_{i}_{wins + j}",
                    winner=1,  # 1=opponent win
                    moves_count=45 + j * 8,
                    duration_seconds=100.0,
                    agent_info=AgentInfo(name="test_agent", version="1.0", metadata={}),
                    opponent_info=OpponentInfo(name="test_opponent", type="random", metadata={}),
                    metadata={"timestamp": base_time + timedelta(days=i, minutes=(wins+j)*5)}
                ))

            # Create summary stats
            total_games = wins + losses
            summary = SummaryStats(
                total_games=total_games,
                agent_wins=wins,
                opponent_wins=losses,
                draws=0,
                win_rate=wins / total_games,
                loss_rate=losses / total_games,
                draw_rate=0.0,
                avg_game_length=50.0,
                total_moves=total_games * 50,
                avg_duration_seconds=110.0
            )

            # Create evaluation result
            result = EvaluationResult(
                context=None, 
                games=games, 
                summary_stats=summary, 
                analytics_data={}
            )
            results.append((base_time + timedelta(days=i), result))

        return results

    def test_enhanced_manager_analytics_integration(self, evaluation_results_series):
        """Test enhanced manager with analytics integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = EnhancedEvaluationManager(
                config=SingleOpponentConfig(num_games=4),
                run_name="analytics_test",
                enable_advanced_analytics=True,
                analytics_output_dir=Path(temp_dir)
            )

            # Verify analytics are available
            assert hasattr(manager, 'advanced_analytics')
            assert manager.advanced_analytics is not None

            # Test analytics report generation using enhanced manager's method
            current_result = evaluation_results_series[-1][1]  # Latest result
            baseline_result = evaluation_results_series[0][1]  # First result as baseline
            
            comparison = manager.compare_performance(
                baseline_results=baseline_result,
                comparison_results=current_result,
                baseline_name="Historical",
                comparison_name="Current"
            )

            assert comparison is not None
            assert hasattr(comparison, 'win_rate_difference')
            assert hasattr(comparison, 'baseline_name')
            assert comparison.baseline_name == "Historical"
            assert comparison.comparison_name == "Current"

    def test_analytics_trend_detection(self, evaluation_results_series):
        """Test trend detection accuracy in analytics."""
        analytics = AdvancedAnalytics()

        # Create trend data with consistent improvement
        trend_data = []
        base_time = datetime.now() - timedelta(days=20)
        
        for i in range(5):
            wins = 3 + i  # Increasing wins: 3, 4, 5, 6, 7
            losses = 7 - wins  # Decreasing losses: 4, 3, 2, 1, 0
            games = []
            
            # Create game results
            for j in range(wins):
                games.append(GameResult(
                    game_id=f"trend_{i}_{j}",
                    winner=0,
                    moves_count=50,
                    duration_seconds=120.0,
                    agent_info=AgentInfo(name="test_agent", version="1.0", metadata={}),
                    opponent_info=OpponentInfo(name="test_opponent", type="random", metadata={})
                ))
            
            for j in range(losses):
                games.append(GameResult(
                    game_id=f"trend_{i}_{wins + j}",
                    winner=1,
                    moves_count=50,
                    duration_seconds=120.0,
                    agent_info=AgentInfo(name="test_agent", version="1.0", metadata={}),
                    opponent_info=OpponentInfo(name="test_opponent", type="random", metadata={})
                ))

            summary = SummaryStats(
                total_games=wins + losses,
                agent_wins=wins,
                opponent_wins=losses,
                draws=0,
                win_rate=wins / (wins + losses),
                loss_rate=losses / (wins + losses),
                draw_rate=0.0,
                avg_game_length=50.0,
                total_moves=(wins + losses) * 50,
                avg_duration_seconds=120.0
            )
            
            result = EvaluationResult(
                context=None, 
                games=games, 
                summary_stats=summary, 
                analytics_data={}
            )
            trend_data.append((base_time + timedelta(days=i*3), result))

        trend = analytics.analyze_trends(trend_data, "win_rate")

        assert trend.metric_name == "win_rate"
        assert trend.trend_direction == "increasing"  # Should detect improvement
        assert trend.data_points == len(trend_data)

    def test_analytics_performance_comparison_accuracy(self, evaluation_results_series):
        """Test analytics performance comparison accuracy."""
        analytics = AdvancedAnalytics()

        # Split into baseline and comparison, extracting games from evaluation results
        baseline_data = evaluation_results_series[:5]
        comparison_data = evaluation_results_series[5:]
        
        # Extract games from the evaluation results
        baseline_games = []
        for timestamp, eval_result in baseline_data:
            baseline_games.extend(eval_result.games)
            
        comparison_games = []
        for timestamp, eval_result in comparison_data:
            comparison_games.extend(eval_result.games)

        comparison = analytics.compare_performance(
            baseline_results=baseline_games,
            comparison_results=comparison_games,
            comparison_name="Recent Performance"
        )

        assert comparison.comparison_name == "Recent Performance"
        assert hasattr(comparison, 'win_rate_difference')
        assert hasattr(comparison, 'statistical_tests')
        assert len(baseline_games) > 0
        assert len(comparison_games) > 0

    def test_analytics_report_persistence(self, evaluation_results_series):
        """Test analytics report saving and loading."""
        analytics = AdvancedAnalytics()

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "analytics_report.json"

            # Generate automated report
            report = analytics.generate_automated_report(
                current_results=evaluation_results_series[-1][1],
                historical_data=evaluation_results_series[:-1],
                output_file=report_path
            )

            # Verify report was saved
            assert report_path.exists()

            # Load and verify content
            with open(report_path, 'r') as f:
                saved_report = json.load(f)

            assert "report_metadata" in saved_report
            assert "current_performance" in saved_report
            assert "advanced_metrics" in saved_report

    def test_analytics_memory_efficiency(self):
        """Test that analytics processing is memory efficient."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        analytics = AdvancedAnalytics()

        # Create large dataset for analytics processing
        large_dataset = []
        base_time = datetime.now() - timedelta(days=100)

        for i in range(50):  # 50 evaluation sessions
            games = []
            for j in range(20):  # 20 games each
                games.append(GameResult(
                    game_id=f"memory_test_{i}_{j}",
                    winner=0 if j % 2 == 0 else 1,
                    moves_count=50 + j,
                    duration_seconds=120.0,
                    agent_info=AgentInfo(name="test_agent", version="1.0", metadata={}),
                    opponent_info=OpponentInfo(name="test_opponent", type="random", metadata={})
                ))

            summary = SummaryStats(
                total_games=20,
                agent_wins=10,
                opponent_wins=10,
                draws=0,
                win_rate=0.5,
                loss_rate=0.5,
                draw_rate=0.0,
                avg_game_length=59.5,
                total_moves=1190,
                avg_duration_seconds=134.5
            )
            
            result = EvaluationResult(
                context=None, 
                games=games, 
                summary_stats=summary, 
                analytics_data={}
            )
            large_dataset.append((base_time + timedelta(days=i), result))

        # Process large dataset
        trend = analytics.analyze_trends(large_dataset, "win_rate")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Should not increase memory by more than 100MB
        assert memory_increase < 100 * 1024 * 1024

    def test_analytics_error_handling(self):
        """Test analytics graceful error handling."""
        analytics = AdvancedAnalytics()

        # Test with empty data
        empty_trend = analytics.analyze_trends([], "win_rate")
        assert empty_trend.data_points == 0
        assert empty_trend.trend_direction == "insufficient_data"

        # Test with invalid metric
        single_result = EvaluationResult(
            context=None,
            games=[GameResult(
                game_id="test",
                winner=0,
                moves_count=50,
                duration_seconds=120.0,
                agent_info=AgentInfo(name="test_agent", version="1.0", metadata={}),
                opponent_info=OpponentInfo(name="test_opponent", type="random", metadata={})
            )],
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
        data = [(datetime.now(), single_result)]

        # Should handle gracefully
        trend = analytics.analyze_trends(data, "invalid_metric")
        assert trend.trend_direction in ["stable", "insufficient_data"]

    def test_analytics_performance_benchmarks(self, evaluation_results_series):
        """Test analytics performance benchmarks."""
        import time
        
        analytics = AdvancedAnalytics()

        # Time trend analysis
        start_time = time.time()
        trend = analytics.analyze_trends(evaluation_results_series, "win_rate")
        trend_time = time.time() - start_time

        # Should complete quickly
        assert trend_time < 1.0  # Less than 1 second

        # Time comparison analysis - extract games from evaluation results
        baseline_data = evaluation_results_series[:5]
        comparison_data = evaluation_results_series[5:]
        
        baseline_games = []
        for timestamp, eval_result in baseline_data:
            baseline_games.extend(eval_result.games)
            
        comparison_games = []
        for timestamp, eval_result in comparison_data:
            comparison_games.extend(eval_result.games)

        start_time = time.time()
        comparison = analytics.compare_performance(
            baseline_results=baseline_games,
            comparison_results=comparison_games,
            comparison_name="Performance Test"
        )
        comparison_time = time.time() - start_time

        # Should complete quickly
        assert comparison_time < 1.0  # Less than 1 second
