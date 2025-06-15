"""
Integration tests for AdvancedAnalytics components.

This module tests end-to-end workflows combining multiple analytics features
including trend analysis, performance comparison, and automated reporting.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from keisei.evaluation.analytics.advanced_analytics import AdvancedAnalytics
from keisei.evaluation.core import EvaluationResult, GameResult
from keisei.evaluation.core.evaluation_result import SummaryStats


class TestAdvancedAnalyticsIntegration:
    """Integration tests combining multiple features."""

    def test_complete_analysis_workflow(
        self,
        analytics,
        sample_evaluation_result,
        baseline_game_results,
        historical_evaluation_data,
    ):
        """Test complete analysis workflow with all features."""
        # Create baseline result
        baseline_context = Mock()
        baseline_stats = SummaryStats(
            total_games=len(baseline_game_results),
            agent_wins=sum(1 for g in baseline_game_results if g.is_agent_win),
            opponent_wins=len(baseline_game_results)
            - sum(1 for g in baseline_game_results if g.is_agent_win),
            draws=0,
            win_rate=sum(1 for g in baseline_game_results if g.is_agent_win)
            / len(baseline_game_results),
            loss_rate=(
                len(baseline_game_results)
                - sum(1 for g in baseline_game_results if g.is_agent_win)
            )
            / len(baseline_game_results),
            draw_rate=0.0,
            avg_game_length=sum(g.moves_count for g in baseline_game_results)
            / len(baseline_game_results),
            total_moves=sum(g.moves_count for g in baseline_game_results),
            avg_duration_seconds=sum(
                g.duration_seconds for g in baseline_game_results  # CHANGED
            )
            / len(baseline_game_results),
        )

        baseline_result = EvaluationResult(
            context=baseline_context,
            games=baseline_game_results,
            summary_stats=baseline_stats,
            analytics_data={},
        )

        # Generate complete report
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "complete_analysis.json"

            report = analytics.generate_automated_report(
                current_results=sample_evaluation_result,
                baseline_results=baseline_result,
                historical_data=historical_evaluation_data,
                output_file=output_file,
            )

            # Verify all sections present
            assert "report_metadata" in report
            assert "current_performance" in report
            assert "advanced_metrics" in report
            assert "performance_comparison" in report
            assert "trend_analysis" in report
            assert "insights_and_recommendations" in report

            # Verify file output
            assert output_file.exists()

            # Verify report structure
            assert (
                report["performance_comparison"]["win_rate_difference"] > 0
            )  # Should show improvement
            assert len(report["trend_analysis"]) >= 2  # Win rate and game length trends
            assert len(report["insights_and_recommendations"]) > 0

    def test_error_handling_in_workflow(self, analytics):
        """Test error handling throughout the analysis workflow."""
        # Create minimal/invalid data to test error handling
        context = Mock()
        empty_stats = SummaryStats(
            total_games=0,
            agent_wins=0,
            opponent_wins=0,
            draws=0,
            win_rate=0,
            loss_rate=0,
            draw_rate=0,
            avg_game_length=0,
            total_moves=0,
            avg_duration_seconds=0,
        )

        empty_result = EvaluationResult(
            context=context,
            games=[],
            summary_stats=empty_stats,
            analytics_data={},
        )

        # Should handle empty results gracefully
        report = analytics.generate_automated_report(current_results=empty_result)

        assert isinstance(report, dict)
        assert "current_performance" in report
        assert report["current_performance"]["total_games"] == 0

    def test_performance_metrics_validation(self, analytics, sample_evaluation_result):
        """Test validation of performance metrics in reports."""
        report = analytics.generate_automated_report(
            current_results=sample_evaluation_result
        )

        # Validate performance metrics
        perf = report["current_performance"]
        assert 0 <= perf["win_rate"] <= 1
        assert perf["total_games"] == perf["wins"] + perf["losses"] + perf["draws"]
        assert perf["avg_game_length"] > 0
        assert perf["total_games"] > 0

    def test_trend_analysis_edge_cases(self, analytics):
        """Test trend analysis with edge case data."""
        # Create data with perfect correlation
        historical_data = []
        for i in range(5):
            timestamp = datetime.now() - timedelta(days=i * 10)

            context = Mock()

            # Create proper mock games with required attributes
            games = []
            for j in range(10):
                game = Mock(spec=GameResult)
                game.is_agent_win = j < (i * 2)
                game.is_opponent_win = j >= (i * 2)
                game.is_draw = False
                game.moves_count = 50 + (j % 10)
                game.duration_seconds = 30.0 + (j % 15)
                games.append(game)

            # Let SummaryStats calculate from games
            summary_stats = SummaryStats.from_games(games)

            result = EvaluationResult(
                context=context,
                games=games,
                summary_stats=summary_stats,
                analytics_data={},
            )

            historical_data.append((timestamp, result))

        trend = analytics.analyze_trends(historical_data, "win_rate")

        # Should detect strong correlation
        assert trend.trend_direction in ["increasing", "decreasing"]
        assert trend.r_squared > 0.5  # Strong correlation

    def test_comparison_statistical_significance(self, analytics):
        """Test statistical significance detection in performance comparison."""
        # Create clearly different result sets
        baseline_results = []
        comparison_results = []

        # Baseline: 20% win rate
        for i in range(100):
            game = Mock(spec=GameResult)
            game.is_agent_win = i < 20
            game.moves_count = 50
            game.duration_seconds = 30.0  # CHANGED
            baseline_results.append(game)

        # Comparison: 80% win rate
        for i in range(100):
            game = Mock(spec=GameResult)
            game.is_agent_win = i < 80
            game.moves_count = 50
            game.duration_seconds = 30.0  # CHANGED
            comparison_results.append(game)

        comparison = analytics.compare_performance(
            baseline_results=baseline_results,
            comparison_results=comparison_results,
        )

        # Should detect significant difference
        assert comparison.practical_significance == True
        assert abs(comparison.win_rate_difference - 0.6) < 0.001  # 60% improvement
        assert len(comparison.statistical_tests) > 0

        # At least one test should be significant
        assert any(test.is_significant for test in comparison.statistical_tests)

    def test_multi_metric_trend_analysis(self, analytics, historical_evaluation_data):
        """Test trend analysis across multiple metrics simultaneously."""
        # Test multiple metrics in one analysis
        metrics = ["win_rate", "avg_game_length"]
        trends = {}

        for metric in metrics:
            trend = analytics.analyze_trends(historical_evaluation_data, metric)
            trends[metric] = trend
            assert trend.metric_name == metric
            assert isinstance(trend.data_points, int)

        # Should have analyzed both metrics
        assert len(trends) == 2
        assert "win_rate" in trends
        assert "avg_game_length" in trends

    def test_cross_session_comparison_workflow(self, analytics):
        """Test comparison workflow across multiple evaluation sessions."""
        # Create multiple session results
        sessions = []
        for session_id in range(3):
            games = []
            win_rate = 0.3 + (session_id * 0.2)  # Improving sessions

            for i in range(20):
                game = Mock(spec=GameResult)
                game.is_agent_win = i < (win_rate * 20)
                game.moves_count = 50 + session_id * 5
                game.duration_seconds = (
                    30.0 + session_id * 2
                )  # Changed from game_length_seconds
                # Add other potentially missing attributes for SummaryStats
                game.is_opponent_win = i >= (win_rate * 20)
                game.is_draw = False
                games.append(game)

            summary_stats = SummaryStats.from_games(games)
            result = EvaluationResult(
                context=Mock(),
                games=games,
                summary_stats=summary_stats,
                analytics_data={},
            )
            sessions.append(result)

        # Compare first and last session
        comparison = analytics.compare_performance(
            baseline_results=sessions[0].games,
            comparison_results=sessions[-1].games,
            baseline_name="Initial Session",
            comparison_name="Final Session",
        )

        assert comparison.baseline_name == "Initial Session"
        assert comparison.comparison_name == "Final Session"
        assert comparison.win_rate_difference > 0  # Should show improvement

    def test_comprehensive_report_with_all_features(
        self,
        analytics,
        sample_evaluation_result,
        baseline_evaluation_result,
        historical_evaluation_data,
    ):
        """Test generating a comprehensive report with all analytics features."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "comprehensive_report.json"

            report = analytics.generate_automated_report(
                current_results=sample_evaluation_result,
                baseline_results=baseline_evaluation_result,
                historical_data=historical_evaluation_data,
                output_file=output_file,
            )

            # Verify comprehensive report structure
            expected_sections = [
                "report_metadata",
                "current_performance",
                "advanced_metrics",
                "performance_comparison",
                "trend_analysis",
                "insights_and_recommendations",
            ]

            for section in expected_sections:
                assert section in report, f"Missing section: {section}"

            # Verify file was created
            assert output_file.exists()

            # Verify insights were generated
            insights = report["insights_and_recommendations"]
            assert isinstance(insights, list)
            assert len(insights) > 0

    def test_memory_efficiency_large_dataset(self, analytics):
        """Test memory efficiency with large analytics datasets."""
        # Create large historical dataset
        large_historical_data = []
        base_time = datetime.now() - timedelta(days=365)

        for day in range(365):  # One year of data
            timestamp = base_time + timedelta(days=day)

            # Create minimal games to reduce memory overhead
            games = []
            for i in range(5):  # Smaller games per day
                game = Mock(spec=GameResult)
                game.is_agent_win = i < 3  # 60% win rate
                game.moves_count = 50
                game.duration_seconds = 30.0  # Changed from game_length_seconds
                # Add other potentially missing attributes for SummaryStats
                game.is_opponent_win = i >= 3
                game.is_draw = False
                games.append(game)

            summary_stats = SummaryStats.from_games(games)
            result = EvaluationResult(
                context=Mock(),
                games=games,
                summary_stats=summary_stats,
                analytics_data={},
            )
            large_historical_data.append((timestamp, result))

        # Should handle large dataset without memory issues
        trend = analytics.analyze_trends(large_historical_data, "win_rate")
        assert trend.data_points == 365
        assert isinstance(trend.trend_direction, str)

    def test_analytics_pipeline_resilience(self, analytics):
        """Test analytics pipeline resilience to data anomalies."""
        # Create dataset with various edge cases
        anomalous_data = []
        base_time = datetime.now() - timedelta(days=30)

        scenarios = [
            {"games": 0, "description": "No games"},
            {"games": 1, "description": "Single game"},
            {"games": 100, "description": "Many games"},
        ]

        for i, scenario in enumerate(scenarios):
            timestamp = base_time + timedelta(days=i * 10)

            games = []
            for j in range(scenario["games"]):
                game = Mock(spec=GameResult)
                game.is_agent_win = j % 2 == 0  # 50% win rate
                game.moves_count = 50
                game.duration_seconds = 30.0  # Changed from game_length_seconds
                # Add other potentially missing attributes for SummaryStats
                game.is_opponent_win = j % 2 != 0
                game.is_draw = False
                games.append(game)

            if games:  # Only create result if we have games
                summary_stats = SummaryStats.from_games(games)
                result = EvaluationResult(
                    context=Mock(),
                    games=games,
                    summary_stats=summary_stats,
                    analytics_data={},
                )
                anomalous_data.append((timestamp, result))

        # Should handle anomalous data gracefully
        if anomalous_data:
            trend = analytics.analyze_trends(anomalous_data, "win_rate")
            assert isinstance(trend.trend_direction, str)
            assert trend.data_points == len(anomalous_data)
