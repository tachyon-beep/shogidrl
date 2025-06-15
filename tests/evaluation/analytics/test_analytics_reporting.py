"""Analytics reporting functionality tests."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from keisei.evaluation.core import EvaluationResult, GameResult
from keisei.evaluation.core.evaluation_result import SummaryStats
from .conftest import create_mock_evaluation_result


class TestAdvancedAnalyticsReporting:
    """Test automated reporting functionality."""

    def test_generate_automated_report_basic(self, analytics, sample_evaluation_result):
        """Test basic automated report generation."""
        report = analytics.generate_automated_report(
            current_results=sample_evaluation_result
        )

        assert isinstance(report, dict)

        # Check required sections
        assert "report_metadata" in report
        assert "current_performance" in report
        assert "advanced_metrics" in report
        assert "insights_and_recommendations" in report

        # Check metadata
        metadata = report["report_metadata"]
        assert "generated_at" in metadata
        assert "analysis_type" in metadata
        assert "keisei_version" in metadata

        # Check current performance
        perf = report["current_performance"]
        assert perf["total_games"] == 10
        assert perf["win_rate"] == 0.6
        assert perf["wins"] == 6
        assert perf["losses"] == 4
        assert perf["draws"] == 0
        assert perf["avg_game_length"] > 0

    def test_generate_automated_report_with_baseline(
        self, analytics, sample_evaluation_result, baseline_evaluation_result
    ):
        """Test automated report generation with baseline comparison."""
        report = analytics.generate_automated_report(
            current_results=sample_evaluation_result,
            baseline_results=baseline_evaluation_result,
        )

        # Should include performance comparison section
        assert "performance_comparison" in report
        comparison = report["performance_comparison"]
        assert "win_rate_difference" in comparison
        assert "statistical_tests" in comparison
        assert "recommendation" in comparison

        # Should show improvement from 40% to 60%
        assert comparison["win_rate_difference"] > 0

    def test_generate_automated_report_with_historical_data(
        self, analytics, sample_evaluation_result, historical_evaluation_data
    ):
        """Test automated report generation with historical trend analysis."""
        report = analytics.generate_automated_report(
            current_results=sample_evaluation_result,
            historical_data=historical_evaluation_data,
        )

        # Should include trend analysis section
        assert "trend_analysis" in report
        trend_data = report["trend_analysis"]
        assert "win_rate_trend" in trend_data
        assert "game_length_trend" in trend_data

        # Check trend structure
        win_rate_trend = trend_data["win_rate_trend"]
        assert "direction" in win_rate_trend
        assert "strength" in win_rate_trend
        assert "r_squared" in win_rate_trend

    def test_generate_automated_report_with_file_output(
        self, analytics, sample_evaluation_result, temp_output_dir
    ):
        """Test automated report generation with file output."""
        output_file = temp_output_dir / "test_report.json"

        report = analytics.generate_automated_report(
            current_results=sample_evaluation_result,
            output_file=output_file,
        )

        # Should create file
        assert output_file.exists()

        # Should contain valid JSON
        with open(output_file, "r") as f:
            saved_report = json.load(f)

        assert saved_report == report

    def test_generate_automated_report_file_error(
        self, analytics, sample_evaluation_result
    ):
        """Test automated report generation with file write error."""
        # Try to write to invalid path
        invalid_path = Path("/invalid/path/report.json")

        with patch(
            "keisei.evaluation.analytics.advanced_analytics.logger"
        ) as mock_logger:
            report = analytics.generate_automated_report(
                current_results=sample_evaluation_result,
                output_file=invalid_path,
            )

            # Should still return report but log error
            assert isinstance(report, dict)
            mock_logger.error.assert_called()

    def test_automated_insights_generation(self, analytics, sample_evaluation_result):
        """Test automated insights generation."""
        report = analytics.generate_automated_report(
            current_results=sample_evaluation_result
        )

        insights = report["insights_and_recommendations"]
        assert isinstance(insights, list)
        assert len(insights) > 0

        # Should contain performance insights
        insight_text = " ".join(insights)
        assert (
            "performance" in insight_text.lower() or "win rate" in insight_text.lower()
        )

    def test_insights_excellent_performance(self, analytics):
        """Test insights for excellent performance (>70% win rate)."""
        # Create high-performance result
        games = []
        for i in range(10):
            game = Mock(spec=GameResult)
            game.is_agent_win = i < 8  # 80% win rate
            game.moves_count = 50
            game.game_length_seconds = 30.0
            games.append(game)

        context = Mock()
        summary_stats = SummaryStats(
            total_games=10,
            agent_wins=8,
            opponent_wins=2,
            draws=0,
            win_rate=0.8,
            loss_rate=0.2,
            draw_rate=0.0,
            avg_game_length=50,
            total_moves=500,
            avg_duration_seconds=30.0,
        )

        result = EvaluationResult(
            context=context,
            games=games,
            summary_stats=summary_stats,
            analytics_data={},
        )

        report = analytics.generate_automated_report(current_results=result)
        insights = report["insights_and_recommendations"]

        # Should contain excellent performance insight
        insight_text = " ".join(insights)
        assert "excellent" in insight_text.lower() or "70%" in insight_text

    def test_insights_poor_performance(self, analytics):
        """Test insights for poor performance (<45% win rate)."""
        # Create low-performance result
        games = []
        for i in range(10):
            game = Mock(spec=GameResult)
            game.is_agent_win = i < 3  # 30% win rate
            game.moves_count = 50
            game.game_length_seconds = 30.0
            games.append(game)

        context = Mock()
        summary_stats = SummaryStats(
            total_games=10,
            agent_wins=3,
            opponent_wins=7,
            draws=0,
            win_rate=0.3,
            loss_rate=0.7,
            draw_rate=0.0,
            avg_game_length=50,
            total_moves=500,
            avg_duration_seconds=30.0,
        )

        result = EvaluationResult(
            context=context,
            games=games,
            summary_stats=summary_stats,
            analytics_data={},
        )

        report = analytics.generate_automated_report(current_results=result)
        insights = report["insights_and_recommendations"]

        # Should contain performance warning
        insight_text = " ".join(insights)
        assert (
            "below average" in insight_text.lower()
            or "adjustments" in insight_text.lower()
        )

    def test_report_metadata_completeness(self, analytics, sample_evaluation_result):
        """Test that report metadata is comprehensive and accurate."""
        report = analytics.generate_automated_report(
            current_results=sample_evaluation_result
        )

        metadata = report["report_metadata"]
        required_fields = [
            "generated_at",
            "analysis_type", 
            "keisei_version",
            "analytics_config"
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"

        # Verify timestamp format
        assert isinstance(metadata["generated_at"], str)
        assert "T" in metadata["generated_at"]  # ISO format

    def test_report_performance_metrics_calculation(self, analytics):
        """Test accurate calculation of performance metrics in reports."""
        # Create precisely controlled game results
        games = []
        for i in range(20):
            game = Mock(spec=GameResult)
            game.is_agent_win = i < 12  # Exactly 60% win rate
            game.moves_count = 50 + i  # Incrementing game length
            game.duration_seconds = 30.0 + (i * 2) # Changed from game_length_seconds
            # Add other potentially missing attributes for SummaryStats
            game.is_opponent_win = i >= 12
            game.is_draw = False
            games.append(game)
    
        summary_stats = SummaryStats.from_games(games)
        result = EvaluationResult(
            context=Mock(),
            games=games,
            summary_stats=summary_stats,
            analytics_data={},
        )

        report = analytics.generate_automated_report(current_results=result)
        perf = report["current_performance"]

        # Verify exact calculations
        assert perf["total_games"] == 20
        assert perf["wins"] == 12
        assert perf["losses"] == 8
        assert perf["draws"] == 0
        assert abs(perf["win_rate"] - 0.6) < 0.001  # 60% exactly

    def test_insights_context_sensitivity(self, analytics):
        """Test that insights are contextually appropriate."""
        # Test with moderate performance (should give balanced insights)
        result = create_mock_evaluation_result(
            win_rate=0.55,  # Moderate performance
            total_games=20,
            session_id="context_test"
        )

        report = analytics.generate_automated_report(current_results=result)
        insights = report["insights_and_recommendations"]

        assert len(insights) > 0
        insight_text = " ".join(insights).lower()
        
        # Should not contain extreme language for moderate performance
        assert "excellent" not in insight_text
        assert "poor" not in insight_text
        assert "terrible" not in insight_text

        # Should contain balanced assessment
        assert any(word in insight_text for word in ["moderate", "average", "balanced", "reasonable"])
