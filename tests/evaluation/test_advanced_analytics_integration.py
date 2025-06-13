"""
Integration tests for AdvancedAnalytics components - comprehensive coverage.

This module provides full integration test coverage for the AdvancedAnalytics 
system including trend analysis, performance comparison, statistical testing,
and automated reporting functionality.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from keisei.evaluation.core import EvaluationResult, GameResult
from keisei.evaluation.core.evaluation_result import SummaryStats
from keisei.evaluation.analytics.advanced_analytics import (
    AdvancedAnalytics,
    PerformanceComparison,
    StatisticalTest,
    TrendAnalysis,
)


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
        game.game_length_seconds = 30.0 + (i * 2.0)
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
        avg_game_length=sum(game.moves_count for game in sample_game_results) / total_games,
        total_moves=sum(game.moves_count for game in sample_game_results),
        avg_duration_seconds=sum(game.game_length_seconds for game in sample_game_results) / total_games,
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
        game.game_length_seconds = 35.0 + (i * 1.5)
        game.timestamp = datetime.now() - timedelta(hours=i + 20)
        results.append(game)
    
    return results


@pytest.fixture
def historical_evaluation_data(sample_evaluation_result):
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
            game.game_length_seconds = 30.0 + (j % 15)
            games.append(game)
        
        result = EvaluationResult(
            context=context,
            games=games,
            summary_stats=summary_stats,
            analytics_data={},
        )
        
        historical_data.append((timestamp, result))
    
    return historical_data


class TestAdvancedAnalyticsCore:
    """Test core AdvancedAnalytics functionality."""

    def test_initialization_default_params(self):
        """Test AdvancedAnalytics initialization with default parameters."""
        analytics = AdvancedAnalytics()
        
        assert analytics.significance_level == 0.05
        assert analytics.min_practical_difference == 0.05
        assert analytics.trend_window_days == 30

    def test_initialization_custom_params(self):
        """Test AdvancedAnalytics initialization with custom parameters."""
        analytics = AdvancedAnalytics(
            significance_level=0.01,
            min_practical_difference=0.1,
            trend_window_days=60,
        )
        
        assert analytics.significance_level == 0.01
        assert analytics.min_practical_difference == 0.1
        assert analytics.trend_window_days == 60

    def test_compare_performance_basic(self, analytics, sample_game_results, baseline_game_results):
        """Test basic performance comparison functionality."""
        comparison = analytics.compare_performance(
            baseline_results=baseline_game_results,
            comparison_results=sample_game_results,
            baseline_name="Baseline",
            comparison_name="Current",
        )
        
        assert isinstance(comparison, PerformanceComparison)
        assert comparison.baseline_name == "Baseline"
        assert comparison.comparison_name == "Current"
        
        # Should show improvement (60% vs 40%)
        assert comparison.win_rate_difference > 0
        assert comparison.win_rate_change_percent > 0
        
        # Should have statistical tests
        assert isinstance(comparison.statistical_tests, list)
        assert len(comparison.statistical_tests) > 0
        
        # Should have confidence interval
        assert isinstance(comparison.confidence_interval, tuple)
        assert len(comparison.confidence_interval) == 2
        
        # Should have recommendation
        assert isinstance(comparison.recommendation, str)
        assert len(comparison.recommendation) > 0

    def test_compare_performance_empty_baseline(self, analytics, sample_game_results):
        """Test performance comparison with empty baseline."""
        comparison = analytics.compare_performance(
            baseline_results=[],
            comparison_results=sample_game_results,
        )
        
        assert comparison.win_rate_difference == 0.6  # 60% vs 0%
        assert comparison.win_rate_change_percent == float("inf")

    def test_compare_performance_empty_comparison(self, analytics, baseline_game_results):
        """Test performance comparison with empty comparison results."""
        comparison = analytics.compare_performance(
            baseline_results=baseline_game_results,
            comparison_results=[],
        )
        
        assert comparison.win_rate_difference == -0.4  # 0% vs 40%

    def test_analyze_trends_insufficient_data(self, analytics):
        """Test trend analysis with insufficient data."""
        historical_data = [
            (datetime.now() - timedelta(days=1), Mock()),
            (datetime.now(), Mock()),
        ]
        
        trend = analytics.analyze_trends(historical_data, "win_rate")
        
        assert isinstance(trend, TrendAnalysis)
        assert trend.metric_name == "win_rate"
        assert trend.trend_direction == "insufficient_data"
        assert trend.trend_strength == 0.0
        assert trend.data_points == 2
        assert trend.time_span_days == 0.0

    def test_analyze_trends_win_rate(self, analytics, historical_evaluation_data):
        """Test trend analysis for win rate metric."""
        trend = analytics.analyze_trends(historical_evaluation_data, "win_rate")
        
        assert isinstance(trend, TrendAnalysis)
        assert trend.metric_name == "win_rate"
        assert trend.trend_direction in ["increasing", "decreasing", "stable"]
        assert 0 <= trend.trend_strength <= 1
        assert trend.data_points == len(historical_evaluation_data)
        assert trend.time_span_days > 0
        
        # Should detect increasing trend due to our test data
        assert trend.trend_direction == "increasing"
        assert trend.slope > 0

    def test_analyze_trends_game_length(self, analytics, historical_evaluation_data):
        """Test trend analysis for game length metric."""
        trend = analytics.analyze_trends(historical_evaluation_data, "avg_game_length")
        
        assert trend.metric_name == "avg_game_length"
        assert trend.trend_direction == "increasing"  # Our test data increases
        assert trend.slope > 0

    def test_analyze_trends_unknown_metric(self, analytics, historical_evaluation_data):
        """Test trend analysis with unknown metric."""
        with patch('keisei.evaluation.analytics.advanced_analytics.logger') as mock_logger:
            trend = analytics.analyze_trends(historical_evaluation_data, "unknown_metric")
            
            assert trend.metric_name == "unknown_metric"
            # Should still work but with default values
            mock_logger.warning.assert_called()

    def test_analyze_trends_with_scipy(self, analytics, historical_evaluation_data):
        """Test trend analysis with scipy available."""
        with patch('keisei.evaluation.analytics.advanced_analytics.SCIPY_AVAILABLE', True):
            with patch('keisei.evaluation.analytics.advanced_analytics.scipy_stats') as mock_scipy:
                mock_scipy.linregress.return_value = (0.1, 0.5, 0.8, 0.01, 0.02)
                
                trend = analytics.analyze_trends(historical_evaluation_data, "win_rate")
                
                assert trend.slope == 0.1
                assert abs(trend.r_squared - 0.64) < 0.01  # 0.8^2

    def test_analyze_trends_without_scipy(self, analytics, historical_evaluation_data):
        """Test trend analysis without scipy (fallback implementation)."""
        with patch('keisei.evaluation.analytics.advanced_analytics.SCIPY_AVAILABLE', False):
            trend = analytics.analyze_trends(historical_evaluation_data, "win_rate")
            
            # Should still work with basic linear regression
            assert isinstance(trend.slope, float)
            assert isinstance(trend.r_squared, float)
            assert 0 <= trend.r_squared <= 1


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

    def test_generate_automated_report_with_baseline(self, analytics, sample_evaluation_result):
        """Test automated report generation with baseline comparison."""
        # Create baseline result
        baseline_context = Mock()
        baseline_games = []
        for i in range(5):
            game = Mock(spec=GameResult)
            game.is_agent_win = i < 2  # 40% win rate
            game.moves_count = 45 + i
            baseline_games.append(game)
        
        baseline_stats = SummaryStats(
            total_games=5,
            agent_wins=2,
            opponent_wins=3,
            draws=0,
            win_rate=0.4,
            loss_rate=0.6,
            draw_rate=0.0,
            avg_game_length=47,
            total_moves=235,
            avg_duration_seconds=30.0,
        )
        
        baseline_result = EvaluationResult(
            context=baseline_context,
            games=baseline_games,
            summary_stats=baseline_stats,
            analytics_data={},
        )
        
        report = analytics.generate_automated_report(
            current_results=sample_evaluation_result,
            baseline_results=baseline_result,
        )
        
        # Should include performance comparison section
        assert "performance_comparison" in report
        comparison = report["performance_comparison"]
        assert "win_rate_difference" in comparison
        assert "statistical_tests" in comparison
        assert "recommendation" in comparison

    def test_generate_automated_report_with_historical_data(self, analytics, sample_evaluation_result, historical_evaluation_data):
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

    def test_generate_automated_report_with_file_output(self, analytics, sample_evaluation_result):
        """Test automated report generation with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "test_report.json"
            
            report = analytics.generate_automated_report(
                current_results=sample_evaluation_result,
                output_file=output_file,
            )
            
            # Should create file
            assert output_file.exists()
            
            # Should contain valid JSON
            with open(output_file, 'r') as f:
                saved_report = json.load(f)
            
            assert saved_report == report

    def test_generate_automated_report_file_error(self, analytics, sample_evaluation_result):
        """Test automated report generation with file write error."""
        # Try to write to invalid path
        invalid_path = Path("/invalid/path/report.json")
        
        with patch('keisei.evaluation.analytics.advanced_analytics.logger') as mock_logger:
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
        assert "performance" in insight_text.lower() or "win rate" in insight_text.lower()

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
        assert "below average" in insight_text.lower() or "adjustments" in insight_text.lower()


class TestAdvancedAnalyticsStatisticalTests:
    """Test statistical analysis functionality."""

    def test_two_proportion_z_test(self, analytics):
        """Test two-proportion z-test implementation."""
        # Use reflection to access private method for testing
        z_test = analytics._two_proportion_z_test(60, 100, 40, 100)
        
        assert isinstance(z_test, StatisticalTest)
        assert z_test.test_name == "two_proportion_z_test"
        assert isinstance(z_test.p_value, float)
        assert isinstance(z_test.is_significant, bool)
        assert isinstance(z_test.interpretation, str)

    def test_mann_whitney_test_with_scipy(self, analytics):
        """Test Mann-Whitney U test with scipy available."""
        with patch('keisei.evaluation.analytics.advanced_analytics.SCIPY_AVAILABLE', True):
            with patch('keisei.evaluation.analytics.advanced_analytics.scipy_stats') as mock_scipy:
                # Return tuple instead of Mock with attributes
                mock_scipy.mannwhitneyu.return_value = (150, 0.03)
                
                test = analytics._mann_whitney_test([1, 2, 3, 4, 5], [3, 4, 5, 6, 7], "test_metric")
                
                assert test.test_name == "mann_whitney_test_metric"
                assert test.p_value == 0.03
                assert test.is_significant is True  # p < 0.05

    def test_mann_whitney_test_without_scipy(self, analytics):
        """Test Mann-Whitney U test fallback without scipy."""
        with patch('keisei.evaluation.analytics.advanced_analytics.SCIPY_AVAILABLE', False):
            test = analytics._mann_whitney_test([1, 2, 3], [4, 5, 6], "test_metric")
            
            assert test.test_name == "mann_whitney_test_metric"
            assert test.interpretation == "Test skipped (scipy not available)"

    def test_confidence_interval_calculation(self, analytics):
        """Test confidence interval calculation for win rate difference."""
        ci = analytics._calculate_win_rate_difference_ci(60, 100, 40, 100)
        
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]  # Lower bound < upper bound

    def test_confidence_interval_edge_cases(self, analytics):
        """Test confidence interval calculation with edge cases."""
        # Zero baseline
        ci = analytics._calculate_win_rate_difference_ci(0, 100, 40, 100)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        
        # Zero comparison
        ci = analytics._calculate_win_rate_difference_ci(60, 100, 0, 100)
        assert isinstance(ci, tuple)
        assert len(ci) == 2

    def test_performance_recommendation_generation(self, analytics):
        """Test performance recommendation generation."""
        # Test with significant improvement
        tests = [
            StatisticalTest(
                test_name="Test",
                statistic=1.96,
                p_value=0.01,
                is_significant=True,
                confidence_level=0.95,
                interpretation="Significant",
                effect_size=0.6,
            )
        ]
        
        recommendation = analytics._generate_performance_recommendation(
            win_rate_difference=0.1,  # 10% improvement
            statistical_tests=tests,
            practical_significance=True,
        )
        
        assert isinstance(recommendation, str)
        assert len(recommendation) > 0
        assert "improvement" in recommendation.lower()

    def test_performance_recommendation_no_significance(self, analytics):
        """Test performance recommendation with no statistical significance."""
        tests = [
            StatisticalTest(
                test_name="Test",
                statistic=0.5,
                p_value=0.8,
                is_significant=False,
                confidence_level=0.95,
                interpretation="Not significant",
            )
        ]
        
        recommendation = analytics._generate_performance_recommendation(
            win_rate_difference=0.01,  # Small change
            statistical_tests=tests,
            practical_significance=False,
        )
        
        assert "no statistically significant" in recommendation.lower()


class TestAdvancedAnalyticsIntegration:
    """Integration tests combining multiple features."""

    def test_complete_analysis_workflow(self, analytics, sample_evaluation_result, baseline_game_results, historical_evaluation_data):
        """Test complete analysis workflow with all features."""
        # Create baseline result
        baseline_context = Mock()
        baseline_stats = SummaryStats(
            total_games=len(baseline_game_results),
            agent_wins=sum(1 for g in baseline_game_results if g.is_agent_win),
            opponent_wins=len(baseline_game_results) - sum(1 for g in baseline_game_results if g.is_agent_win),
            draws=0,
            win_rate=sum(1 for g in baseline_game_results if g.is_agent_win) / len(baseline_game_results),
            loss_rate=(len(baseline_game_results) - sum(1 for g in baseline_game_results if g.is_agent_win)) / len(baseline_game_results),
            draw_rate=0.0,
            avg_game_length=sum(g.moves_count for g in baseline_game_results) / len(baseline_game_results),
            total_moves=sum(g.moves_count for g in baseline_game_results),
            avg_duration_seconds=sum(g.game_length_seconds for g in baseline_game_results) / len(baseline_game_results),
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
            assert report["performance_comparison"]["win_rate_difference"] > 0  # Should show improvement
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
        report = analytics.generate_automated_report(current_results=sample_evaluation_result)
        
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
            game.game_length_seconds = 30.0
            baseline_results.append(game)
        
        # Comparison: 80% win rate  
        for i in range(100):
            game = Mock(spec=GameResult)
            game.is_agent_win = i < 80
            game.moves_count = 50
            game.game_length_seconds = 30.0
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
